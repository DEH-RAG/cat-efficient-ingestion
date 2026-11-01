"""Efficient re-embed engine (owned by the efficient_ingestion plugin).

Implements ``EfficientIngestionEngine`` — the replaceable, more efficient
implementation of the core ``BaseIngestionEngine``. It is the ONE phase machine
for the whole ingestion lifecycle: fresh uploads, recovery, re-embed and
re-ingest all go through the same two-phase flow, driven by the
ingestion-status doc (see :mod:`registry`):

  - ``parsing_chunking``: clean-sweep of every artifact produced by this phase
    AND the following ones (text chunk points, image points, saved image files),
    then parse + chunk + ALWAYS extract the images (via the cat-multimodal-ingestion
    plugin) and store the text chunks AND image points with EMPTY vectors
    (``vector={}``). No embedding is computed here.
  - ``embedding``: recompute the embeddings of the stored text chunks and image
    points (``embed_documents`` / ``embed_images``), replace them, then mark the
    source ``completed``.

Recovering/resuming in a phase deletes its artifacts (and the ones of the later
phases) and restarts from the beginning of that phase.

Status is written through the plugin's own ``registry``. Import-safe: nothing
runs at import time.

URL sources are fetched ONCE: the resolved bytes are persisted in the agent's
file storage under a deterministic, sanitized name derived from the URL, and a
later run reads that copy back instead of re-downloading. This makes a
re-ingest deterministic (a URL whose content changes no longer silently changes
what gets indexed) and lets other components (e.g. CATALOG's card) read the
original content. The core source resolver remains a FALLBACK when the
persisted copy is missing (e.g. it was deleted manually), and that fallback is
logged because the fetched content may have changed.

Retention: the persisted bytes occupy the agent's file storage like any other
uploaded file. There is NO automatic eviction here — they live until the file is
deleted (or the agent is removed), so operators should account for their size.
"""

import asyncio
import copy
import hashlib
import io
import math
import mimetypes
import os
import time
import uuid
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Optional
from urllib.parse import urlparse

from langchain_core.documents import Document
from langchain_core.documents.base import Blob
from redis.exceptions import LockError

from cat.core_plugins.base_plugin.parsers import MimeTypeBasedParser
from cat.db.cruds import settings as crud_settings
from cat.env import get_env_int
from cat.exceptions import CustomHookVetoException, CustomNotFoundException
from cat.log import log
from cat.looking_glass.models import StoredSourceWithMetadata
from cat.services.factory.embedder import is_multimodal_embedder
from cat.services.factory.ingestion import BaseIngestionEngine
from cat.routes.routes_utils import sanitize_source_name
from cat.services.memory.models import PointStruct, VectorMemoryType
from cat.utils import get_nlp_object_name, guess_file_type, is_url

from cat.core_plugins.cat_multimodal_ingestion.ingestion import (
    collect_document_images,
    image_file_name,
    strip_image_payload,
)

from .hooks import (
    execute_phase_accumulator,
    execute_phase_hook_on_all,
    execute_shared_hook,
    phase_settings_key,
)
from .ingestion_executor import run_in_ingestion_executor
from .phases import (
    compute_tags,
    dependency_outcome,
    execute_phase_on_owner,
    get_phases,
    has_live_running_phase,
    merged_phases,
)
from .registry import (
    PHASE_EMBEDDING,
    PHASE_PARSING_CHUNKING,
    PHASE_STATE_COMPLETE,
    PHASE_STATE_ERROR,
    PHASE_STATE_SKIPPED,
    IngestionStatus,
    backfill_completed_phases,
    claim_source_for_resume,
    derive_source_status,
    get_completed_phases,
    get_status,
    ingestion_canceled,
    phase_lock,
    phase_state,
    record_phase_completed,
    record_phase_error,
    record_phase_processing,
    record_phase_skipped,
    reset_phase_entries,
    set_phase,
    set_status,
    should_abandon,
    source_lock,
)
from .split import split_oversized

# Time budget for one external phase to report ``{"status": "not_ready"}``.
# Unlike a retry COUNT, this is a wall-clock bound: a phase that is legitimately
# waiting for upstream work (e.g. the MyGRAPH extraction phase waiting for the
# embedded chunks to land) may legitimately stay pending for minutes. Only a
# row still not ready after this generous bound is marked ERROR. Overridable
# with ``CAT_INGESTION_NOT_READY_MAX_SECONDS`` (default 3600).
_NOT_READY_MAX_SECONDS_DEFAULT = 3600.0
_NOT_READY_RETRY_SECONDS_DEFAULT = 5.0

#: Status of a row the machine must never rewrite.
_TERMINAL_STATUSES = (IngestionStatus.COMPLETED.value, IngestionStatus.ERROR.value)


class _NothingStored(Exception):
    """Signal that a phase body produced no storeable chunk.

    Raised BEFORE any destructive write so old points stay intact. The
    dispatcher treats it exactly like an external ``not_ready`` response:
    the row stays non-terminal and is retried until the time budget expires.
    """

    def __init__(self, reason: str, retry_after: float = _NOT_READY_RETRY_SECONDS_DEFAULT):
        super().__init__(reason)
        self.reason = reason
        self.retry_after = retry_after


def _not_ready_max_seconds() -> float:
    """Wall-clock budget a row may stay ``not_ready`` before it becomes ERROR.

    ``CAT_INGESTION_NOT_READY_MAX_SECONDS`` (default 3600). An explicit ``0``
    means "expire immediately" (used by tests); unset/negative falls back to
    the default.
    """
    value = get_env_int("CAT_INGESTION_NOT_READY_MAX_SECONDS")
    if value is not None and value >= 0:
        return float(value)
    return _NOT_READY_MAX_SECONDS_DEFAULT


def _not_ready_retry_after(result) -> float:
    """Delay before re-running a not-ready phase (from the hook's ``retry_after``)."""
    value = (result or {}).get("retry_after") if isinstance(result, dict) else None
    try:
        retry = float(value)
    except (TypeError, ValueError):
        retry = _NOT_READY_RETRY_SECONDS_DEFAULT
    return retry if retry > 0 else _NOT_READY_RETRY_SECONDS_DEFAULT


async def _wait_not_ready(
    ccat,
    scope: str,
    source_name: str,
    phase: str,
    chat_id,
    reason: str,
    retry_after: float,
) -> bool:
    """Keep a not-ready row NON-TERMINAL and bounded by wall-clock time.

    Refreshes the row's ``phase_retry`` hint (``status`` stays ``processing``,
    ``phase`` unchanged) so the sweep can resume it, and only marks ERROR once
    the elapsed time since the FIRST wait exceeds the configured budget. Logs
    INFO on the first wait and WARNING exactly when the bound is exceeded.

    Returns ``True`` when the caller should wait and retry, ``False`` when the
    row is terminal (never rewritten) or has just been marked ERROR.
    """
    doc = await get_status(ccat.agent_key, scope, source_name)
    if doc is not None and doc.get("status") in _TERMINAL_STATUSES:
        # never overwrite a terminal row (e.g. another worker finished it)
        return False

    now = time.time()
    hint = (doc or {}).get("phase_retry") if isinstance(doc, dict) else None
    hint = hint if isinstance(hint, dict) else {}
    try:
        since = float(hint.get("since"))
    except (TypeError, ValueError):
        since = None
    first_wait = hint.get("phase") != phase or since is None
    if first_wait:
        # first wait for THIS phase: restart the clock and log once
        since = now
        attempts = 1
        log.info(
            f"Agent id: {ccat._id}. Phase {phase} for {source_name} is not ready; "
            f"keeping it pending (retry every {retry_after:.0f}s, max "
            f"{_not_ready_max_seconds():.0f}s)"
        )
    else:
        try:
            attempts = int(hint.get("attempts") or 0) + 1
        except (TypeError, ValueError):
            attempts = 1

    elapsed = max(0.0, now - since)
    bound = _not_ready_max_seconds()
    if elapsed >= bound:
        msg = (
            f"phase {phase} not ready after {elapsed:.0f}s (max {bound:.0f}s): {reason}"
        )
        log.warning(f"Agent id: {ccat._id}. Source {source_name}: {msg}")
        await _set_status(ccat, source_name, IngestionStatus.ERROR, error=msg, chat_id=chat_id)
        return False

    try:
        await set_status(
            ccat.agent_key,
            scope,
            source_name,
            type_="url" if is_url(source_name) else "file",
            status=IngestionStatus.PROCESSING,
            chat_id=chat_id,
            phase=phase,
            extra={
                "phase_retry": {
                    "phase": phase,
                    "since": since,
                    "attempts": attempts,
                    "retry_after": retry_after,
                    "reason": reason,
                }
            },
        )
    except Exception as e:  # noqa: BLE001 - a hint write must never break the machine
        log.warning(
            f"Agent id: {ccat._id}. Could not refresh the not-ready hint for "
            f"{source_name}: {e}"
        )
    return True


def _is_valid_embedding_vector(vector) -> bool:
    """Whether a vector is a non-empty, finite, non-zero L2-norm sequence.

    Mirrors the vector-handler guard: dense stores drop zero/non-finite vectors
    silently, so the embedding phase must detect them BEFORE the destructive
    replace and keep the source non-terminal when nothing valid remains.
    An empty mapping ``{}`` is the intentional payload-only vector and is NOT
    considered here (callers only pass vectors they actually computed).
    """
    if vector is None or isinstance(vector, dict):
        return False
    try:
        values = [float(v) for v in vector]
    except (TypeError, ValueError):
        return False
    if not values:
        return False
    if not all(math.isfinite(v) for v in values):
        return False
    return math.sqrt(sum(v * v for v in values)) > 1e-12


class _IngestionCanceled(Exception):
    """Internal signal: the agent was canceled (deleted) mid-phase.

    Raised by the parse/embed loops as soon as ``ingestion_canceled`` flips, so
    a long phase stops BEFORE it writes points instead of only at a status
    boundary. The dispatcher swallows it: a canceled agent gets no further
    status write (``set_status`` is itself guarded).
    """


async def _raise_if_canceled(ccat) -> None:
    """Raise :class:`_IngestionCanceled` when the agent must self-abort."""
    agent_key = getattr(ccat, "agent_key", None)
    if agent_key is not None and await ingestion_canceled(agent_key):
        raise _IngestionCanceled(f"agent {agent_key} canceled mid-phase")


async def _resolve_rabbit_hole(ccat):
    """Return a RabbitHole bound to ``ccat``, creating one when none exists.

    The app creates ``lizard.rabbit_hole`` during ``BillTheLizard.bootstrap``.
    A process that builds a CheshireCat outside the app (e.g. the maintenance
    CLI in a fresh ``docker exec``) only discovers the plugins, so
    ``ccat.rabbit_hole`` stays ``None``. Create a per-call instance instead of
    touching the shared singleton, which keeps concurrent per-agent parsing
    isolated (``_run_parsing_phase`` still copies it per call).
    """
    rabbit_hole = getattr(ccat, "rabbit_hole", None)
    if rabbit_hole is not None:
        return rabbit_hole
    from cat.rabbit_hole import RabbitHole

    rabbit_hole = RabbitHole()
    await rabbit_hole.setup(ccat)
    return rabbit_hole


async def _set_status(ccat, source: str, status: IngestionStatus, error: str | None = None, chat_id: str | None = None) -> None:
    """Best-effort status write via the plugin's own registry."""
    scope = str(chat_id) if chat_id else "agent"
    try:
        await set_status(
            ccat.agent_key,
            scope,
            source,
            type_="url" if is_url(source) else "file",
            status=status,
            chat_id=chat_id,
            error=error,
        )
    except Exception as e:  # noqa: BLE001 - status must never break the re-embed pass
        log.error(f"Agent id: {ccat._id}. Failed to write ingestion status for {source}: {e}")


def _claim_stale_after() -> float:
    """Staleness gate for the per-source claim during the re-embed pass.

    Reuses the resume threshold so a source being actively re-embedded by
    another worker is not re-claimed. ``<=0`` via env disables the gate.
    """
    value = get_env_int("CAT_INGESTION_RESUME_STALE_SECONDS")
    return float(value) if value and value > 0 else 0.0


def _heartbeat_interval() -> float:
    """Interval (seconds) at which an in-flight ingestion row is refreshed."""
    value = get_env_int("CAT_INGESTION_HEARTBEAT_SECONDS")
    return float(value) if value and value > 0 else 30.0


async def _start_heartbeat(ccat, source_name: str, scope: str, cat) -> None:
    """Start refreshing the claimed row while a batch/recovery phase runs.

    ``engine.run`` and the resume sweep call :func:`reembed_sources` directly
    (not through ``ingest_file``), so without this the row has no heartbeat and
    a phase longer than the boot-stale threshold (60 s) could be re-claimed by
    the next periodic sweep. Best-effort: a missing plugin manager (unit tests)
    or hook never breaks the pass.
    """
    plugin_manager = getattr(ccat, "plugin_manager", None)
    if plugin_manager is None:
        return
    try:
        await execute_shared_hook(
            cat,
            "rabbithole_processing_heartbeat_start",
            source_name, scope, _heartbeat_interval(),
        )
    except Exception as e:  # noqa: BLE001 - a heartbeat must never break ingestion
        log.warning(
            f"Agent id: {ccat._id}. Could not start the ingestion heartbeat "
            f"for {source_name}: {e}"
        )


async def _stop_heartbeat(ccat, source_name: str, scope: str, cat) -> None:
    """Cancel the heartbeat started by :func:`_start_heartbeat` (best-effort)."""
    plugin_manager = getattr(ccat, "plugin_manager", None)
    if plugin_manager is None:
        return
    try:
        await execute_shared_hook(
            cat,
            "rabbithole_processing_heartbeat_stop",
            source_name, scope,
        )
    except Exception as e:  # noqa: BLE001 - stopping must never break the pass
        log.warning(
            f"Agent id: {ccat._id}. Could not stop the ingestion heartbeat "
            f"for {source_name}: {e}"
        )


def _source_metadata(source_name: str, chat_id) -> dict:
    """Vector-memory filter for ONE source in ONE scope.

    The same file name can exist in several conversations (and in the agent
    KB). Every query/delete that targets a source MUST also target its
    ``chat_id`` when chat-scoped, otherwise ingesting one chat would read or
    delete another chat's chunks and card.
    """
    metadata = {"source": source_name}
    if chat_id:
        metadata["chat_id"] = chat_id
    return metadata


def _point_matches_source(point, source_name: str, chat_id) -> bool:
    """Whether a stored point belongs to ``source_name`` in the given scope."""
    meta = (point.payload or {}).get("metadata", {}) or {}
    if meta.get("source") != source_name:
        return False
    if chat_id:
        return str(meta.get("chat_id")) == str(chat_id)
    return meta.get("chat_id") in (None, "")


async def _source_has_stored_points(
    ccat, handler, collection_name, source_name: str, chat_id
) -> bool:
    """Store-grounded freshness probe for ONE source.

    The completed-phases diary is bookkeeping, NOT proof the points still
    exist: a wipe that leaves the diary intact would otherwise make every
    phase look fresh and permanently suppress re-embedding (the source keeps
    zero chunk points, the embedder is never called, and later phases such as
    ``catalog_cards`` fail forever).

    This asks the ACTIVE vector-memory handler whether the source still has at
    least one stored CHUNK point (catalogue cards excluded), with ONE bounded
    query per source. It is called ONLY when the diary already says fresh.

    Returns True (populated) whenever the store cannot be proven empty, so an
    unavailable/erroring handler never triggers a destructive re-parse.
    """
    count_source_chunk_points = getattr(handler, "count_source_chunk_points", None)
    try:
        if count_source_chunk_points is not None:
            count = await count_source_chunk_points(
                str(collection_name), source_name, chat_id=chat_id
            )
            return int(count or 0) > 0
        # Generic handlers have no catalogue cards: a bounded get (limit 1) is
        # enough to tell "some point exists" from "no point at all".
        points, _ = await handler.get_all_tenant_points(
            str(collection_name), limit=1,
            metadata=_source_metadata(source_name, chat_id),
            with_vectors=False,
        )
    except Exception as e:  # noqa: BLE001 - a probe error must never wipe
        log.warning(
            f"Agent id: {ccat._id}. Source {source_name}: store freshness probe "
            f"failed ({e}); assuming the stored points are present"
        )
        return True
    return bool(points)


async def _cleanup_orphan_images(ccat, collection_name, source_name, chat_id) -> None:
    """Remove image points + saved image files of a source before a full re-ingest.

    A full re-ingest (parsing_chunking) regenerates the chunks AND re-extracts
    the images from scratch. The image points and their saved ``image_file``
    from the PREVIOUS (possibly incomplete) parse are therefore stale: delete
    the points in the target first and the files from the agent storage, so the
    re-ingest starts clean and no orphan image file lingers on disk.
    """
    filter_ = {**_source_metadata(source_name, chat_id), "image": True}
    try:
        points, _ = await ccat.vector_memory_handler.get_all_tenant_points(
            str(collection_name), with_vectors=False,
            metadata=filter_,
        )
    except Exception:  # noqa: BLE001 - cleanup must never break the pass
        return
    image_files = [
        (p.payload or {}).get("metadata", {}).get("image_file")
        for p in points
        if (p.payload or {}).get("metadata", {}).get("image_file")
    ]
    if not image_files:
        return
    root_dir = ccat.agent_key
    if chat_id:
        root_dir = os.path.join(root_dir, str(chat_id))
    for image_file in image_files:
        try:
            ccat.file_manager.remove_file(os.path.join(root_dir, image_file))
        except Exception:  # noqa: BLE001,S110 - best-effort, cleanup must never break the pass
            pass
    await ccat.vector_memory_handler.delete_tenant_points(
        str(collection_name), metadata=filter_
    )


async def _resolve_callers(ccat, chat_id):
    """Return the RabbitHole/plugin-manager caller and the embedding cat for a source.

    Agent-scoped sources use the CheshireCat itself (scope ``"agent"``); a
    chat-scoped source needs its StrayCat for the correct ``scope``/hook caller.
    Returns ``(cat, scope, chat_id)`` where ``cat`` is what receives the
    ``after_rabbithole_stored_documents`` hook caller, or ``(None, None, None)``
    when the chat does not (or no longer) exist.
    """
    if not chat_id:
        return ccat, "agent", None
    stray_cat = await ccat._find_stray_cat(str(chat_id))
    if stray_cat is None:
        return None, str(chat_id), chat_id
    return stray_cat, str(chat_id), chat_id


async def _clear_source_artifacts(ccat, collection_name, source_name, chat_id) -> None:
    """Remove EVERY artifact of a source: text chunk points, image points and
    saved image files. Used when a phase is restarted (clean-sweep of the phase
    and all the later ones)."""
    # image files + image points first (the point metadata carries image_file)
    await _cleanup_orphan_images(ccat, collection_name, source_name, chat_id)
    # then any remaining point (text chunks and any leftover image point)
    await ccat.vector_memory_handler.delete_tenant_points(
        str(collection_name), metadata=_source_metadata(source_name, chat_id)
    )


async def _chunk_without_finalize(ccat, rabbit_hole, super_docs):
    """Chunk parsed docs WITHOUT applying the token-budget finalize here.

    The efficient phase applies the token-budget split exactly ONCE through the
    emitted ``before_rabbithole_stores_documents`` hook (after chunking). The
    core ``RabbitHole._split_text`` would also run ``finalize_oversized_chunks``,
    applying the split a second time, so the same chunking steps are invoked
    directly here (splits hook + chunker + short-chunk merge) and the finalize
    step is left to the store hook. The caller matches what the copied
    ``RabbitHole._split_text`` would use (``self.stray or self.cat`` -> ccat).
    """
    docs = await ccat.plugin_manager.execute_hook(
        "before_rabbithole_splits_documents", super_docs, caller=ccat,
    )
    docs = await ccat.chunker.split_documents(docs)
    try:
        docs = rabbit_hole._merge_short_chunks(docs, ccat.chunker)
    except Exception as e:  # noqa: BLE001 - mirrors the core fallback
        log.warning(f"Error merging short chunks: {e}. Proceeding with original chunks.")
    return docs


async def _parse_and_chunk(
    ccat, rabbit_hole, source, file_bytes, content_type, cat
):
    """Parse the source bytes into chunked LangChain docs, ALWAYS extracting images.

    Mirrors the core ``_parse_to_docs`` but collects the extracted images
    UNCONDITIONALLY (the cat-multimodal-ingestion plugin's pure helper reads the
    ``image_base64`` payload the parser attached), strips the transient payload
    from the docs, and returns ``(docs, images)``.
    """
    source_name = source.name
    fh = await ccat.file_handlers()
    super_docs = await run_in_ingestion_executor(
        lambda: MimeTypeBasedParser(handlers=fh).parse(
            Blob(data=file_bytes, mimetype=content_type).from_data(
                data=file_bytes, mime_type=content_type, path=source_name
            )
        )
    )
    for doc in super_docs:
        if isinstance(doc.metadata, dict):
            doc.metadata.setdefault("source", source_name)

    # ALWAYS extract the images (regardless of the active embedder): the
    # embedding phase decides later whether to embed them as images or keep
    # them payload-only.
    images = collect_document_images(super_docs)
    strip_image_payload(super_docs)

    docs = await _chunk_without_finalize(ccat, rabbit_hole, super_docs)
    return docs, images


async def _store_empty_vectors(
    ccat, collection_name, source_name, chat_id, docs, images,
    file_hash=None, metadata=None,
):
    """Store the parsed text chunks and image points with EMPTY vectors.

    Phase ``parsing_chunking`` output: the points carry the full payload (text
    page_content + image file references in metadata) but ``vector={}`` — no
    embedding is computed here. The image files are persisted via ``save_file``
    so the embedding phase can re-read them.

    Returns the list of stored ``PointStruct``.
    """
    points: list[PointStruct] = []

    for doc in docs:
        # A delete/cancel mid-parse stops before ANY point is written.
        await _raise_if_canceled(ccat)
        # enrich the metadata as the core store_documents does: source, when,
        # hash (deduped compute) and chat_id for chat-scoped sources.
        doc.metadata = (
            doc.metadata
            | (metadata or {})
            | {"source": source_name, "when": time.time(), "hash": file_hash}
            | ({"chat_id": chat_id} if chat_id else {})
        )
        # Honour a document-provided id (e.g. a deterministic card id) so a
        # re-ingest upserts instead of duplicating; plain chunks keep uuid4.
        points.append(
            PointStruct(id=doc.id or uuid.uuid4().hex, payload=doc.model_dump(), vector={})
        )

    if images:
        for idx, img in enumerate(images):
            await _raise_if_canceled(ccat)
            image_bytes = img["image_bytes"]
            mime_type = img["image_mime_type"]
            img_file = image_file_name(source_name, idx, mime_type, image_bytes)
            await ccat.save_file(image_bytes, mime_type, img_file, chat_id)
            points.append(
                PointStruct(
                    id=uuid.uuid4().hex,
                    payload={
                        "page_content": f"[Image] {source_name}",
                        "metadata": {
                            **(metadata or {}),
                            "source": source_name,
                            "when": time.time(),
                            "image": True,
                            "image_file": img_file,
                            **({"chat_id": chat_id} if chat_id else {}),
                        },
                    },
                    vector={},
                )
            )

    # Final guard immediately before the write: the marker may have flipped
    # while the last chunks/images were being prepared.
    await _raise_if_canceled(ccat)
    await ccat.vector_memory_handler.add_points_to_tenant(
        collection_name=str(collection_name), points=points
    )
    return points


async def _embed_phase(
    ccat, collection_name, source_name, source_points, embedder, chat_id, cat, rabbit_hole,
    zero_norm_summary: dict | None = None, counts: dict | None = None,
):
    """Recompute the embeddings of the stored points and mark the source completed.

    Phase ``embedding``: rebuild LangChain docs from the stored text points and
    recompute their vectors via ``embed_documents``; for image points, embed via
    ``embed_images`` when the active embedder is multimodal, otherwise keep them
    payload-only (``vector={}``). The old points are deleted and replaced. Fires
    ``after_rabbithole_stored_documents`` so analytics and the other listeners
    are informed, then returns the stored points.

    Zero/non-finite L2-norm vectors are counted and logged at WARNING with the
    source and chunk ids (dense stores would otherwise drop them silently). If
    a source would end up with NO valid chunk at all, :class:`_NothingStored`
    is raised BEFORE the destructive replace, so the caller keeps the row
    non-terminal and retries later instead of marking an empty source complete.

    Returns the list of replaced points (with vectors), or None when the
    embedding failed (the source is left for a later pass).
    """
    await _raise_if_canceled(ccat)
    source_points_text = [p for p in source_points if not (p.payload or {}).get("metadata", {}).get("image")]
    source_points_image = [p for p in source_points if (p.payload or {}).get("metadata", {}).get("image")]

    points: list[PointStruct] = []
    # Per-source counters for the run summary: a silently empty source must be
    # visible, not only a per-chunk warning buried in the log.
    zero_norm_count = 0
    zero_norm_ids: list[str] = []
    intended_vectors = 0
    valid_vectors = 0

    # --- text points: recompute the vector from the stored page_content ---
    if source_points_text:
        docs = [
            Document(
                # Carry a stored id through the re-embed so an id-bearing point
                # (e.g. a catalogue card) keeps its identity across passes. The
                # payload id is authoritative; the stored point id is the
                # fallback for handlers that keep it only at the point level.
                id=(p.payload or {}).get("id") or getattr(p, "id", None),
                page_content=(p.payload or {}).get("page_content", ""),
                metadata=dict((p.payload or {}).get("metadata", {})),
            )
            for p in source_points_text
        ]
        # Re-chunk only chunks exceeding the current embedder's input limit.
        # The parsing-phase store hook already applied the token-budget split,
        # so this is a no-op on a fresh ingest; it is the re-split path when an
        # embedder-only change re-runs this phase with a smaller window (the
        # stored payload is reused, the files are not parsed again).
        # Catalogue-card segments are EXEMPT: the phase already split the card
        # into stable, id-bearing segments and marked them, so re-splitting them
        # here (e.g. after an embedder change) would mint a different segment
        # count and drop the deterministic ``:s<start_index>`` id suffix.
        kept = []
        for doc in docs:
            if doc.metadata.get("is_catalogue_card_segment") is True:
                kept.append(doc)
            else:
                kept.extend(split_oversized([doc], embedder))
        docs = kept
        vectors = await run_in_ingestion_executor(
            embedder.embed_documents, [d.page_content for d in docs]
        )
        intended_vectors += len(docs)
        for index, (doc, vector) in enumerate(zip(docs, vectors)):
            if _is_valid_embedding_vector(vector):
                valid_vectors += 1
            else:
                zero_norm_count += 1
                zero_norm_ids.append(str(doc.id or f"#{index}"))
        points.extend(
            PointStruct(id=d.id or uuid.uuid4().hex, payload=d.model_dump(), vector=vector)
            for d, vector in zip(docs, vectors)
        )

    # --- image points: embed via embed_images when multimodal, else payload-only ---
    if source_points_image:
        if is_multimodal_embedder(embedder):
            recoverable = []
            for p in source_points_image:
                await _raise_if_canceled(ccat)
                meta = (p.payload or {}).get("metadata", {})
                image_file = meta.get("image_file")
                root_dir = ccat.agent_key
                if metadata_chat := meta.get("chat_id"):
                    root_dir = os.path.join(root_dir, str(metadata_chat))
                image_bytes = ccat.file_manager.read_file(image_file, root_dir) if image_file else None
                if image_bytes is None:
                    # H2 fallback: the image file is gone; keep the point
                    # payload-only (no vector) instead of failing the source.
                    points.append(
                        PointStruct(
                            id=uuid.uuid4().hex,
                            payload={"page_content": f"[Image] {source_name}", "metadata": dict(meta)},
                            vector={},
                        )
                    )
                else:
                    recoverable.append((p, image_bytes))
            if recoverable:
                image_vectors = await run_in_ingestion_executor(
                    embedder.embed_images, [b for _, b in recoverable]
                )
                if len(image_vectors) != len(recoverable):
                    raise ValueError(
                        f"embed_images returned {len(image_vectors)} vectors "
                        f"for {len(recoverable)} images"
                    )
                for (p, _b), vector in zip(recoverable, image_vectors):
                    meta = dict((p.payload or {}).get("metadata", {}))
                    intended_vectors += 1
                    if _is_valid_embedding_vector(vector):
                        valid_vectors += 1
                    else:
                        zero_norm_count += 1
                        zero_norm_ids.append(
                            str((p.payload or {}).get("id") or meta.get("image_file") or p.id)
                        )
                    points.append(
                        PointStruct(
                            id=uuid.uuid4().hex,
                            payload={"page_content": f"[Image] {source_name}", "metadata": meta},
                            vector=vector,
                        )
                    )
        else:
            # Non-multimodal embedder: preserve the image points payload-only.
            for p in source_points_image:
                meta = dict((p.payload or {}).get("metadata", {}))
                points.append(
                    PointStruct(
                        id=uuid.uuid4().hex,
                        payload={"page_content": f"[Image] {source_name}", "metadata": meta},
                        vector={},
                    )
                )

    if zero_norm_count:
        sample = ", ".join(zero_norm_ids[:10])
        more = "" if len(zero_norm_ids) <= 10 else f" (+{len(zero_norm_ids) - 10} more)"
        log.warning(
            f"Agent id: {ccat._id}. Source {source_name}: {zero_norm_count} chunk(s) "
            f"have zero/non-finite L2-norm and will be dropped by the vector store "
            f"(ids: {sample}{more})"
        )
        if zero_norm_summary is not None:
            zero_norm_summary[source_name] = {
                "count": zero_norm_count,
                "ids": list(zero_norm_ids),
                "valid": valid_vectors,
                "intended": intended_vectors,
            }

    if intended_vectors and not valid_vectors:
        # EVERY computed vector is unusable: storing these points would leave
        # the source with NO embedding at all (dense stores drop zero vectors
        # silently). Abort BEFORE the destructive replace so the previous
        # points/vectors stay intact and the dispatcher can retry the phase.
        raise _NothingStored(
            f"embedding phase produced 0 valid vectors for {intended_vectors} "
            f"chunk(s): all have zero/non-finite L2-norm",
            retry_after=_not_ready_retry_after(None),
        )

    # All vectors were computed BEFORE the delete so a failure leaves the old
    # points intact (compute-before-delete). A cancel that flipped during the
    # (possibly long) embedding must stop here, before ANY vector write.
    await _raise_if_canceled(ccat)
    await ccat.vector_memory_handler.delete_tenant_points(
        str(collection_name), metadata=_source_metadata(source_name, chat_id)
    )
    await ccat.vector_memory_handler.add_points_to_tenant(
        collection_name=str(collection_name), points=points
    )

    # Token accounting + completion signal.
    await execute_shared_hook(
        cat, "after_rabbithole_stored_documents", source_name, points,
    )
    if counts is not None:
        # expected: the stored chunks that needed vectors for this source;
        # processed: how many items were actually embedded/stored (a re-split
        # may produce more stored points than input chunks — all input chunks
        # were nevertheless processed).
        counts["expected"] = len(source_points)
        counts["processed"] = len(points)
    return points


def _should_stop_for_error(doc) -> bool:
    """True when the status doc is an absorbing ``error`` row.

    The phase machine never advances an ``error`` row: a phase transition must
    not resurrect it (stale-state protection).
    """
    return bool(doc) and doc.get("status") == IngestionStatus.ERROR.value


async def _probe_pending(ccat, cat, source_name, doc) -> list:
    """Clock-free phase probe: diary -> hook list shape -> accumulator.

    Converts the completed-phases diary of ``doc`` (backfilling legacy
    completed rows) to the ``ingestion_phase_pending`` list shape (every entry
    carries a ``"phase"`` key), threads it into the accumulator hook and
    returns the filtered stale phases. Malformed results (None / entries
    without a ``"phase"`` key) are dropped, never raised on.
    """
    completed = await backfill_completed_phases(
        doc or {}, list((await merged_phases(ccat, cat)).keys())
    )
    completed_as_list = [dict(e, phase=p) for p, e in completed.items()]
    pending = await execute_phase_accumulator(
        cat,
        "ingestion_phase_pending",
        [],
        source_name,
        completed_as_list,
        default=list,
    )
    return [p for p in (pending or []) if p and p.get("phase")]


def _point_id(record):
    """Best-effort id of a stored point record (handler or payload)."""
    value = getattr(record, "id", None) or (record.payload or {}).get("id")
    return value


def _point_meta(record) -> dict:
    return (record.payload or {}).get("metadata", {}) or {}


async def _purge_parsing_phase(ccat, collection_name, source_name, chat_id) -> int:
    """PURGE ``parsing_chunking``: the source's staging/chunk artifacts only.

    Deletes every stored point of THIS source in THIS collection EXCEPT the
    catalogue-card segments (``is_catalogue_card``), which are another phase's
    product. Image points/files are parsing products too, so the saved image
    files are cleaned up first. Never touches other sources.
    """
    handler = ccat.vector_memory_handler
    await _cleanup_orphan_images(ccat, collection_name, source_name, chat_id)
    records, _ = await handler.get_all_tenant_points(
        str(collection_name),
        metadata=_source_metadata(source_name, chat_id),
        with_vectors=True,
    )
    ids = []
    for record in records or []:
        if _point_meta(record).get("is_catalogue_card") is True:
            continue
        point_id = _point_id(record)
        if point_id is not None:
            ids.append(point_id)
    if ids:
        await handler.delete_tenant_points_by_ids(str(collection_name), ids)
    return len(ids)


async def _purge_embedding_phase(ccat, collection_name, source_name, chat_id) -> int:
    """PURGE ``embedding``: ONLY the source's points with a REAL vector.

    A staging point carries ``vector={}`` (payload only) and MUST survive: it
    is parsing's product and the parsing purge cleans it. Graph artifacts are
    not vector points and are never touched here.
    """
    handler = ccat.vector_memory_handler
    records, _ = await handler.get_all_tenant_points(
        str(collection_name),
        metadata=_source_metadata(source_name, chat_id),
        with_vectors=True,
    )
    ids = []
    for record in records or []:
        if not _is_valid_embedding_vector(getattr(record, "vector", None)):
            continue
        point_id = _point_id(record)
        if point_id is not None:
            ids.append(point_id)
    if ids:
        await handler.delete_tenant_points_by_ids(str(collection_name), ids)
    return len(ids)


async def _purge_external_phase(ccat, cat, source_name, phase) -> bool:
    """PURGE an external phase via ``ingestion_phase_purge`` on its owner.

    Returns False (and logs a WARNING naming phase/source) when the phase has
    no purge implementation: recovery must then leave the entry ``processing``,
    not delete anything blindly, not re-run and not mark it complete.
    """
    result = await execute_phase_on_owner(
        cat if cat is not None else ccat,
        "ingestion_phase_purge",
        phase,
        phase,
        source_name,
        default=None,
        caller=cat,
    )
    if result is None:
        log.warning(
            f"Agent id: {ccat._id}. Phase {phase} for {source_name} has no "
            f"purge implementation; leaving its processing entry in place "
            f"(not re-running, not completing)"
        )
        return False
    return True


async def _purge_phase(ccat, cat, collection_name, scope, source_name, chat_id, phase) -> bool:
    """Dispatch the PURGE for one phase, source-scoped.

    Built-in phases have an explicit product boundary; everything else goes
    through the external hook. Returns False when no purge implementation
    exists (the caller must leave the processing entry untouched).
    """
    if phase == PHASE_PARSING_CHUNKING:
        count = await _purge_parsing_phase(ccat, collection_name, source_name, chat_id)
        log.info(
            f"Agent id: {ccat._id}. Purged parsing_chunking artifacts for "
            f"{source_name}: {count} staging/chunk point(s)"
        )
        return True
    if phase == PHASE_EMBEDDING:
        count = await _purge_embedding_phase(ccat, collection_name, source_name, chat_id)
        log.info(
            f"Agent id: {ccat._id}. Purged embedding artifacts for "
            f"{source_name}: {count} real-vector point(s)"
        )
        return True
    return await _purge_external_phase(ccat, cat, source_name, phase)


async def _delete_phase(
    ccat, cat, collection_name, scope, source_name, chat_id, phase
) -> bool:
    """Delete one phase's artifacts through its :meth:`IngestionPhase.delete`.

    The phase OBJECT is resolved from the same registry the dispatch uses, so
    an external phase owns its delete implementation. Built-in phases raise
    ``NotImplementedError`` from ``delete`` (their bodies still live in this
    legacy dispatcher): for those the source-scoped legacy purge is the delete
    operation. Returns False when no purge implementation exists; the caller
    must leave the ``processing`` entry untouched.
    """
    phase_obj = None
    try:
        for candidate in await get_phases(cat):
            if getattr(candidate, "id", None) == phase:
                phase_obj = candidate
                break
    except Exception as exc:  # noqa: BLE001 - registry failure falls back to legacy purge
        log.warning(
            f"Agent id: {ccat._id}. Could not resolve phase object for "
            f"{phase} of {source_name}: {exc}; using the legacy purge"
        )
    if phase_obj is not None:
        try:
            await phase_obj.delete(source_name, cat)
            return True
        except NotImplementedError:
            # Built-in phase: its delete is the legacy dispatcher purge below.
            pass
    return await _purge_phase(
        ccat, cat, collection_name, scope, source_name, chat_id, phase
    )


async def _recover_abandoned_phases(
    ccat, cat, collection_name, scope, source_name, chat_id, doc
) -> bool | None:
    """Delete + reset every ``processing`` phase owned by a PREVIOUS boot.

    Called at startup (through the startup pass) and BEFORE a source is
    dispatched (``reembed_sources``), never while a live owner holds the phase:
    :func:`should_abandon` decides ownership, so an entry written by this
    process (same ``BOOT_EPOCH``) or by a live pod/pid is left untouched.

    The earliest abandoned ``processing`` entry in execution order is the
    abandoned phase; every later phase that has a diary entry (its downstream
    products) is deleted with it, in REVERSE execution order (later phases
    first), because those products may depend on the abandoned one. Only after
    every delete succeeds are the entries reset (dropped -> ``pending``) so the
    normal dispatch re-runs them from zero. Recovery NEVER writes ``complete``.

    A delete that raises is logged as a WARNING and that entry is left
    ``processing`` (never reset, never re-run); the sweep continues instead of
    failing as a whole. A phase with no purge implementation (``delete``
    returns False) is NOT deleted blindly and stops this source, preserving the
    existing conservative contract.

    Returns:
        ``None`` when nothing is abandoned (normal path), ``True`` after a
        successful delete+reset, ``False`` when a phase has no purge
        implementation (its WARNING was logged; the caller must stop this
        source and leave the entry processing).
    """
    diary = get_completed_phases(doc)
    if not diary:
        return None
    merged = await merged_phases(ccat, cat)
    order = list(merged.keys())
    abandoned = [
        phase for phase in order
        if isinstance(diary.get(phase), dict)
        and should_abandon(diary[phase])
    ]
    if not abandoned:
        return None

    first = abandoned[0]
    try:
        start = order.index(first)
    except ValueError:
        order.append(first)
        start = len(order) - 1
    # Only phases that actually have an entry (ran at least once) are purged:
    # a later phase that never started has no products to remove.
    to_purge = [
        phase for phase in order[start:]
        if isinstance(diary.get(phase), dict)
    ]

    failed = set()
    for phase in reversed(to_purge):
        try:
            deleted = await _delete_phase(
                ccat, cat, collection_name, scope, source_name, chat_id, phase
            )
        except Exception as exc:  # noqa: BLE001 - one delete must not fail the sweep
            log.warning(
                f"Agent id: {ccat._id}. delete() for phase {phase} of "
                f"{source_name} raised ({exc}); leaving its entry processing "
                f"and continuing the recovery sweep"
            )
            failed.add(phase)
            continue
        if not deleted:
            # No purge implementation: do NOT delete blindly and do NOT
            # re-run; the abandoned entry stays ``processing`` and visible.
            return False

    to_reset = [phase for phase in to_purge if phase not in failed]
    if not to_reset:
        return True
    await reset_phase_entries(ccat.agent_key, scope, source_name, to_reset)
    log.warning(
        f"Agent id: {ccat._id}. Abandoned phase {first} for {source_name} "
        f"(previous boot/dead owner); deleted {list(reversed(to_purge))}, "
        f"reset {to_reset} to pending and re-running it"
    )
    return True


async def _mark_phase_running(ccat, scope, source_name, phase, chat_id, type_) -> None:
    """Write ``state: processing`` BEFORE the phase body starts."""
    await record_phase_processing(
        ccat.agent_key, scope, source_name, phase, type_=type_, chat_id=chat_id
    )


async def _mark_phase_error(ccat, scope, source_name, phase, reason, chat_id, type_) -> None:
    """Write ``state: error`` (with the reason) for a failed phase.

    Also writes the DERIVED source status: any phase in ``error`` makes the
    source ``error`` (the phase never writes the source status itself).
    """
    await record_phase_error(
        ccat.agent_key, scope, source_name, phase, reason,
        type_=type_, chat_id=chat_id,
    )
    await _set_status(
        ccat, source_name, IngestionStatus.ERROR, error=reason, chat_id=chat_id
    )


async def _mark_phase_skipped(ccat, scope, source_name, phase, reason, chat_id, type_) -> None:
    """Write ``state: skipped`` (with the reason) for a non-applicable phase."""
    await record_phase_skipped(
        ccat.agent_key, scope, source_name, phase, reason,
        type_=type_, chat_id=chat_id,
    )


def _phase_count(value):
    """Return ``value`` when it is a genuine integer count, else ``None``.

    ``bool`` is an ``int`` subclass but is never a count; a non-integer
    (missing, string, float, ...) means the phase did not report it.
    """
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _unpack_parsing_result(result):
    """Unpack a ``_run_parsing_phase`` result, tolerating legacy 2-tuples.

    The real body returns ``(source_name, stored_points, expected, processed)``.
    A test/external double returning the legacy ``(source_name, stored_points)``
    (or a 3-tuple) is accepted with the counts unknown, preserving behaviour.
    """
    if len(result) == 4:
        return result
    if len(result) == 3:
        source_name, stored_points, expected = result
        return source_name, stored_points, expected, None
    source_name, stored_points = result
    return source_name, stored_points, None, None


async def _finish_phase(
    ccat, cat, scope, source_name, phase, chat_id, products,
    expected=None, processed=None,
) -> bool:
    """Record ``state: complete`` because the phase body returned ``done``.

    Completion is "the worker finished without error". It does NOT depend on
    how many items were produced: a phase with ZERO items still completes, and
    an item-count mismatch is never a reason to stay ``processing``.

    ``expected`` / ``processed`` are TELEMETRY ONLY: when the body reported
    them they are logged at INFO, otherwise a DEBUG line says they were
    unavailable. They never feed a decision here.
    """
    if expected is None and processed is None:
        log.debug(
            f"Agent id: {ccat._id}. Phase {phase} for {source_name} "
            "completed; processed/expected counts unavailable"
        )
    else:
        log.info(
            f"Agent id: {ccat._id}. Phase {phase} for {source_name} "
            f"completed: processed {processed!r} of expected {expected!r} "
            "item(s) (telemetry only)"
        )
    # ``products`` is kept in the signature for callers that still report the
    # body output; completion no longer inspects it.
    _ = products
    await _record_phase(ccat, cat, scope, source_name, phase, chat_id=chat_id)
    return True


async def _run_external_phase(
    ccat, cat, scope, source_name, phase, chat_id, counts=None, outcome=None
) -> bool:
    """Dispatch one external (non-``PHASES``) phase through the run hook.

    Handles the contract of ``ingestion_phase_run`` (hooks.py):

      - ``{"status": "done"}`` -> returns ``True`` and records the outcome
        ``done``; the caller writes the diary ``complete`` entry and re-probes.
        The optional ``processed`` / ``expected`` fields, when both are
        integers, are copied into ``counts`` as TELEMETRY only (they never
        decide completion);
      - ``{"status": "skipped", "reason": "..."}`` -> returns ``True`` with
        outcome ``skipped``; the caller writes a ``skipped`` diary entry with
        the reason;
      - ``{"status": "not_ready", "retry_after": N}`` -> the phase is retried
        after ``N`` seconds, keeping the row NON-TERMINAL (status
        ``processing``, phase unchanged) until the wall-clock budget
        ``CAT_INGESTION_NOT_READY_MAX_SECONDS`` (default 3600) is exceeded;
        only then is the row marked ERROR with the phase and elapsed time;
      - ``{"status": "error", "reason": "..."}`` -> FAIL-HARD: the row is
        marked ERROR with ``reason`` and ``False`` is returned;
      - ``None``, a raise, or any other return -> FAIL-HARD: the row is marked
        ERROR and ``False`` is returned. A phase nobody implements is an error,
        NEVER a silent success (misleading-success guard).

    An ``error`` row is never resurrected: returning ``False`` leaves the
    absorbing ERROR in place and the caller stops the machine.

    ``outcome`` (optional dict) receives ``{"status": "done"|"skipped"|
    "error", "reason": ...}`` so the caller can write the matching diary state
    instead of inferring it from the boolean.

    Returns:
        ``True`` when the phase produced ``done``/``skipped``; ``False`` when
        the row was marked ERROR.
    """
    outcome = outcome if outcome is not None else {}

    def _set_outcome(status, reason=None):
        outcome["status"] = status
        outcome["reason"] = reason

    doc = await get_status(ccat.agent_key, scope, source_name)
    completed = await backfill_completed_phases(
        doc or {}, list((await merged_phases(ccat, cat)).keys())
    )
    completed_as_list = [dict(e, phase=p) for p, e in completed.items()]

    while True:
        try:
            result = await execute_phase_on_owner(
                cat if cat is not None else ccat,
                "ingestion_phase_run",
                phase,
                phase,
                source_name,
                completed_as_list,
                default=None,
                caller=cat,
            )
        except Exception as e:  # noqa: BLE001 - a raising registrant is a permanent failure
            log.error(
                f"Agent id: {ccat._id}. External phase {phase} for {source_name} raised: {e}"
            )
            _set_outcome("error", str(e))
            await _set_status(ccat, source_name, IngestionStatus.ERROR, error=str(e), chat_id=chat_id)
            return False

        if isinstance(result, dict) and result.get("status") == "done":
            if counts is not None:
                # Telemetry only: an integer pair is a usable report, anything
                # else leaves the counts unknown. It never decides completion.
                counts["expected"] = _phase_count(result.get("expected"))
                counts["processed"] = _phase_count(result.get("processed"))
            _set_outcome("done")
            return True
        if isinstance(result, dict) and result.get("status") == "skipped":
            _set_outcome("skipped", result.get("reason"))
            return True
        if isinstance(result, dict) and result.get("status") == "not_ready":
            retry_after = _not_ready_retry_after(result)
            reason = result.get("reason") or "external phase is not ready"
            if not await _wait_not_ready(
                ccat, scope, source_name, phase, chat_id, reason, retry_after
            ):
                _set_outcome("error", reason)
                return False
            await asyncio.sleep(retry_after)
            continue
        if isinstance(result, dict) and result.get("status") == "error":
            reason = result.get("reason") or f"external phase {phase} reported an error"
            msg = f"external phase {phase} failed: {reason}"
            log.error(f"Agent id: {ccat._id}. {msg} for {source_name}")
            _set_outcome("error", reason)
            await _set_status(ccat, source_name, IngestionStatus.ERROR, error=reason, chat_id=chat_id)
            return False

        # None or any other return: fail-hard (unimplemented / misleading).
        msg = f"external phase {phase} returned {result!r} (expected a status dict)"
        log.error(f"Agent id: {ccat._id}. {msg} for {source_name}")
        _set_outcome("error", msg)
        await _set_status(ccat, source_name, IngestionStatus.ERROR, error=msg, chat_id=chat_id)
        return False


async def _record_phase(ccat, cat, scope, source_name, phase, chat_id=None) -> dict:
    """Record the successful completion of one phase in the diary (atomic).

    Computes the CURRENT inputs AT COMPLETION and writes the single diary
    entry via ``record_phase_completed``:

    - ``marker``: the plugin-defined marker from the
      ``ingestion_phase_settings_marker`` hook;
    - ``settings_version``: the OPAQUE ``updated_at`` of the settings entry for
      ``spec.settings_category`` (used only for equality); ``None`` when the
      phase declares no settings category;
    - ``deps``: ``{upstream: <current diary marker>}`` for each upstream in
      ``spec.depends_on``, copied from the CURRENT diary;
    - ``tag``: the phase's recursive tag from :func:`compute_tags` (own
      material folded with every dependency's tag), so a later deep change
      invalidates this entry.

    The spec is looked up in the MERGED phase set (:func:`merged_phases`):
    built-in ``PHASES`` plus every ``PhaseSpec`` registered through the
    ``ingestion_phase_specs`` hook. A registered phase therefore gets a REAL
    marker (from the hook) and its declared ``deps``, exactly like a built-in
    one; only a truly unknown phase (not in the merged set) falls back to the
    minimal ``{marker: None, settings_version: None, deps: {}}`` entry.

    This is the ONLY diary write for a phase and it happens AFTER the phase
    body succeeded — NEVER at phase start (atomic-completion invariant): a
    phase interrupted before this write has no entry and is stale on the next
    pass, so a restart re-runs it.

    Args:
        ccat: the CheshireCat.
        cat: the hook caller (StrayCat/CheshireCat) that issued the pass.
        scope: ``"agent"`` or a conversation ``chat_id``.
        source_name: the source being processed.
        phase: the phase id that just completed (e.g. ``parsing_chunking``).
        chat_id: the conversation id, when chat-scoped.

    Returns:
        The stored status doc (``record_phase_completed`` output).
    """
    spec = (await merged_phases(ccat, cat)).get(phase)
    if spec is None:
        # truly unknown phase (not in the merged built-in + registered set):
        # record a minimal diary entry so the machine can advance past it (no
        # marker / settings version / deps — nobody owns the phase's material
        # inputs). The atomic-completion invariant still holds: the entry is
        # written ONLY after the phase body reported ``done``.
        return await record_phase_completed(
            ccat.agent_key, scope, source_name, phase,
            marker=None, settings_version=None, deps={},
            type_="url" if is_url(source_name) else "file",
            chat_id=chat_id,
        )

    marker = await execute_phase_on_owner(
        cat if cat is not None else ccat,
        "ingestion_phase_settings_marker",
        phase,
        phase,
        default=None,
        caller=cat,
    )
    settings_version = None
    if spec.settings_category is not None:
        agent_key = getattr(cat, "agent_key", None) or ccat.agent_key
        settings_key = phase_settings_key(spec.id, agent_key)
        settings_doc = await crud_settings.get_settings_by_category(
            settings_key, spec.settings_category
        )
        if isinstance(settings_doc, dict):
            settings_version = settings_doc.get("updated_at")

    # deps: upstream markers copied from the CURRENT diary at completion only
    current = await get_status(ccat.agent_key, scope, source_name) or {}
    diary = get_completed_phases(current)
    deps = {}
    for upstream in spec.depends_on:
        upstream_entry = diary.get(upstream)
        deps[upstream] = upstream_entry.get("marker") if isinstance(upstream_entry, dict) else None

    tag = ""
    try:
        caller = cat if cat is not None else ccat
        tags = await compute_tags(caller, await get_phases(caller))
        tag = tags.get(phase, "")
    except Exception as exc:  # noqa: BLE001 - a tag must never break completion
        log.debug(f"Agent id: {ccat._id}. Could not resolve tag for phase {phase}: {exc}")

    return await record_phase_completed(
        ccat.agent_key,
        scope,
        source_name,
        phase,
        marker=marker,
        settings_version=settings_version,
        deps=deps,
        tag=tag,
        type_="url" if is_url(source_name) else "file",
        chat_id=chat_id,
    )


async def _apply_dependency_outcome(
    ccat, cat, scope, source_name, phase, diary, chat_id, type_
) -> bool:
    """Record a dependency's outcome on ``phase`` BEFORE its body can run.

    Looks the phase spec up in the merged set (built-in + external) and asks
    the pure :func:`phases.dependency_outcome`. When a dependency is ``error``
    the phase is recorded ``error`` (reason names that dependency); when it is
    ``skipped`` the phase is recorded ``skipped``. Returns True when the phase
    was terminalized (the caller must NOT run its body) and False when no
    dependency is failed.
    """
    spec = (await merged_phases(ccat, cat)).get(phase)
    if spec is None:
        return False
    outcome = dependency_outcome(spec, diary)
    if outcome is None:
        return False
    state, reason = outcome
    if state == PHASE_STATE_ERROR:
        await _mark_phase_error(
            ccat, scope, source_name, phase, reason, chat_id, type_
        )
    else:
        await _mark_phase_skipped(
            ccat, scope, source_name, phase, reason, chat_id, type_
        )
    return True


async def _phase_applies(ccat, cat, phase, diary) -> tuple[bool, Optional[str]]:
    """Ask the registered phase whether it applies to this source.

    Returns ``(True, None)`` when no registered implementation is resolvable
    (the dispatcher then runs the legacy flow and fails hard if the phase is
    truly unimplemented). A phase whose ``applies_to`` returns false is
    recorded as ``skipped`` with its reason instead of being run.
    """
    try:
        for registered in await get_phases(cat if cat is not None else ccat):
            if registered.id != phase:
                continue
            applies, reason = await registered.applies_to(
                cat if cat is not None else ccat, diary
            )
            return bool(applies), reason
    except Exception as exc:  # noqa: BLE001 - applicability must never break the pass
        log.warning(
            f"Agent id: {ccat._id}. Could not evaluate applies_to for phase "
            f"{phase}: {exc}"
        )
    return True, None


def _first_error_reason(diary) -> Optional[str]:
    """Reason of the first diary entry in ``error`` (None when absent)."""
    for entry in (diary or {}).values():
        if phase_state(entry) == PHASE_STATE_ERROR:
            return (entry or {}).get("reason")
    return None


async def _refresh_source_status(ccat, cat, scope, source_name, chat_id) -> Optional[str]:
    """Derive the source status from the diary and write it (dispatcher-owned).

    The rule is explicit and shared (:func:`registry.derive_source_status`):
    ``error`` if any phase is ``error``; ``complete`` iff no phase is
    ``pending``/``processing``; ``processing`` otherwise. A ``complete``
    derivation still passes through the completion gate (:func:`_complete_source`).
    """
    doc = await get_status(ccat.agent_key, scope, source_name)
    if not doc:
        return None
    phase_ids = list((await merged_phases(ccat, cat)).keys())
    diary = get_completed_phases(doc)
    derived = derive_source_status(diary, phase_ids)
    if derived == PHASE_STATE_ERROR:
        await _set_status(
            ccat, source_name, IngestionStatus.ERROR,
            error=_first_error_reason(diary) or "a phase failed",
            chat_id=chat_id,
        )
    elif derived == PHASE_STATE_COMPLETE:
        await _complete_source(ccat, cat, scope, source_name, chat_id)
    else:
        await _set_status(
            ccat, source_name, IngestionStatus.PROCESSING, chat_id=chat_id
        )
    return derived


async def _complete_source(ccat, cat, scope, source_name, chat_id) -> None:
    """Write the terminal COMPLETED status (dispatcher-owned).

    Runs the ``before_ingestion_status_completed`` gate first: a registrant
    that raises forces the source to ERROR instead of completing. The caller
    re-probes with the UPDATED diary before invoking this, so nothing is stale
    anymore and a double COMPLETED write is harmless (idempotent).
    ``clear_phase=True`` drops the work phase: a ``completed`` row proves every
    phase finished, so no phase is pending. Never overwrites an ``error`` row.
    """
    try:
        await execute_phase_hook_on_all(
            cat if cat is not None else ccat,
            "before_ingestion_status_completed",
            source_name,
        )
    except CustomHookVetoException as e:
        # a hook refused the completion: honour it and record the reason
        log.warning(
            f"Agent id: {ccat._id}. Ingestion completion vetoed for {source_name}: {e}"
        )
        await _set_status(ccat, source_name, IngestionStatus.ERROR, error=str(e), chat_id=chat_id)
        return
    except Exception as e:  # noqa: BLE001 - the gate decides the terminal state
        log.error(
            f"Agent id: {ccat._id}. before_ingestion_status_completed gate failed "
            f"for {source_name}: {e}"
        )
        await _set_status(ccat, source_name, IngestionStatus.ERROR, error=str(e), chat_id=chat_id)
        return
    try:
        await set_status(
            ccat.agent_key,
            scope,
            source_name,
            type_="url" if is_url(source_name) else "file",
            status=IngestionStatus.COMPLETED,
            chat_id=chat_id,
            clear_phase=True,
        )
    except Exception as e:  # noqa: BLE001 - status must never break the re-embed pass
        log.error(f"Agent id: {ccat._id}. Failed to write ingestion status for {source_name}: {e}")


def _persisted_url_file_name(source_url: str, root_dir: str) -> str:
    """Deterministic, sanitized file name for the persisted bytes of a URL.

    The name is derived from the URL itself (so the same URL always maps to the
    same file) plus its path basename/extension (so the parser can recover the
    content type on a later run). It is validated with the core
    ``sanitize_source_name`` to guarantee it cannot escape the agent storage.
    """
    parsed = urlparse(source_url)
    raw_stem = os.path.basename(parsed.path) or "index"
    stem, ext = os.path.splitext(raw_stem)
    ext = "".join(ch for ch in ext if ch.isalnum() or ch == ".")[:12]
    digest = hashlib.sha256(source_url.encode("utf-8")).hexdigest()[:16]
    candidate = f"url_{digest}_{stem[:80]}{ext}"
    candidate = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in candidate)
    return sanitize_source_name(candidate, path=root_dir)


async def _run_parsing_phase(
    ccat, collection_name, source, source_name, chat_id, cat, rabbit_hole, active_chunker_name,
):
    """Run the ``parsing_chunking`` phase body (clean-sweep -> parse -> store).

    Deletes every artifact of this phase and the following ones (text chunk
    points, image points, saved image files), resolves the source bytes (from
    the in-memory ``source.content``, the persisted copy of a previously
    downloaded URL, the persisted file on disk, or — as a logged fallback — the
    URL itself), parses + chunks ALWAYS extracting the images, and stores the
    text chunks and image points with EMPTY vectors.

    Returns:
        ``(resolved_source_name, stored_points, expected, processed)`` — the
        (possibly re-resolved) source name, the stored empty-vector points
        (which the embedding phase consumes on the next loop iteration), the
        number of text chunks the parser produced (``expected``) and the number
        of those chunks actually stored (``processed``).
    """
    # 1. clean-sweep: remove every artifact of this phase and the following ones.
    await _raise_if_canceled(ccat)
    await _clear_source_artifacts(ccat, collection_name, source_name, chat_id)

    # Use a PER-CALL copy of the RabbitHole bound to this ccat. The lizard owns
    # ONE RabbitHole shared by every agent, and ``_split_text`` reads
    # ``self.cat.chunker``: mutating the shared instance would let two
    # concurrent per-agent re-embeds chunk one agent's source with another
    # agent's chunker (and still mark the phase up to date). The shallow copy
    # is cheap, matches ``RabbitHole.ingest_memory``'s own pattern, and leaves
    # the shared object untouched.
    if rabbit_hole is None:
        rabbit_hole = await _resolve_rabbit_hole(ccat)
    rabbit_hole = copy.copy(rabbit_hole)
    rabbit_hole.cat = ccat
    rabbit_hole.stray = None

    # 2. resolve the source bytes: file already on disk (via the resume/upload),
    #    a persisted copy of a previously downloaded URL, or a URL to download.
    if source.content is not None:
        file_io = source.content
        file_bytes = file_io.read()
        content_type, _ = guess_file_type(file_io)
    elif is_url(source_name):
        path = ccat.agent_key
        if chat_id:
            path = os.path.join(path, str(chat_id))
        persisted_name = _persisted_url_file_name(source_name, path)
        # Prefer the bytes fetched on the first ingest: a re-ingest must index
        # the SAME content even if the URL changed in the meantime.
        file_bytes = ccat.file_manager.read_file(persisted_name, path)
        if file_bytes is not None:
            content_type = mimetypes.guess_type(persisted_name)[0]
            if content_type is None:
                content_type, _ = guess_file_type(io.BytesIO(file_bytes))
        else:
            # FALLBACK: no persisted copy -> (re)download via the core resolver.
            # This means the content may have changed since the last ingest.
            log.warning(
                f"Agent id: {ccat._id}. No persisted bytes for URL '{source_name}'; "
                "downloading it now (the content may have changed since the last ingest)."
            )
            source_name_resolved, file_bytes, content_type, _ = await rabbit_hole._resolve_source_bytes(
                source_name, source_name, None
            )
            if file_bytes is None:
                raise Exception(f"Something went wrong with the source '{source_name}'")
            if source_name_resolved:
                source_name = source_name_resolved
            # Persist the fetched bytes so a later run is deterministic.
            if not ccat.file_manager.write_file(file_bytes, persisted_name, path):
                log.error(
                    f"Agent id: {ccat._id}. Could not persist URL bytes for "
                    f"'{source_name}' at '{persisted_name}'; the next run will re-download."
                )
    else:
        # re-read from disk (the persisted file)
        path = ccat.agent_key
        if chat_id:
            path = os.path.join(path, str(chat_id))
        file_bytes = ccat.file_manager.read_file(source_name, path)
        if file_bytes is None:
            raise Exception(f"File '{source_name}' not found on disk; cannot re-ingest.")
        content_type = None

    # 3. parse + chunk + ALWAYS extract images.
    docs, images = await _parse_and_chunk(
        ccat, rabbit_hole, source, file_bytes, content_type, cat
    )
    if not docs:
        raise Exception(f"No valid chunks found in the file '{source_name}'.")

    # 3b. Emit the store hook EXACTLY ONCE for this source, after chunking and
    #     before the first (empty-vector) store. It is also the SINGLE
    #     token-budget split path for this engine (EffING's handler), and it
    #     lets other plugins append or rewrite the documents; the embedding
    #     phase rebuilds points from the stored payload, so it must NOT emit
    #     the hook again.
    docs = await execute_shared_hook(
        cat, "before_rabbithole_stores_documents", docs,
    )
    if not docs:
        raise Exception(f"No valid chunks found in the file '{source_name}'.")

    # 4. store text chunks + image points with EMPTY vectors.
    sha256 = hashlib.sha256()
    sha256.update(file_bytes or b"")
    file_hash = sha256.hexdigest()
    stored = await _store_empty_vectors(
        ccat, collection_name, source_name, chat_id,
        docs, images, file_hash=file_hash, metadata=source.metadata or {},
    )
    # Report the phase counts for the completion predicate: the parser produced
    # ``len(docs)`` text chunks; ``processed`` is how many of those text chunks
    # were actually stored (image points are not chunks and are excluded from
    # both sides).
    chunks_stored = sum(
        1 for p in stored if not (p.payload or {}).get("metadata", {}).get("image")
    )
    return source_name, stored, len(docs), chunks_stored


@asynccontextmanager
async def _phase_guard(ccat, scope, source_name, phase):
    """Acquire the per-``(source, phase)`` lock and re-check the diary under it.

    Yields ``True`` when the caller must run the phase body and ``False`` when
    the body must be skipped because another worker already terminalized the
    entry (``complete`` / ``error`` / ``skipped``) or the lock timed out.

    **ORDERING RULE** (see :func:`registry.source_lock`): the caller must
    already hold the per-source lock. This guard takes the per-phase lock
    INSIDE it — source FIRST, phase SECOND, never the reverse.

    On a lock timeout the warning names agent/source/phase and the diary entry
    is left EXACTLY as it was: no error write and no failure of the source.
    """
    lock_cm = phase_lock(ccat.agent_key, source_name, phase)
    try:
        await lock_cm.__aenter__()
    except LockError as exc:
        log.warning(
            f"Agent id: {ccat._id}. Could not acquire the phase lock for "
            f"source {source_name} phase {phase} within the timeout ({exc}); "
            f"leaving the entry untouched and continuing with the next phase"
        )
        yield False
        return
    try:
        # ---- RE-CHECK UNDER THE LOCK ----
        # Between the probe that selected this phase and the lock acquisition
        # another worker may have run it to completion. Re-read the diary and
        # skip the body when the entry is already terminal.
        doc = await get_status(ccat.agent_key, scope, source_name)
        state = phase_state(get_completed_phases(doc).get(phase))
        if state in (PHASE_STATE_COMPLETE, PHASE_STATE_ERROR, PHASE_STATE_SKIPPED):
            log.info(
                f"Agent id: {ccat._id}. Source {source_name}: phase {phase} is "
                f"already {state!r} under the lock; skipping its body"
            )
            yield False
            return
        yield True
    finally:
        await lock_cm.__aexit__(None, None, None)


async def reembed_sources(
    ccat,
    collection_name: VectorMemoryType,
    stored_sources: list[StoredSourceWithMetadata],
    stale_after: float | None = None,
    caller_cat=None,
) -> None:
    """
    Run the ONE two-phase ingestion machine for a set of stored sources.

    ``caller_cat``: optional pre-resolved caller (the StrayCat/CheshireCat that
    issued the ingestion). When provided for chat-scoped sources it is used as
    the hook caller instead of re-resolving it via ``_resolve_callers`` (which
    is needed by the background re-embed/resume passes but would fail for a
    brand-new chat upload).

    Phase decision (per source), clock-free:
      - status doc present: the ``completed_phases`` diary is read (backfilling
        legacy completed rows), converted to the hook list shape (each entry
        carries a ``"phase"`` key) and threaded into the
        ``ingestion_phase_pending`` accumulator hook. Registrants (this
        plugin's own plus any external one, e.g. MyGRAPH) compare the diary
        against the CURRENT settings markers/versions and report the stale
        phases. No stale phases -> the source is up to date and is SKIPPED
        WITHOUT claiming; otherwise the row is claimed
        (``claim_completed=True`` for completed rows) and the serial dispatch
        loop below runs the stale phases.
      - no status doc: no diary to probe against -> ``embedding`` if reusable
        points exist, else ``parsing_chunking`` (legacy heuristic); the first
        pending entry is synthesized from that decision.

    Serial probe-driven dispatch (per claimed source): one phase at a time, in
    probe order. Each phase body runs (``parsing_chunking``: clean-sweep the
    source's artifacts then parse + chunk + ALWAYS extract the images and store
    text chunks and image points with EMPTY vectors; ``embedding``: recompute
    the vectors of the stored points and mark the source completed). After each
    phase succeeds, its diary entry (``marker`` / ``settings_version`` / ``deps``)
    is recorded ATOMICALLY at completion — never at phase start — and the probe
    is re-run against the UPDATED diary. When the re-probe is empty the
    terminal COMPLETED status is written (after the
    ``before_ingestion_status_completed`` gate), with the work phase cleared. A
    phase interrupted before its diary write has NO entry -> stale on the next
    pass, so a restart re-runs it. An ``error`` row is never advanced nor
    resurrected to COMPLETED.
    """
    log.info(f"Agent id: {ccat._id}. Embedding stored files to the vector memory")

    # Additive registry wiring: list the phases the registry exposes. This is a
    # DEBUG log only — the dispatch loop below is unchanged and still runs the
    # legacy flow. A failure here must never affect ingestion.
    try:
        registered_phases = await get_phases(caller_cat)
        log.debug(
            f"Agent id: {ccat._id}. Registered ingestion phases: "
            f"{[phase.id for phase in registered_phases]}"
        )
    except Exception as exc:  # noqa: BLE001 - a debug listing must never break ingestion
        log.warning(f"Agent id: {ccat._id}. Could not list ingestion phases: {exc}")

    existing_points, _ = await ccat.vector_memory_handler.get_all_tenant_points(
        str(collection_name), with_vectors=False
    )

    rabbit_hole = await _resolve_rabbit_hole(ccat)
    embedder = await ccat.embedder()
    active_embedder_name = getattr(embedder, "name", None) or get_nlp_object_name(embedder, "default_embedder")
    chunker = getattr(ccat, "chunker", None)
    active_chunker_name = (
        str(chunker.name) if chunker is not None and getattr(chunker, "name", None) else
        get_nlp_object_name(chunker, "default_chunker")
    )
    owner = f"reembed-{os.getpid()}"
    counter = 0
    # Per-source zero/non-finite vector counts, reported in the run summary so
    # a source that ended up empty is visible even when each drop was logged.
    zero_norm_summary: dict = {}

    for source in stored_sources:
        source_name = source.name
        chat_id = source.metadata.get("chat_id")

        if caller_cat is not None:
            # ingestion flow: the caller (StrayCat/CheshireCat) is already
            # resolved; use it directly and derive the scope from it
            cat = caller_cat
            scope = str(cat.id) if hasattr(cat, "id") and getattr(cat, "id", None) else "agent"
            chat_id = chat_id if chat_id else (getattr(cat, "id", None) if hasattr(cat, "id") else None)
        else:
            cat, scope, chat_id = await _resolve_callers(ccat, chat_id)
            if cat is None:
                # the chat this episodic source belongs to no longer exists (or
                # never had a persisted conversation, e.g. a chat-scoped upload
                # with no chat message ever sent): the source is orphaned, so
                # clean it up instead of leaving stale points untouched forever
                # (mirrors the chat-existence check in the chunk-reuse path below).
                log.warning(
                    f"Stray cat with id {chat_id} not found. Cleaning up {source.path}/{source.name}"
                )
                await ccat.vector_memory_handler.delete_tenant_points(
                    str(collection_name), metadata=_source_metadata(source_name, chat_id)
                )
                await _set_status(ccat, source_name, IngestionStatus.COMPLETED, chat_id=chat_id)
                continue

        # ---- decide the start phase from the status doc + phase probe ----
        doc = await get_status(ccat.agent_key, scope, source_name)
        doc_status = (doc or {}).get("status")
        doc_embedder = (doc or {}).get("embedder_name")
        doc_chunker = (doc or {}).get("chunker_name")
        doc_phase = (doc or {}).get("phase")

        # Set when the diary says fresh but the store holds no chunk point: the
        # forced re-parse must be followed by a forced re-embed in THIS pass
        # (the diary's own re-probe cannot see the wipe).
        store_empty_forced = False

        if doc is not None:
            # Clock-free phase probe: read the completed-phases diary (backfilling
            # legacy completed rows that predate the diary), convert it to the hook
            # list shape (every entry carries a "phase" key) and let the
            # ``ingestion_phase_pending`` accumulator registrants — this plugin's
            # own (phases.py) plus any external one (e.g. MyGRAPH) — compare it
            # against the CURRENT settings markers/versions and report the stale
            # phases. No timestamps are used for correctness anywhere.
            # clock-free phase probe (diary -> hook list shape -> accumulator)
            pending = await _probe_pending(ccat, cat, source_name, doc)
            if not pending and not await _source_has_stored_points(
                ccat, ccat.vector_memory_handler, collection_name,
                source_name, chat_id,
            ):
                # The diary says every phase is fresh, but the source has ZERO
                # stored chunk points: the store was wiped while the diary
                # survived. Trust the STORE, not the diary: force a full
                # re-parse + re-embed, otherwise the embed phase never runs and
                # later phases (catalog_cards) fail forever.
                log.info(
                    f"Agent id: {ccat._id}. Source {source_name}: "
                    f"diary-fresh-store-empty; forcing parsing_chunking + "
                    f"embedding"
                )
                pending = [
                    {"phase": PHASE_PARSING_CHUNKING},
                    {"phase": PHASE_EMBEDDING},
                ]
                store_empty_forced = True
            if not pending:
                # A ``processing`` entry with a LIVE lease means another worker
                # (or this process) is still inside the phase body: leave the
                # source alone, never terminalize it mid-phase.
                if has_live_running_phase(doc):
                    log.debug(
                        f"Agent id: {ccat._id}. Source {source_name}: a live "
                        f"processing phase lease is held; leaving it alone"
                    )
                    continue
                # every recorded phase is still fresh against the current settings
                # -> the source is up to date: skip WITHOUT claiming.
                # A non-terminal row whose diary is already complete (e.g. left
                # ``processing`` by a run interrupted between the last diary
                # write and the terminal write) must be terminalized here, or
                # the sweep would re-hand it forever without ever advancing it.
                # Terminal rows are left untouched: an ``error`` row is never
                # resurrected and a ``completed`` row needs no write (no churn).
                if doc_status not in (
                    IngestionStatus.COMPLETED.value,
                    IngestionStatus.ERROR.value,
                ):
                    log.info(
                        f"Agent id: {ccat._id}. Source {source_name}: every phase is "
                        f"already complete (status {doc_status!r}); writing the "
                        f"derived status"
                    )
                    await _refresh_source_status(
                        ccat, cat, scope, source_name, chat_id
                    )
                else:
                    log.debug(
                        f"Agent id: {ccat._id}. Source {source_name}: no stale phases "
                        f"(status {doc_status!r}, diary {sorted(get_completed_phases(doc))}), skipping"
                    )
                continue
            start_phase = pending[0]["phase"]
        else:
            # no status doc: no diary, so the probe has nothing to compare against.
            # Fall back to the legacy heuristic: embedding if reusable points exist
            # (chunk-reuse), else a full re-ingest from parsing_chunking.
            has_points = any(
                _point_matches_source(p, source_name, chat_id)
                for p in existing_points
            )
            start_phase = PHASE_EMBEDDING if has_points else PHASE_PARSING_CHUNKING
            # no diary to probe against: synthesize the first pending entry so
            # the serial dispatcher loop has a single entry point.
            pending = [{"phase": start_phase}]

        # ---- log the phase transition (debug) ----
        log.debug(
            f"Agent id: {ccat._id}. Source {source_name}: ingestion phase "
            f"{doc_phase or '(none)'} -> {start_phase} (status {doc_status or '(none)'} -> "
            f"{IngestionStatus.PROCESSING.value}, embedder {doc_embedder!r} -> {active_embedder_name!r}, "
            f"chunker {doc_chunker!r} -> {active_chunker_name!r})"
        )

        # ---- ORDERING RULE: source lock FIRST, phase lock SECOND ----
        # The per-source lock is held for the WHOLE per-source phase sequence
        # (claim + every phase body). Each phase body then takes its own
        # per-(source, phase) lock INSIDE it (see ``_phase_guard``) — never the
        # reverse. The delete path takes ONLY this source lock, never a phase
        # lock. A timeout here leaves nothing touched: continue with the next
        # source instead of failing it.
        source_locks = AsyncExitStack()
        try:
            await source_locks.enter_async_context(
                source_lock(ccat.agent_key, source_name)
            )
        except LockError as exc:
            log.warning(
                f"Agent id: {ccat._id}. Could not acquire the source lock for "
                f"{source_name} within the timeout ({exc}); leaving it "
                f"untouched and continuing with the next source"
            )
            continue

        # ---- claim the per-source work (skip if another worker holds it) ----
        # A tracked source is claimed under its existing row; an UNTRACKED one
        # (legacy stored source with no status doc) has its row CREATED
        # atomically by the claim, so two concurrent passes cannot both proceed.
        claimed = await claim_source_for_resume(
            ccat.agent_key,
            scope,
            source_name,
            stale_after=_claim_stale_after() if stale_after is None else stale_after,
            owner=owner,
            claim_completed=(doc_status == IngestionStatus.COMPLETED.value),
            create_if_missing=doc is None,
            type_="url" if is_url(source_name) else "file",
            chat_id=chat_id,
        )
        if claimed is None:
            # another worker is already (re)processing this source
            await source_locks.aclose()
            continue

        # Batch/recovery passes do not go through ``ingest_file``: keep the
        # claimed row fresh here, otherwise a phase longer than the boot-stale
        # threshold could be re-claimed by the next periodic sweep.
        await _start_heartbeat(ccat, source_name, scope, cat)
        # The phase body currently in flight (None between bodies). Used to
        # attribute a raised exception to the phase that produced it so the
        # machine writes ``state: error`` with the reason.
        current_phase = None
        try:
            # ---- RECOVERY SWEEP: startup pass + before this source dispatch ----
            # An entry left ``processing`` by a PREVIOUS boot / dead owner is
            # abandoned: delete its products (later phases first) and reset the
            # entries to ``pending`` BEFORE anything is dispatched, so the
            # normal dispatch re-runs it from zero. A live owner (the CURRENT
            # ``BOOT_EPOCH``, or a live pod/pid) is never touched. The startup
            # pass (``after_lizard_bootstrap`` -> ``_startup_pass`` -> here)
            # therefore sweeps every candidate source at boot, and every source
            # is swept again immediately before it is dispatched.
            recovered = await _recover_abandoned_phases(
                ccat, cat, collection_name, scope, source_name, chat_id,
                await get_status(ccat.agent_key, scope, source_name),
            )
            if recovered is False:
                # No delete implementation: leave the entry ``processing`` and
                # do not dispatch (or re-run) this source.
                continue
            if recovered is True:
                doc = await get_status(ccat.agent_key, scope, source_name)
                pending = await _probe_pending(ccat, cat, source_name, doc)

            # ---- SERIAL PROBE-DRIVEN PHASE DISPATCH ----
            # One phase at a time, in probe order. After each phase body
            # succeeds its diary entry (marker / settings_version / deps) is
            # recorded ATOMICALLY and the probe is re-run against the UPDATED
            # diary. When the re-probe is empty the terminal COMPLETED status
            # is written (after the ``before_ingestion_status_completed``
            # gate). A phase interrupted before its diary write has NO entry ->
            # stale on the next pass, so a restart re-runs it. An ``error`` row
            # is never advanced nor resurrected.
            stored_points = None
            while pending:
                doc = await get_status(ccat.agent_key, scope, source_name)
                diary = get_completed_phases(doc)

                # ---- DEPENDENCY PROPAGATION: BEFORE any phase body ----
                # A phase whose dependency errored/skipped inherits that
                # outcome — the reason names the offending dependency — and its
                # body NEVER runs. Scanning the whole pending list (not only
                # its head) makes the result independent of probe order; each
                # pass terminalizes exactly one phase, so a chain propagates one
                # link per iteration and always makes progress.
                propagated = False
                for candidate in pending:
                    if await _apply_dependency_outcome(
                        ccat, cat, scope, source_name, candidate["phase"],
                        diary, chat_id, "url" if is_url(source_name) else "file",
                    ):
                        propagated = True
                        break
                if propagated:
                    doc = await get_status(ccat.agent_key, scope, source_name)
                    pending = await _probe_pending(ccat, cat, source_name, doc)
                    if not pending:
                        # The derived status reflects the propagated outcome:
                        # any ``error`` -> source ``error``; all
                        # terminal-non-error -> ``complete``.
                        await _refresh_source_status(
                            ccat, cat, scope, source_name, chat_id
                        )
                        break
                    continue

                if _should_stop_for_error(doc):
                    # ERROR absorbing: never advance a failed row
                    break
                entry = pending[0]
                phase = entry["phase"]
                current_phase = None
                phase_products = True
                # Counts REPORTED by the phase body: TELEMETRY ONLY, logged by
                # ``_finish_phase`` and never used as a completion criterion.
                phase_expected = None
                phase_processed = None

                # Embedding with no reusable points is a full re-ingest: pick
                # ``parsing_chunking`` BEFORE taking the phase lock, so no phase
                # lock is ever held while another phase's body runs.
                if phase == PHASE_EMBEDDING:
                    embedding_points = stored_points or [
                        p for p in existing_points
                        if _point_matches_source(p, source_name, chat_id)
                    ]
                    if not embedding_points:
                        phase = PHASE_PARSING_CHUNKING

                # ---- ORDERING RULE: phase lock INSIDE the source lock ----
                # The per-source lock is already held above. ``_phase_guard``
                # takes this phase's own lock, then RE-READS the diary and
                # returns False when another worker already terminalized the
                # entry or the lock timed out (entry left untouched).
                async with _phase_guard(
                    ccat, scope, source_name, phase
                ) as should_run:
                    if not should_run:
                        doc = await get_status(ccat.agent_key, scope, source_name)
                        if phase_state(get_completed_phases(doc).get(phase)) in (
                            PHASE_STATE_COMPLETE,
                            PHASE_STATE_ERROR,
                            PHASE_STATE_SKIPPED,
                        ):
                            # Another worker terminalized it under the lock:
                            # re-probe and pick the next pending phase.
                            pending = await _probe_pending(ccat, cat, source_name, doc)
                            if not pending:
                                if (
                                    not _should_stop_for_error(doc)
                                    and not has_live_running_phase(doc)
                                ):
                                    await _refresh_source_status(
                                        ccat, cat, scope, source_name, chat_id
                                    )
                                break
                            continue
                        # Lock timeout: the entry was left untouched. Do NOT
                        # retry the SAME phase in this pass (it would spin on
                        # the held lock); continue with the NEXT phase.
                        pending = [p for p in pending if p.get("phase") != phase]
                        if not pending:
                            break
                        continue

                    # Use the post-lock diary for the applicability check.
                    doc = await get_status(ccat.agent_key, scope, source_name)

                    # A phase may declare itself not applicable (e.g. a graph
                    # phase on a Qdrant agent): record ``skipped`` with the reason
                    # instead of running the body. The machine then re-probes; if
                    # every remaining phase is terminal-non-error the source
                    # completes through the derived-status path.
                    applies, skip_reason = await _phase_applies(
                        ccat, cat, phase, get_completed_phases(doc)
                    )
                    if not applies:
                        await _mark_phase_skipped(
                            ccat, scope, source_name, phase, skip_reason, chat_id,
                            "url" if is_url(source_name) else "file",
                        )
                        doc = await get_status(ccat.agent_key, scope, source_name)
                        pending = await _probe_pending(ccat, cat, source_name, doc)
                        if (
                            not pending
                            and not _should_stop_for_error(doc)
                            and not has_live_running_phase(doc)
                        ):
                            await _refresh_source_status(
                                ccat, cat, scope, source_name, chat_id
                            )
                            break
                        continue

                    if phase == PHASE_EMBEDDING:
                        # chunks (text) already stored for this source: chunk-reuse
                        # path. For episodic sources, the chat must still exist.
                        if chat_id and not (await ccat._find_stray_cat(str(chat_id))):
                            log.warning(
                                f"Stray cat with id {chat_id} not found. Cleaning up {source.path}/{source.name}"
                            )
                            await ccat.vector_memory_handler.delete_tenant_points(
                                str(collection_name), metadata=_source_metadata(source_name, chat_id)
                            )
                            await _set_status(ccat, source_name, IngestionStatus.COMPLETED, chat_id=chat_id)
                            break

                        await set_phase(
                            ccat.agent_key, scope, source_name,
                            PHASE_EMBEDDING,
                            embedder_name=active_embedder_name,
                            type_="url" if is_url(source_name) else "file",
                            chat_id=chat_id,
                        )
                        source_points = embedding_points
                        if source_points:
                            # ``processing`` is written only for the phase body that
                            # actually runs: the fall-through below re-parses
                            # instead and must not leave a stale embedding entry.
                            await _mark_phase_running(
                                ccat, scope, source_name, PHASE_EMBEDDING, chat_id,
                                "url" if is_url(source_name) else "file",
                            )
                            current_phase = PHASE_EMBEDDING
                            try:
                                embed_counts: dict = {}
                                phase_products = await _embed_phase(
                                    ccat, collection_name, source_name, source_points,
                                    embedder, chat_id, cat, rabbit_hole,
                                    zero_norm_summary=zero_norm_summary,
                                    counts=embed_counts,
                                )
                                phase_expected = embed_counts.get("expected")
                                phase_processed = embed_counts.get("processed")
                            except _NothingStored as exc:
                                # No valid vector at all: keep the row non-terminal
                                # and retry the SAME phase until the time budget
                                # expires (old points were NOT deleted). This is the
                                # not_ready path, never a silent empty completion.
                                if not await _wait_not_ready(
                                    ccat, scope, source_name, PHASE_EMBEDDING,
                                    chat_id, exc.reason, exc.retry_after,
                                ):
                                    break
                                await asyncio.sleep(exc.retry_after)
                                continue
                            stored_points = None
                            counter += 1
                        else:
                            # Unreachable: no reusable points is pre-selected as
                            # ``parsing_chunking`` above, BEFORE the phase lock. If
                            # it ever happens, re-probe instead of running another
                            # phase's body while holding the embedding lock.
                            continue
                    elif phase == PHASE_PARSING_CHUNKING:
                        # ---- parsing_chunking phase body ----
                        await set_phase(
                            ccat.agent_key, scope, source_name,
                            PHASE_PARSING_CHUNKING,
                            chunker_name=active_chunker_name,
                            type_="url" if is_url(source_name) else "file",
                            chat_id=chat_id,
                        )
                        await _mark_phase_running(
                            ccat, scope, source_name, PHASE_PARSING_CHUNKING, chat_id,
                            "url" if is_url(source_name) else "file",
                        )
                        current_phase = PHASE_PARSING_CHUNKING
                        source_name, stored_points, phase_expected, phase_processed = (
                            _unpack_parsing_result(
                                await _run_parsing_phase(
                                    ccat, collection_name, source, source_name,
                                    chat_id, cat, rabbit_hole, active_chunker_name,
                                )
                            )
                        )
                        phase_products = stored_points
                    else:
                        # external phase (e.g. MyGRAPH): dispatched through the
                        # ``ingestion_phase_run`` hook (tri-state, hooks.py).
                        # A phase nobody implements (hook returns None) fails hard
                        # — it is an error, never a silent success. A phase that
                        # reports ``done`` is recorded by the shared ATOMIC
                        # completion write below and re-probed; ``not_ready`` is
                        # retried (bounded) and an exceeded bound marks the row
                        # ERROR. The machine stops on any ERROR (absorbing).
                        await set_phase(
                            ccat.agent_key, scope, source_name, phase,
                            type_="url" if is_url(source_name) else "file",
                            chat_id=chat_id,
                        )
                        await _mark_phase_running(
                            ccat, scope, source_name, phase, chat_id,
                            "url" if is_url(source_name) else "file",
                        )
                        current_phase = phase
                        external_counts: dict = {}
                        external_outcome: dict = {}
                        if not await _run_external_phase(
                            ccat, cat, scope, source_name, phase, chat_id,
                            counts=external_counts, outcome=external_outcome,
                        ):
                            # the row was marked ERROR (fail-hard / retry bound):
                            # write the matching diary state and stop the machine;
                            # the ERROR-absorbing check above never advances nor
                            # resurrects it.
                            await _mark_phase_error(
                                ccat, scope, source_name, phase,
                                external_outcome.get("reason")
                                or f"external phase {phase} failed",
                                chat_id, "url" if is_url(source_name) else "file",
                            )
                            current_phase = None
                            break
                        if external_outcome.get("status") == "skipped":
                            await _mark_phase_skipped(
                                ccat, scope, source_name, phase,
                                external_outcome.get("reason"), chat_id,
                                "url" if is_url(source_name) else "file",
                            )
                            current_phase = None
                            doc = await get_status(ccat.agent_key, scope, source_name)
                            pending = await _probe_pending(ccat, cat, source_name, doc)
                            if (
                                not pending
                                and not _should_stop_for_error(doc)
                                and not has_live_running_phase(doc)
                            ):
                                await _refresh_source_status(
                                    ccat, cat, scope, source_name, chat_id
                                )
                                break
                            continue
                        phase_products = True
                        phase_expected = external_counts.get("expected")
                        phase_processed = external_counts.get("processed")

                    # ---- ATOMIC COMPLETION: the body returned ``done``, so the
                    # ---- phase is complete regardless of how many items it made
                    # ---- (item counts are telemetry only).
                    if not await _finish_phase(
                        ccat, cat, scope, source_name, phase, chat_id, phase_products,
                        expected=phase_expected, processed=phase_processed,
                    ):
                        break
                    current_phase = None

                    # ---- re-probe with the updated diary ----
                    doc = await get_status(ccat.agent_key, scope, source_name)
                    pending = await _probe_pending(ccat, cat, source_name, doc)

                    # The diary was fresh while the store was empty: the re-parse
                    # just repopulated the points, but the diary still says
                    # ``embedding`` is fresh (its dependency marker did not change),
                    # so force the matching re-embed in THIS pass. The store check
                    # alone cannot catch this: the staging pass writes empty-vector
                    # points, which the store probe counts as present.
                    if store_empty_forced and phase == PHASE_PARSING_CHUNKING:
                        if not any(p.get("phase") == PHASE_EMBEDDING for p in pending):
                            pending = [{"phase": PHASE_EMBEDDING}, *pending]

                    # ---- terminal COMPLETED when nothing is stale anymore ----
                    if (
                        not pending
                        and not _should_stop_for_error(doc)
                        and not has_live_running_phase(doc)
                    ):
                        # Derived source status: ``complete`` only because no
                        # phase is pending/processing and none errored.
                        await _refresh_source_status(
                            ccat, cat, scope, source_name, chat_id
                        )
                        break
                    if not pending:
                        # an error row appeared while recording: never resurrect it
                        break

        except _IngestionCanceled:
            # The agent was deleted mid-phase: stop this source (and the pass)
            # without writing anything else. set_status is already a no-op for
            # a canceled agent, so no ERROR write is attempted.
            log.info(
                f"Agent id: {ccat._id}. Ingestion canceled while processing "
                f"{source_name}; stopping"
            )
            return
        except Exception as e:  # noqa: BLE001 - a failing source must not abort the pass
            log.error(f"Agent id: {ccat._id}. Error re-embedding source {source_name}: {e}")
            if current_phase is not None:
                # Attribute the failure to the phase body that raised: the
                # diary carries ``state: error`` with the reason and the
                # derived source status becomes ``error``.
                try:
                    await _mark_phase_error(
                        ccat, scope, source_name, current_phase, str(e),
                        chat_id, "url" if is_url(source_name) else "file",
                    )
                    await _refresh_source_status(
                        ccat, cat, scope, source_name, chat_id
                    )
                except Exception as state_error:  # noqa: BLE001 - status must never break the pass
                    log.error(
                        f"Agent id: {ccat._id}. Could not record phase error for "
                        f"{current_phase} of {source_name}: {state_error}"
                    )
            else:
                await _set_status(ccat, source_name, IngestionStatus.ERROR, error=str(e), chat_id=chat_id)
        finally:
            await _stop_heartbeat(ccat, source_name, scope, cat)
            await source_locks.aclose()

    log.info(f"Agent id: {ccat._id}. Embedded {counter} files to the vector memory")
    for source_name, info in zero_norm_summary.items():
        log.warning(
            f"Agent id: {ccat._id}. Run summary: source {source_name} had "
            f"{info['count']} zero/non-finite vector chunk(s) "
            f"({info['valid']}/{info['intended']} valid)"
        )


class EfficientIngestionEngine(BaseIngestionEngine):
    """Efficient re-embed engine (one two-phase phase machine).

    Pluggable implementation of the base ingestion engine, registered by the
    plugin through the ``factory_allowed_ingestions`` hook as
    ``EfficientIngestionConfiguration``. Runs the same phase machine for fresh
    ingestion (via ``ingest_file``) and for the re-embed pass (via ``run``),
    writes the ingestion status through the plugin's own ``registry`` and honors
    a configurable concurrency cap (settings category ``ingestion``).
    """

    def __init__(self, ingestion_max_concurrency: int = 5, reembed_max_concurrency: int | None = None, **kwargs):
        super().__init__(**kwargs)
        # legacy alias: pre-rename configs saved `reembed_max_concurrency`
        if (
            reembed_max_concurrency is not None
            and reembed_max_concurrency > 0
            and ingestion_max_concurrency == 5
        ):
            ingestion_max_concurrency = reembed_max_concurrency
        self.ingestion_max_concurrency = max(1, int(ingestion_max_concurrency))

    async def ingest_file(
        self,
        cat,
        file,
        filename: str | None = None,
        metadata: dict | None = None,
        store_file: bool = True,
        content_type: str | None = None,
    ) -> None:
        """Ingest a single file through the two-phase phase machine.

        Similar lifecycle to the core flow: fires the ingestion/processing hooks,
        persists the file (when ``store_file``), then hands the source to the
        phase machine (parsing_chunking -> embedding -> completed).
        """
        source_name = filename or (file if isinstance(file, str) else None)
        if not source_name:
            raise ValueError("No filename provided.")

        # normalize the caller: the routes pass a StrayCat (chat scope) or a
        # CheshireCat (agent scope). The phase machine works on the CheshireCat
        # (ccat) and derives the chat scope from the source metadata.
        if hasattr(cat, "agent_key") and not hasattr(cat, "_id"):
            # StrayCat: resolve its CheshireCat and keep the chat id
            ccat = await cat.lizard.get_cheshire_cat(cat.agent_key)
            chat_id = cat.id
        else:
            ccat = cat
            chat_id = None
        if ccat is None:
            raise ValueError(f"Agent '{getattr(cat, 'agent_key', None)}' not found; cannot ingest.")

        scope = str(chat_id) if chat_id else "agent"
        collection_name = VectorMemoryType.EPISODIC if chat_id else VectorMemoryType.DECLARATIVE

        # Materialize the file bytes once (the BytesIO from the route is consumed
        # by reads); both persistence and the phase machine use a fresh copy.
        if is_url(source_name):
            file_data = None
            content = None
        elif isinstance(file, bytes):
            file_data = file
            content = io.BytesIO(file)
        elif hasattr(file, "read"):
            file_data = file.read()
            content = io.BytesIO(file_data)
        else:
            file_data = None
            content = file

        # lifecycle: start + persist (durable across restarts) + processing
        # Shared hooks run on the agent manager (other plugins) AND the system
        # manager (EffING's own status/heartbeat), each plugin exactly once.
        await execute_shared_hook(
            cat, "rabbithole_ingestion_start", source_name, metadata or {},
            is_url(source_name),
        )
        if store_file and file_data is not None:
            await ccat.save_file(file_data, content_type, source_name, chat_id)
        await execute_shared_hook(
            cat, "rabbithole_ingestion_processing", source_name,
        )
        heartbeat_interval = get_env_int("CAT_INGESTION_HEARTBEAT_SECONDS") or 30
        await execute_shared_hook(
            cat, "rabbithole_processing_heartbeat_start",
            source_name, scope, heartbeat_interval,
        )

        try:
            source_obj = StoredSourceWithMetadata(
                name=source_name, path=source_name, content=content,
                metadata={**(metadata or {}), **({"chat_id": chat_id} if chat_id else {})},
            )
            await reembed_sources(
                ccat, collection_name, [source_obj], stale_after=0.0,
                caller_cat=cat,
            )
        finally:
            await execute_shared_hook(
                cat, "rabbithole_processing_heartbeat_stop",
                source_name, scope,
            )

    async def run(self, lizard) -> bool:
        """Resolve the current embedder and run the re-embed pass."""
        success = False
        try:
            embedder = await lizard.embedder()
            embedder_name = embedder.name
            embedder_size = embedder.size

            ccat_ids = [a["agent_id"] for a in await crud_settings.get_agents()]
            stored_files_by_ccat = []
            # first, get all the stored files from all the Cheshire Cats with the
            # metadata stored within the vector memory; nothing is removed from
            # the latter to avoid any race condition
            for ccat_id in ccat_ids:
                if await ingestion_canceled(ccat_id):
                    # agent is being deleted (marker) or dead (ghost): never re-embed it
                    continue
                try:
                    ccat = await lizard.get_cheshire_cat(ccat_id)
                except CustomNotFoundException:
                    # ghost agent: skip without crashing the pass
                    continue
                if ccat is None:
                    continue
                stored_files_by_ccat.append({
                    "ccat": ccat,
                    "stored_sources": await ccat.get_stored_sources_with_metadata(),
                })

            # re-initialize all the vector databases in a serialized way, outside
            # threads to avoid race conditions
            for entry in stored_files_by_ccat:
                await entry["ccat"].vector_memory_handler.initialize(embedder_name, embedder_size)

            # then re-embed every stored file/procedure, limiting concurrent
            # embeddings to avoid overwhelming resources (tunable via the plugin
            # settings, category ingestion)
            semaphore = asyncio.Semaphore(self.ingestion_max_concurrency)

            async def embed_with_limit(entry_):
                async with semaphore:
                    tasks = [
                        reembed_sources(entry_["ccat"], collection_name, sources)
                        for collection_name, sources in entry_["stored_sources"].items()
                        if sources
                    ] + [entry_["ccat"].embed_procedures()]
                    await asyncio.gather(*tasks)

            await asyncio.gather(*[embed_with_limit(entry) for entry in stored_files_by_ccat])

            success = True
        except Exception as e:  # noqa: BLE001 - surfaced on the hook, never raised
            log.error(f"Error embedding all stored files: {e}")

        await lizard.plugin_manager.execute_hook(
            "after_all_cheshire_cats_embedded", success, caller=lizard,
        )
        return success
