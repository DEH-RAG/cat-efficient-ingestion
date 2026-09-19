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
"""

import asyncio
import hashlib
import io
import os
import time
import uuid

from langchain_core.documents import Document
from langchain_core.documents.base import Blob

from cat.core_plugins.base_plugin.parsers import MimeTypeBasedParser
from cat.db.cruds import settings as crud_settings
from cat.env import get_env_int
from cat.exceptions import CustomNotFoundException
from cat.log import log
from cat.looking_glass.models import StoredSourceWithMetadata
from cat.services.factory.embedder import is_multimodal_embedder
from cat.services.factory.ingestion import BaseIngestionEngine
from cat.services.memory.models import PointStruct, VectorMemoryType
from cat.utils import get_nlp_object_name, guess_file_type, is_url

from cat.plugins.cat_multimodal_ingestion.ingestion import (
    collect_document_images,
    image_file_name,
    strip_image_payload,
)

from .ingestion_executor import run_in_ingestion_executor
from .phases import PHASES
from .registry import (
    PHASE_EMBEDDING,
    PHASE_PARSING_CHUNKING,
    IngestionStatus,
    backfill_completed_phases,
    claim_source_for_resume,
    get_completed_phases,
    get_status,
    ingestion_canceled,
    record_phase_completed,
    set_phase,
    set_status,
)
from .split import split_oversized

# Maximum number of ``{"status": "not_ready"}`` retries allowed for ONE
# external phase before the machine gives up and marks the row ERROR. The
# bound keeps a stuck external registrant from looping forever.
_PHASE_RUN_MAX_RETRIES = 5


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


async def _cleanup_orphan_images(ccat, collection_name, source_name, chat_id) -> None:
    """Remove image points + saved image files of a source before a full re-ingest.

    A full re-ingest (parsing_chunking) regenerates the chunks AND re-extracts
    the images from scratch. The image points and their saved ``image_file``
    from the PREVIOUS (possibly incomplete) parse are therefore stale: delete
    the points in the target first and the files from the agent storage, so the
    re-ingest starts clean and no orphan image file lingers on disk.
    """
    try:
        points, _ = await ccat.vector_memory_handler.get_all_tenant_points(
            str(collection_name), with_vectors=False,
            metadata={"source": source_name, "image": True},
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
        str(collection_name), metadata={"source": source_name, "image": True}
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
        str(collection_name), metadata={"source": source_name}
    )


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

    docs = await rabbit_hole._split_text(super_docs)
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
        # enrich the metadata as the core store_documents does: source, when,
        # hash (deduped compute) and chat_id for chat-scoped sources.
        doc.metadata = (
            doc.metadata
            | (metadata or {})
            | {"source": source_name, "when": time.time(), "hash": file_hash}
            | ({"chat_id": chat_id} if chat_id else {})
        )
        points.append(
            PointStruct(id=uuid.uuid4().hex, payload=doc.model_dump(), vector={})
        )

    if images:
        for idx, img in enumerate(images):
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

    await ccat.vector_memory_handler.add_points_to_tenant(
        collection_name=str(collection_name), points=points
    )
    return points


async def _embed_phase(ccat, collection_name, source_name, source_points, embedder, chat_id, cat, rabbit_hole):
    """Recompute the embeddings of the stored points and mark the source completed.

    Phase ``embedding``: rebuild LangChain docs from the stored text points and
    recompute their vectors via ``embed_documents``; for image points, embed via
    ``embed_images`` when the active embedder is multimodal, otherwise keep them
    payload-only (``vector={}``). The old points are deleted and replaced. Fires
    ``after_rabbithole_stored_documents`` so analytics and the other listeners
    are informed, then returns the stored points.

    Returns the list of replaced points (with vectors), or None when the
    embedding failed (the source is left for a later pass).
    """
    source_points_text = [p for p in source_points if not (p.payload or {}).get("metadata", {}).get("image")]
    source_points_image = [p for p in source_points if (p.payload or {}).get("metadata", {}).get("image")]

    points: list[PointStruct] = []

    # --- text points: recompute the vector from the stored page_content ---
    if source_points_text:
        docs = [
            Document(
                page_content=(p.payload or {}).get("page_content", ""),
                metadata=dict((p.payload or {}).get("metadata", {})),
            )
            for p in source_points_text
        ]
        # Re-chunk only chunks exceeding the current embedder's input limit.
        docs = split_oversized(docs, embedder)
        vectors = await run_in_ingestion_executor(
            embedder.embed_documents, [d.page_content for d in docs]
        )
        points.extend(
            PointStruct(id=uuid.uuid4().hex, payload=d.model_dump(), vector=vector)
            for d, vector in zip(docs, vectors)
        )

    # --- image points: embed via embed_images when multimodal, else payload-only ---
    if source_points_image:
        if is_multimodal_embedder(embedder):
            recoverable = []
            for p in source_points_image:
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

    # All vectors were computed BEFORE the delete so a failure leaves the old
    # points intact (compute-before-delete).
    await ccat.vector_memory_handler.delete_tenant_points(
        str(collection_name), metadata={"source": source_name}
    )
    await ccat.vector_memory_handler.add_points_to_tenant(
        collection_name=str(collection_name), points=points
    )

    # Token accounting + completion signal.
    await ccat.plugin_manager.execute_hook(
        "after_rabbithole_stored_documents", source_name, points, caller=cat,
    )
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
    completed = await backfill_completed_phases(doc or {}, list(PHASES.keys()))
    completed_as_list = [dict(e, phase=p) for p, e in completed.items()]
    pending = await ccat.plugin_manager.execute_hook(
        "ingestion_phase_pending", [], source_name, completed_as_list, caller=cat
    )
    return [p for p in (pending or []) if p and p.get("phase")]


async def _run_external_phase(ccat, cat, scope, source_name, phase, chat_id) -> bool:
    """Dispatch one external (non-``PHASES``) phase through the run hook.

    Handles the tri-state contract of ``ingestion_phase_run`` (hooks.py):

      - ``{"status": "done"}`` -> returns ``True``: the caller records the
        diary entry and re-probes;
      - ``{"status": "not_ready", "retry_after": N}`` -> the phase is retried
        after ``N`` seconds, bounded by ``_PHASE_RUN_MAX_RETRIES`` attempts;
        exceeding the bound marks the row ERROR and returns ``False``;
      - ``None``, a raise, or any other return -> FAIL-HARD: the row is marked
        ERROR and ``False`` is returned. A phase nobody implements is an error,
        NEVER a silent success (misleading-success guard).

    An ``error`` row is never resurrected: returning ``False`` leaves the
    absorbing ERROR in place and the caller stops the machine.

    Returns:
        ``True`` when the phase completed (diary recording is the caller's
        job); ``False`` when the row was marked ERROR.
    """
    doc = await get_status(ccat.agent_key, scope, source_name)
    completed = await backfill_completed_phases(doc or {}, list(PHASES.keys()))
    completed_as_list = [dict(e, phase=p) for p, e in completed.items()]

    retries = 0
    while True:
        try:
            result = await ccat.plugin_manager.execute_hook(
                "ingestion_phase_run", phase, source_name, completed_as_list, caller=cat
            )
        except Exception as e:  # noqa: BLE001 - a raising registrant is a permanent failure
            log.error(
                f"Agent id: {ccat._id}. External phase {phase} for {source_name} raised: {e}"
            )
            await _set_status(ccat, source_name, IngestionStatus.ERROR, error=str(e), chat_id=chat_id)
            return False

        if isinstance(result, dict) and result.get("status") == "done":
            return True
        if isinstance(result, dict) and result.get("status") == "not_ready":
            retries += 1
            if retries > _PHASE_RUN_MAX_RETRIES:
                msg = f"external phase {phase} not ready after {_PHASE_RUN_MAX_RETRIES} retries"
                log.error(f"Agent id: {ccat._id}. {msg} for {source_name}")
                await _set_status(ccat, source_name, IngestionStatus.ERROR, error=msg, chat_id=chat_id)
                return False
            retry_after = result.get("retry_after") or 0
            if retry_after > 0:
                await asyncio.sleep(retry_after)
            continue

        # None or any other return: fail-hard (unimplemented / misleading).
        msg = f"external phase {phase} returned {result!r} (expected a status dict)"
        log.error(f"Agent id: {ccat._id}. {msg} for {source_name}")
        await _set_status(ccat, source_name, IngestionStatus.ERROR, error=msg, chat_id=chat_id)
        return False


async def _record_phase(ccat, cat, scope, source_name, phase, chat_id=None) -> dict:
    """Record the successful completion of one phase in the diary (atomic).

    Computes the CURRENT inputs AT COMPLETION and writes the single diary
    entry via ``record_phase_completed``:

    - ``marker``: the plugin-defined marker from the
      ``ingestion_phase_settings_marker`` hook;
    - ``settings_version``: the OPAQUE ``updated_at`` of the settings entry for
      ``PHASES[phase].settings_category`` (used only for equality);
    - ``deps``: ``{upstream: <current diary marker>}`` for each upstream in
      ``PHASES[phase].depends_on``, copied from the CURRENT diary.

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
    spec = PHASES.get(phase)
    if spec is None:
        # unknown (external) phase: record a minimal diary entry so the
        # machine can advance past it (no marker / settings version / deps —
        # the external registrant owns the phase's material inputs). The
        # atomic-completion invariant still holds: the entry is written ONLY
        # after the external dispatch reported ``done``.
        return await record_phase_completed(
            ccat.agent_key, scope, source_name, phase,
            marker=None, settings_version=None, deps={},
        )

    marker = await ccat.plugin_manager.execute_hook(
        "ingestion_phase_settings_marker", phase, caller=cat
    )
    settings_version = None
    if spec.settings_category is not None:
        settings_doc = await crud_settings.get_settings_by_category(
            getattr(cat, "agent_key", None) or ccat.agent_key, spec.settings_category
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

    return await record_phase_completed(
        ccat.agent_key,
        scope,
        source_name,
        phase,
        marker=marker,
        settings_version=settings_version,
        deps=deps,
    )


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
        await ccat.plugin_manager.execute_hook(
            "before_ingestion_status_completed", source_name, caller=cat
        )
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


async def _run_parsing_phase(
    ccat, collection_name, source, source_name, chat_id, cat, rabbit_hole, active_chunker_name,
):
    """Run the ``parsing_chunking`` phase body (clean-sweep -> parse -> store).

    Deletes every artifact of this phase and the following ones (text chunk
    points, image points, saved image files), resolves the source bytes (from
    the in-memory ``source.content``, a re-downloaded URL, or the persisted
    file on disk), parses + chunks ALWAYS extracting the images, and stores the
    text chunks and image points with EMPTY vectors.

    Returns:
        ``(resolved_source_name, stored_points)`` — the (possibly re-resolved)
        source name and the stored empty-vector points, which the embedding
        phase consumes on the next loop iteration.
    """
    # 1. clean-sweep: remove every artifact of this phase and the following ones.
    await _clear_source_artifacts(ccat, collection_name, source_name, chat_id)

    # ensure the rabbit_hole context is wired on the ccat (URL download +
    # parsing helpers read ``self.cat``)
    if rabbit_hole.cat is None:
        await rabbit_hole.setup(ccat)

    # 2. resolve the source bytes: file already on disk (via the resume/upload),
    #    or a URL to (re)download.
    if source.content is not None:
        file_io = source.content
        file_bytes = file_io.read()
        content_type, _ = guess_file_type(file_io)
    elif is_url(source_name):
        # URL: re-download via the core source resolver.
        source_name_resolved, file_bytes, content_type, _ = await rabbit_hole._resolve_source_bytes(
            source_name, source_name, None
        )
        if file_bytes is None:
            raise Exception(f"Something went wrong with the source '{source_name}'")
        if source_name_resolved:
            source_name = source_name_resolved
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

    # 4. store text chunks + image points with EMPTY vectors.
    sha256 = hashlib.sha256()
    sha256.update(file_bytes or b"")
    file_hash = sha256.hexdigest()
    stored = await _store_empty_vectors(
        ccat, collection_name, source_name, chat_id,
        docs, images, file_hash=file_hash, metadata=source.metadata or {},
    )
    return source_name, stored


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

    existing_points, _ = await ccat.vector_memory_handler.get_all_tenant_points(
        str(collection_name), with_vectors=False
    )

    rabbit_hole = ccat.rabbit_hole
    embedder = await ccat.embedder()
    active_embedder_name = getattr(embedder, "name", None) or get_nlp_object_name(embedder, "default_embedder")
    chunker = getattr(ccat, "chunker", None)
    active_chunker_name = (
        str(chunker.name) if chunker is not None and getattr(chunker, "name", None) else
        get_nlp_object_name(chunker, "default_chunker")
    )
    owner = f"reembed-{os.getpid()}"
    counter = 0

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
                    str(collection_name), metadata={"source": source_name}
                )
                await _set_status(ccat, source_name, IngestionStatus.COMPLETED, chat_id=chat_id)
                continue

        # ---- decide the start phase from the status doc + phase probe ----
        doc = await get_status(ccat.agent_key, scope, source_name)
        doc_status = (doc or {}).get("status")
        doc_embedder = (doc or {}).get("embedder_name")
        doc_chunker = (doc or {}).get("chunker_name")
        doc_phase = (doc or {}).get("phase")

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
            if not pending:
                # every recorded phase is still fresh against the current settings
                # -> the source is up to date: skip WITHOUT claiming.
                log.debug(
                    f"Agent id: {ccat._id}. Source {source_name}: no stale phases "
                    f"(diary {sorted(get_completed_phases(doc))}), skipping"
                )
                continue
            start_phase = pending[0]["phase"]
        else:
            # no status doc: no diary, so the probe has nothing to compare against.
            # Fall back to the legacy heuristic: embedding if reusable points exist
            # (chunk-reuse), else a full re-ingest from parsing_chunking.
            has_points = any(
                (p.payload or {}).get("metadata", {}).get("source") == source_name
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

        # ---- claim the per-source work (skip if another worker holds it) ----
        if doc is not None:
            claimed = await claim_source_for_resume(
                ccat.agent_key,
                scope,
                source_name,
                stale_after=_claim_stale_after() if stale_after is None else stale_after,
                owner=owner,
                claim_completed=(doc_status == IngestionStatus.COMPLETED.value),
            )
            if claimed is None:
                # another worker is already (re)processing this source
                continue

        try:
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
                if _should_stop_for_error(doc):
                    # ERROR absorbing: never advance a failed row
                    break
                entry = pending[0]
                phase = entry["phase"]

                if phase == PHASE_EMBEDDING:
                    # chunks (text) already stored for this source: chunk-reuse
                    # path. For episodic sources, the chat must still exist.
                    if chat_id and not (await ccat._find_stray_cat(str(chat_id))):
                        log.warning(
                            f"Stray cat with id {chat_id} not found. Cleaning up {source.path}/{source.name}"
                        )
                        await ccat.vector_memory_handler.delete_tenant_points(
                            str(collection_name), metadata={"source": source_name}
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
                    source_points = stored_points or [
                        p for p in existing_points
                        if (p.payload or {}).get("metadata", {}).get("source") == source_name
                    ]
                    if source_points:
                        await _embed_phase(
                            ccat, collection_name, source_name, source_points,
                            embedder, chat_id, cat, rabbit_hole,
                        )
                        stored_points = None
                        counter += 1
                    else:
                        # no stored points (fresh source / points were cleared):
                        # fall through to a full re-ingest from parsing_chunking.
                        phase = PHASE_PARSING_CHUNKING
                        await set_phase(
                            ccat.agent_key, scope, source_name,
                            PHASE_PARSING_CHUNKING,
                            chunker_name=active_chunker_name,
                            type_="url" if is_url(source_name) else "file",
                            chat_id=chat_id,
                        )
                        source_name, stored_points = await _run_parsing_phase(
                            ccat, collection_name, source, source_name, chat_id, cat,
                            rabbit_hole, active_chunker_name,
                        )
                elif phase == PHASE_PARSING_CHUNKING:
                    # ---- parsing_chunking phase body ----
                    await set_phase(
                        ccat.agent_key, scope, source_name,
                        PHASE_PARSING_CHUNKING,
                        chunker_name=active_chunker_name,
                        type_="url" if is_url(source_name) else "file",
                        chat_id=chat_id,
                    )
                    source_name, stored_points = await _run_parsing_phase(
                        ccat, collection_name, source, source_name, chat_id, cat,
                        rabbit_hole, active_chunker_name,
                    )
                else:
                    # external phase (e.g. MyGRAPH): dispatched through the
                    # ``ingestion_phase_run`` hook (tri-state, hooks.py).
                    # A phase nobody implements (hook returns None) fails hard
                    # — it is an error, never a silent success. A phase that
                    # reports ``done`` is recorded by the shared ATOMIC
                    # completion write below and re-probed; ``not_ready`` is
                    # retried (bounded) and an exceeded bound marks the row
                    # ERROR. The machine stops on any ERROR (absorbing).
                    if not await _run_external_phase(
                        ccat, cat, scope, source_name, phase, chat_id
                    ):
                        # the row was marked ERROR (fail-hard / retry bound):
                        # stop the machine; the ERROR-absorbing check above
                        # never advances nor resurrects it.
                        break

                # ---- ATOMIC COMPLETION: record the diary entry (never at
                # ---- phase start): marker + settings_version + deps ----
                await _record_phase(ccat, cat, scope, source_name, phase, chat_id=chat_id)

                # ---- re-probe with the updated diary ----
                doc = await get_status(ccat.agent_key, scope, source_name)
                pending = await _probe_pending(ccat, cat, source_name, doc)

                # ---- terminal COMPLETED when nothing is stale anymore ----
                if not pending and not _should_stop_for_error(doc):
                    await _complete_source(ccat, cat, scope, source_name, chat_id)
                    break
                if not pending:
                    # an error row appeared while recording: never resurrect it
                    break

        except Exception as e:  # noqa: BLE001 - a failing source must not abort the pass
            log.error(f"Agent id: {ccat._id}. Error re-embedding source {source_name}: {e}")
            await _set_status(ccat, source_name, IngestionStatus.ERROR, error=str(e), chat_id=chat_id)

    log.info(f"Agent id: {ccat._id}. Embedded {counter} files to the vector memory")


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
        await ccat.plugin_manager.execute_hook(
            "rabbithole_ingestion_start", source_name, metadata or {},
            is_url(source_name), caller=cat,
        )
        if store_file and file_data is not None:
            await ccat.save_file(file_data, content_type, source_name, chat_id)
        await ccat.plugin_manager.execute_hook(
            "rabbithole_ingestion_processing", source_name, caller=cat,
        )
        heartbeat_interval = get_env_int("CAT_INGESTION_HEARTBEAT_SECONDS") or 30
        await ccat.plugin_manager.execute_hook(
            "rabbithole_processing_heartbeat_start",
            source_name, scope, heartbeat_interval, caller=cat,
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
            await ccat.plugin_manager.execute_hook(
                "rabbithole_processing_heartbeat_stop",
                source_name, scope, caller=cat,
            )

    async def run(self, lizard) -> bool:
        """Resolve the current embedder and run the re-embed pass."""
        success = False
        try:
            embedder = await lizard.embedder()
            embedder_name = embedder.name
            embedder_size = embedder.size

            ccat_ids = await crud_settings.get_agents_main_keys()
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
