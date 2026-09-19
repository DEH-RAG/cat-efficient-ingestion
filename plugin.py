"""Efficient ingestion — replaceable ingestion engine.

Registers ``EfficientIngestionConfiguration`` through the
``factory_allowed_ingestions`` hook: the core resolves the engine through the
ServiceFactory (``ingestion`` category) when the embedder changes and, when
this plugin is present, prefers our efficient implementation.

The plugin is system-level: factory entries and the engine selection live in
the global ``system:agent`` store under the ``ingestion`` category (see
``configs.py`` and the plugin's settings endpoints).

It also owns the token-budget chunk split (moved from the core
``RabbitHole._split_oversized``, a MyCAT-only addition): the
``finalize_oversized_chunks`` and ``before_rabbithole_stores_documents`` hooks
resolve the active embedder and split oversized chunks into budget-compliant
sub-chunks. With this plugin disabled the core returns to upstream parity (no
token-budget split).
"""

from typing import List
import asyncio
from langchain_core.documents import Document

from cat import hook
from cat.db import crud
from cat.db.models import generate_timestamp
from cat.log import log

from .configs import EfficientIngestionConfiguration
from .phases import PHASES
from .registry import (
    PHASE_DOWNLOADING,
    PHASE_PARSING_CHUNKING,
    IngestionStatus,
    backfill_completed_phases,
    clear_agent,
    delete_status,
    get_status,
    ingestion_canceled,
    set_phase,
    set_status,
    status_key,
)
from .split import split_oversized


def _scope_and_chat(cat):
    """Resolve ``(scope, chat_id)`` from the hook caller.

    A StrayCat carries a conversation ``id``; a CheshireCat does not, so the
    scope is the agent KB. Mirrors ``chat_id = self.stray.id if self.stray else None``.
    """
    if hasattr(cat, "id"):
        return cat.id, cat.id
    return "agent", None


def _source_type(source: str, is_url: bool = False) -> str:
    if is_url or source.startswith("http"):
        return "url"
    return "file"


def _chunker_name(cat) -> str | None:
    """Best-effort name of the active chunker, read from the caller.

    The chunker is resolved lazily by the core (``cat.chunker`` on
    BotMixin); a failure here must never break a status write.
    """
    try:
        chunker = getattr(cat, "chunker", None)
        if chunker is None:
            return None
        name = getattr(chunker, "name", None)
        return str(name) if name else None
    except Exception:  # noqa: BLE001 - status writes must never fail
        return None


_heartbeat_tasks: dict = {}


def _heartbeat_key(agent_id: str, scope: str, source: str) -> str:
    return f"{agent_id}:{scope}:{source}"


async def _cancel_heartbeat(agent_id: str, scope: str, source: str) -> None:
    task = _heartbeat_tasks.pop(_heartbeat_key(agent_id, scope, source), None)
    if task is None:
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


async def _heartbeat_status(agent_id: str, scope: str, source: str, interval: float) -> None:
    """Periodically bump ``updated_at`` of a PROCESSING row while it is being handled.

    A long parse/embed can otherwise make the row look stale and get re-claimed
    by another worker (see ``claim_source_for_resume``). Stops as soon as the
    row leaves the PROCESSING state.

    The deletion marker is checked on EVERY tick: this is the fast-abort path
    for long phases (embedding can take minutes), so a deleted agent's worker
    self-aborts within one interval instead of letting the delete's bounded
    quiesce-wait time out.
    """
    while True:
        await asyncio.sleep(interval)
        if await ingestion_canceled(agent_id):
            # Agent deleted mid-ingestion: mark the row ERROR and stop. The
            # ERROR write goes through a DIRECT crud.store because set_status
            # is itself guarded by ingestion_canceled (it returns {} without
            # writing when canceled) — the direct store bypasses the guard so
            # the row ends terminal instead of dangling as PROCESSING. Only
            # overwrite a row still in PROCESSING (the loop's own invariant).
            current = await get_status(agent_id, scope, source)
            if current and current.get("status") == IngestionStatus.PROCESSING.value:
                await crud.store(
                    status_key(agent_id, scope, source),
                    {
                        **current,
                        "status": IngestionStatus.ERROR.value,
                        "error": "Agent deleted — ingestion aborted",
                        "updated_at": generate_timestamp(),
                    },
                )
            return
        current = await get_status(agent_id, scope, source)
        if current and current.get("status") == IngestionStatus.PROCESSING.value:
            await set_status(
                agent_id,
                scope,
                source,
                type_=current.get("type", "file"),
                status=IngestionStatus.PROCESSING,
                chat_id=current.get("chat_id"),
            )
        else:
            return


@hook(priority=0)
def factory_allowed_ingestions(allowed, lizard):
    """Register the efficient ingestion engine as a factory option."""
    return list(allowed) + [EfficientIngestionConfiguration]


@hook(priority=0)
async def finalize_oversized_chunks(docs: List[Document], cat) -> List[Document]:
    """Split oversized chunks so none exceeds the active embedder's input limit.

    Fired by the core RabbitHole after chunking/splitting: resolves the active
    embedder and applies the token-budget split (pure fold, metadata carried
    forward). No-op-compatible when the embedder has no ``max_input_tokens``.
    """
    embedder = await cat.lizard.embedder()
    return split_oversized(docs, embedder)


@hook(priority=100)
async def before_rabbithole_stores_documents(docs: List[Document], cat) -> List[Document]:
    """Apply the token-budget split on the store path of the core engine.

    Fired by ``RabbitHole.store_documents`` before the embed/store loop (e.g.
    also covering documents added by other post-chunking hooks): split oversized
    chunks so the 1:1 embed/store pairing is never broken.
    """
    embedder = await cat.lizard.embedder()
    return split_oversized(docs, embedder)


@hook(priority=0)
async def rabbithole_ingestion_start(source, metadata, is_url, cat) -> None:
    """Record that an ingestion is about to begin (source known, nothing stored yet)."""
    if await ingestion_canceled(cat.agent_key):
        # Agent deleted (or master key gone): never write an UPLOADED row for
        # a dead agent — the teardown wipes the namespace anyway.
        log.debug(f"ingestion aborted: agent {cat.agent_key} deleted")
        return
    scope, chat_id = _scope_and_chat(cat)
    await set_status(
        cat.agent_key,
        scope,
        source,
        type_=_source_type(source, is_url),
        status=IngestionStatus.UPLOADED,
        chat_id=chat_id,
    )


@hook(priority=0)
async def rabbithole_url_downloading(url, filename, cat) -> None:
    """Record that a URL download is about to start."""
    if await ingestion_canceled(cat.agent_key):
        # Same gate as rabbithole_ingestion_start: no DOWNLOADING row for a
        # deleted agent.
        log.debug(f"ingestion aborted: agent {cat.agent_key} deleted")
        return
    scope, chat_id = _scope_and_chat(cat)
    await set_status(
        cat.agent_key,
        scope,
        url,
        type_="url",
        status=IngestionStatus.DOWNLOADING,
        chat_id=chat_id,
        phase=PHASE_DOWNLOADING,
    )


@hook(priority=0)
async def rabbithole_url_download_completed(url, filename, cat) -> None:
    """Record that a URL download completed successfully."""
    scope, chat_id = _scope_and_chat(cat)
    await set_status(
        cat.agent_key,
        scope,
        url,
        type_="url",
        status=IngestionStatus.DOWNLOADED,
        chat_id=chat_id,
        phase=PHASE_DOWNLOADING,
    )


@hook(priority=0)
async def rabbithole_ingestion_processing(source, cat) -> None:
    """Record that the source is being parsed, chunked and embedded."""
    if await ingestion_canceled(cat.agent_key):
        # Agent deleted mid-ingestion: skip the write entirely. Marking the
        # row ERROR via set_status would be a NO-OP here — set_status itself
        # aborts (returns {}) when ingestion_canceled — and the teardown
        # wipes the whole namespace anyway, so the row is left as-is.
        return
    scope, chat_id = _scope_and_chat(cat)
    await set_status(
        cat.agent_key,
        scope,
        source,
        type_=_source_type(source),
        status=IngestionStatus.PROCESSING,
        chat_id=chat_id,
        phase=PHASE_PARSING_CHUNKING,
        chunker_name=_chunker_name(cat),
    )


@hook(priority=0)
async def after_rabbithole_stored_documents(source, stored_points, cat) -> None:
    """Record that the source was stored successfully (phase-probe gated).

    The hook fires in the ``finally`` block of ``ingest_file`` and mid-loop
    from the EffING engine's ``_embed_phase``, so it must NOT write COMPLETED
    while any work phase is still pending: it probes
    ``ingestion_phase_pending`` first (clock-free, diary backfilled and
    converted to the MyGRAPH-compatible ``list[dict]`` shape).

    - **Pending phases**: the row is advanced to the next pending phase via
      ``set_phase`` and stays PROCESSING — the phase machine (dispatcher)
      owns the terminal write, and this hook never double-completes.
    - **Nothing pending**: the ``before_ingestion_status_completed`` gate runs
      first — a raise forces ERROR, otherwise the row is written COMPLETED
      with the phase diary cleared. Re-probing first makes any double write
      harmless (idempotent).
    - Never overwrites an already-recorded ERROR state, and ignores the
      unresolved empty source. A canceled agent (delete marker / missing
      master key) is skipped entirely: no phase advance, no COMPLETED.
    """
    if not source:
        return
    if await ingestion_canceled(cat.agent_key):
        # Agent deleted mid-ingestion: skip the write (same reasoning as
        # rabbithole_ingestion_processing — set_status would no-op and the
        # teardown wipes the namespace). A canceled agent never gets a phase
        # advance nor a terminal COMPLETED.
        return
    scope, chat_id = _scope_and_chat(cat)
    current = await get_status(cat.agent_key, scope, source)
    if current and current.get("status") == IngestionStatus.ERROR.value:
        return

    # Clock-free phase probe: backfill the diary, convert to the hook list
    # shape, thread it into the accumulator and keep only the stale phases.
    completed = await backfill_completed_phases(current or {}, list(PHASES.keys()))
    completed_as_list = [dict(e, phase=p) for p, e in completed.items()]
    pending = await cat.plugin_manager.execute_hook(
        "ingestion_phase_pending", [], source, completed_as_list, caller=cat
    )
    pending = [p for p in (pending or []) if p and p.get("phase")]

    if pending:
        # A phase is still stale: advance to it and leave the row PROCESSING;
        # the phase machine owns the terminal COMPLETED write.
        await set_phase(
            cat.agent_key,
            scope,
            source,
            pending[0]["phase"],
            type_=_source_type(source),
            chat_id=chat_id,
        )
        return

    # Nothing pending: final gate, then the terminal state.
    try:
        await cat.plugin_manager.execute_hook(
            "before_ingestion_status_completed", source, caller=cat
        )
    except Exception as e:  # noqa: BLE001 - the gate decides the terminal state
        await set_status(
            cat.agent_key,
            scope,
            source,
            type_=_source_type(source),
            status=IngestionStatus.ERROR,
            chat_id=chat_id,
            error=str(e),
            clear_phase=True,
        )
        return
    await set_status(
        cat.agent_key,
        scope,
        source,
        type_=_source_type(source),
        status=IngestionStatus.COMPLETED,
        chat_id=chat_id,
        clear_phase=True,
    )


@hook(priority=0)
async def rabbithole_ingestion_error(source, error, cat) -> None:
    """Record that the ingestion failed, with the error message."""
    scope, chat_id = _scope_and_chat(cat)
    await set_status(
        cat.agent_key,
        scope,
        source,
        type_=_source_type(source),
        status=IngestionStatus.ERROR,
        chat_id=chat_id,
        error=str(error),
    )

@hook(priority=0)
def rabbithole_processing_heartbeat_start(source: str, scope: str, interval: float, cat) -> None:
    """Spawn the heartbeat task that keeps the PROCESSING row fresh."""
    if cat is None:
        return
    agent_id = getattr(cat, "agent_key", None)
    if not agent_id:
        return
    key = _heartbeat_key(agent_id, scope, source)
    # safety: stop any pre-existing heartbeat for the same row
    if key in _heartbeat_tasks:
        _heartbeat_tasks[key].cancel()
    _heartbeat_tasks[key] = asyncio.ensure_future(
        _heartbeat_status(agent_id, scope, source, interval)
    )


@hook(priority=0)
async def rabbithole_processing_heartbeat_stop(source: str, scope: str, cat) -> None:
    """Cancel the heartbeat task spawned on processing start."""
    if cat is None:
        return
    agent_id = getattr(cat, "agent_key", None)
    if not agent_id:
        return
    await _cancel_heartbeat(agent_id, scope, source)


@hook(priority=0)
async def after_file_manager_file_deleted(filename: str, scope: str, cat) -> None:
    """Drop the per-source status row when the file is deleted."""
    if cat is None:
        return
    agent_id = getattr(cat, "agent_key", None)
    if not agent_id:
        return
    await delete_status(agent_id, scope, filename)


@hook(priority=0)
async def after_cheshire_cat_destroy(agent_id: str, cat) -> None:
    """Drop every ingestion-status row of a destroyed agent.

    Fired by ``CheshireCat.destroy()`` once the agent's resources are gone;
    the whole ``agents:<agent_id>:ingestion:*`` namespace belongs to this
    plugin, so the cleanup lives here (the core only fires the hook).
    """
    await clear_agent(agent_id)
