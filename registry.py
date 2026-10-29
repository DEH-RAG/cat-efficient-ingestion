"""Redis-backed ingestion-status registry of the ``efficient_ingestion`` core plugin.

Key namespace (owned by this plugin; consumed by the destroy purge and the
re-embed flow): ``agents:{agent_id}:ingestion:{scope}:{sha256(source)}`` where
``scope`` is ``"agent"`` for agent-KB ingestion or a conversation ``chat_id``.

Every status transition is written as a Redis-JSON document via ``cat.db.crud``
(no raw Redis client is used here). Enumeration is crud-based too:
``list_statuses`` walks the ``agents:{agent_id}:ingestion:*`` namespace through
the agent-scoped host CRUD API (``get_ingestion_status_keys``) and reads every
doc via ``crud.read``; ``clear_agent`` wipes the namespace via
``crud.destroy`` and re-stores the delete marker. No raw Redis client.

**Stuck download policy: resumable, never purgeable while in flight.**
``downloading``/``downloaded`` rows are NOT dead ends: :func:`claim_source_for_resume`
accepts them (like ``uploaded``/``processing``/``error``) so the resume sweep
reclaims a download that crashed between download and processing and restarts
it from the URL. The ``updated_at`` staleness gate is the age/lease threshold,
and the downloading writer keeps the row fresh through a real heartbeat
(:func:`plugin._heartbeat_status`, started by ``rabbithole_url_downloading``),
so an active download is never stolen; a failed download's heartbeat is
time-bounded and the row ages out (becomes resumable). In-flight rows are still never purged by
the reconcile/DELETE paths (the DELETE endpoint keeps refusing ``uploaded`` /
``downloading`` / ``downloaded`` / ``processing``); recovery is the resume
sweep's job, not the purge path's.
"""
import hashlib
from typing import Dict, List, Optional

from cat.db import crud
from cat.db.crud import escape_glob
from cat.db.cruds import settings as crud_settings
from cat.db.models import generate_timestamp
from cat.env import get_env_int
from cat.utils import Enum


class IngestionStatus(Enum):
    """Lifecycle states of a source ingestion.

    Files: ``uploaded -> processing -> completed | error``.
    URLs add ``downloading -> downloaded`` between ``uploaded`` and ``processing``.
    """
    UPLOADED = "uploaded"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"


#: Work phases (diary of what the source is doing), stored in ``doc["phase"]``.
#: ``parsing_chunking`` and ``embedding`` are the recovery-relevant phases: a
#: crash while ``phase=embedding`` means chunks are intact and only the vectors
#: must be recomputed; ``phase=parsing_chunking`` means chunks must be
#: regenerated (full re-ingest). ``downloading`` applies to URLs.
PHASE_DOWNLOADING = "downloading"
PHASE_PARSING_CHUNKING = "parsing_chunking"
PHASE_EMBEDDING = "embedding"


def status_key(agent_id: str, scope: str, source: str) -> str:
    """Redis key for a source's ingestion status.

    Args:
        agent_id: The agent (chatbot) id.
        scope: ``"agent"`` for the agent KB, or a conversation ``chat_id``.
        source: The ingested file name or URL.

    Returns:
        ``agents:{agent_id}:ingestion:{scope}:{sha256(source)}``
    """
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    return f"agents:{agent_id}:ingestion:{scope}:{digest}"


#: Key part of the deletion marker, stored INSIDE the ``agents:{agent_id}:ingestion:*``
#: namespace on purpose: the namespace wipe includes it, and the agent master
#: keys (``agents:*:agent``) never list it.
DELETE_MARKER_KEY_PART = "delete"


def delete_marker_key(agent_id: str) -> str:
    """Redis key of the deletion marker for an agent.

    The marker is persistent (no TTL): written by the delete flow, removed
    ONLY at teardown completion. Invariant: marker present <=> teardown
    incomplete.

    Args:
        agent_id: The agent (chatbot) id.

    Returns:
        ``agents:{agent_id}:ingestion:delete``
    """
    return f"agents:{agent_id}:ingestion:{DELETE_MARKER_KEY_PART}"


async def _list_status_keys(agent_id: str) -> List[str]:
    """List the agent's ``agents:{agent_id}:ingestion:*`` keys (crud-only).

    Enumeration goes through the host CRUD API
    ``crud_settings.get_ingestion_status_keys``: a read-only, agent-scoped
    listing that returns the full keys of the agent's ingestion namespace
    (delete marker included, filtered by the callers) and glob-escapes the
    agent id. Never a raw client.
    """
    return await crud_settings.get_ingestion_status_keys(agent_id)


async def set_delete_marker(agent_id: str) -> None:
    """Write the deletion marker for an agent (idempotent)."""
    await crud.store(delete_marker_key(agent_id), {"delete_marker": True})


async def has_delete_marker(agent_id: str) -> bool:
    """Whether the deletion marker is present for an agent."""
    return await crud.read(delete_marker_key(agent_id)) is not None


async def clear_delete_marker(agent_id: str) -> None:
    """Remove the deletion marker for an agent (teardown completion)."""
    await crud.delete(delete_marker_key(agent_id))


async def ingestion_canceled(agent_id: str) -> bool:
    """Whether ingestion for an agent must self-abort.

    True when the deletion marker is present OR the agent's master key
    (``agents:{agent_id}:agent``) is absent — the master-key check makes
    ghost agents (peripheral keys without a master) also "canceled", so no
    worker ever writes status for a dead agent. Absence is detected through
    the official ``cat.db.crud`` API (``read(...) is None``), never a raw
    client. This is an EXACT-key read (not a glob pattern), so the id is used
    literally; only the scan/destroy patterns below are glob-escaped.
    """
    return await has_delete_marker(agent_id) or (
        await crud.read(f"agents:{agent_id}:agent")
    ) is None


async def _read_doc(key: str) -> Optional[Dict]:
    """Read a status doc, unwrapping the RedisJSON array wrapper."""
    value = await crud.read(key)
    if value is None:
        return None
    if isinstance(value, list):
        return value[0] if value else None
    return value


async def set_status(
    agent_id: str,
    scope: str,
    source: str,
    *,
    type_: str,
    status: IngestionStatus,
    chat_id: Optional[str] = None,
    error: Optional[str] = None,
    phase: str | None = None,
    embedder_name: str | None = None,
    chunker_name: str | None = None,
    clear_phase: bool = False,
    extra: Optional[Dict] = None,
    completed_phases: Optional[Dict] = None,
) -> Dict:
    """Write (or update) the ingestion-status doc for a source.

    ``created_at`` is preserved across updates; ``updated_at`` is bumped on
    every write; ``error_at`` is set when ``status`` is ERROR.

    **Merge semantics (important)**: ``phase`` / ``embedder_name`` /
    ``chunker_name`` are updated ONLY when explicitly provided (not None).
    Existing callers that write status without these fields (heartbeat,
    ``after_rabbithole_stored_documents``, ``mark_file_missing``) therefore
    never clobber the engine-written phase/embedder/chunker of a row.
    ``clear_phase`` is the explicit way to drop the phase when the source
    reaches a terminal state (e.g. ``completed``).

    **Completed-phases diary**: ``completed_phases`` is a clock-free diary of
    which work phases completed successfully, keyed by phase id::

        completed_phases: {phase: {"marker": Any, "settings_version": str|None,
                                   "deps": {upstream_phase: upstream_marker}}}

    It is MERGED by phase: an incoming key replaces only its own entry, every
    other phase entry is preserved, and a write WITHOUT ``completed_phases``
    never wipes an existing diary. ``settings_version`` is an OPAQUE copy of
    the settings entry's ``updated_at``, used only for equality, never as a
    time; correctness never depends on timestamps.

    Args:
        agent_id: The agent (chatbot) id.
        scope: ``"agent"`` for the agent KB, or a conversation ``chat_id``.
        source: The ingested file name or URL.
        type_: ``"file"`` or ``"url"``.
        status: The new lifecycle state.
        chat_id: The conversation id, when the ingestion is chat-scoped.
        error: The error message, when ``status`` is ERROR.
        phase: The current work phase (``downloading``, ``parsing_chunking``,
            ``embedding``) — see :func:`set_phase`. Updated only when provided.
        embedder_name: The embedder that produced the stored vectors. Updated
            only when provided.
        chunker_name: The chunker that produced the stored chunks. Updated
            only when provided.
        clear_phase: When True, drop the ``phase`` field from the doc.
        extra: Optional extra fields merged into the stored doc.
        completed_phases: Optional diary updates, merged by phase: each key
            replaces only its own entry; other phase entries are preserved.
            Omitted (None) -> the existing diary is left untouched.

    Returns:
        The stored status document.
    """
    if await ingestion_canceled(agent_id):
        return {}

    key = status_key(agent_id, scope, source)
    now = generate_timestamp()

    existing = await _read_doc(key)
    created_at = existing.get("created_at", now) if existing else now

    # Start from a copy of the existing doc so fields NOT provided in this
    # write (phase/embedder_name/chunker_name/extra) survive — the merge
    # semantics that lets a plain lifecycle write coexist with engine-written
    # metadata. Fields explicitly overwritten below always win.
    doc = dict(existing or {})
    doc.update({
        "source": source,
        "scope": scope,
        "chat_id": chat_id,
        "type": type_,
        "status": status,
        "error": error,
        "error_at": now if status == IngestionStatus.ERROR else None,
        "created_at": created_at,
        "updated_at": now,
    })
    # Merge semantics: only update phase/embedder/chunker when explicitly given
    if phase is not None:
        doc["phase"] = phase
    if embedder_name is not None:
        doc["embedder_name"] = embedder_name
    if chunker_name is not None:
        doc["chunker_name"] = chunker_name
    if clear_phase:
        doc.pop("phase", None)
    # A row created by the no-doc claim path carries ``resume_created`` so a
    # second worker cannot immediately re-claim it. Once the row reaches a
    # terminal state the transient marker is meaningless and must go, otherwise
    # a later completed-row re-embed claim would be refused.
    if status in (IngestionStatus.COMPLETED, IngestionStatus.ERROR):
        doc.pop("resume_created", None)
    if extra:
        doc.update(extra)
    # Completed-phases diary: merge by phase so a re-recorded phase replaces
    # only its own entry and never clobbers the rest of the diary. The diary
    # param wins over any legacy `extra` field.
    if completed_phases:
        stored = doc.get("completed_phases")
        diary = dict(stored) if isinstance(stored, dict) else {}
        for phase_id, entry in completed_phases.items():
            diary[phase_id] = entry
        doc["completed_phases"] = diary
    await crud.store(key, doc)
    return doc


async def set_phase(
    agent_id: str,
    scope: str,
    source: str,
    phase: str,
    *,
    embedder_name: str | None = None,
    chunker_name: str | None = None,
    type_: str = "file",
    chat_id: str | None = None,
    completed_phases: Optional[Dict] = None,
) -> dict:
    """Transition a source into a work phase.

    Atomic ``status=PROCESSING`` + ``phase=<phase>`` write, bumping
    ``updated_at`` so the row is restartable on crash and never looks stale
    to another worker mid-phase. Records ``embedder_name`` (for the
    ``embedding`` phase) or ``chunker_name`` (for the ``parsing_chunking``
    phase) at the START of the phase that consumes them.

    Args:
        agent_id: The agent (chatbot) id.
        scope: ``"agent"`` or a conversation ``chat_id``.
        source: The ingested file name or URL.
        phase: ``downloading`` | ``parsing_chunking`` | ``embedding``.
        embedder_name: Recorded when provided (used at ``embedding`` start).
        chunker_name: Recorded when provided (used at ``parsing_chunking`` start).
        type_: ``"file"`` or ``"url"``.
        chat_id: The conversation id, when chat-scoped.
        completed_phases: Optional diary updates, forwarded to
            :func:`set_status` (merged by phase, see there).

    Returns:
        The stored status document.
    """
    return await set_status(
        agent_id,
        scope,
        source,
        type_=type_,
        status=IngestionStatus.PROCESSING,
        chat_id=chat_id,
        phase=phase,
        embedder_name=embedder_name,
        chunker_name=chunker_name,
        completed_phases=completed_phases,
    )


async def get_status(agent_id: str, scope: str, source: str) -> Optional[Dict]:
    """Read the ingestion-status doc for a source, or None if absent."""
    return await _read_doc(status_key(agent_id, scope, source))


def get_completed_phases(doc: Optional[Dict]) -> dict:
    """Return the completed-phases diary of a status doc, or ``{}``.

    Clock-free diary of which work phases completed successfully, keyed by
    phase id::

        {phase: {"marker": Any, "settings_version": str|None,
                 "deps": {upstream_phase: upstream_marker}}}

    ``marker`` is the plugin-defined marker of the settings the phase ran
    with; ``settings_version`` is an OPAQUE copy of the settings entry's
    ``updated_at`` (used only for equality, never as a time); ``deps`` maps
    each upstream phase to the marker it ran with. No timestamps are used for
    correctness.

    Args:
        doc: The status document (as returned by :func:`get_status`), or None.

    Returns:
        The diary dict, or ``{}`` when absent or malformed.
    """
    if not doc:
        return {}
    diary = doc.get("completed_phases")
    return diary if isinstance(diary, dict) else {}


async def backfill_completed_phases(doc: Optional[Dict], phase_ids) -> dict:
    """Return the completed-phases diary, backfilling legacy completed rows.

    Rows completed BEFORE the clock-free ``completed_phases`` diary existed
    carry no diary. For those rows (status COMPLETED with an absent/empty
    diary) this returns a SENTINEL diary with every ``phase_ids`` phase present
    but marked ``marker=None`` / ``settings_version=None`` / ``deps={}``: their
    settings version is unknown, so the recovery/re-embed path conservatively
    re-runs each phase exactly once (one-time revalidation).

    Non-completed rows and completed rows that already carry a diary are
    returned UNCHANGED — the backfill never overwrites an existing diary
    (stale-state protection).

    Never raises: a ``None`` or malformed ``doc`` yields ``{}``; ``phase_ids``
    may be ``None``/empty (then the sentinel is empty too).

    Args:
        doc: The status document (as returned by :func:`get_status`), or None.
        phase_ids: Iterable of phase ids to backfill (e.g. ``parsing_chunking``,
            ``embedding``).

    Returns:
        The diary dict: the sentinel backfill for legacy completed rows, the
        existing diary when one is present, or ``{}`` otherwise.
    """
    diary = get_completed_phases(doc)
    if diary:
        return diary
    if not doc or doc.get("status") != IngestionStatus.COMPLETED.value:
        return {}
    return {
        phase: {"marker": None, "settings_version": None, "deps": {}}
        for phase in (phase_ids or [])
    }


async def record_phase_completed(
    agent_id: str,
    scope: str,
    source: str,
    phase: str,
    *,
    marker: object = None,
    settings_version: Optional[str] = None,
    deps: Optional[Dict] = None,
) -> dict:
    """Record the successful completion of one work phase in the diary.

    Writes the entry ``{phase: {"marker": marker, "settings_version":
    settings_version, "deps": deps or {}}}`` into the status doc's
    ``completed_phases`` diary, merging by phase (re-recording the same phase
    replaces only its own entry). Uses the plugin-defined ``marker`` only; no
    full settings fingerprint is stored. ``settings_version`` is an OPAQUE
    copy of the settings entry's ``updated_at`` used only for equality, never
    as a time — no timestamps are used for correctness.

    **ATOMIC COMPLETION INVARIANT**: the diary entry is written ONLY on
    successful completion of the phase, in a single atomic write — NEVER at
    phase start. A phase interrupted by a restart therefore has no entry and
    is re-run on recovery.

    Args:
        agent_id: The agent (chatbot) id.
        scope: ``"agent"`` or a conversation ``chat_id``.
        source: The ingested file name or URL.
        phase: The phase id that completed (e.g. ``parsing_chunking``).
        marker: The plugin-defined marker of the settings the phase ran with.
        settings_version: Opaque copy of the settings entry's ``updated_at``.
        deps: ``{upstream_phase: upstream_marker}`` of the phases this phase
            consumed. ``None``/missing upstreams are stored as ``{}``.

    Returns:
        The stored status document. A malformed (missing/invalid) ``phase``
        is a no-op that never crashes and returns the current doc (or ``{}``).
    """
    if not phase or not isinstance(phase, str):
        return await get_status(agent_id, scope, source) or {}
    entry = {
        "marker": marker,
        "settings_version": settings_version,
        "deps": deps or {},
    }
    return await set_status(
        agent_id,
        scope,
        source,
        type_="file",
        status=IngestionStatus.PROCESSING,
        completed_phases={phase: entry},
    )


def _resume_created_lease_seconds() -> float:
    """Time-bounded lease for the ``resume_created`` guard.

    A row created by the no-doc claim path is refused to a second claim while
    it is still being worked on. The lease makes that refusal FINITE even when
    the staleness gate is disabled (``stale_after <= 0``): the live writer keeps
    ``updated_at`` fresh through the heartbeat, and once the writer stops (crash
    or completion) the row becomes reclaimable after the lease.
    """
    heartbeat = get_env_int("CAT_INGESTION_HEARTBEAT_SECONDS")
    return float(max(30, 2 * heartbeat)) if heartbeat and heartbeat > 0 else 60.0


async def claim_source_for_resume(
    agent_id: str,
    scope: str,
    source: str,
    *,
    stale_after: float,
    owner: str,
    claim_completed: bool = False,
    create_if_missing: bool = False,
    type_: str = "file",
    chat_id: Optional[str] = None,
) -> Optional[Dict]:
    """Atomically claim a source for (re)ingestion under a per-source lock.

    Lock granularity is ``<agent>:<scope>:<sha256(source)>``, so many sources
    of the same agent can be (re)ingested concurrently by different workers,
    while two workers can never claim the SAME source at the same time.

    A source is claimable only when its current status allows a (re)start and
    its last update is older than ``stale_after`` (seconds): a fresh
    ``uploaded``/``processing``/``downloading`` row means another worker is
    already handling it, so this call returns None instead of
    double-ingesting.

    ``downloading``/``downloaded`` are resumable: a crash between the download
    and the processing phase leaves such a row behind, and the resume sweep
    must be able to reclaim it. The same ``stale_after`` gate protects a
    source that is actively downloading from being stolen.

    When ``claim_completed`` is True a ``completed`` row is ALSO claimable
    (regardless of staleness): used by the ingestion engine to re-embed a
    source whose embedder/chunker changed. The engine must re-check the
    mismatch itself before claiming. ``resume.py`` keeps the default False.

    On success the row is reset to PROCESSING with ``resume_owner`` /
    ``resume_at`` / bumped ``updated_at`` so other workers see it as taken.

    When no status row exists yet (a legacy stored source) and
    ``create_if_missing`` is True, the claim CREATES the row atomically with
    ``crud.store(..., nx=True)``: only one worker can win the create, so two
    concurrent passes can never process the same untracked source twice. The
    created row carries ``resume_created``; a second worker that finds it while
    the work is in flight is refused (see below).

    Args:
        agent_id: The agent (chatbot) id.
        scope: ``"agent"`` or a conversation ``chat_id``.
        source: The ingested file name or URL.
        stale_after: Minimum age of ``updated_at`` (seconds) for the entry to
            be claimable — protects in-flight work from being double-processed.
        owner: Identifier of the claiming worker (e.g. ``pid``).
        claim_completed: Whether a ``completed`` row may be claimed for
            re-embedding (engine passes True after its own mismatch check).
        create_if_missing: When True and no row exists, atomically create the
            PROCESSING row as the claim itself (returns None if another worker
            won the create).
        type_: ``"file"`` or ``"url"`` recorded on a created row.
        chat_id: Conversation id recorded on a created row (chat scope).

    Returns:
        The claimed (updated) status doc, or None when not claimable.
    """
    import time

    if await ingestion_canceled(agent_id):
        return None

    key = status_key(agent_id, scope, source)
    lock_pattern = f"ingestion-resume:{agent_id}:{scope}:{source}"
    async with crud.distributed_lock(lock_pattern, timeout=30, blocking_timeout=15):
        doc = await _read_doc(key)
        if doc is None:
            if not create_if_missing:
                return None
            now = generate_timestamp()
            created = {
                "source": source,
                "scope": scope,
                "chat_id": chat_id,
                "type": type_,
                "status": IngestionStatus.PROCESSING.value,
                "error": None,
                "error_at": None,
                "created_at": now,
                "updated_at": now,
                "resume_owner": owner,
                "resume_at": now,
                # Marker used to refuse an immediate second claim: the create
                # itself is the acquisition, so a concurrent worker must not
                # treat the just-created PROCESSING row as fresh-but-claimable.
                "resume_created": True,
            }
            stored = await crud.store(key, created, nx=True)
            return created if stored is not None else None

        status = doc.get("status")
        if doc.get("resume_created") and status == IngestionStatus.PROCESSING.value:
            # Created by the no-doc claim path and still in flight: refuse an
            # immediate second claim, but never forever. With a positive
            # staleness gate the gate itself bounds the refusal; with the gate
            # disabled a time-bounded lease does (the live writer refreshes
            # ``updated_at`` via the heartbeat).
            try:
                created_updated_at = float(doc.get("updated_at") or 0)
            except (TypeError, ValueError):
                created_updated_at = 0
            lease = stale_after if stale_after > 0 else _resume_created_lease_seconds()
            if time.time() - created_updated_at < lease:
                return None
        if status == IngestionStatus.COMPLETED.value:
            # Completed rows are never re-claimed by the resume sweep; the
            # engine may claim them for re-embedding only when it already
            # verified a real embedder/chunker mismatch.
            if not claim_completed:
                return None
        elif status not in (
            IngestionStatus.UPLOADED.value,
            IngestionStatus.DOWNLOADING.value,
            IngestionStatus.DOWNLOADED.value,
            IngestionStatus.PROCESSING.value,
            IngestionStatus.ERROR.value,
        ):
            return None

        # Fresh in-flight work (uploaded/downloading/downloaded/processing) is
        # only claimable when stale; completed rows (engine re-embed) have no
        # in-flight counterpart, so they bypass the staleness gate: a
        # completed row is never "in flight".
        if status != IngestionStatus.COMPLETED.value:
            try:
                updated_at = float(doc.get("updated_at") or 0)
            except (TypeError, ValueError):
                updated_at = 0
            if time.time() - updated_at < stale_after:
                # fresh: another worker is handling it right now
                return None

        now = generate_timestamp()
        doc.update({
            "status": IngestionStatus.PROCESSING.value,
            "error": None,
            "error_at": None,
            "resume_owner": owner,
            "resume_at": now,
            "updated_at": now,
        })
        await crud.store(key, doc)
        return doc


async def delete_status(agent_id: str, scope: str, source: str) -> None:
    """Delete the ingestion-status doc for a source."""
    await crud.delete(status_key(agent_id, scope, source))


async def list_statuses(agent_id: str, chat_id: Optional[str] = None) -> List[Dict]:
    """List the ingestion-status docs for an agent.

    With no ``chat_id`` only agent-scope entries are returned; with a
    ``chat_id`` only that conversation's entries are returned.

    Enumeration is crud-based: the ``agents:{agent_id}:ingestion:*`` namespace
    is listed through ``get_ingestion_status_keys`` and each status doc is
    read via ``crud.read`` — no raw Redis client. Keys that read as ``None``
    (deleted/never written) are skipped.
    """
    results: List[Dict] = []
    for key in await _list_status_keys(agent_id):
        if key == delete_marker_key(agent_id):
            # the deletion-marker doc is not a status entry
            continue
        doc = await _read_doc(key)
        if not doc:
            continue
        scope = doc.get("scope")
        if chat_id is None:
            if scope != "agent":
                continue
        elif scope != chat_id:
            continue
        results.append(doc)
    return results


async def clear_agent(agent_id: str) -> int:
    """Delete every ingestion-status key for an agent (all scopes).

    The deletion marker (``agents:{agent_id}:ingestion:delete``) is
    deliberately PRESERVED: the teardown removes it explicitly as the LAST
    step, so a wipe here must not clear it. Crud-only: the marker is read, the
    whole namespace is wiped via ``crud.destroy``, then the marker is
    re-stored if it was present.

    Returns:
        The number of keys deleted (marker excluded).
    """
    marker_key = delete_marker_key(agent_id)
    marker = await _read_doc(marker_key)
    deleted = await crud.destroy(f"agents:{escape_glob(agent_id)}:ingestion:*")
    if marker is not None:
        await crud.store(marker_key, marker)
    # ``crud.destroy`` counted every key in the namespace; the marker
    # (re-stored above) is excluded from the returned count.
    return deleted - (1 if marker is not None else 0)


async def clear_chat(agent_id: str, chat_id: str) -> int:
    """Delete every ingestion-status key for one conversation scope.

    Returns:
        The number of keys deleted.
    """
    return await crud.destroy(f"agents:{escape_glob(agent_id)}:ingestion:{escape_glob(chat_id)}:*")