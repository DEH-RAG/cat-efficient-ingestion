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
import os
import socket
import time
import uuid
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

    ``deleting`` is the file-level deletion state: written BEFORE any phase
    ``delete()`` runs, and the row is removed only after every phase delete
    succeeded. A crash therefore leaves the row ``deleting`` (resumable) instead
    of claiming a state the artifacts no longer match. It is never claimable by
    the resume sweep (it is not an in-flight ingestion status).
    """
    UPLOADED = "uploaded"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
    DELETING = "deleting"


#: Work phases (diary of what the source is doing), stored in ``doc["phase"]``.
#: ``parsing_chunking`` and ``embedding`` are the recovery-relevant phases: a
#: crash while ``phase=embedding`` means chunks are intact and only the vectors
#: must be recomputed; ``phase=parsing_chunking`` means chunks must be
#: regenerated (full re-ingest). ``downloading`` applies to URLs.
PHASE_DOWNLOADING = "downloading"
PHASE_PARSING_CHUNKING = "parsing_chunking"
PHASE_EMBEDDING = "embedding"

#: Per-(source, phase) diary state machine. Every diary entry carries an
#: explicit ``state``: ``processing`` is written BEFORE the phase body starts,
#: ``complete`` when the body returned ``done``, ``error`` with a reason when
#: it failed, and ``skipped`` with a reason when ``applies_to`` says the phase
#: does not apply. An ABSENT entry means ``pending``. An entry written by an
#: older release has no ``state`` and is read as ``complete`` (backward
#: compatible) and normalized on the next write.
PHASE_STATE_PENDING = "pending"
PHASE_STATE_PROCESSING = "processing"
PHASE_STATE_COMPLETE = "complete"
PHASE_STATE_ERROR = "error"
PHASE_STATE_SKIPPED = "skipped"

#: The explicit states stored in a diary entry (``pending`` is the absence of
#: an entry, so it is never written).
PHASE_STATES = (
    PHASE_STATE_PROCESSING,
    PHASE_STATE_COMPLETE,
    PHASE_STATE_ERROR,
    PHASE_STATE_SKIPPED,
)

#: Deprecated alias: the old "running" state is now ``processing``.
PHASE_STATE_RUNNING = PHASE_STATE_PROCESSING

#: Lease of a ``processing`` diary entry, in seconds (env-overridable). A
#: ``processing`` entry older than this is considered abandoned: its phase is
#: purged and re-run from scratch. Default 30 minutes.
PHASE_LEASE_SECONDS_DEFAULT = 1800

#: Identity of THIS process/boot, generated ONCE at import time and stable for
#: the lifetime of the process. A restart (new process, or a new container)
#: produces a DIFFERENT epoch, which is how recovery tells an entry left behind
#: by a previous boot from one written by a live worker. It is written into
#: every ``processing`` diary entry as ``owner.boot``.
BOOT_EPOCH = uuid.uuid4().hex

#: Backward-compatible alias for the previous private name.
_BOOT_ID = BOOT_EPOCH

#: Default timeout (seconds) of the ingestion locks. ``distributed_lock`` uses
#: the same value as the lock's auto-release TTL and as the acquisition
#: blocking timeout. Generous because a phase body can run for minutes.
#: Env-overridable through ``CAT_INGESTION_LOCK_TIMEOUT_SECONDS``.
INGESTION_LOCK_TIMEOUT_DEFAULT = 300


def ingestion_lock_timeout_seconds() -> float:
    """Timeout used by :func:`source_lock` and :func:`phase_lock` (env-overridable)."""
    value = get_env_int("CAT_INGESTION_LOCK_TIMEOUT_SECONDS")
    return float(value) if value is not None and value > 0 else float(INGESTION_LOCK_TIMEOUT_DEFAULT)


def source_lock(agent_id: str, source_key: str):
    """Per-source ingestion lock.

    **ORDERING RULE**: a caller acquires this SOURCE lock FIRST and, INSIDE it,
    a :func:`phase_lock`; never the reverse. The delete path takes ONLY this
    source lock (never a phase lock).

    ``source_key`` is escaped with :func:`escape_glob`, so a source whose name
    contains a glob metacharacter (``* ? [ ]``) cannot widen the lock key and
    collide with another source. The lock is acquired through
    ``cat.db.crud.distributed_lock`` — the only sanctioned locking primitive.
    """
    timeout = ingestion_lock_timeout_seconds()
    return crud.distributed_lock(
        f"ingestion-source:{agent_id}:{escape_glob(source_key)}",
        timeout=timeout,
        blocking_timeout=timeout,
    )


def phase_lock(agent_id: str, source_key: str, phase_id: str):
    """Per-``(source, phase)`` ingestion lock.

    **ORDERING RULE**: always acquired INSIDE :func:`source_lock` (source
    FIRST, phase SECOND); never the reverse and never on its own in the delete
    path. ``source_key`` is escaped with :func:`escape_glob` (see
    :func:`source_lock`).
    """
    timeout = ingestion_lock_timeout_seconds()
    return crud.distributed_lock(
        f"ingestion-phase:{agent_id}:{escape_glob(source_key)}:{phase_id}",
        timeout=timeout,
        blocking_timeout=timeout,
    )


def _pod_name() -> str:
    """Best-effort pod/container name of the current process."""
    return os.environ.get("HOSTNAME") or socket.gethostname()


def _pid_alive(pid: object) -> bool:
    """Whether ``pid`` is a live process on this host (best effort).

    ``os.kill(pid, 0)`` performs the existence check without delivering a
    signal. A ``PermissionError`` means the process exists but belongs to
    another user (still alive); any other ``OSError`` (notably
    ``ProcessLookupError``) means it is gone.
    """
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except OSError:
        return False
    return True



def phase_lease_seconds() -> float:
    """Lease duration of a ``processing`` phase entry (env-overridable)."""
    value = get_env_int("CAT_INGESTION_PHASE_LEASE_SECONDS")
    return float(value) if value is not None and value > 0 else float(PHASE_LEASE_SECONDS_DEFAULT)


def current_phase_owner() -> Dict[str, object]:
    """Identity of the process writing a ``processing`` entry.

    ``pod`` is the container/pod name, ``pid`` the process id and ``boot`` a
    per-process UUID generated at import time (a restart gets a new boot).
    """
    return {"pod": _pod_name(), "pid": os.getpid(), "boot": BOOT_EPOCH}


def phase_state(entry: Optional[Dict]) -> str:
    """Return the explicit state of a diary entry.

    An absent/malformed entry is ``pending``. A legacy entry without ``state``
    reads as ``complete`` (the pre-state-machine contract). The old ``running``
    value reads as ``processing`` so an entry written by the previous release
    is never mistaken for a finished phase.
    """
    if not isinstance(entry, dict):
        return PHASE_STATE_PENDING
    state = entry.get("state")
    if state in PHASE_STATES:
        return state
    if state == "running":
        return PHASE_STATE_PROCESSING
    return PHASE_STATE_COMPLETE


def normalize_phase_entry(entry: Optional[Dict]) -> Optional[Dict]:
    """Return ``entry`` with the explicit five-state shape, without mutating it.

    Legacy entries (no ``state``) gain ``state: complete`` and the missing
    shape fields with their neutral defaults, so the next diary write persists
    the normalized contract.
    """
    if not isinstance(entry, dict):
        return entry
    normalized = dict(entry)
    normalized["state"] = phase_state(entry)
    normalized.setdefault("reason", None)
    normalized.setdefault("started_at", None)
    normalized.setdefault("owner", None)
    normalized.setdefault("tag", "")
    normalized.setdefault("attempt", 1)
    return normalized


def derive_source_status(diary: Optional[Dict], phase_ids=None) -> str:
    """Derive the source status from the phase diary.

    - ``error`` if ANY phase is ``error``;
    - ``complete`` iff no phase is ``pending`` or ``processing``;
    - ``processing`` otherwise.

    ``phase_ids`` is the authoritative phase set for the source; an ABSENT
    entry is ``pending``. When ``phase_ids`` is omitted the diary keys are
    used (useful when no registry is resolvable).
    """
    diary = diary if isinstance(diary, dict) else {}
    ids = list(phase_ids) if phase_ids is not None else list(diary.keys())
    states = [phase_state(diary.get(phase)) for phase in ids]
    if PHASE_STATE_ERROR in states:
        return PHASE_STATE_ERROR
    if any(state in (PHASE_STATE_PENDING, PHASE_STATE_PROCESSING) for state in states):
        return PHASE_STATE_PROCESSING
    return PHASE_STATE_COMPLETE



def _entry_started_at(entry: Dict) -> Optional[float]:
    try:
        return float(entry.get("started_at"))
    except (TypeError, ValueError):
        return None


def is_phase_lease_expired(entry: Optional[Dict], now: Optional[float] = None) -> bool:
    """Whether a ``processing`` diary entry's lease is expired.

    Both contract conditions are evaluated explicitly:

    - ``started_at`` is older than ``phase_lease_seconds()`` (or missing /
      malformed -> expired: an entry with no clock cannot be trusted) OR
    - the recorded ``owner`` is NOT the current process and the entry is older
      than the same threshold.

    A non-``processing`` entry (``complete`` / ``error`` / ``skipped`` or a
    legacy no-state entry) never expires: expiry only applies to the
    ``processing`` state machine.
    """
    if phase_state(entry) != PHASE_STATE_PROCESSING:
        return False
    now = time.time() if now is None else now
    lease = phase_lease_seconds()

    started_at = _entry_started_at(entry)
    age = None if started_at is None else now - started_at
    started_expired = age is None or age > lease

    owner = entry.get("owner")
    owner_is_current = isinstance(owner, dict) and owner == current_phase_owner()
    owner_expired = (not owner_is_current) and age is not None and age > lease

    return started_expired or owner_expired


def should_abandon(entry: Optional[Dict], now: Optional[float] = None) -> bool:
    """Whether a ``processing`` entry belongs to a PREVIOUS boot and may be
    abandoned (purged, reset to ``pending`` and re-run from zero).

    Pure decision function (no I/O) with strict precedence:

    1. the entry is not ``processing`` -> ``False`` (terminal states are
       never abandoned);
    2. ``owner`` is missing/malformed -> lease fallback
       (:func:`is_phase_lease_expired`): an entry without an owner can only be
       judged by its age;
    3. ``owner.boot == BOOT_EPOCH`` -> ``False``: this is the CURRENT process,
       which is by definition alive — never touch it, even past the lease;
    4. the owner lives in this pod (same pod name) AND its pid is alive ->
       ``False``: another live worker of this pod owns the phase;
    5. a foreign pod cannot be probed locally, so its lease is the liveness
       signal: an unexpired lease keeps it untouched (alive), an expired one
       marks a dead previous boot;
    6. otherwise (previous boot, same pod, dead pid) -> ``True``.

    A live owner is never abandoned; only an entry whose owner provably belongs
    to a previous boot (or whose owner cannot be shown alive) is.
    """
    if phase_state(entry) != PHASE_STATE_PROCESSING:
        return False
    if not isinstance(entry, dict):
        return False
    now = time.time() if now is None else now

    owner = entry.get("owner")
    if not isinstance(owner, dict) or not owner.get("boot"):
        return is_phase_lease_expired(entry, now)

    if owner.get("boot") == BOOT_EPOCH:
        return False

    pod = owner.get("pod")
    current_pod = _pod_name()
    if pod == current_pod and _pid_alive(owner.get("pid")):
        return False

    if pod and pod != current_pod:
        # Foreign pod: no local liveness probe is possible. The lease is the
        # conservative signal — only an expired lease proves a dead boot.
        return is_phase_lease_expired(entry, now)

    # Same pod, previous boot, pid no longer alive.
    return True




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
        # Normalize EVERY entry on write: a legacy entry without ``state`` is
        # persisted as ``complete`` the next time the diary is touched.
        diary = {
            phase_id: normalize_phase_entry(entry)
            for phase_id, entry in diary.items()
        }
        for phase_id, entry in completed_phases.items():
            diary[phase_id] = normalize_phase_entry(entry)
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
    if not isinstance(diary, dict):
        return {}
    # Read-as-complete for the pre-state-machine entries: the returned copy
    # always carries an explicit ``state`` without rewriting Redis.
    return {
        phase: normalize_phase_entry(entry)
        for phase, entry in diary.items()
    }


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
        phase: {
            "state": PHASE_STATE_COMPLETE,
            "reason": None,
            "started_at": None,
            "owner": None,
            "tag": "",
            "attempt": 1,
            "marker": None,
            "settings_version": None,
            "deps": {},
        }
        for phase in (phase_ids or [])
    }



async def _previous_diary_entry(agent_id, scope, source, phase):
    return get_completed_phases(
        await get_status(agent_id, scope, source)
    ).get(phase)


def _attempt(previous) -> int:
    """Attempt counter for the next entry (previous + 1, or 1)."""
    try:
        return int(previous.get("attempt") or 0) + 1 if isinstance(previous, dict) else 1
    except (TypeError, ValueError):
        return 1


def _carried(previous, *, started_at, owner, tag, attempt) -> dict:
    """Shape fields carried from the previous entry, with sane defaults."""
    previous = previous if isinstance(previous, dict) else {}
    return {
        "started_at": (
            previous.get("started_at") if started_at is None else started_at
        ),
        "owner": previous.get("owner") if owner is None else owner,
        "tag": tag if tag else str(previous.get("tag") or ""),
        "attempt": int(previous.get("attempt") or attempt),
    }


async def _write_phase_entry(
    agent_id: str,
    scope: str,
    source: str,
    phase: str,
    entry: dict,
    *,
    type_: str = "file",
    chat_id: Optional[str] = None,
) -> dict:
    """Write ONE diary entry (merged by phase) and nothing else.

    The SOURCE status is derived and written by the dispatcher, never by a
    phase: this helper only records the phase state and keeps the row alive as
    ``processing`` while the phase machine is running.
    """
    return await set_status(
        agent_id,
        scope,
        source,
        type_=type_,
        status=IngestionStatus.PROCESSING,
        chat_id=chat_id,
        completed_phases={phase: entry},
    )


async def record_phase_completed(
    agent_id: str,
    scope: str,
    source: str,
    phase: str,
    *,
    marker: object = None,
    settings_version: Optional[str] = None,
    deps: Optional[Dict] = None,
    tag: str = "",
    type_: str = "file",
    chat_id: Optional[str] = None,
) -> dict:
    """Record ``state: complete`` for one phase (body returned ``done``).

    Completion is "the worker finished without error", NOT an item count: the
    caller records this entry whenever the phase body returned ``done``,
    including when it produced zero items. The entry keeps the material marker
    fields used by staleness (``marker`` / ``settings_version`` / ``deps``)
    plus the explicit five-state shape::

        {phase: {"state": "complete", "reason": None,
                 "started_at": <epoch float|None>,
                 "owner": {"pod", "pid", "boot"}|None,
                 "tag": <str>, "attempt": <int>,
                 "marker": ..., "settings_version": ..., "deps": {...}}}

    **ATOMIC COMPLETION INVARIANT**: the entry is written ONLY after the phase
    body succeeded, never at phase start. A phase interrupted before this
    write has no entry and is re-run on recovery.
    """
    if not phase or not isinstance(phase, str):
        return await get_status(agent_id, scope, source) or {}
    previous = await _previous_diary_entry(agent_id, scope, source, phase)
    carried = _carried(
        previous, started_at=None, owner=None, tag=tag, attempt=_attempt(previous)
    )
    entry = {
        "state": PHASE_STATE_COMPLETE,
        "reason": None,
        **carried,
        "marker": marker,
        "settings_version": settings_version,
        "deps": deps or {},
    }
    return await _write_phase_entry(
        agent_id, scope, source, phase, entry, type_=type_, chat_id=chat_id
    )


async def record_phase_processing(
    agent_id: str,
    scope: str,
    source: str,
    phase: str,
    *,
    type_: str = "file",
    chat_id: Optional[str] = None,
    started_at: Optional[float] = None,
    tag: str = "",
) -> dict:
    """Mark a phase ``processing`` BEFORE its body starts.

    Writes the diary entry::

        {phase: {"state": "processing", "reason": None,
                 "started_at": <epoch float>,
                 "owner": {"pod", "pid", "boot"},
                 "tag": <str>, "attempt": <n>}}

    replacing any previous entry for that phase (a stale ``complete``, an
    ``error``, a ``skipped`` or an abandoned ``processing`` one). ``attempt``
    is the previous attempt count + 1, or 1 for a fresh phase, so a recovery
    re-run is visible in the diary. A ``processing`` entry is NOT proof the
    phase completed.
    """
    if not phase or not isinstance(phase, str):
        return await get_status(agent_id, scope, source) or {}
    previous = await _previous_diary_entry(agent_id, scope, source, phase)
    entry = {
        "state": PHASE_STATE_PROCESSING,
        "reason": None,
        "started_at": generate_timestamp() if started_at is None else float(started_at),
        "owner": current_phase_owner(),
        "tag": tag if tag else str((previous or {}).get("tag") or ""),
        "attempt": _attempt(previous),
    }
    return await _write_phase_entry(
        agent_id, scope, source, phase, entry, type_=type_, chat_id=chat_id
    )


async def record_phase_error(
    agent_id: str,
    scope: str,
    source: str,
    phase: str,
    reason: str,
    *,
    type_: str = "file",
    chat_id: Optional[str] = None,
    tag: str = "",
) -> dict:
    """Record ``state: error`` (with the reason) for one phase."""
    if not phase or not isinstance(phase, str):
        return await get_status(agent_id, scope, source) or {}
    previous = await _previous_diary_entry(agent_id, scope, source, phase)
    carried = _carried(
        previous, started_at=None, owner=None, tag=tag, attempt=_attempt(previous)
    )
    entry = {
        "state": PHASE_STATE_ERROR,
        "reason": reason,
        **carried,
    }
    return await _write_phase_entry(
        agent_id, scope, source, phase, entry, type_=type_, chat_id=chat_id
    )


async def record_phase_skipped(
    agent_id: str,
    scope: str,
    source: str,
    phase: str,
    reason: Optional[str] = None,
    *,
    type_: str = "file",
    chat_id: Optional[str] = None,
    tag: str = "",
) -> dict:
    """Record ``state: skipped`` (with the reason) for a non-applicable phase."""
    if not phase or not isinstance(phase, str):
        return await get_status(agent_id, scope, source) or {}
    previous = await _previous_diary_entry(agent_id, scope, source, phase)
    carried = _carried(
        previous, started_at=None, owner=None, tag=tag, attempt=_attempt(previous)
    )
    entry = {
        "state": PHASE_STATE_SKIPPED,
        "reason": reason,
        **carried,
    }
    return await _write_phase_entry(
        agent_id, scope, source, phase, entry, type_=type_, chat_id=chat_id
    )


#: Backward-compatible alias: "running" is spelled ``processing`` now.
record_phase_running = record_phase_processing


async def record_phase_purge_failed(
    agent_id: str,
    scope: str,
    source: str,
    phase: str,
    *,
    reason: Optional[str] = None,
    type_: str = "file",
    chat_id: Optional[str] = None,
) -> dict:
    """Leave a phase ``processing`` after a FAILED purge, so the next pass retries.

    Purge and state must move in lockstep: a ``complete`` entry whose ``delete``
    raised would keep describing data that may already be gone (or that the
    next pass must still remove). The entry is therefore rewritten to
    ``processing`` — never ``complete`` — with:

    - ``owner = None`` and ``started_at`` already past the lease, so
      :func:`should_abandon` reclaims it on the very next recovery sweep
      (a live owner would make the retry wait for a lease that never expires
      in this process);
    - the previous ``tag`` / ``attempt`` carried over.

    Completion is never implied by this write; the next successful purge drops
    the entry (``pending``) and the normal dispatch re-runs the phase.
    """
    if not phase or not isinstance(phase, str):
        return await get_status(agent_id, scope, source) or {}
    previous = await _previous_diary_entry(agent_id, scope, source, phase)
    expired_started_at = time.time() - phase_lease_seconds() - 1.0
    entry = {
        "state": PHASE_STATE_PROCESSING,
        "reason": reason,
        "started_at": expired_started_at,
        "owner": None,
        "tag": str((previous or {}).get("tag") or ""),
        "attempt": _attempt(previous),
    }
    return await _write_phase_entry(
        agent_id, scope, source, phase, entry, type_=type_, chat_id=chat_id
    )


async def reset_phase_entries(
    agent_id: str, scope: str, source: str, phases
) -> dict:
    """Remove the diary entries of ``phases`` for a source (purge reset).

    Used by the abandoned-lease recovery: after the phase products were
    purged, the entries are dropped so the phases are stale again and re-run
    from scratch. A missing row/doc/diary is a no-op (returns ``{}``).
    """
    phases = set(phases or [])
    if not phases:
        return await get_status(agent_id, scope, source) or {}
    key = status_key(agent_id, scope, source)
    doc = await _read_doc(key)
    if not isinstance(doc, dict):
        return {}
    diary = doc.get("completed_phases")
    if not isinstance(diary, dict):
        return doc
    for phase in phases:
        diary.pop(phase, None)
    doc["completed_phases"] = diary
    await crud.store(key, doc)
    return doc


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
        if doc.get("resume_created") and status in (
            IngestionStatus.UPLOADED.value,
            IngestionStatus.DOWNLOADING.value,
            IngestionStatus.DOWNLOADED.value,
            IngestionStatus.PROCESSING.value,
        ):
            # Created by the no-doc claim path and still in flight: refuse an
            # immediate second claim, but never forever. The lease must cover
            # EVERY in-flight status, not just PROCESSING: a URL row created by
            # this path flips to DOWNLOADING/DOWNLOADED before it returns to
            # PROCESSING, and with the staleness gate disabled only the lease
            # keeps a live download from being stolen. With a positive
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


async def _delete_status_doc(agent_id: str, scope: str, source: str) -> None:
    """Remove the status doc WITHOUT taking the source lock.

    Only for callers that already HOLD :func:`source_lock` for the whole
    delete operation (see ``phases.delete_source``); taking the lock twice in
    the same task would deadlock. Every other caller uses
    :func:`delete_status`.
    """
    await crud.delete(status_key(agent_id, scope, source))


async def delete_status(agent_id: str, scope: str, source: str) -> None:
    """Delete the ingestion-status doc for a source.

    **ORDERING RULE**: the delete path takes ONLY the per-source lock and
    NEVER a per-(source, phase) lock: it removes the whole row, not one phase.
    Taking the source lock makes a delete wait for any in-flight per-source
    phase sequence to finish instead of racing it.
    """
    async with source_lock(agent_id, source):
        await _delete_status_doc(agent_id, scope, source)


async def clear_completed_phases(
    agent_id: str, scope: str, source: str, phase_ids=None
) -> bool:
    """Clear the completed-phases diary of a source (crud-only, no raw Redis).

    **ORDERING RULE**: this is a STATE write. The caller MUST already have
    deleted the data it describes (points, graph artifacts, files); never call
    it before the purge, or the diary would claim work that still exists is
    pending (harmless) while the real risk is the opposite — a diary that keeps
    claiming ``complete`` after the data is gone.

    The diary is BOOKKEEPING, not proof the points exist. ``--reingest`` wipes
    a source's points and must also clear its diary, so the phases re-run
    deterministically even if the status row survives (e.g. ``delete_status``
    fails after the points are gone) instead of looking fresh forever.

    With ``phase_ids`` omitted the WHOLE diary is cleared: the CLI reingest
    wiped every phase's products, so every phase must look pending again
    (never a hardcoded pair). Pass an explicit iterable only for a deletion
    that invalidates a KNOWN subset (e.g. the ``--graph`` wipe resets just the
    graph phases).

    Returns True when at least one diary entry was cleared, False when the
    row/doc/diary was absent (or none of ``phase_ids`` was present).
    """
    key = status_key(agent_id, scope, source)
    doc = await _read_doc(key)
    if not isinstance(doc, dict) or not doc.get("completed_phases"):
        return False
    diary = doc["completed_phases"]
    if not isinstance(diary, dict):
        return False
    if phase_ids is None:
        doc["completed_phases"] = {}
    else:
        wanted = set(phase_ids)
        before = len(diary)
        for phase in wanted:
            diary.pop(phase, None)
        if len(diary) == before:
            return False
        doc["completed_phases"] = diary
    await crud.store(key, doc)
    return True


async def reset_phase_entries_for_agent(agent_id: str, phases) -> int:
    """Drop ``phases`` diary entries from EVERY status doc of an agent.

    **ORDERING RULE**: a STATE write used after a deletion that is not
    source-scoped (e.g. the manual ``--graph`` wipe removes the tenant's graph
    artifacts, which invalidates the graph phases for every source). The caller
    MUST delete the data first; this only makes the surviving diary stop
    claiming it exists.

    Enumerates the agent's whole ingestion namespace (all scopes, via
    :func:`_list_status_keys`), so a conversation-scoped source is reset too.
    The deletion marker is skipped. Each doc is read, its diary stripped of the
    given phase ids and re-stored only when something actually changed.

    Returns:
        The number of status docs whose diary was changed.
    """
    wanted = set(phases or ())
    if not wanted:
        return 0
    changed = 0
    marker_key = delete_marker_key(agent_id)
    for key in await _list_status_keys(agent_id):
        if key == marker_key:
            continue
        doc = await _read_doc(key)
        if not isinstance(doc, dict):
            continue
        diary = doc.get("completed_phases")
        if not isinstance(diary, dict):
            continue
        if not any(phase in diary for phase in wanted):
            continue
        for phase in wanted:
            diary.pop(phase, None)
        doc["completed_phases"] = diary
        await crud.store(key, doc)
        changed += 1
    return changed


async def purge_source_for_reingest(
    agent_id: str,
    scope: str,
    source: str,
    delete_data,
    *,
    phase_ids=None,
) -> None:
    """Wipe a source's data, THEN its diary, THEN its status row.

    **ORDERING INVARIANT**: the data deletion is awaited FIRST. Only after it
    completed does this write any state (the diary clear, then the row delete),
    so a crash between the steps can only leave state claiming LESS than the
    data says — never a ``complete`` diary over wiped data. The CLI reingest
    MUST go through this helper instead of hand-rolling the sequence: a
    reordered sequence is exactly how wiped data silently keeps looking fresh.

    ``delete_data`` is an async callable performing the actual wipe (e.g. the
    vector handler's ``delete_tenant_points`` bound to the source). It MUST
    raise on failure; this function never swallows it, so a failed wipe leaves
    the diary intact and the source re-runs from its current checkpoint rather
    than from a false ``pending``.

    Args:
        agent_id: The agent (chatbot) id.
        scope: ``"agent"`` or a conversation ``chat_id``.
        source: The ingested file name or URL.
        delete_data: ``async () -> None`` that removes the source's data.
        phase_ids: Optional subset for the diary clear; ``None`` clears every
            phase (the reingest wiped every phase's products).
    """
    await delete_data()
    if phase_ids is None:
        await clear_completed_phases(agent_id, scope, source)
    else:
        await clear_completed_phases(
            agent_id, scope, source, phase_ids=phase_ids
        )
    await delete_status(agent_id, scope, source)


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