"""Recovery loop for the ``efficient_ingestion`` plugin — ONE state machine.

The recovery loop of the ingestion lifecycle lives ONLY in this plugin:
the ``after_lizard_bootstrap`` hook schedules a fire-and-forget background pass
(per agent) that:

  (a) resumes stale ``uploaded``/``downloading``/``downloaded``/``processing``/
      ``error`` ingestions by handing each candidate to the SAME phase machine
      used by the re-embed pass
      (:func:`reembed_sources <cat.core_plugins.cat_efficient_ingestion.reembed.reembed_sources>`);
      the machine re-reads the file from disk (or re-downloads the URL), claims
      the per-source work, records the phase, cleans up orphan images and
      performs the full re-ingest;
  (b) REVALIDATES ``completed`` rows: the ``ingestion_phase_pending`` probe is
      re-run READ-ONLY for every completed row and only the rows whose probe
      comes back non-empty (a chunker/embedder/settings change made a phase
      stale) are claimed and handed to the phase machine — a genuinely-fresh
      completed row is never touched; and
  (c) purges status entries whose source is absent from the canonical lists
      (files on disk, URLs in the vector store, existing conversations) via the
      plugin's own shared helper :func:`reconcile_agent`.

Canceled agents (deletion marker present, or ghost agents whose master key is
gone) are skipped entirely, and a ghost agent that raises
``CustomNotFoundException`` on resolution is treated as a skip, never a crash.

The same pass is re-run periodically (every
``CAT_INGESTION_RESUME_INTERVAL_SECONDS``, default 60) so a stale entry is
recovered without a manual restart; set the interval to ``0`` to disable the
periodic sweep.

The core stays untouched by this loop: it keeps its upstream ingestion flow
(plain ``ingest_file``, no state machine) and is "replaced" only when this
plugin drives the lifecycle (recovery, re-embed, re-ingest all reuse the one
machine).

Importing this module has zero side effects: the loop only starts from the
``after_lizard_bootstrap`` hook.
"""

import asyncio
import os
import time
from io import BytesIO
from typing import Any

from cat import BillTheLizard, hook, log
from cat.db import crud
from cat.db.cruds import settings as crud_settings
from cat.env import get_env_int
from cat.exceptions import CustomNotFoundException
from cat.services.memory.models import VectorMemoryType
from cat.utils import is_url

from .hooks import execute_phase_accumulator
from .phases import PHASES
from .reconcile import reconcile_agent
from .reembed import reembed_sources
from .registry import (
    IngestionStatus,
    backfill_completed_phases,
    claim_source_for_resume,
    ingestion_canceled,
    list_statuses,
    set_status,
)


def _resume_enabled() -> bool:
    """``CAT_INGESTION_RESUME_ON_STARTUP`` flag, defaulting to true."""
    value = os.getenv("CAT_INGESTION_RESUME_ON_STARTUP")
    return value is None or value.lower() in ("1", "true", "yes", "on")


def _gc_enabled() -> bool:
    """``CAT_INGESTION_STATUS_GC_ON_STARTUP`` flag, defaulting to true."""
    value = os.getenv("CAT_INGESTION_STATUS_GC_ON_STARTUP")
    return value is None or value.lower() in ("1", "true", "yes", "on")


def _stale_seconds() -> int:
    """Staleness threshold for resume, defaulting to 300 seconds."""
    value = get_env_int("CAT_INGESTION_RESUME_STALE_SECONDS")
    return value if value is not None and value > 0 else 300


def _boot_stale_seconds() -> int:
    """Staleness threshold for the resume pass right after a container restart.

    A freshly restarted container inherits rows left in ``processing`` by a
    worker that died with the previous process: those must be recoverable
    quickly, not after the full ``_stale_seconds()`` (default 300s) budget.
    Any *live* worker (another replica still ingesting) keeps its rows fresh
    with the heartbeat (``CAT_INGESTION_HEARTBEAT_SECONDS``, default 30s), so
    requiring the row to be older than ``2 x heartbeat`` is enough to never
    steal live work while making dead-worker rows claimable at the first boot
    pass. Configurable with ``CAT_INGESTION_RESUME_BOOT_STALE_SECONDS``.
    """
    value = get_env_int("CAT_INGESTION_RESUME_BOOT_STALE_SECONDS")
    if value is not None and value > 0:
        return value
    heartbeat = get_env_int("CAT_INGESTION_HEARTBEAT_SECONDS")
    if heartbeat is not None and heartbeat > 0:
        return max(30, 2 * heartbeat)
    return 60


def _maintenance_marker_key(agent_id: str) -> str:
    """Redis key shared with the maintenance CLI (``maintenance_agent.py``)."""
    return f"agents:{agent_id}:ingestion:maintenance"


async def _maintenance_marker_active(agent_id: str) -> bool:
    """Whether a maintenance CLI is currently processing this agent.

    The marker is written by the GraphRAG maintenance CLI through
    ``cat.db.crud`` with a 2h TTL and deleted at the end of the run. While it
    is present the periodic sweep MUST NOT touch the agent: it would otherwise
    steal/retry rows the CLI is actively rebuilding.

    Fail-safe: a read failure is logged and treated as "no marker" so a Redis
    hiccup cannot block the sweep forever; a crashed CLI's marker expires by
    its own TTL.
    """
    try:
        value = await crud.read(_maintenance_marker_key(agent_id))
    except Exception as e:  # noqa: BLE001 - never block the sweep on a read error
        log.warning(
            f"Ingestion sweep: could not read the maintenance marker for agent "
            f"{agent_id}; proceeding: {e}"
        )
        return False
    if isinstance(value, list):
        value = value[0] if value else None
    return value is not None


async def mark_file_missing(agent_id: str, scope: str, source: str) -> None:
    """Mark a source whose file is missing on disk as ``error``.

    A stale ``uploaded``/``processing`` entry whose source file no longer
    exists on disk cannot be resumed; instead of leaving it ``processing``
    forever, transition it to ``error`` so the teacher sees why and can
    re-upload (or remove the file to abandon it).
    """
    await set_status(
        agent_id,
        scope,
        source,
        type_="file",
        status=IngestionStatus.ERROR,
        chat_id=None if scope == "agent" else scope,
        error="Source file does not exist on disk; cannot resume. Remove the file to abandon it.",
    )


def _collection_for_scope(scope: str) -> VectorMemoryType:
    """Vector-memory collection for an ingestion scope ("agent" -> declarative, chat -> episodic)."""
    return VectorMemoryType.DECLARATIVE if scope == "agent" else VectorMemoryType.EPISODIC


async def _pending_phases(ccat, entry: dict) -> list:
    """READ-ONLY phase probe for one status entry.

    Backfills the completed-phases diary (legacy completed rows get a sentinel so
    they are conservatively stale once), converts it to the hook list shape and
    lets the ``ingestion_phase_pending`` accumulator registrants report which
    phases must be re-run against the CURRENT settings. Returns the filtered
    stale-phase list; an empty list means the row is genuinely up to date.
    """
    completed = await backfill_completed_phases(entry, list(PHASES.keys()))
    completed_as_list = [dict(e, phase=p) for p, e in completed.items()]
    pending = await execute_phase_accumulator(
        ccat,
        "ingestion_phase_pending",
        [],
        entry.get("source"),
        completed_as_list,
        default=list,
    )
    return [p for p in (pending or []) if p and p.get("phase")]


def _source_from_entry(ccat, agent_id: str, scope: str, source: str, type_: str = "file"):
    """Build a ``StoredSourceWithMetadata`` for an in-flight entry, re-reading the file.

    Files are re-read from disk (persisted bytes wrapped in ``BytesIO``, the
    same shape ``get_stored_sources_with_metadata`` produces), so the phase
    machine performs the full re-ingest from the actual file content. URLs are
    passed through with ``content=None`` (the machine re-downloads them).
    Returns None (and logs) when the file is missing.
    """
    from cat.looking_glass.models import StoredSourceWithMetadata

    if (type_ or "") == "url" or is_url(source):
        return StoredSourceWithMetadata(
            name=source, path=source, content=None,
            metadata={"chat_id": scope} if scope != "agent" else {},
        )

    path = agent_id
    if scope != "agent":
        path = os.path.join(path, str(scope))
    file_bytes = ccat.file_manager.read_file(source, path)
    if file_bytes is None:
        log.warning(f"Ingestion resume: file {source} missing for agent {agent_id}; skipping")
        return None
    metadata = {}
    if scope != "agent":
        metadata["chat_id"] = scope
    return StoredSourceWithMetadata(
        name=source, path=path, content=BytesIO(file_bytes), metadata=metadata,
    )


async def _resume_agent(lizard: BillTheLizard, agent_id: str, ccat: Any = None) -> None:
    """Re-trigger stale ingestions for one agent through the ONE state machine.

    Only entries in ``uploaded``/``downloading``/``downloaded``/``processing``/
    ``error`` older than the stale threshold are candidates; each is handed to
    ``reembed_sources`` (the same phase machine used for re-embed): it decides
    the phase from the status doc, claims the per-source work atomically (so
    two workers never double-process), records the phase, cleans up any orphan
    images and performs the full re-ingest from disk (or URL re-download).
    An ``error`` row is retried ONLY while its diary still shows a stale phase,
    and the phase machine terminalizes an up-to-date ``processing`` row on its
    first pass, so a fully-processed row is never re-handed.
    Fresh entries and ``completed`` entries are left untouched (completed rows
    are revalidated by :func:`_revalidate_agent`). A canceled agent (deletion
    marker or ghost) is skipped entirely.

    ``downloading``/``downloaded`` are resumable (a crash between download and
    processing would otherwise be a permanent dead end). They have no PROCESSING
    heartbeat, so they are gated by the regular resume threshold
    (``CAT_INGESTION_RESUME_STALE_SECONDS``, default 300s) — never the shorter
    boot threshold — so an actively downloading source is not stolen.
    """
    if await ingestion_canceled(agent_id):
        # agent is being deleted (marker) or dead (ghost): never resume it
        return
    if await _maintenance_marker_active(agent_id):
        # a maintenance CLI owns this agent right now: never steal its rows
        log.debug(
            f"Ingestion sweep for agent {agent_id} skipped: maintenance marker present"
        )
        return
    if ccat is None:
        try:
            ccat = await lizard.get_cheshire_cat(agent_id)
        except CustomNotFoundException:
            # ghost agent (master key gone): treat as a skip, never a crash
            return
    if ccat is None:
        return

    boot_stale = _boot_stale_seconds()
    # downloading/downloaded have no heartbeat: use the larger resume threshold
    # so a live download is never re-claimed before it looks genuinely stale.
    download_stale = max(boot_stale, float(_stale_seconds()))
    now = time.time()
    entries = await list_statuses(agent_id)
    for entry in entries:
        status = entry.get("status")
        if status not in (
            IngestionStatus.UPLOADED.value,
            IngestionStatus.DOWNLOADING.value,
            IngestionStatus.DOWNLOADED.value,
            IngestionStatus.PROCESSING.value,
            IngestionStatus.ERROR.value,
        ):
            continue

        # An ``error`` row is terminal unless its diary still shows a stale
        # phase: retrying it while every phase is fresh would resurrect a row the
        # machine already exhausted (and re-hand it forever without any
        # progress). A failed phase leaves its diary entry missing, so a genuinely
        # retryable error still probes as stale and is handed below.
        if status == IngestionStatus.ERROR.value:
            try:
                if not await _pending_phases(ccat, entry):
                    continue
            except Exception as e:  # noqa: BLE001 - a probe failure must not resurrect
                log.warning(
                    f"Ingestion resume: could not probe stale phases for {entry.get('source')} "
                    f"(agent {agent_id}); leaving it error: {e}"
                )
                continue

        # A download has no PROCESSING heartbeat: only touch it once it is
        # older than the download threshold (an active download stays fresh).
        if status in (
            IngestionStatus.DOWNLOADING.value,
            IngestionStatus.DOWNLOADED.value,
        ):
            try:
                updated_at = float(entry.get("updated_at") or 0)
            except (TypeError, ValueError):
                updated_at = 0
            if now - updated_at < download_stale:
                continue
            claim_stale = download_stale
        else:
            claim_stale = boot_stale

        source = entry.get("source")
        scope = entry.get("scope")
        type_ = entry.get("type")
        if not source:
            continue

        # Re-read the file from disk (or pass the URL through); skip entries
        # whose file is gone on disk (they are marked error so the teacher can
        # re-upload or abandon).
        source_obj = _source_from_entry(ccat, agent_id, str(scope), source, type_)
        if source_obj is None:
            if type_ != "url" and not is_url(source):
                await mark_file_missing(agent_id, str(scope), source)
            continue

        log.info(
            f"Ingestion resume: handing {source} (status {status}, phase {entry.get('phase')!r}) "
            f"to the ingestion phase machine for {agent_id}"
        )
        # The phase machine claims the per-source work itself, using the same
        # staleness budget chosen above for this entry's status.
        await reembed_sources(
            ccat, _collection_for_scope(str(scope)), [source_obj],
            stale_after=claim_stale,
        )


async def _revalidate_agent(lizard: BillTheLizard, agent_id: str, ccat: Any = None) -> None:
    """Revalidation pass: re-probe completed rows and re-ingest the stale ones.

    A ``completed`` row is only authoritative while its recorded phase diary
    matches the CURRENT settings. This pass re-runs the clock-free
    ``ingestion_phase_pending`` probe READ-ONLY for every completed row
    (diary backfilled for legacy rows, converted to the hook list shape) and,
    when the probe returns non-empty (a chunker/embedder/settings change made
    a phase stale), claims the row (``claim_completed=True`` — ONLY after the
    non-empty probe) and hands it to the ONE phase machine, the same path the
    recovery sweep uses (:func:`_source_from_entry` + ``reembed_sources``).

    Rows whose probe is empty are NEVER claimed: a genuinely-fresh completed
    row is left untouched. The probe governs COMPLETED revalidation while the
    ``stale_after`` claim in :func:`_resume_agent` governs crashed
    PROCESSING recovery — the two coexist without conflict. ``error`` rows
    are not revalidated (absorbing, handled by the stale-error sweep).
    Per-row exceptions are tolerated and counted, so one bad row never aborts
    the pass. A canceled agent (deletion marker or ghost) is skipped entirely.
    """
    if await ingestion_canceled(agent_id):
        # agent is being deleted (marker) or dead (ghost): never revalidate it
        return
    if await _maintenance_marker_active(agent_id):
        # a maintenance CLI owns this agent right now: never revalidate it
        log.debug(
            f"Ingestion revalidation for agent {agent_id} skipped: maintenance marker present"
        )
        return
    if ccat is None:
        try:
            ccat = await lizard.get_cheshire_cat(agent_id)
        except CustomNotFoundException:
            # ghost agent (master key gone): treat as a skip, never a crash
            return
    if ccat is None:
        return

    entries = await list_statuses(agent_id)
    revalidated = 0
    for entry in entries:
        if entry.get("status") != IngestionStatus.COMPLETED.value:
            continue
        source = entry.get("source")
        scope = entry.get("scope")
        type_ = entry.get("type")
        if not source or not scope:
            continue
        try:
            # READ-ONLY probe: backfill the diary (legacy completed rows get a
            # sentinel -> conservatively stale once), convert it to the hook
            # list shape (every entry carries a "phase" key) and let the
            # accumulator registrants report the stale phases. A non-empty
            # result means at least one phase must be re-run.
            pending = await _pending_phases(ccat, entry)
            if not pending:
                # probe empty: the row is genuinely complete — never claim it
                continue
            # claim ONLY after a non-empty probe (atomic across workers; the
            # agent-level sweep lock already serializes the pass per agent)
            claimed = await claim_source_for_resume(
                agent_id, str(scope), source,
                stale_after=0.0, owner=f"revalidate-{os.getpid()}",
                claim_completed=True,
            )
            if claimed is None:
                # another worker already claimed it (or the agent was canceled
                # in the meantime)
                continue
            source_obj = _source_from_entry(ccat, agent_id, str(scope), source, type_)
            if source_obj is None:
                if type_ != "url" and not is_url(source):
                    await mark_file_missing(agent_id, str(scope), source)
                continue
            log.info(
                f"Ingestion revalidation: handing {source} (stale phase(s) "
                f"{[p.get('phase') for p in pending]}) to the ingestion phase "
                f"machine for {agent_id}"
            )
            # stale_after=0: the sweep already claimed the row, so the machine's
            # internal claim must not refuse it as "fresh in-flight work"
            await reembed_sources(
                ccat, _collection_for_scope(str(scope)), [source_obj],
                stale_after=0.0,
            )
            revalidated += 1
        except Exception as e:  # noqa: BLE001 - a failing row must not abort the pass
            log.error(f"Ingestion revalidation failed for {source} (agent {agent_id}): {e}")
    log.info(
        f"Ingestion revalidation for agent {agent_id}: {revalidated} completed "
        f"row(s) re-ingested"
    )


async def _pass_for_agent(lizard: BillTheLizard, agent_id: str) -> None:
    """Run the recovery + GC pass for one agent.

    Only the enumeration is guarded by a short agent-level distributed lock;
    actual (re)ingestion of each candidate happens under its own per-source
    lock (inside the phase machine's claim), so different sources of the same
    agent are processed concurrently while the same source can never be
    double-processed by another worker. A canceled agent (deletion marker or
    ghost) is skipped entirely.
    """
    if await ingestion_canceled(agent_id):
        # agent is being deleted (marker) or dead (ghost): never sweep it
        return
    if await _maintenance_marker_active(agent_id):
        # A maintenance CLI owns this agent: skip the whole sweep ONCE per pass
        # (the inner _resume_agent/_revalidate_agent checks keep direct callers
        # safe without adding a second log line to the periodic sweep).
        log.debug(
            f"Ingestion sweep for agent {agent_id} skipped: maintenance marker present"
        )
        return
    try:
        async with crud.distributed_lock(f"ingestion-sweep:{agent_id}", timeout=30, blocking_timeout=5):
            try:
                ccat = await lizard.get_cheshire_cat(agent_id)
            except CustomNotFoundException:
                # ghost agent (master key gone): treat as a skip, never a crash
                return
            if ccat is None:
                return
            if _resume_enabled():
                await _resume_agent(lizard, agent_id, ccat=ccat)
                await _revalidate_agent(lizard, agent_id, ccat=ccat)
            if _gc_enabled():
                await reconcile_agent(agent_id, ccat=ccat)
    except crud.LockError:
        # Another replica is already running this sweep: expected contention
        # across workers, not a fault. Log as info and move on.
        log.info(f"Ingestion sweep for agent {agent_id} skipped: lock held by another worker")
    except Exception as e:
        log.error(f"Ingestion startup pass failed for agent {agent_id}: {e}")


async def _startup_pass(lizard: BillTheLizard) -> None:
    """Enumerate agents and run the recovery + GC pass for each."""
    try:
        agent_ids = [a["agent_id"] for a in await crud_settings.get_agents()]
    except Exception as e:
        log.error(f"Ingestion startup pass: failed to enumerate agents: {e}")
        return
    for agent_id in agent_ids:
        if await ingestion_canceled(agent_id):
            # agent is being deleted (marker) or dead (ghost): skip it
            continue
        await _pass_for_agent(lizard, agent_id)


async def _periodic_sweep_loop(lizard: BillTheLizard) -> None:
    """Re-run the recovery + GC pass every ``CAT_INGESTION_RESUME_INTERVAL_SECONDS``.

    Fire-and-forget: never blocks bootstrap. Each pass is safe against
    double-processing across replicas because ``_pass_for_agent`` is guarded by
    a per-agent distributed lock and each source claim by the phase machine.
    """
    interval = get_env_int("CAT_INGESTION_RESUME_INTERVAL_SECONDS")
    if interval is None or interval <= 0:
        return
    while True:
        await _startup_pass(lizard)
        await asyncio.sleep(interval)


@hook
async def after_lizard_bootstrap(lizard: BillTheLizard):
    """Schedule the fire-and-forget startup pass (never blocks bootstrap)."""
    asyncio.ensure_future(_startup_pass(lizard))
    interval = get_env_int("CAT_INGESTION_RESUME_INTERVAL_SECONDS")
    if interval is not None and interval > 0:
        asyncio.ensure_future(_periodic_sweep_loop(lizard))