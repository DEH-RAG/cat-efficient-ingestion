"""Phase DAG and clock-free two-level staleness for the efficient-ingestion plugin.

Declares the built-in ingestion phases (``parsing_chunking``, ``embedding``)
and the accumulator registrant of ``ingestion_phase_pending`` that reports
which of them are stale for a source, using ONLY content comparison — no
timestamps are ever used for correctness.

**Clock-free two-level staleness.** A phase's diary entry (see
``registry.completed_phases``) records the explicit state plus the
material ``tag`` and ``{"marker", "settings_version", "deps"}``. A phase is
stale when:

- (a) **dependency**: any upstream in ``spec.depends_on`` is missing from the
  diary, or the recorded ``deps[upstream]`` differs from the upstream's
  current diary ``marker`` (upstream re-ran -> this phase must re-run too);
- (b) **settings fast-path**: the settings entry for ``spec.settings_category``
  was NOT rewritten since the phase ran (``settings_version`` equals the
  current entry's ``updated_at``, used as an OPAQUE token) -> FRESH, no
  marker comparison needed;
- (c) **slow-path**: the plugin-defined marker from
  ``ingestion_phase_settings_marker`` differs from the recorded ``marker``
  (the settings entry WAS rewritten; the phase-owning plugin decides whether
  the change is material).

A missing settings entry means the fast-path cannot apply (no current
version) and the phase falls to the slow-path with the marker the plugin
returns — conservatively stale when the marker is unknown.

**Phase registration.** Every phase is exposed as an
:class:`IngestionPhase` instance through :func:`get_phases`: the built-ins (as
classes), then the instances a plugin returns from the
``ingestion_phase_objects`` getter hook, then the legacy ``ingestion_phase_specs``
declarations wrapped in a :class:`HookPhase` adapter (so an unmigrated plugin
keeps working). External plugins declare their own legacy phases through the
``ingestion_phase_specs`` accumulator hook (see ``hooks.py``);
:func:`merged_phases` merges the returned ``PhaseSpec`` objects into the
built-in ``PHASES`` at probe time — built-ins win on duplicate ids — and the
``ingestion_phase_pending`` registrant computes staleness for ALL merged
phases, so a provider only declares a spec + a marker and never re-implements
staleness. The class API is additive for now: the dispatch loop still runs the
legacy flow.

**Mandatory cleanup.** :meth:`IngestionPhase.delete` is abstract: a subclass
without it cannot be instantiated. :class:`HookPhase.delete` calls the legacy
purge hook and raises :class:`PhaseDeleteError` when the hook is absent (or its
no-op returns ``None``) — a phase never silently skips cleanup.

**Third-party extension contract.** A plugin exposing a phase must (1) declare
a dependency on ``cat_efficient_ingestion`` — the FOLDER id with underscores,
not the hyphenated repo name, because ``Plugin.missing_dependencies`` compares
against plugin folder names — (2) import :class:`PhaseSpec` from the canonical
path ``cat.core_plugins.cat_efficient_ingestion.phases`` inside a module-top
``try/except ImportError``, (3) define its phase hooks (or no-op stand-ins)
ONLY when that import succeeded, and (4) keep the classic CAT ingestion hooks
so basic ingestion still works when EffING is absent. The canonical path is
mandatory: the host loads ``cat/core_plugins/`` in preference to
``cat/plugins/`` and derives the module name from the folder, so importing the
same file through another path creates a SECOND module identity and
:func:`merged_phases`' ``isinstance(spec, PhaseSpec)`` check would silently
drop the spec.

Import-safe: the only top-level statements are the dataclass, the ``PHASES``
data structure and the ``@hook`` decorator — no Redis, no network, no
filesystem access at import time.
"""

import inspect
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

from cat import hook
from cat.db.cruds import settings as crud_settings
from cat.log import log

from .hooks import (
    _accepts_caller,
    execute_hook_compat,
    phase_managers,
    phase_settings_key,
)
from .registry import (
    PHASE_STATE_ERROR,
    PHASE_STATE_PROCESSING,
    PHASE_STATE_SKIPPED,
    get_completed_phases,
    is_phase_lease_expired,
    phase_state,
)


@dataclass(frozen=True)
class PhaseSpec:
    """Declaration of one ingestion phase and its invalidation inputs.

    Attributes:
        id: The phase id (also the diary key, e.g. ``"embedding"``).
        settings_category: The settings category whose ``updated_at`` is the
            opaque fast-path token (``"chunker"``, ``"embedder"``), or None
            for phases with no settings entry (marker-only invalidation).
        depends_on: Upstream phase ids whose recorded markers this phase
            consumed; a change in any upstream marker makes this phase stale.
    """

    id: str
    settings_category: str | None
    depends_on: tuple[str, ...]


@dataclass(frozen=True)
class PhaseResult:
    """Outcome of running one phase.

    ``processed`` / ``expected`` are TELEMETRY ONLY: they may be logged or
    surfaced to an operator, but they are never a completion criterion. A
    phase is complete because ``status == "done"``, not because a count
    matches.
    """

    status: Literal["done", "not_ready", "error", "skipped"]
    reason: str | None = None
    processed: int | None = None
    expected: int | None = None


class IngestionPhase(ABC):
    """One ingestion phase, owning its applicability, work and cleanup.

    Every phase MUST implement all four operations. ``delete`` is deliberately
    abstract: a phase that cannot clean up its own artifacts would make the
    recovery path delete data blindly, so it cannot exist. There is no no-op
    default anywhere — a subclass without ``delete`` raises ``TypeError`` at
    instantiation time.
    """

    #: Stable phase id (the diary key, e.g. ``"embedding"``).
    id: str = ""
    #: Upstream phase ids this phase consumes.
    depends_on: tuple[str, ...] = ()
    #: Settings category whose material configuration feeds :meth:`tag`.
    settings_category: str | None = None

    @abstractmethod
    async def applies_to(self, cat, deps_state) -> tuple[bool, str | None]:
        """Whether this phase applies, with an optional human-readable reason."""

    @abstractmethod
    async def process(self, source: str, cat) -> PhaseResult:
        """Run the phase body for ``source``."""

    @abstractmethod
    async def delete(self, source: str, cat) -> None:
        """Delete ONLY the artifacts this phase produced for ``source``.

        Must raise when the phase has no cleanup implementation: a silent
        no-op would let recovery delete data blindly.
        """

    @abstractmethod
    def tag(self, cat) -> str:
        """Deterministic identity of the phase's material configuration."""


#: The built-in phase DAG. ``parsing_chunking`` depends only on the chunker
#: settings; ``embedding`` depends on the embedder settings AND the recorded
#: ``parsing_chunking`` marker — so an embedder-only change re-runs only
#: ``embedding``, while a chunker change re-runs ``parsing_chunking`` first
#: and ``embedding`` follows on the next probe (its dep marker is recomputed
#: against the new parsing marker once parsing is recorded).
PHASES = {
    "parsing_chunking": PhaseSpec("parsing_chunking", "chunker", ()),
    "embedding": PhaseSpec("embedding", "embedder", ("parsing_chunking",)),
}


class _BuiltinPhase(IngestionPhase):
    """Declarative built-in phase.

    The built-in phase BODIES still live in the legacy ``reembed`` dispatcher
    (this todo is additive): :meth:`process` / :meth:`delete` are implemented
    so the ABC contract is complete, but they refuse to run here rather than
    duplicating the dispatcher's source-scoped logic. Wiring the dispatcher to
    these methods is the next todo, not this one.
    """

    async def applies_to(self, cat, deps_state) -> tuple[bool, str | None]:
        return True, None

    async def process(self, source: str, cat) -> PhaseResult:
        raise NotImplementedError(
            f"Built-in phase '{self.id}' is executed by the legacy reembed "
            "dispatcher until the dispatch migration; use HookPhase for "
            "hook-declared phases."
        )

    async def delete(self, source: str, cat) -> None:
        raise NotImplementedError(
            f"Built-in phase '{self.id}' is purged by the legacy reembed "
            "dispatcher until the dispatch migration; use HookPhase for "
            "hook-declared phases."
        )

    def tag(self, cat) -> str:
        return _declared_tag(self.id, self.settings_category, self.depends_on)


class ParsingChunkingPhase(_BuiltinPhase):
    id = "parsing_chunking"
    depends_on = ()
    settings_category = "chunker"


class EmbeddingPhase(_BuiltinPhase):
    id = "embedding"
    depends_on = ("parsing_chunking",)
    settings_category = "embedder"


#: Built-in phase classes in execution order. ``get_phases`` instantiates them
#: fresh on every call so callers can never mutate a shared phase object.
_BUILTIN_PHASE_CLASSES: tuple[type[IngestionPhase], ...] = (
    ParsingChunkingPhase,
    EmbeddingPhase,
)


async def _registered_specs(cat) -> list[tuple[PhaseSpec, object]]:
    """Collect every registered ``PhaseSpec`` together with its OWNER manager.

    Phase providers are agent-level plugins: MyGRAPH and CATALOG register their
    specs on the caller's agent manager. EffING itself is system-only, so its
    built-ins are not registered this way (they live in :data:`PHASES`). The
    spec origin is what later tells the run/marker dispatch which manager owns
    a phase, so it must not be flattened away.

    Managers are visited system-first, then agent; a duplicate phase id is kept
    from the FIRST manager that registered it (deterministic ownership).
    Malformed hook output is skipped, never raised on.
    """
    seeded: set[str] = set()
    collected: list[tuple[PhaseSpec, object]] = []
    for manager in phase_managers(cat):
        # No registrant on this manager is normal (EffING is system-only, the
        # providers are agent-level): skip silently, never warn per phase.
        ran, specs = await execute_hook_compat(
            manager, "ingestion_phase_specs", [], caller=cat
        )
        if not ran or not isinstance(specs, list):
            continue
        for spec in specs:
            if isinstance(spec, PhaseSpec) and spec.id not in seeded:
                seeded.add(spec.id)
                collected.append((spec, manager))
    return collected


#: Sentinel distinguishing "no marker hook resolvable" from a marker that is
#: legitimately ``None`` ("unknown").
_MISSING = object()


def _declared_tag(phase_id: str, settings_category: str | None, depends_on) -> str:
    """Stable tag from a phase's declared configuration inputs.

    Used when no synchronous material marker can be resolved (the legacy
    marker hooks are ``async``). Deterministic and dependency-aware; the
    recursive settings hash is the next todo's job.
    """
    return f"{phase_id}:{settings_category or '-'}:{','.join(depends_on)}"


def _stable_marker(marker: Any) -> str:
    """Render a marker deterministically (dicts are sorted by key)."""
    if isinstance(marker, str):
        return marker
    try:
        return json.dumps(marker, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(marker)


def _as_count(value: Any) -> int | None:
    """Return ``value`` only when it is a genuine integer count (not bool)."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _coerce_phase_result(phase_id: str, result: Any) -> PhaseResult:
    """Normalize a legacy ``ingestion_phase_run`` return into ``PhaseResult``.

    Legacy registrants return a status dict (or the fail-hard ``None``). An
    unknown/invalid return becomes an explicit ``error`` — never a silent
    success, mirroring the dispatcher's hard-fail policy.
    """
    if isinstance(result, PhaseResult):
        return result
    if isinstance(result, dict):
        status = result.get("status")
        if status not in ("done", "not_ready", "error", "skipped"):
            return PhaseResult(
                status="error",
                reason=f"phase '{phase_id}' returned invalid status {status!r}",
            )
        return PhaseResult(
            status=status,
            reason=result.get("reason"),
            processed=_as_count(result.get("processed")),
            expected=_as_count(result.get("expected")),
        )
    if result is None or result is _MISSING:
        return PhaseResult(
            status="error",
            reason=f"phase '{phase_id}' has no run implementation",
        )
    return PhaseResult(
        status="error",
        reason=f"phase '{phase_id}' returned {result!r} (expected a status dict)",
    )


def _call_sync_legacy_hook(manager, hook_name: str, phase: str, cat):
    """Call a SYNC legacy hook registrant directly, or return ``_MISSING``.

    ``tag`` is synchronous by contract, so it cannot await an ``async``
    marker hook. When the marker registrant on this manager is synchronous it
    is used verbatim; otherwise the caller falls back to the declared tag.
    """
    hooks = getattr(manager, "hooks", None)
    registered = hooks.get(hook_name) if isinstance(hooks, dict) else None
    if not registered:
        return _MISSING
    keyword = getattr(manager, "context_execute_hook", "cat")
    for registered_hook in registered:
        function = getattr(registered_hook, "function", registered_hook)
        if inspect.iscoroutinefunction(function):
            continue
        if not _accepts_caller(function, keyword):
            continue
        try:
            return function(phase, **{keyword: cat})
        except Exception as exc:  # noqa: BLE001 - a marker must never break tag()
            log.warning(f"Legacy {hook_name} for phase '{phase}' failed: {exc}")
    return _MISSING


class PhaseDeleteError(RuntimeError):
    """A phase has no delete/purge implementation: refuse to delete blindly."""


class HookPhase(IngestionPhase):
    """Adapter keeping one legacy hook-declared phase working.

    Wraps a single :class:`PhaseSpec` plus the legacy hooks
    (``ingestion_phase_run``, ``ingestion_phase_settings_marker``,
    ``ingestion_phase_purge``). Legacy behaviour is preserved:

    - :meth:`applies_to` is ``(True, None)`` (legacy phases were always
      considered applicable);
    - :meth:`process` maps the legacy status dict to :class:`PhaseResult`;
    - :meth:`delete` calls the purge hook and RAISES when it is absent (or
      when the legacy no-op returns ``None``) — never a silent no-op;
    - :meth:`tag` uses the settings marker when a synchronous registrant is
      available, else a deterministic declared tag.
    """

    def __init__(
        self,
        spec: PhaseSpec,
        *,
        cat=None,
        manager=None,
        run_hook=None,
        purge_hook=None,
        settings_marker=None,
    ):
        self.spec = spec
        self.id = spec.id
        self.depends_on = tuple(spec.depends_on)
        self.settings_category = spec.settings_category
        self._cat = cat
        self._manager = manager
        self._run_hook = run_hook
        self._purge_hook = purge_hook
        self._settings_marker = settings_marker

    async def applies_to(self, cat, deps_state) -> tuple[bool, str | None]:
        return True, None

    async def process(self, source: str, cat) -> PhaseResult:
        used = cat if cat is not None else self._cat
        if self._run_hook is not None:
            result = self._run_hook(self.id, source, [], used)
            if inspect.isawaitable(result):
                result = await result
        elif self._manager is not None:
            ran, result = await execute_hook_compat(
                self._manager,
                "ingestion_phase_run",
                self.id,
                self.id,
                source,
                [],
                caller=used,
            )
            if not ran:
                result = None
        else:
            result = await execute_phase_on_owner(
                used,
                "ingestion_phase_run",
                self.id,
                self.id,
                source,
                [],
                default=_MISSING,
                caller=used,
            )
        return _coerce_phase_result(self.id, result)

    async def delete(self, source: str, cat) -> None:
        used = cat if cat is not None else self._cat
        if self._purge_hook is not None:
            result = self._purge_hook(self.id, source, used)
            if inspect.isawaitable(result):
                result = await result
        elif self._manager is not None:
            ran, result = await execute_hook_compat(
                self._manager,
                "ingestion_phase_purge",
                self.id,
                self.id,
                source,
                caller=used,
            )
            if not ran:
                result = None
        else:
            result = await execute_phase_on_owner(
                used,
                "ingestion_phase_purge",
                self.id,
                self.id,
                source,
                default=_MISSING,
                caller=used,
            )
        if result is None or result is _MISSING:
            raise PhaseDeleteError(
                f"Ingestion phase '{self.id}' has no delete/purge "
                f"implementation; refusing to delete its artifacts silently"
            )

    def tag(self, cat) -> str:
        used = cat if cat is not None else self._cat
        marker = self._resolve_marker(used)
        if marker is _MISSING:
            return _declared_tag(self.id, self.settings_category, self.depends_on)
        if marker is None:
            return f"{self.id}:unknown"
        return f"{self.id}:{_stable_marker(marker)}"

    def _resolve_marker(self, cat):
        if self._settings_marker is not None:
            try:
                return self._settings_marker(self.id)
            except Exception as exc:  # noqa: BLE001 - a marker must never break tag()
                log.warning(f"Settings marker for phase '{self.id}' failed: {exc}")
                return _MISSING
        if self._manager is not None:
            return _call_sync_legacy_hook(
                self._manager, "ingestion_phase_settings_marker", self.id, cat
            )
        return _MISSING


async def _plugin_phase_objects(cat) -> list[IngestionPhase]:
    """Collect phase instances contributed through ``ingestion_phase_objects``.

    The hook is a getter (``ingestion_phase_objects(cat) -> list``), not an
    accumulator: every registrant on every manager is called and its returned
    list is merged, so two plugins can both contribute without clobbering each
    other. Non-list / non-``IngestionPhase`` output is skipped, never raised
    on.
    """
    if cat is None:
        return []
    collected: list[IngestionPhase] = []
    for manager in phase_managers(cat):
        hooks = getattr(manager, "hooks", None)
        registered = (
            hooks.get("ingestion_phase_objects") if isinstance(hooks, dict) else None
        )
        if not registered:
            continue
        keyword = getattr(manager, "context_execute_hook", "cat")
        for registered_hook in registered:
            function = getattr(registered_hook, "function", registered_hook)
            if not _accepts_caller(function, keyword):
                continue
            try:
                value = function(**{keyword: cat})
                if inspect.isawaitable(value):
                    value = await value
            except Exception as exc:  # noqa: BLE001 - one bad plugin must not break the registry
                plugin_id = getattr(registered_hook, "plugin_id", None) or "?"
                log.error(
                    f"Error in plugin {plugin_id}::ingestion_phase_objects: {exc}"
                )
                continue
            if isinstance(value, list):
                collected.extend(
                    phase for phase in value if isinstance(phase, IngestionPhase)
                )
    return collected


async def phase_owner(cat, phase: str):
    """Return the plugin manager that owns ``phase``.

    Built-ins belong to the system manager (EffING's own hooks). A phase
    declared through ``ingestion_phase_specs`` belongs to the manager its
    provider registered on (agent-level for MyGRAPH/CATALOG). Unknown phases
    fall back to the system manager, preserving the fail-hard default.
    """
    managers = phase_managers(cat)
    if phase in PHASES:
        return managers[0] if managers else None
    for spec, manager in await _registered_specs(cat):
        if spec.id == phase:
            return manager
    return managers[0] if managers else None


async def execute_phase_on_owner(
    cat, hook_name: str, phase: str, *args, default=None, caller=None
):
    """Execute ``hook_name`` on the manager that owns ``phase``.

    Reaching the owning manager is what keeps a built-in phase (system) and an
    external MyGRAPH/CATALOG phase (agent) both working: a single-manager
    dispatch would silently run the wrong registrant and drop the other.
    """
    manager = await phase_owner(cat, phase)
    if manager is not None:
        ran, value = await execute_hook_compat(
            manager, hook_name, *args, caller=caller
        )
        if ran:
            return value
    # no manager to ask (unit test fake / bootstrap), or no compatible
    # registrant: same conservative degradation as a missing hook, never an
    # AttributeError.
    from cat.log import log
    log.warning(
        f"Phase hook {hook_name} is not present in any active plugin: "
        f"treating it as no external phases"
    )
    return default() if callable(default) else default


async def merged_phases(ccat, cat) -> dict[str, PhaseSpec]:
    """Merge the registered phase specs into the built-in ``PHASES``.

    Starts from a copy of the built-in ``PHASES`` and appends every
    ``PhaseSpec`` returned by the ``ingestion_phase_specs`` accumulator hook
    from BOTH managers (system + agent) whose ``id`` is not already present —
    built-ins win, a registered spec with a duplicate id is IGNORED. Collecting
    from both is what keeps agent-level MyGRAPH/CATALOG phases alive while
    EffING itself is system-only. Malformed hook output (``None``,
    non-``PhaseSpec`` entries) is skipped, never raised on.

    Args:
        ccat: the cat that issued the pass (unused except as a fallback when
            ``cat`` is missing; both resolve the same managers).
        cat: the caller whose ``lizard`` (system manager) and
            ``plugin_manager`` (agent manager) are queried.

    Returns:
        The merged ``dict[str, PhaseSpec]`` (a fresh dict; ``PHASES`` is
        never mutated).
    """
    merged = dict(PHASES)
    resolver = cat if cat is not None else ccat
    if resolver is None:
        return merged
    for spec, _manager in await _registered_specs(resolver):
        if spec.id not in merged:
            merged[spec.id] = spec
    return merged


async def get_phases(cat) -> list[IngestionPhase]:
    """Return every ingestion phase as an :class:`IngestionPhase` instance.

    Order: the built-in phases (as classes, in execution order), then the
    instances contributed by plugins through the ``ingestion_phase_objects``
    hook, then the legacy ``ingestion_phase_specs`` declarations wrapped in a
    :class:`HookPhase` adapter. Built-ins win on duplicate ids; a phase id is
    returned exactly once.

    ``cat`` may be ``None`` (unit tests / bootstrap): only the built-ins are
    returned. This is purely additive — the dispatch loop still runs the
    legacy flow; the registry is consumed only where a debug listing is safe.
    """
    phases: list[IngestionPhase] = [cls() for cls in _BUILTIN_PHASE_CLASSES]
    seeded = {phase.id for phase in phases}

    for phase in await _plugin_phase_objects(cat):
        if phase.id in seeded:
            continue
        seeded.add(phase.id)
        phases.append(phase)

    for spec, manager in await _registered_specs(cat):
        if spec.id in seeded:
            continue
        seeded.add(spec.id)
        phases.append(HookPhase(spec, cat=cat, manager=manager))
    return phases


def _diary_from_completed_phases(completed_phases: Any) -> dict[str, Any]:
    """Normalize the hook's ``completed_phases`` argument to a diary dict.

    The hook contract threads ``completed_phases`` as a ``list[dict]`` where
    every entry carries at least a ``"phase"`` key (MyGRAPH compatibility);
    a status doc (dict with a ``"completed_phases"`` key) and a raw diary
    dict are also accepted defensively. Malformed input (``None``, non-dict
    entries, entries without a ``"phase"`` key) is skipped, never raised on.

    Args:
        completed_phases: ``list[dict]`` of completed phases, a status doc,
            a diary dict, or None.

    Returns:
        The diary dict keyed by phase id (``{}`` when absent/malformed).
    """
    if isinstance(completed_phases, dict):
        if "completed_phases" in completed_phases:
            return get_completed_phases(completed_phases)
        return {k: v for k, v in completed_phases.items() if isinstance(v, dict)}
    if isinstance(completed_phases, list):
        return {
            p["phase"]: p
            for p in completed_phases
            if isinstance(p, dict) and p.get("phase") is not None
        }
    return {}


def is_phase_stale(
    spec: PhaseSpec,
    completed_phases: dict[str, Any],
    current_settings_updated_at: object,
    current_marker: object,
) -> bool:
    """Decide whether one phase is stale, purely (no I/O).

    Clock-free two-level check over the diary entry of ``spec.id``:

    - (a) dependency: for each upstream in ``spec.depends_on``, stale when the
      upstream entry is missing, when ``deps[upstream]`` is missing, or when
      it differs from the upstream's current diary ``marker``;
    - (b) settings fast-path: when ``spec.settings_category`` is set AND a
      current settings version is available, equal ``settings_version`` ->
      FRESH (the settings entry was not rewritten since the phase ran);
    - (c) slow-path: the plugin marker — equal to the recorded ``marker`` ->
      fresh, different -> stale.

    A missing diary entry (phase never completed) is always stale. A missing
    current settings version (no settings entry) skips the fast-path and
    falls to the slow-path — conservative.

    Args:
        spec: The phase declaration to evaluate.
        completed_phases: The diary dict keyed by phase id.
        current_settings_updated_at: The current settings entry's
            ``updated_at`` (opaque token, only ever compared for equality),
            or None when the settings entry is missing.
        current_marker: The plugin-defined marker from
            ``ingestion_phase_settings_marker`` (None = unknown).

    Returns:
        True when the phase must be re-run, False when it is fresh.
    """
    entry = completed_phases.get(spec.id)
    if not isinstance(entry, dict):
        return True

    # (state machine) A ``processing`` entry is stale only when its lease
    # expired: an abandoned run must be purged and re-run from scratch, while
    # a LIVE run (another worker, or this process still inside the phase body)
    # is left alone. ``error`` / ``skipped`` are terminal-non-complete states:
    # they are never stale by themselves (the source status carries the error,
    # and a skipped phase does not apply), while an ABSENT entry is pending and
    # therefore stale.
    state = phase_state(entry)
    if state == PHASE_STATE_PROCESSING:
        return is_phase_lease_expired(entry)
    if state in (PHASE_STATE_ERROR, PHASE_STATE_SKIPPED):
        return False

    # (a) dependency: upstream marker identity, never timestamps
    for upstream in spec.depends_on:
        upstream_entry = completed_phases.get(upstream)
        if not isinstance(upstream_entry, dict):
            return True
        deps = entry.get("deps")
        recorded = deps.get(upstream) if isinstance(deps, dict) else None
        if recorded is None or recorded != upstream_entry.get("marker"):
            return True

    # (b) settings fast-path: opaque version equality
    if spec.settings_category is not None and current_settings_updated_at is not None:
        if entry.get("settings_version") == current_settings_updated_at:
            return False

    # (c) slow-path: plugin marker comparison
    return entry.get("marker") != current_marker


def has_live_running_phase(completed_phases: Any) -> bool:
    """Whether the diary holds a ``processing`` entry with a LIVE lease.

    Such a source is being worked on by another worker (or by this process,
    earlier in the same pass): the machine must leave it alone and must NOT
    write a terminal COMPLETED while the lease is live. (The name is kept for
    API compatibility; ``processing`` is the current spelling of ``running``.)
    """
    diary = _diary_from_completed_phases(completed_phases)
    return any(
        phase_state(entry) == PHASE_STATE_PROCESSING
        and not is_phase_lease_expired(entry)
        for entry in diary.values()
    )


@hook(priority=0)
async def ingestion_phase_pending(
    pending, source, completed_phases, cat=None, lizard=None
) -> list[dict[str, Any]]:
    """Accumulator registrant declaring the stale phases for ``source``.

    Accumulator convention (see ``hooks.py``): ``pending`` is the mutable
    list threaded through every registrant by ``execute_hook``; this
    registrant appends ``{"phase": <id>}`` for each phase that is stale and
    returns the (possibly extended) list. The default no-op in ``hooks.py``
    (same priority) returns ``pending`` unchanged, so the two compose safely
    in any execution order.

    The phases probed are the MERGED set from :func:`merged_phases`: the
    built-in ``PHASES`` plus every ``PhaseSpec`` registered through the
    ``ingestion_phase_specs`` accumulator hook (built-ins win on duplicate
    ids). For each ``PhaseSpec`` the current inputs are resolved live:

    - ``current_settings_updated_at``: the ``updated_at`` of the settings
      entry for ``spec.settings_category`` (via ``crud_settings`` — the
      official CRUD API, no raw Redis), used only as an opaque equality
      token; a missing settings entry yields None and falls to the slow-path;
      registered phases with ``settings_category=None`` skip the fast-path
      entirely (their marker IS the material settings fingerprint);
    - ``current_marker``: the plugin-defined marker from the
      ``ingestion_phase_settings_marker`` hook (None = unknown ->
      conservative stale).

    Args:
        pending: list of stale-phase dicts accumulated so far (``None`` is
            normalized to ``[]``).
        source: the source being checked (e.g. ``"doc.pdf"``).
        completed_phases: ``list[dict]`` of completed phases for this source
            (each with at least a ``"phase"`` key), a status doc, or a diary
            dict — see :func:`_diary_from_completed_phases`.
        cat: the CheshireCat instance (may be ``None`` in unit tests; then
            every current input is unknown and the diary alone decides).

    Returns:
        The extended ``pending`` list.
    """
    if pending is None:
        pending = []
    cat = cat if cat is not None else lizard
    diary = _diary_from_completed_phases(completed_phases)

    for spec in (await merged_phases(cat, cat)).values():
        current_settings_updated_at = None
        current_marker = None
        if cat is not None:
            if spec.settings_category is not None:
                settings_key = phase_settings_key(spec.id, getattr(cat, "agent_key", None))
                settings_doc = await crud_settings.get_settings_by_category(
                    settings_key, spec.settings_category
                )
                if isinstance(settings_doc, dict):
                    current_settings_updated_at = settings_doc.get("updated_at")
            plugin_manager = await phase_owner(cat, spec.id)
            if plugin_manager is not None:
                ran_marker, marker = await execute_hook_compat(
                    plugin_manager,
                    "ingestion_phase_settings_marker",
                    spec.id,
                    caller=cat,
                )
                if ran_marker:
                    current_marker = marker
        if is_phase_stale(spec, diary, current_settings_updated_at, current_marker):
            pending.append({"phase": spec.id})
    return pending