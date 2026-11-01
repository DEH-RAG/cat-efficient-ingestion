"""Phase DAG and clock-free two-level staleness for the efficient-ingestion plugin.

Declares the built-in ingestion phases (``parsing_chunking``, ``embedding``)
and the accumulator registrant of ``ingestion_phase_pending`` that reports
which of them are stale for a source, using ONLY content comparison — no
timestamps are ever used for correctness.

**Recursive dependency tags.** A phase's diary entry (see
``registry.completed_phases``) records the explicit state plus a material
``tag``. The tag is a SHA-256 of the phase's OWN material (its settings
marker, or a declared ``id:category:deps`` fallback when no marker hook
resolves) folded with the tags of every dependency in ``spec.depends_on``.
Because dependencies are resolved first, a change deep in the chain changes
the tag of every dependent, transitively. ``compute_tags`` performs this
resolution in dependency order and raises on a cycle.

A phase is stale when:

- its own entry is absent (pending) or its ``processing`` lease expired;
- any dependency is absent (pending), ``processing``, ``error`` or
  ``skipped`` — there is no completed upstream work to consume;
- the stored ``tag`` differs from the recursively computed current tag, or is
  missing (legacy entry, stale exactly once until rewritten).

The legacy flat ``settings_version`` field is no longer part of the decision:
the tag already folds in every material input. The field stays readable on
old entries only.

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

import contextvars
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal, Optional

from cat import hook
from cat.log import log

from .hooks import (
    _accepts_caller,
    execute_hook_compat,
    phase_managers,
)
from .phase_tag_resolver import (
    PhaseTagCycleError,
    PhaseTagError,
    compose_tag,
    own_tag,
    relevant_settings,
    resolve_tags,
)
from .registry import (
    PHASE_STATE_ERROR,
    PHASE_STATE_PROCESSING,
    PHASE_STATE_SKIPPED,
    IngestionStatus,
    _delete_status_doc,
    get_completed_phases,
    get_status,
    phase_state,
    set_status,
    should_abandon,
    source_lock,
)

#: Scope (``"agent"`` or a chat id) of the file deletion currently running.
#: ``delete(source, cat)`` has no scope argument, so the delete orchestrator
#: publishes it for the built-in phases, which purge agent/chat-scoped vector
#: collections. Task-local (never shared across workers) and reset in a
#: ``finally`` block.
_DELETE_SCOPE: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "effing_delete_scope", default=None
)

#: Configuration class name of the efficient ingestion engine in the SYSTEM
#: ``ingestion`` category. Mirrors the MyGRAPH ``_phase_path_active`` predicate.
_EFFICIENT_INGESTION_CONFIG_NAME = "EfficientIngestionConfiguration"


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
        settings_check: Optional narrowing of the settings that feed
            :meth:`IngestionPhase.tag`: a key collection keeps only those
            keys, a (possibly async) callable returns the value that
            participates. ``None`` means the whole settings value.
    """

    id: str
    settings_category: str | None
    depends_on: tuple[str, ...]
    settings_check: Any = None


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
    #: Optional narrowing of the settings that feed :meth:`tag` (see
    #: :class:`PhaseSpec`). ``None`` uses the whole settings value.
    settings_check: Any = None

    @property
    def name(self) -> str:
        """Stable phase name (alias of :attr:`id`)."""
        return self.id

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

    async def tag(self, settings: Any = None) -> str:
        """Detached identity of this phase's OWN material configuration.

        ``sha256(kind + ":" + canonical_json(relevant_settings))`` where
        ``relevant_settings`` is ``settings_check``-narrowed when declared,
        else the whole ``settings`` value. Async because a declared
        ``settings_check`` may be a coroutine. Only configuration is read —
        never an LLM call. Folding in dependency tags is
        :func:`phase_tag_resolver.resolve_tags`' job.
        """
        relevant = await relevant_settings(
            {} if settings is None else settings, self.settings_check
        )
        return own_tag(self.id, relevant)


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

    The built-in phase BODIES still live in the legacy ``reembed`` dispatcher:
    :meth:`process` refuses to run here rather than duplicating the dispatcher,
    while :meth:`delete` delegates to the same source-scoped legacy purge the
    recovery path uses, so the reverse-order file deletion can call EVERY
    phase's ``delete`` (built-ins included).
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
        """Delete this phase's source-scoped products through the legacy purge.

        The phase body still lives in ``reembed`` (the dispatcher owns the
        source-scoped product boundary), so :meth:`delete` delegates to the
        SAME legacy purge instead of duplicating it. The delete orchestrator
        publishes the scope in :data:`_DELETE_SCOPE`; without it a caller falls
        back to the caller's own scope (a ``StrayCat.id`` is chat-scoped, a
        ``CheshireCat`` is agent-scoped).

        Raises:
            PhaseDeleteError: when the legacy purge reports no implementation.
        """
        from cat.services.memory.models import VectorMemoryType

        from . import reembed

        scope = _DELETE_SCOPE.get()
        if scope is None:
            caller_id = getattr(cat, "id", None)
            scope = str(caller_id) if caller_id else "agent"
        chat_id = None if scope == "agent" else scope
        collection_name = (
            VectorMemoryType.EPISODIC if chat_id else VectorMemoryType.DECLARATIVE
        )
        ccat = cat
        if not hasattr(ccat, "_id") and hasattr(ccat, "lizard"):
            ccat = await ccat.lizard.get_cheshire_cat(ccat.agent_key)
        deleted = await reembed._purge_phase(
            ccat, cat, collection_name, scope, source, chat_id, self.id
        )
        if not deleted:
            raise PhaseDeleteError(
                f"Built-in phase '{self.id}' has no delete/purge "
                f"implementation; refusing to delete its artifacts silently"
            )

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
    """Fallback material for a phase with no resolvable settings marker."""
    return f"{phase_id}:{settings_category or '-'}:{','.join(depends_on)}"


async def _material_settings(cat, phase) -> Any:
    """Resolve the settings value handed to ``phase.tag`` for this run.

    A HookPhase with an in-instance ``settings_marker`` resolves its own marker
    in :meth:`HookPhase.tag`; here it gets ``{}``. For every other phase the
    material settings marker is the configuration fingerprint: the same
    configuration-only value the marker hook produces (no LLM call). A missing
    marker falls back to the declared tag, and an unknown marker to an explicit
    ``<id>:unknown`` sentinel, so a phase never silently hashes nothing.
    """
    if getattr(phase, "_settings_marker", None) is not None:
        return {}
    if cat is None:
        return _declared_tag(
            phase.id, getattr(phase, "settings_category", None), phase.depends_on
        )
    marker = await execute_phase_on_owner(
        cat,
        "ingestion_phase_settings_marker",
        phase.id,
        default=_MISSING,
        caller=cat,
    )
    if marker is _MISSING:
        return _declared_tag(
            phase.id, getattr(phase, "settings_category", None), phase.depends_on
        )
    if marker is None:
        return f"{phase.id}:unknown"
    return marker


async def compute_tags(cat, phases) -> dict[str, str]:
    """Resolve every phase's recursive tag in dependency order.

    Reads each phase's material configuration ONCE (the marker hook, or the
    declared fallback) and delegates the dependency fold to
    :func:`phase_tag_resolver.resolve_tags`. A change deep in the chain changes
    every dependent's tag.

    Raises:
        PhaseTagCycleError: when the dependency graph cycles, naming the cycle.
        PhaseTagError: when a dependency is not declared.
    """
    phase_list = list(phases)
    settings = {
        phase.id: await _material_settings(cat, phase) for phase in phase_list
    }
    return await resolve_tags(settings, phase_list)


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


async def _call_legacy_hook_async(manager, hook_name: str, phase: str, cat):
    """Call a legacy marker registrant, awaiting it when it is a coroutine.

    The marker hook only reads configuration (it performs no LLM call), so
    awaiting it is cheap and deterministic. A plain function is called
    directly; the first compatible registrant wins. Returns ``_MISSING`` when
    the manager has no compatible registrant.
    """
    hooks = getattr(manager, "hooks", None)
    registered = hooks.get(hook_name) if isinstance(hooks, dict) else None
    if not registered:
        return _MISSING
    keyword = getattr(manager, "context_execute_hook", "cat")
    for registered_hook in registered:
        function = getattr(registered_hook, "function", registered_hook)
        if not _accepts_caller(function, keyword):
            continue
        try:
            value = function(phase, **{keyword: cat})
            if inspect.isawaitable(value):
                value = await value
            return value
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
    - :meth:`tag` reads the settings marker (awaiting it when the registrant
      is a coroutine) and folds the dependency tags into a recursive hash; when
      no marker hook resolves it falls back to the declared
      ``id:category:deps`` material (documented, never silent).
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
        self.settings_check = getattr(spec, "settings_check", None)
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
                source,
                default=_MISSING,
                caller=used,
            )
        if result is None or result is _MISSING:
            raise PhaseDeleteError(
                f"Ingestion phase '{self.id}' has no delete/purge "
                f"implementation; refusing to delete its artifacts silently"
            )

    async def tag(self, settings: Any = None) -> str:
        if self._settings_marker is not None:
            marker = await self._resolve_marker(self._cat)
            if marker is _MISSING:
                settings = _declared_tag(
                    self.id, self.settings_category, self.depends_on
                )
            elif marker is None:
                settings = f"{self.id}:unknown"
            else:
                settings = marker
        return await super().tag(settings)

    async def _resolve_marker(self, cat):
        if self._settings_marker is not None:
            try:
                value = self._settings_marker(self.id)
                if inspect.isawaitable(value):
                    value = await value
                return value
            except Exception as exc:  # noqa: BLE001 - a marker must never break tag()
                log.warning(f"Settings marker for phase '{self.id}' failed: {exc}")
                return _MISSING
        if self._manager is not None:
            return await _call_legacy_hook_async(
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
            manager, hook_name, phase, *args, caller=caller
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


async def _resolved_system_ingestion_config_name(cat) -> Optional[str]:
    """Effective SYSTEM-level ``ingestion`` engine name (EffING's own gate).

    Mirrors MyGRAPH's ``_phase_path_active`` helper: prefer the host's
    ``resolved_config_name(lizard)`` when it exists, otherwise read the single
    system-level ``ingestion`` category entry directly. Both paths return the
    same value. Any failure is ``None`` (the gate then stays inactive).
    """
    try:
        from cat.services.factory.ingestion import resolved_config_name
    except ImportError:
        resolved_config_name = None

    if resolved_config_name is not None:
        try:
            return await resolved_config_name(cat)
        except Exception:  # noqa: BLE001 - fall back to the direct settings read
            pass

    try:
        from cat.db.cruds import settings as crud_settings
        from cat.db.database import DEFAULT_SYSTEM_KEY

        saved = await crud_settings.get_settings_by_category(
            DEFAULT_SYSTEM_KEY, "ingestion"
        )
    except Exception as exc:  # noqa: BLE001 - an unresolvable gate is inactive
        log.warning(
            f"Could not resolve the system ingestion engine ({exc!r}); "
            f"using the standard file deletion."
        )
        return None
    return (saved or {}).get("name")


async def phase_delete_active(cat) -> bool:
    """Whether EffING owns the reverse-order phase deletion for this caller.

    Single source of truth for the two deletion modes (mirrors MyGRAPH's
    ``_phase_path_active``):

    ==========================  ======================  ==================
    EffING installed/loadable   SYSTEM ``ingestion``    deletion mode
    ==========================  ======================  ==================
    no                          any                     standard
    yes                         core engine             standard
    yes                         efficient engine        phase reverse
    ==========================  ======================  ==================

    Gate FALSE means the STANDARD deletion applies: no ``deleting`` state is
    written, nothing is raised, no residue is left behind.
    """
    try:
        import cat.core_plugins.cat_efficient_ingestion  # noqa: F401
    except Exception:  # noqa: BLE001 - EffING not importable
        return False

    try:
        name = await _resolved_system_ingestion_config_name(cat)
    except Exception as exc:  # noqa: BLE001 - an unresolvable gate is inactive
        log.warning(
            f"Could not resolve the system ingestion engine ({exc!r}); "
            f"using the standard file deletion."
        )
        return False
    return name == _EFFICIENT_INGESTION_CONFIG_NAME


def _delete_scope(cat, scope: Optional[str]) -> str:
    """Resolve the source scope (``"agent"`` or a chat id) for a deletion."""
    if scope:
        return str(scope)
    caller_id = getattr(cat, "id", None)
    return str(caller_id) if caller_id else "agent"


def _delete_source_type(source: str) -> str:
    return "url" if source.startswith("http") else "file"


def phase_execution_order(phases) -> list:
    """Return ``phases`` in dependency (execution) order, stable by input.

    A phase's depth is ``0`` without dependencies, else ``1 + max(depth(dep))``;
    sorting by ``(depth, input position)`` yields a deterministic topological
    order. An unknown dependency or a cycle contributes no edge instead of
    raising, so deletion always has an order to reverse.
    """
    phase_list = list(phases)
    by_id: dict[str, Any] = {}
    for phase in phase_list:
        by_id.setdefault(phase.id, phase)

    depth: dict[str, int] = {}

    def _depth(phase_id: str, visiting: set[str]) -> int:
        if phase_id in depth:
            return depth[phase_id]
        phase = by_id.get(phase_id)
        if phase is None or phase_id in visiting:
            return 0
        visiting.add(phase_id)
        deps = [
            dep
            for dep in (getattr(phase, "depends_on", ()) or ())
            if dep in by_id
        ]
        value = 0 if not deps else 1 + max(_depth(dep, visiting) for dep in deps)
        visiting.discard(phase_id)
        depth[phase_id] = value
        return value

    indexed = list(enumerate(phase_list))
    indexed.sort(key=lambda item: (_depth(item[1].id, set()), item[0]))
    return [phase for _index, phase in indexed]


async def delete_source(source: str, cat, scope: Optional[str] = None) -> bool:
    """Delete ONE source through EVERY registered phase in REVERSE order.

    This is EffING's single deletion path: it applies its own gate and falls
    back to the STANDARD deletion when the phase path does not apply. The steps
    of the phase path are the contract:

    1. gate: :func:`phase_delete_active` must be True, at least one
       plugin-registered phase (beyond the always-present built-ins) must be in
       :func:`get_phases`, and the status diary must not reference an
       UNINSTALLED phase. Otherwise ``False`` is returned and the STANDARD
       deletion (the core wipe + :func:`delete_status`) applies unchanged.
    2. write the file-level ``deleting`` state BEFORE any phase delete.
    3. call each phase's ``delete(source, cat)`` in REVERSE execution order,
       recording every completed phase so a re-run resumes with the remaining
       ones. A phase without a delete implementation (or whose delete raises)
       is surfaced as :class:`PhaseDeleteError`; the ``deleting`` row stays so
       the operation is retried, and no later delete runs (the remaining
       upstream inputs must stay resolvable).
    4. remove the status record LAST, after every delete succeeded.

    Returns:
        ``True`` when the phase path ran and the record was removed, ``False``
        when the standard deletion applies instead.
    """
    agent_id = getattr(cat, "agent_key", None)
    if not agent_id:
        return False

    scope = _delete_scope(cat, scope)

    # ORDERING RULE: the delete path takes ONLY the per-source lock (never a
    # phase lock) and holds it for the WHOLE operation — the ``deleting``
    # write, every phase ``delete()`` and the final row removal. A concurrent
    # re-embed pass holds the same lock for its whole per-source sequence, so
    # it cannot write artifacts that this purge has just removed (and cannot
    # re-create them after the row is gone).
    async with source_lock(agent_id, source):
        return await _delete_source_locked(source, cat, agent_id, scope)


async def _delete_source_locked(
    source: str, cat, agent_id: str, scope: str
) -> bool:
    """The phase-path body of :func:`delete_source`, run under the source lock."""
    if not await phase_delete_active(cat):
        # Gate FALSE: STANDARD deletion. No ``deleting`` row, no error, and
        # the record (including a stale ``deleting`` one) is removed.
        await _delete_status_doc(agent_id, scope, source)
        return False

    try:
        phases = await get_phases(cat)
    except Exception as exc:  # noqa: BLE001 - a broken registry is an error
        log.error(
            f"Agent id: {agent_id}. Could not resolve the ingestion phases "
            f"for {source}: {exc}"
        )
        raise PhaseDeleteError(
            f"Could not resolve the ingestion phases for {source}: {exc}"
        ) from exc

    plugin_phases = [phase for phase in phases if phase.id not in PHASES]
    if not plugin_phases:
        # Only the always-present built-ins: the standard wipe owns them.
        await _delete_status_doc(agent_id, scope, source)
        return False

    doc = await get_status(agent_id, scope, source)
    diary = (doc or {}).get("completed_phases")
    diary_ids = set(diary) if isinstance(diary, dict) else set()
    registered_ids = {phase.id for phase in phases}
    uninstalled = sorted(diary_ids - registered_ids)
    if uninstalled:
        log.warning(
            f"Agent id: {agent_id}. Source {source} has diary phases that are "
            f"no longer installed ({uninstalled}); using the standard deletion."
        )
        await _delete_status_doc(agent_id, scope, source)
        return False

    already: set[str] = set()
    stored_deleted = (doc or {}).get("deleted_phases")
    if isinstance(stored_deleted, (list, tuple, set)):
        already = {str(phase_id) for phase_id in stored_deleted}

    await set_status(
        agent_id,
        scope,
        source,
        type_=_delete_source_type(source),
        status=IngestionStatus.DELETING,
        chat_id=None if scope == "agent" else scope,
        clear_phase=True,
        extra={"deleted_phases": sorted(already)},
    )

    token = _DELETE_SCOPE.set(scope)
    try:
        for phase in reversed(phase_execution_order(phases)):
            if phase.id in already:
                continue

            delete = getattr(phase, "delete", None)
            if not callable(delete):
                log.error(
                    f"Agent id: {agent_id}. Ingestion phase '{phase.id}' has no "
                    f"delete implementation; refusing to delete {source} silently"
                )
                raise PhaseDeleteError(
                    f"Ingestion phase '{phase.id}' has no delete implementation"
                )

            try:
                await delete(source, cat)
            except PhaseDeleteError as exc:
                log.error(
                    f"Agent id: {agent_id}. Ingestion phase '{phase.id}' has no "
                    f"delete/purge implementation for {source}: {exc}"
                )
                raise
            except NotImplementedError as exc:
                log.error(
                    f"Agent id: {agent_id}. Ingestion phase '{phase.id}' has no "
                    f"delete implementation for {source}: {exc}"
                )
                raise PhaseDeleteError(
                    f"Ingestion phase '{phase.id}' has no delete implementation"
                ) from exc
            except Exception as exc:  # noqa: BLE001 - surface, never skip
                log.error(
                    f"Agent id: {agent_id}. delete() for phase '{phase.id}' of "
                    f"{source} raised ({exc}); leaving the row deleting for the "
                    f"next pass"
                )
                raise PhaseDeleteError(
                    f"delete() for phase '{phase.id}' of {source} failed: {exc}"
                ) from exc

            already.add(phase.id)
            await set_status(
                agent_id,
                scope,
                source,
                type_=_delete_source_type(source),
                status=IngestionStatus.DELETING,
                chat_id=None if scope == "agent" else scope,
                clear_phase=True,
                extra={"deleted_phases": sorted(already)},
            )
    finally:
        _DELETE_SCOPE.reset(token)

    # STATE LAST: the row is removed only after every phase delete succeeded.
    await _delete_status_doc(agent_id, scope, source)
    return True


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


def downstream_phase_ids(phases, phase_id: str) -> list[str]:
    """Ids that TRANSITIVELY depend on ``phase_id`` (excluding it).

    ``phases`` is the merged ``{id: spec}`` map. Order follows the mapping's
    insertion order, so the result is deterministic. Used by the deletion
    paths: when a phase's products are wiped, every phase that consumed them
    (directly or through another dependency) has had its data invalidated and
    must stop being recorded as ``complete``.

    A malformed entry (no ``depends_on``) simply contributes no edges; the
    function never raises.
    """
    deps_by_id = {
        pid: tuple(getattr(spec, "depends_on", ()) or ())
        for pid, spec in phases.items()
    }
    downstream = set()
    changed = True
    while changed:
        changed = False
        for pid, deps in deps_by_id.items():
            if pid == phase_id or pid in downstream:
                continue
            if phase_id in deps or any(dep in downstream for dep in deps):
                downstream.add(pid)
                changed = True
    return [pid for pid in phases if pid in downstream]


def dependency_outcome(phase, diary) -> Optional[tuple[str, str]]:
    """Propagate a dependency's terminal outcome onto a dependent phase.

    Pure, clock-free decision (no I/O). Iterates ``phase.depends_on`` in its
    DECLARED order and returns the FIRST offending dependency:

    - ``("error", "dependency <dep> is error")`` when that dependency's diary
      entry is ``error``;
    - ``("skipped", "dependency <dep> is skipped")`` when it is ``skipped``;
    - ``None`` when no dependency is terminal-failed.

    A dependent of a failed phase can never run its body or become
    ``complete``; the dispatcher records the returned state instead.
    """
    diary = diary if isinstance(diary, dict) else {}
    for dep in getattr(phase, "depends_on", ()) or ():
        state = phase_state(diary.get(dep))
        if state == PHASE_STATE_ERROR:
            return PHASE_STATE_ERROR, f"dependency {dep} is error"
        if state == PHASE_STATE_SKIPPED:
            return PHASE_STATE_SKIPPED, f"dependency {dep} is skipped"
    return None


def is_phase_stale(
    spec: PhaseSpec,
    completed_phases: dict[str, Any],
    current_tag: object,
) -> bool:
    """Decide whether one phase is stale, purely (no I/O).

    Clock-free decision over the diary entry of ``spec.id``:

    - the entry's OWN state machine is honoured first: a ``processing`` entry
      is stale only when its lease expired (a live run is left alone), while
      ``error`` / ``skipped`` are terminal-non-complete and not stale by
      themselves;
    - a dependency that is absent (pending), ``processing``, ``error`` or
      ``skipped`` makes this phase stale — it cannot have consumed completed
      upstream work;
    - the stored ``tag`` (a recursive hash of this phase's own material plus
      every dependency's tag) must equal ``current_tag``. A MISSING/empty tag
      is a legacy entry and is stale exactly once (the rewrite stores a tag).

    ``settings_version`` is deliberately NOT consulted: the tag already folds
    in every material input, so the old flat version comparison is redundant.
    The field stays readable on legacy entries but never drives the decision.

    Args:
        spec: The phase declaration to evaluate.
        completed_phases: The diary dict keyed by phase id.
        current_tag: The recursively computed tag for ``spec.id`` now.

    Returns:
        True when the phase must be re-run, False when it is fresh.
    """
    entry = completed_phases.get(spec.id)
    if not isinstance(entry, dict):
        return True

    state = phase_state(entry)
    if state == PHASE_STATE_PROCESSING:
        return should_abandon(entry)
    if state in (PHASE_STATE_ERROR, PHASE_STATE_SKIPPED):
        return False

    # Dependency state: pending / processing / error / skipped upstreams mean
    # there is no completed material for this phase to have consumed.
    for upstream in spec.depends_on:
        upstream_entry = completed_phases.get(upstream)
        if not isinstance(upstream_entry, dict):
            return True
        upstream_state = phase_state(upstream_entry)
        if upstream_state in (
            PHASE_STATE_PROCESSING,
            PHASE_STATE_ERROR,
            PHASE_STATE_SKIPPED,
        ):
            return True

    stored_tag = entry.get("tag")
    if not isinstance(stored_tag, str) or not stored_tag:
        return True
    return stored_tag != current_tag


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
        and not should_abandon(entry)
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
    ids). The current recursive tags are resolved ONCE per probe with
    :func:`compute_tags` over :func:`get_phases`; each tag already folds in the
    phase's settings marker and every dependency's tag, so no separate
    settings-version or marker read is needed here.

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

    tags: dict[str, str] = {}
    if cat is not None:
        try:
            tags = await compute_tags(cat, await get_phases(cat))
        except PhaseTagError as exc:
            log.error(f"Cannot compute phase tags: {exc}")
            tags = {}

    for spec in (await merged_phases(cat, cat)).values():
        current_tag = tags.get(spec.id)
        if current_tag is None:
            current_tag = compose_tag(
                own_tag(
                    spec.id,
                    _declared_tag(
                        spec.id, spec.settings_category, spec.depends_on
                    ),
                ),
                (),
                {},
            )
        if is_phase_stale(spec, diary, current_tag):
            pending.append({"phase": spec.id})
    return pending