"""Phase-machine extension hooks for the efficient-ingestion plugin.

These hooks define the contract between the EffING phase machine and any
external registrant (other plugins, e.g. MyGRAPH) that wants to participate in
ingestion phases. They are declared here, in the plugin itself, with
``@hook(priority=0)`` so that external plugins can override them with higher
priorities — no MyCAT core change is needed.

All defaults are conservative no-ops: a phase that nobody implements is
reported as pending (stale), fails hard when run, and its marker is unknown.

Importing this module has zero side effects: the only top-level statements are
the ``@hook`` decorators (which merely wrap the functions in ``CatHook``
instances). No Redis, no network, no filesystem access.
"""

import inspect
import json

from cat import hook
from cat.db.cruds import settings as crud_settings
from cat.db.database import DEFAULT_SYSTEM_KEY
from cat.exceptions import CustomHookVetoException
from cat.log import log
from cat.utils import SUFFIX_TO_CRYPT


def _hook_missing_from_manager(plugin_manager, hook_name: str) -> bool:
    """True when ``plugin_manager`` has no registrant for ``hook_name``."""
    hooks = getattr(plugin_manager, "hooks", None)
    return isinstance(hooks, dict) and hook_name not in hooks


def _is_missing_hook_error(exc: Exception, hook_name: str) -> bool:
    """True for the dispatcher's "hook not present in any plugin" error only."""
    return str(exc) == f"Hook {hook_name} not present in any plugin"


async def execute_phase_hook(
    plugin_manager, hook_name: str, *args, default=None, caller=None
):
    """Execute a phase hook, degrading when the hook is absent.

    A missing phase hook means "no external phases": the dispatcher exception
    is swallowed and ``default`` (or ``default()`` when it is a factory, e.g.
    ``list``) is returned after a warning. Errors raised BY an existing hook
    are never swallowed — they propagate unchanged.
    """
    def _degraded():
        log.warning(
            f"Phase hook {hook_name} is not present in any active plugin: "
            f"treating it as no external phases"
        )
        return default() if callable(default) else default

    if _hook_missing_from_manager(plugin_manager, hook_name):
        return _degraded()

    try:
        return await plugin_manager.execute_hook(hook_name, *args, caller=caller)
    except Exception as exc:  # noqa: BLE001 - only the missing-hook error degrades
        if not _is_missing_hook_error(exc, hook_name):
            raise
        return _degraded()


def _accepts_caller(function, keyword: str) -> bool:
    """True when ``function`` can receive the caller as ``keyword``.

    A hook declared ``def h(..., cat)`` cannot be called as ``h(lizard=...)``.
    The system manager passes ``lizard=`` and the agent manager ``cat=``; an
    agent-level provider (MyGRAPH/CATALOG) that is ALSO loaded on the system
    manager must therefore be skipped there and served by the agent manager,
    instead of logging a ``TypeError`` on every probe. ``**kwargs`` hooks
    accept either name.
    """
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):  # builtins / C callables: assume compatible
        return True
    if keyword in parameters:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())


async def execute_hook_compat(manager, hook_name: str, *args, caller=None):
    """Execute ``hook_name`` on one manager, skipping incompatible registrants.

    Returns ``(ran, value)``: ``ran`` is False when the manager has no
    registrant for the hook (missing hook, or none able to accept the manager's
    caller keyword). The first argument is threaded through the registrants,
    like ``MadHatter.execute_hook``; a generic hook error is logged and the
    chain continues, while a veto/management gate propagates.
    """
    hooks = getattr(manager, "hooks", None)
    if not isinstance(hooks, dict):
        # no registry (unit-test fake): delegate to the manager's dispatcher
        try:
            return True, await manager.execute_hook(hook_name, *args, caller=caller)
        except Exception as exc:  # noqa: BLE001 - only a missing hook is skipped
            if _is_missing_hook_error(exc, hook_name):
                return False, None
            raise

    registered = hooks.get(hook_name) or []
    if not registered:
        return False, None

    keyword = getattr(manager, "context_execute_hook", "cat")
    value = args[0] if args else None
    rest = args[1:]
    ran = False
    for registered_hook in registered:
        function = getattr(registered_hook, "function", registered_hook)
        if not _accepts_caller(function, keyword):
            continue
        ran = True
        try:
            result = function(value, *rest, **{keyword: caller})
            if inspect.isawaitable(result):
                result = await result
            if result is not None:
                value = result
        except CustomHookVetoException:
            raise
        except Exception as exc:  # noqa: BLE001 - same policy as MadHatter
            plugin_id = getattr(registered_hook, "plugin_id", None) or "?"
            log.error(f"Error in plugin {plugin_id}::{hook_name}: {exc}")
    return ran, value


async def execute_shared_hook(cat, hook_name: str, value, *args):
    """Thread a SHARED ingestion lifecycle hook through both managers.

    Hooks such as ``before_rabbithole_stores_documents`` and
    ``after_rabbithole_stored_documents`` are implemented by EffING (now
    system-only) AND by agent-level plugins. The agent manager runs first so
    agent plugins can append/rewrite documents, then the system manager runs
    EffING's own handler (the single token-budget split) on the final list.
    Each plugin lives on exactly one manager, so nothing runs twice.
    """
    ordered = []
    for manager in (agent_plugin_manager(cat), system_plugin_manager(cat)):
        if manager is not None and all(manager is not seen for seen in ordered):
            ordered.append(manager)
    for manager in ordered:
        ran, new_value = await execute_hook_compat(
            manager, hook_name, value, *args, caller=cat
        )
        if ran:
            value = new_value
    return value


def system_plugin_manager(cat):
    """The system-level plugin manager (BillTheLizard's own), or None.

    EffING is a system-only plugin: its own phase hooks are registered there.
    ``cat.lizard`` is the ``BillTheLizard`` singleton for every cat
    (CheshireCat/StrayCat) — verified against ``cat/mixins.py`` (BotMixin.lizard)
    — and ``.plugin_manager`` is its ``MadHatter``.
    """
    lizard = getattr(cat, "lizard", None)
    return getattr(lizard, "plugin_manager", None) if lizard is not None else None


def agent_plugin_manager(cat):
    """The caller's agent-level plugin manager (or None)."""
    return getattr(cat, "plugin_manager", None)


def phase_managers(cat):
    """The managers a phase hook can live on: system first, then agent.

    System-only EffING hooks live on the system manager; the phase PROVIDERS
    (MyGRAPH, CATALOG) stay agent-level plugins, so their hooks live on the
    caller's agent manager. The list is deduplicated by identity (a caller that
    exposes the same manager twice, or a unit-test fake without ``lizard``,
    yields a single entry) and never contains None.
    """
    managers = []
    for manager in (system_plugin_manager(cat), agent_plugin_manager(cat)):
        if manager is not None and all(manager is not seen for seen in managers):
            managers.append(manager)
    return managers


async def execute_phase_hook_on_all(cat, hook_name: str, *args, default=None):
    """Execute a NON-accumulator phase hook on every manager that has it.

    Every registrant must run: EffING's built-in completion gate lives on the
    system manager and agent-level providers (MyGRAPH) register their own gate
    on the agent manager. Exceptions — including a veto — propagate unchanged
    so the caller can honour them. Only when NO manager registers the hook does
    this degrade like :func:`execute_phase_hook`.
    """
    ran_any = False
    result = default() if callable(default) else default
    for manager in phase_managers(cat):
        ran, new_result = await execute_hook_compat(
            manager, hook_name, *args, caller=cat
        )
        if ran:
            ran_any = True
            result = new_result
    if not ran_any:
        log.warning(
            f"Phase hook {hook_name} is not present in any active plugin: "
            f"treating it as no external phases"
        )
    return result


async def execute_phase_accumulator(cat, hook_name: str, value, *args, default=None):
    """Thread an accumulator phase hook through every manager that has it.

    System first, then agent: the accumulated value returned by EffING's own
    registrant (system manager) is handed to any agent-level registrant, so a
    phase provider that also implements the accumulator keeps working. A
    missing hook on one manager is skipped (never a warning) as long as at
    least one manager registers it; if none does, this degrades like
    :func:`execute_phase_hook`.
    """
    ran_any = False
    for manager in phase_managers(cat):
        ran, new_value = await execute_hook_compat(
            manager, hook_name, value, *args, caller=cat
        )
        if ran:
            ran_any = True
            value = new_value
    if not ran_any:
        log.warning(
            f"Phase hook {hook_name} is not present in any active plugin: "
            f"treating it as no external phases"
        )
        return default() if callable(default) else default
    return value

def _caller(cat, lizard):
    """The caller object, whatever name the plugin manager used for it.

    A system plugin manager passes hooks ``lizard=<caller>`` while an agent
    manager passes ``cat=<caller>`` (``MadHatter.context_execute_hook``). EffING
    runs on the system manager but still acts on behalf of the calling agent, so
    the value is the agent cat in both cases; only the keyword name differs.
    """
    return cat if cat is not None else lizard


# The ``completed_phases`` argument threaded through the hooks is a
# ``list[dict]`` where every entry carries at least a ``"phase"`` key
# (MyGRAPH compatibility: its existing hook reads
# ``{p.get("phase") for p in completed_phases if isinstance(p, dict)}``),
# plus optional ``marker``/``deps`` keys. No timestamps are used for
# correctness anywhere in this protocol.


@hook(priority=0)
def ingestion_phase_pending(pending, source, completed_phases, cat=None, lizard=None):
    """Accumulator of stale phases for ``source``.

    ``pending`` is threaded through every registrant: each one appends the
    ``{"phase": <id>, ...}`` entries it considers stale and returns the
    (possibly extended) list. The default implementation is the identity —
    it returns ``pending`` unchanged, so a source with no registrants is
    reported as having nothing pending.

    Args:
        pending: list of stale-phase dicts accumulated so far (may be ``None``
            from a caller that did not initialize it; it is returned as-is).
        source: the source being checked (e.g. ``"doc.pdf"``).
        completed_phases: ``list[dict]`` of completed phases for this source,
            each with at least a ``"phase"`` key.
        cat: the CheshireCat instance (may be ``None`` in unit tests).

    Returns:
        The extended ``pending`` list.
    """
    return pending


@hook(priority=0)
def ingestion_phase_run(phase, source, completed_phases, cat=None, lizard=None):
    """Run one phase for ``source``; tri-state contract.

    A registrant that implements ``phase`` must either:
      - return ``{"status": "done"}`` on success;
      - return ``{"status": "not_ready", "retry_after": N}`` when the phase is
        retriable but cannot run yet (the machine retries after ``N`` seconds);
      - return ``{"status": "error", "reason": "..."}`` on a permanent failure
        (the machine marks the phase ERROR with that reason);
      - raise an exception on permanent failure (the machine marks the phase
        ERROR).

    The default returns ``None``, which the machine treats as fail-hard /
    unimplemented: a phase with no registrant is an error, never a silent
    success.

    Args:
        phase: the phase id to run (e.g. ``"embedding"``).
        source: the source being processed (e.g. ``"doc.pdf"``).
        completed_phases: ``list[dict]`` of completed phases for this source,
            each with at least a ``"phase"`` key.
        cat: the CheshireCat instance (may be ``None`` in unit tests).

    Returns:
        ``None`` (default), or a status dict per the contract above.
    """
    return None


@hook(priority=0)
async def ingestion_phase_purge(phase, source, cat=None, lizard=None):
    """Purge the products of one external phase for ONE source.

    Recovery contract: when a ``running`` diary entry's lease expired, the
    machine calls this hook for the abandoned phase (and any completed later
    phase, in REVERSE execution order) before re-running it. A registrant must
    delete ONLY the artifacts THAT phase produced for THAT source.

    The default is a no-op returning ``None``, which the dispatcher reads as
    "this phase has NO purge implementation": it does NOT delete anything
    blindly, logs a WARNING naming the phase/source and leaves the entry
    ``running`` (visible, not re-run, not completed). A real implementation
    must return a truthy value (e.g. ``{"status": "done"}``) after purging.

    Args:
        phase: the phase id whose products must be purged.
        source: the source whose phase artifacts must be removed.
        cat: the CheshireCat instance (may be ``None`` in unit tests).

    Returns:
        ``None`` when the phase has no purge implementation, else a truthy
        result once the source-scoped purge completed.
    """
    return None


@hook(priority=0)
async def before_ingestion_status_completed(source, cat=None, lizard=None):
    """Final gate before ``source`` is marked COMPLETED.

    Called right before the machine writes the COMPLETED status for a source.
    A registrant that wants to veto the completion raises
    :class:`cat.exceptions.CustomHookVetoException`; the dispatcher propagates
    it (unlike ordinary errors, which are logged and swallowed) so the caller
    marks the source ``error`` with the reason instead of ``completed``.

    The built-in implementation vetoes when the caller exposes pending
    post-phase work through a ``has_pending_source_tasks(source)`` probe
    (sync or async, e.g. a graph handler attached to the cat). Without such a
    probe it is a no-op and the source is allowed to complete.

    Args:
        source: the source about to be marked completed (e.g. ``"doc.pdf"``).
        cat: the CheshireCat instance (may be ``None`` in unit tests).

    Returns:
        ``None`` (default).

    Raises:
        CustomHookVetoException: when the caller reports pending source tasks.
    """
    probe = getattr(_caller(cat, lizard), "has_pending_source_tasks", None)
    if probe is None:
        return

    pending = probe(source)
    if inspect.isawaitable(pending):
        pending = await pending
    if pending:
        raise CustomHookVetoException(
            f"Pending ingestion tasks for source '{source}'"
        )


#: Settings category whose configuration feeds each built-in phase.
_PHASE_SETTINGS_CATEGORY = {
    "parsing_chunking": "chunker",
    "embedding": "embedder",
}

#: SCOPE of each built-in phase's settings entry. The marker and the
#: fast-path ``settings_version`` MUST read the same scope the writer uses, or
#: a real change is invisible:
#:
#: - ``chunker`` is an AGENT-scoped service (``CheshireCat.create`` resolves it
#:   from ``cat.agent_key`` and ``PUT /chunking/settings`` writes it under the
#:   caller's agent key);
#: - ``embedder`` is SYSTEM-scoped (``PUT /embedder/settings`` writes it under
#:   ``DEFAULT_SYSTEM_KEY`` and the re-embed pass resolves it from the system
#:   lizard).
#:
#: Phases not listed here (external ones) default to the caller's agent key.
_PHASE_SETTINGS_SCOPE = {
    "parsing_chunking": "agent",
    "embedding": "system",
}


def phase_settings_key(phase: str, agent_key: str | None) -> str:
    """Return the settings key the phase's material configuration lives under.

    ``chunker`` markers live under the caller's agent key, ``embedder`` markers
    under ``DEFAULT_SYSTEM_KEY`` (see :data:`_PHASE_SETTINGS_SCOPE`). External
    phases default to the caller's agent key; a missing key falls back to the
    system key so the lookup still resolves.

    Args:
        phase: the phase id (e.g. ``"parsing_chunking"``).
        agent_key: the caller's ``agent_key`` (may be ``None``).

    Returns:
        The settings key to read with ``get_settings_by_category``.
    """
    if _PHASE_SETTINGS_SCOPE.get(phase) == "system":
        return DEFAULT_SYSTEM_KEY
    return agent_key or DEFAULT_SYSTEM_KEY


def _material_settings(value: object) -> dict:
    """Return the non-secret settings fields that materially affect a phase.

    Credentials (fields matching ``SUFFIX_TO_CRYPT``) are stripped: they are
    encrypted with a random IV on every write, so including them would make
    the fingerprint non-deterministic for an unchanged configuration. Every
    other field is kept — for the built-in chunker that is
    ``encoding_name``/``chunk_size``/``chunk_overlap``; for embedders it is
    the model/name and, when stored, the embedding size.
    """
    if not isinstance(value, dict):
        return {}
    return {
        k: v
        for k, v in value.items()
        if not (isinstance(k, str) and any(suffix in k for suffix in SUFFIX_TO_CRYPT))
    }


@hook(priority=0)
async def ingestion_phase_settings_marker(phase, cat=None, lizard=None):
    """Plugin-defined material marker for ``phase``.

    The machine compares this marker against the one recorded in the
    ``completed_phases`` diary entry to decide whether the phase's settings
    changed materially (clock-free invalidation). For the built-in phases this
    returns a stable, sorted JSON fingerprint of the service configuration,
    read from the SAME scope the corresponding route writes:

    - ``parsing_chunking`` -> the AGENT-scoped ``chunker`` category (class/config
      name plus the chunker fields, e.g. size and overlap);
    - ``embedding`` -> the SYSTEM-scoped ``embedder`` category (class/config
      name plus the model/name and, when stored, the embedding size).

    Any unknown phase returns ``None`` ("unknown"), which the machine treats
    conservatively as stale. A known phase whose settings entry is missing
    also returns ``None``.

    Args:
        phase: the phase id whose marker is requested (e.g. ``"embedding"``).
        cat: the caller (CheshireCat/StrayCat/lizard) whose ``agent_key``
            selects the chunker scope; may be ``None`` in unit tests.

    Returns:
        A deterministic ``json.dumps(..., sort_keys=True)`` string, or
        ``None`` for an unknown phase / missing settings entry.
    """
    category = _PHASE_SETTINGS_CATEGORY.get(phase)
    if category is None:
        return None

    settings_key = phase_settings_key(phase, getattr(_caller(cat, lizard), "agent_key", None))
    settings_doc = await crud_settings.get_settings_by_category(settings_key, category)
    if not isinstance(settings_doc, dict):
        return None

    return json.dumps(
        {
            "config": settings_doc.get("name"),
            "value": _material_settings(settings_doc.get("value")),
        },
        sort_keys=True,
    )


@hook(priority=0)
def ingestion_phase_specs(specs, cat=None, lizard=None) -> list:
    """Accumulator: plugins append PhaseSpec objects to declare their own
    ingestion phases. EffING merges them into its PHASES at probe time.
    The default returns `specs` unchanged.

    Args:
        specs: list of ``PhaseSpec`` objects accumulated so far (may be
            ``None`` from a caller that did not initialize it; it is returned
            as-is).
        cat: the CheshireCat instance (may be ``None`` in unit tests).

    Returns:
        The extended ``specs`` list.
    """
    return specs