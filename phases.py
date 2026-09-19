"""Phase DAG and clock-free two-level staleness for the efficient-ingestion plugin.

Declares the built-in ingestion phases (``parsing_chunking``, ``embedding``)
and the accumulator registrant of ``ingestion_phase_pending`` that reports
which of them are stale for a source, using ONLY content comparison — no
timestamps are ever used for correctness.

**Clock-free two-level staleness.** A phase's diary entry (see
``registry.completed_phases``) records ``{"marker", "settings_version",
"deps"}``. A phase is stale when:

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

**Phase registration.** External plugins declare their own phases through the
``ingestion_phase_specs`` accumulator hook (see ``hooks.py``);
:func:`merged_phases` merges the returned ``PhaseSpec`` objects into the
built-in ``PHASES`` at probe time — built-ins win on duplicate ids — and the
``ingestion_phase_pending`` registrant computes staleness for ALL merged
phases, so a provider only declares a spec + a marker and never re-implements
staleness.

Import-safe: the only top-level statements are the dataclass, the ``PHASES``
data structure and the ``@hook`` decorator — no Redis, no network, no
filesystem access at import time.
"""

from dataclasses import dataclass
from typing import Any

from cat import hook
from cat.db.cruds import settings as crud_settings

from .registry import get_completed_phases


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


async def merged_phases(ccat, cat) -> dict[str, PhaseSpec]:
    """Merge the registered phase specs into the built-in ``PHASES``.

    Starts from a copy of the built-in ``PHASES`` and appends every
    ``PhaseSpec`` returned by the ``ingestion_phase_specs`` accumulator hook
    whose ``id`` is not already present — built-ins win, a registered spec
    with a duplicate id is IGNORED. Malformed hook output (``None``,
    non-``PhaseSpec`` entries) is skipped, never raised on.

    Args:
        ccat: the cat whose ``plugin_manager`` executes the hook (may be
            ``None`` in unit tests -> built-ins only).
        cat: the caller threaded into ``execute_hook(..., caller=cat)``.

    Returns:
        The merged ``dict[str, PhaseSpec]`` (a fresh dict; ``PHASES`` is
        never mutated).
    """
    merged = dict(PHASES)
    plugin_manager = (
        getattr(ccat, "plugin_manager", None) if ccat is not None else None
    )
    if plugin_manager is None:
        return merged
    specs = await plugin_manager.execute_hook("ingestion_phase_specs", [], caller=cat)
    if not isinstance(specs, list):
        return merged
    for spec in specs:
        if isinstance(spec, PhaseSpec) and spec.id not in merged:
            merged[spec.id] = spec
    return merged


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


@hook(priority=0)
async def ingestion_phase_pending(
    pending, source, completed_phases, cat
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
    diary = _diary_from_completed_phases(completed_phases)

    for spec in (await merged_phases(cat, cat)).values():
        current_settings_updated_at = None
        current_marker = None
        if cat is not None:
            if spec.settings_category is not None:
                settings_doc = await crud_settings.get_settings_by_category(
                    getattr(cat, "agent_key", None), spec.settings_category
                )
                if isinstance(settings_doc, dict):
                    current_settings_updated_at = settings_doc.get("updated_at")
            plugin_manager = getattr(cat, "plugin_manager", None)
            if plugin_manager is not None:
                current_marker = await plugin_manager.execute_hook(
                    "ingestion_phase_settings_marker", spec.id, caller=cat,
                )
        if is_phase_stale(spec, diary, current_settings_updated_at, current_marker):
            pending.append({"phase": spec.id})
    return pending