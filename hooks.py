"""Phase-machine extension hooks for the efficient-ingestion plugin.

These five hooks define the contract between the EffING phase machine and any
external registrant (other plugins, e.g. MyGRAPH) that wants to participate in
ingestion phases. They are declared here, in the plugin itself, with
``@hook(priority=0)`` so that external plugins can override them with higher
priorities — no MyCAT core change is needed.

All defaults are conservative no-ops: a phase that nobody implements is
reported as pending (stale), fails hard when run, and its marker is unknown.

Importing this module has zero side effects: the only top-level statements are
the five ``@hook`` decorators (which merely wrap the functions in ``CatHook``
instances). No Redis, no network, no filesystem access.
"""

from cat import hook

# The ``completed_phases`` argument threaded through the hooks is a
# ``list[dict]`` where every entry carries at least a ``"phase"`` key
# (MyGRAPH compatibility: its existing hook reads
# ``{p.get("phase") for p in completed_phases if isinstance(p, dict)}``),
# plus optional ``marker``/``deps`` keys. No timestamps are used for
# correctness anywhere in this protocol.


@hook(priority=0)
def ingestion_phase_pending(pending, source, completed_phases, cat):
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
def ingestion_phase_run(phase, source, completed_phases, cat):
    """Run one phase for ``source``; tri-state contract.

    A registrant that implements ``phase`` must either:
      - return ``{"status": "done"}`` on success;
      - return ``{"status": "not_ready", "retry_after": N}`` when the phase is
        retriable but cannot run yet (the machine retries after ``N`` seconds);
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
def before_ingestion_status_completed(source, cat):
    """Final gate before ``source`` is marked COMPLETED.

    Called right before the machine writes the COMPLETED status for a source.
    A registrant raises to force the source to ERROR instead; the default is a
    no-op (the source is allowed to complete).

    Args:
        source: the source about to be marked completed (e.g. ``"doc.pdf"``).
        cat: the CheshireCat instance (may be ``None`` in unit tests).

    Returns:
        ``None`` (default).
    """
    return None


@hook(priority=0)
def ingestion_phase_settings_marker(phase, cat):
    """Plugin-defined material marker for ``phase``.

    The machine compares this marker against the one recorded in the
    ``completed_phases`` diary entry to decide whether the phase's settings
    changed materially (clock-free invalidation). The default returns ``None``,
    meaning "unknown" — which the machine treats conservatively as stale.

    Args:
        phase: the phase id whose marker is requested (e.g. ``"embedding"``).
        cat: the CheshireCat instance (may be ``None`` in unit tests).

    Returns:
        ``None`` (default, "unknown" -> conservative stale), or any
        hashable/equatable value the plugin defines.
    """
    return None


@hook(priority=0)
def ingestion_phase_specs(specs, cat) -> list:
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