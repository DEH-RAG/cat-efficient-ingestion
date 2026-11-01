"""Recursive, configuration-only phase tags.

A phase's tag is the SHA-256 of its OWN material configuration::

    own_tag(kind, relevant_settings) =
        sha256(kind + ":" + canonical_json(relevant_settings))

``relevant_settings`` is the whole settings value, unless the phase declares a
``settings_check``: a key collection then narrows the settings to those keys,
while a (possibly async) callable returns the value that participates instead.

The recursive tag of a phase folds in the tags of every phase it depends on::

    sha256(own_tag + "|" + "|".join(f"{dep}={tag_dep}" for dep in sorted(deps)))

Because dependencies are resolved first, a change deep in the chain changes the
tag of every dependent, transitively. :func:`resolve_tags` performs the
dependency-ordered resolution and raises :class:`PhaseTagCycleError` naming the
cycle.

Pure and import-safe: no network, no Redis, no filesystem access, and no LLM
call — the tag only ever reads the configuration handed to it.
"""

import hashlib
import inspect
import json
from collections.abc import Iterable, Mapping
from typing import Any


class PhaseTagError(ValueError):
    """The phase tag graph cannot be resolved (unknown dependency or cycle)."""


class PhaseTagCycleError(PhaseTagError):
    """The phase dependency graph contains a cycle."""


def canonical_json(value: Any) -> str:
    """Render ``value`` as canonical JSON (sorted keys, no incidental spacing)."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def _digest(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def own_tag(kind: str, relevant_settings: Any) -> str:
    """Detached tag of one phase from its kind and relevant settings only."""
    return _digest(f"{kind}:{canonical_json(relevant_settings)}")


def compose_tag(
    own: str, depends: Iterable[str], dep_tags: Mapping[str, str]
) -> str:
    """Fold a phase's own tag with its dependency tags (sorted, deterministic)."""
    dep_part = "|".join(f"{dep}={dep_tags[dep]}" for dep in sorted(depends))
    return _digest(f"{own}|{dep_part}")


async def relevant_settings(settings: Any, settings_check: Any = None) -> Any:
    """Return the settings slice that participates in the tag.

    - ``settings_check`` is ``None``: the whole ``settings`` value.
    - ``settings_check`` is callable: its return value (awaited when it is a
      coroutine) — the async ``settings_check`` contract.
    - ``settings_check`` is a key (or an iterable of keys): only those keys are
      kept from a mapping ``settings``.
    """
    if settings_check is None:
        return settings
    if callable(settings_check):
        value = settings_check(settings)
        if inspect.isawaitable(value):
            value = await value
        return value
    keys = [settings_check] if isinstance(settings_check, str) else list(settings_check)
    if not isinstance(settings, Mapping):
        return {}
    return {key: settings.get(key) for key in keys if key in settings}


async def resolve_tags(settings: Mapping[str, Any], phases) -> dict[str, str]:
    """Resolve every phase's recursive tag in dependency order.

    Args:
        settings: ``{phase_id: settings}`` — the resolved settings value handed
            to each phase's :meth:`tag`. A phase absent from the mapping gets
            an empty settings value (``{}``).
        phases: iterable of phase-like objects exposing ``id``, ``depends_on``
            and an async ``tag(settings) -> str``.

    Returns:
        ``{phase_id: recursive_tag}`` for every phase in ``phases``.

    Raises:
        PhaseTagCycleError: when the dependency graph cycles, naming the cycle
            (e.g. ``a -> b -> a``).
        PhaseTagError: when a phase depends on an id absent from ``phases``.
    """
    phase_by_id: dict[str, Any] = {}
    for phase in phases:
        phase_by_id.setdefault(phase.id, phase)

    resolved: dict[str, str] = {}
    visiting: list[str] = []

    async def _resolve(phase_id: str) -> str:
        if phase_id in resolved:
            return resolved[phase_id]
        if phase_id in visiting:
            start = visiting.index(phase_id)
            cycle = visiting[start:] + [phase_id]
            raise PhaseTagCycleError(
                "phase dependency cycle detected: " + " -> ".join(cycle)
            )
        phase = phase_by_id.get(phase_id)
        if phase is None:
            raise PhaseTagError(
                f"phase '{phase_id}' is a dependency that is not declared"
            )
        visiting.append(phase_id)
        try:
            own = await phase.tag(settings.get(phase_id, {}))
            dep_tags = {dep: await _resolve(dep) for dep in phase.depends_on}
            tag = compose_tag(own, phase.depends_on, dep_tags)
            resolved[phase_id] = tag
            return tag
        finally:
            visiting.pop()

    for phase in phase_by_id.values():
        await _resolve(phase.id)
    return resolved
