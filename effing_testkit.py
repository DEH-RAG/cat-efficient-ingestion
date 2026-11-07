"""Shared, import-safe doubles for the Efficient Ingestion phase contract.

Why this module exists
----------------------
CATALOG and MyGRAPH consume EffING's phase contract (``PhaseSpec``,
``PhaseResult``, ``IngestionPhase``) while staying separate plugins that must
keep importing on a host where EffING is not installed. Each of them used to
hand-roll its own ``sys.modules`` stub of
``cat.core_plugins.cat_efficient_ingestion``; this module is the single
published copy of those doubles.

Import safety
-------------
The CAT host imports **every** ``.py`` file under a plugin folder (recursively)
at activation. At module level this file only defines constants, classes and
functions: it never imports ``cat``, writes the environment, touches Redis or
reads the filesystem. The host is only touched inside
:func:`install_stub_package`, :func:`uninstall_stub_package` and
:class:`IngestionPhase`, which tests call from under
``if __name__ == "__main__":``.

Typical use from another plugin's standalone script::

    import effing_testkit

    effing_testkit.install_stub_package()
    try:
        from cat.core_plugins.cat_efficient_ingestion.phases import PhaseSpec
        ...
    finally:
        effing_testkit.uninstall_stub_package()

``install_stub_package()`` prefers a REAL EffING installation when the host
provides one (the canonical package importable) and only synthesizes the
package for a bare checkout. It is idempotent: a second call is a no-op and
never re-creates (or overwrites) a module.
"""
from __future__ import annotations

import abc as _abc
import hashlib as _hashlib
import importlib as _importlib
import importlib.util as _importlib_util
import json as _json
import os as _os
import sys as _sys
import types as _types
from dataclasses import dataclass, field

#: Canonical host module path EffING is mounted at.
PACKAGE_NAME = "cat.core_plugins.cat_efficient_ingestion"
PHASES_MODULE = PACKAGE_NAME + ".phases"
SPLIT_MODULE = PACKAGE_NAME + ".split"

_MISSING = object()


# ---------------------------------------------------------------------------
# Contract doubles
# ---------------------------------------------------------------------------


class PhaseSpec:
    """Recording stand-in for EffING's ``PhaseSpec`` dataclass.

    Only the declarative input is captured (``id`` / ``settings_category`` /
    ``depends_on`` / ``settings_check``); the phase machine behaviour it drives
    is not reproduced, because this is a contract double, not an engine.
    """

    __slots__: tuple[str, ...] = (
        "id",
        "settings_category",
        "depends_on",
        "settings_check",
    )

    def __init__(
        self,
        id: str,
        settings_category: str | None = None,
        depends_on: tuple[str, ...] = (),
        settings_check: object = None,
    ) -> None:
        self.id = id
        self.settings_category = settings_category
        self.depends_on = tuple(depends_on)
        self.settings_check = settings_check

    def __repr__(self) -> str:
        return (
            f"PhaseSpec(id={self.id!r}, "
            f"settings_category={self.settings_category!r}, "
            f"depends_on={self.depends_on!r})"
        )


class PhaseResult:
    """Stand-in for EffING's ``PhaseResult`` dataclass (telemetry included)."""

    __slots__: tuple[str, ...] = (
        "status",
        "reason",
        "processed",
        "expected",
        "warnings",
    )

    def __init__(
        self,
        status: str,
        reason: str | None = None,
        processed: int | None = None,
        expected: int | None = None,
        warnings: list[dict[str, object]] | None = None,
    ) -> None:
        self.status = status
        self.reason = reason
        self.processed = processed
        self.expected = expected
        self.warnings = warnings

    def __repr__(self) -> str:
        return (
            f"PhaseResult(status={self.status!r}, reason={self.reason!r}, "
            f"processed={self.processed!r}, expected={self.expected!r})"
        )


class IngestionPhase(_abc.ABC):
    """Stand-in for EffING's ``IngestionPhase`` ABC.

    Mirrors the surface a phase subclass relies on: the declarative attributes,
    :attr:`name` / :meth:`display_label`, the three abstract operations and the
    concrete :meth:`tag`. ``tag`` is the same configuration-only SHA-256 the
    real base class computes, so a consumer can pin tag stability without the
    real plugin installed.
    """

    id: str = ""
    label: str | None = None
    depends_on: tuple[str, ...] = ()
    settings_category: str | None = None
    settings_check: object = None

    @property
    def name(self) -> str:
        return self.id

    def display_label(self) -> str:
        return self.label or self.id

    @_abc.abstractmethod
    async def applies_to(self, cat: object, deps_state: object) -> tuple[bool, str | None]:
        raise NotImplementedError

    @_abc.abstractmethod
    async def process(self, source: str, cat: object) -> PhaseResult:
        raise NotImplementedError

    @_abc.abstractmethod
    async def delete(self, source: str, cat: object) -> None:
        raise NotImplementedError

    async def tag(self, settings: object = None) -> str:
        relevant = {} if settings is None else settings
        payload = _json.dumps(
            relevant, sort_keys=True, separators=(",", ":"), default=str
        )
        return _hashlib.sha256(f"{self.id}:{payload}".encode()).hexdigest()


def make_diary_entry(
    phase: str,
    *,
    gen: object = _MISSING,
    marker: object = _MISSING,
    tag: str = "",
    state: str = "complete",
    reason: str | None = None,
    started_at: float | None = None,
    owner: str | None = None,
    attempt: int = 1,
    warnings: list[dict[str, object]] | None = None,
    **extra: object,
) -> dict[str, object]:
    """Build one completed-phases diary entry.

    The hook contract threads ``completed_phases`` as a ``list[dict]`` where
    every entry carries at least a ``"phase"`` key; EffING normalizes that list
    to its diary dict. This factory produces that entry shape (plus the v2
    ``state``/``tag``/``attempt`` fields) without importing the real registry.

    ``gen`` (MyGRAPH compatibility) and ``marker`` are only emitted when passed.
    """
    entry: dict[str, object] = {
        "phase": phase,
        "state": state,
        "reason": reason,
        "started_at": started_at,
        "owner": owner,
        "tag": tag,
        "attempt": attempt,
    }
    if gen is not _MISSING:
        entry["gen"] = gen
    if marker is not _MISSING:
        entry["marker"] = marker
    if warnings:
        entry["warnings"] = list(warnings)
    entry.update(extra)
    return entry


def completed_phases(*entries: object) -> list[object]:
    """Wrap diary entries into the hook's ``list[dict]`` ``completed_phases``."""
    return list(entries)


# ---------------------------------------------------------------------------
# Synthetic package install / uninstall
# ---------------------------------------------------------------------------


@dataclass
class _InstallState:
    """Tracks what the testkit registered so ``uninstall`` can undo it exactly."""

    active: bool = False
    mode: str | None = None
    saved: dict[str, object] = field(default_factory=dict)
    created: tuple[str, ...] = ()
    attrs: tuple[tuple[str, str, object], ...] = ()


_STATE = _InstallState()


def _real_package_available() -> bool:
    """Whether the host already provides the real EffING package."""
    try:
        return _importlib_util.find_spec(PACKAGE_NAME) is not None
    except (ImportError, AttributeError, ValueError, ModuleNotFoundError):
        return False


def _new_package(name: str) -> _types.ModuleType:
    module = _types.ModuleType(name)
    module.__path__ = []
    return module


def _build_phases_module() -> _types.ModuleType:
    module = _new_package(PHASES_MODULE)
    setattr(module, "PhaseSpec", PhaseSpec)
    setattr(module, "PhaseResult", PhaseResult)
    setattr(module, "IngestionPhase", IngestionPhase)
    return module


def _load_real_split() -> _types.ModuleType | None:
    """Load EffING's real ``split.py`` when it is importable in this process."""
    path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "split.py")
    if not _os.path.isfile(path):
        return None
    try:
        spec = _importlib_util.spec_from_file_location(SPLIT_MODULE, path)
        if spec is None or spec.loader is None:
            return None
        module = _importlib_util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception:  # noqa: BLE001 - optional dependency (langchain, cat.log)
        return None


def _build_split_module() -> _types.ModuleType:
    module = _new_package(SPLIT_MODULE)
    real = _load_real_split()
    if real is not None:
        setattr(module, "doc_tokens", getattr(real, "doc_tokens", None))
        setattr(module, "split_to_budget", getattr(real, "split_to_budget", None))
        setattr(module, "split_oversized", getattr(real, "split_oversized", None))
        return module

    def split_oversized(docs: list[object], embedder: object) -> list[object]:
        return list(docs)

    def split_to_budget(doc: object, embedder: object) -> list[object]:
        return [doc]

    setattr(module, "doc_tokens", None)
    setattr(module, "split_to_budget", split_to_budget)
    setattr(module, "split_oversized", split_oversized)
    return module


def install_stub_package() -> str:
    """Make ``cat.core_plugins.cat_efficient_ingestion`` importable.

    Prefers the REAL package when the host provides it (nothing is registered:
    ``uninstall`` then does nothing). For a bare checkout it registers the
    synthetic package plus the ``.phases`` and ``.split`` submodules.

    Idempotent: calling it twice leaves the exact same module objects in place.

    Returns:
        ``"real"`` when the real installation was used, ``"stub"`` when the
        synthetic package was registered, or the previous result on a repeat
        call.
    """
    if _STATE.active:
        return _STATE.mode or "stub"

    if _real_package_available():
        _importlib.import_module(PACKAGE_NAME)
        _STATE.active = True
        _STATE.mode = "real"
        _STATE.saved = {}
        _STATE.created = ()
        _STATE.attrs = ()
        return "real"

    phases = _build_phases_module()
    split = _build_split_module()
    package = _new_package(PACKAGE_NAME)
    setattr(package, "phases", phases)
    setattr(package, "split", split)

    plan = (
        ("cat", _new_package("cat")),
        ("cat.core_plugins", _new_package("cat.core_plugins")),
        (PACKAGE_NAME, package),
        (PHASES_MODULE, phases),
        (SPLIT_MODULE, split),
    )

    saved: dict[str, object] = {}
    created: list[str] = []
    attrs: list[tuple[str, str, object]] = []
    for name, module in plan:
        parent_name, _, attr = name.rpartition(".")
        parent = _sys.modules.get(parent_name)

        saved[name] = _sys.modules.get(name, _MISSING)
        installed = _sys.modules.get(name)
        if installed is None:
            _sys.modules[name] = module
            created.append(name)
            installed = module

        if parent is not None:
            attrs.append((parent_name, attr, getattr(parent, attr, _MISSING)))
            setattr(parent, attr, installed)

    # ``cat.core_plugins`` must exist before the package below it is importable;
    # the loop above registers in order, so re-attach the package now.
    core_plugins = _sys.modules.get("cat.core_plugins")
    if core_plugins is not None:
        setattr(core_plugins, "cat_efficient_ingestion", _sys.modules[PACKAGE_NAME])
    cat_mod = _sys.modules.get("cat")
    if cat_mod is not None:
        setattr(cat_mod, "core_plugins", _sys.modules["cat.core_plugins"])

    _STATE.active = True
    _STATE.mode = "stub"
    _STATE.saved = saved
    _STATE.created = tuple(created)
    _STATE.attrs = tuple(attrs)
    return "stub"


def uninstall_stub_package() -> None:
    """Undo :func:`install_stub_package`, restoring the previous module state.

    Only the modules and parent attributes created by ``install`` are removed or
    restored; a real installation (mode ``"real"``) is left untouched.
    """
    if not _STATE.active:
        return

    for name in reversed(_STATE.created):
        _sys.modules.pop(name, None)

    for name, module in _STATE.saved.items():
        if module is _MISSING:
            _sys.modules.pop(name, None)
        elif isinstance(module, _types.ModuleType):
            _sys.modules[name] = module

    for parent_name, attr, previous in reversed(_STATE.attrs):
        parent = _sys.modules.get(parent_name)
        if parent is None:
            continue
        if previous is _MISSING:
            try:
                delattr(parent, attr)
            except AttributeError:
                pass
        else:
            setattr(parent, attr, previous)

    _STATE.active = False
    _STATE.mode = None
    _STATE.saved = {}
    _STATE.created = ()
    _STATE.attrs = ()
