"""Import-safety tests for the third-party phase-extension contract.

A consumer plugin declares an ingestion phase by importing ``PhaseSpec`` from
the CANONICAL module path behind a guarded import (EffING optional) and by
keeping the classic CAT ingestion hooks so basic ingestion still works when
EffING is absent. These tests exercise the tiny fixture plugin both ways:

- with EffING importable: the phase hook is registered and ``merged_phases``
  accepts its spec (``isinstance`` true — one module identity);
- with EffING blocked via a process-local ``sys.modules`` guard: the module
  still imports and the classic hooks exist.

The guard patches ``sys.modules`` inside this test process ONLY and restores it
in ``finally``; the host never mutates ``sys.modules`` at plugin import time.

Standalone and import-safe (the CAT loader imports every ``.py`` under the
plugin folder): the body lives under ``if __name__ == "__main__":``. Run it
from the host repository::

    cd /root/git/grinning-cat-core
    .venv/bin/python /root/git/plugins/cat-efficient-ingestion/test_phase_extension_contract.py
"""
if __name__ == "__main__":
    import asyncio
    import importlib.util
    import json
    import os
    import sys
    import traceback
    from pathlib import Path
    from types import SimpleNamespace
    from typing import Any

    os.environ.setdefault("CAT_REDIS_HOST", "localhost")
    os.environ.setdefault("CAT_REDIS_PORT", "6379")
    os.environ.setdefault("CAT_JWT_SECRET", "test_only_jwt_secret_0123456789_abcdefghij")
    os.environ["CAT_REDIS_DB"] = "1"

    from cat.core_plugins.cat_efficient_ingestion import phases

    FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "phase_consumer"
    MODULE_PATH = FIXTURE_DIR / "consumer.py"
    CANONICAL = "cat.core_plugins.cat_efficient_ingestion.phases"
    PHASE_PLUGIN_ID = "cat_efficient_ingestion"
    _MISSING = object()

    def _load(name: str):
        """Execute the fixture module under ``name`` and return it (fresh each call)."""
        spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
        return module

    class _ConsumerPluginManager:
        """Dispatch hooks to the fixture module only (no real plugin load)."""

        def __init__(self, module):
            self.module = module

        async def execute_hook(self, name, *args, caller=None):
            decorated = getattr(self.module, name, None)
            if decorated is None:
                raise RuntimeError(f"Hook {name} not present in any plugin")
            return decorated.function(*args, cat=caller)

    async def test_phase_hook_registered_and_spec_accepted():
        """With EffING importable the phase hook exists and its spec is merged."""
        module = _load("_effing_phase_consumer_present")
        try:
            assert module.PhaseSpec is phases.PhaseSpec
            assert sys.modules[CANONICAL] is phases
            assert hasattr(module, "ingestion_phase_specs")

            cat = SimpleNamespace(plugin_manager=_ConsumerPluginManager(module))

            merged = await phases.merged_phases(cat, cat)

            spec = merged[module.PHASE_ID]
            assert isinstance(spec, phases.PhaseSpec)
            assert spec == module.PhaseSpec(module.PHASE_ID, None, ("embedding",))
        finally:
            sys.modules.pop("_effing_phase_consumer_present", None)

    async def test_module_imports_and_classic_hooks_survive_without_effing():
        """With EffING blocked the module still imports and serves classic hooks."""
        saved: Any = sys.modules.get(CANONICAL, _MISSING)
        sys.modules[CANONICAL] = None  # type: ignore[assignment]
        try:
            module = _load("_effing_phase_consumer_absent")
            try:
                assert module.PhaseSpec is None
                assert not hasattr(module, "ingestion_phase_specs")

                classic = module.rabbithole_ingestion_processing
                assert classic.function("doc.pdf", object()) is None
            finally:
                sys.modules.pop("_effing_phase_consumer_absent", None)
        finally:
            if saved is _MISSING:
                sys.modules.pop(CANONICAL, None)
            else:
                sys.modules[CANONICAL] = saved

    def test_dependency_uses_the_plugin_folder_id():
        """A consumer declares EffING by its folder id, never the hyphenated name."""
        manifest = json.loads((FIXTURE_DIR / "plugin.json").read_text())
        assert manifest["dependencies"] == [PHASE_PLUGIN_ID]
        assert "-" not in PHASE_PLUGIN_ID

        from cat import utils

        core_plugins = Path(utils.__file__).resolve().parent / "core_plugins"
        assert (core_plugins / PHASE_PLUGIN_ID).is_dir()

        effing_manifest = json.loads((core_plugins / PHASE_PLUGIN_ID / "plugin.json").read_text())
        for dependency in effing_manifest["dependencies"]:
            assert "-" not in dependency
            assert (core_plugins / dependency).is_dir()

    _tests = [
        test_phase_hook_registered_and_spec_accepted,
        test_module_imports_and_classic_hooks_survive_without_effing,
        test_dependency_uses_the_plugin_folder_id,
    ]

    _passed = 0
    _failures = []
    for _test in _tests:
        try:
            if asyncio.iscoroutinefunction(_test):
                asyncio.run(_test())
            else:
                _test()
        except BaseException as exc:  # noqa: BLE001 - report every failure
            _failures.append((_test.__name__, exc, traceback.format_exc()))
            print(f"FAIL {_test.__name__}: {type(exc).__name__}: {exc}")
        else:
            _passed += 1
            print(f"PASS {_test.__name__}")

    print(f"{_passed}/{len(_tests)} passed")
    if _failures:
        for _name, _exc, _tb in _failures:
            print(f"\n--- {_name} ---\n{_tb}")
        sys.exit(1)
