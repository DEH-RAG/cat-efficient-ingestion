"""Standalone, import-safe tests for the legacy ``HookPhase`` hook adapter.

The adapter has two dispatch paths that must both reach the real hook with
exactly the documented arguments:

  * the MANAGER branch (``HookPhase(..., manager=...)``), which is what
    ``get_phases()`` always builds in production;
  * the ``execute_phase_on_owner`` fallback (no manager), used for a unit-test
    fake / bootstrap.

The CAT plugin loader imports EVERY ``.py`` under a plugin folder at
activation, so the whole file is inert on import: every import, class and test
lives inside ``if __name__ == "__main__":``. Run it as a plain script from the
host repository (the plugin is symlinked under ``cat/core_plugins``)::

    cd /root/git/grinning-cat-core
    .venv/bin/python /root/git/plugins/cat-efficient-ingestion/test_hookphase_adapter.py
"""

if __name__ == "__main__":
    import asyncio
    import sys
    import types

    from cat.core_plugins.cat_efficient_ingestion import hooks, phases

    class RecordingLog:
        """Capture the dispatcher's logs so a swallowed TypeError is visible."""

        def __init__(self):
            self.errors = []
            self.warnings = []

        def error(self, msg):
            self.errors.append(str(msg))

        def warning(self, msg):
            self.warnings.append(str(msg))

        def info(self, msg):
            pass

        def debug(self, msg):
            pass

    class FakeHook:
        def __init__(self, function, plugin_id="fake"):
            self.function = function
            self.plugin_id = plugin_id

    class FakeManager:
        context_execute_hook = "cat"

        def __init__(self, hooks=None):
            self.hooks = hooks or {}

    def _spec(phase_id):
        return phases.PhaseSpec(phase_id, None, ())

    def _assert_real_arguments(calls, expected, log):
        assert calls == [expected], calls
        assert not any("multiple values" in err for err in log.errors), log.errors

    async def test_manager_purge_runs_the_hook_and_purges():
        artifacts = {"artifact:A"}
        calls = []
        log = RecordingLog()
        original_log = hooks.log
        hooks.log = log

        async def purge(phase, source, cat=None, lizard=None):
            calls.append((phase, source, cat))
            artifacts.discard("artifact:A")
            return {"status": "done"}

        manager = FakeManager(
            {"ingestion_phase_purge": [FakeHook(purge)]}
        )
        cat = types.SimpleNamespace(plugin_manager=manager)
        hook_phase = phases.HookPhase(_spec("legacy_purge"), manager=manager)

        try:
            await hook_phase.delete("doc.txt", cat)
        finally:
            hooks.log = original_log

        _assert_real_arguments(calls, ("legacy_purge", "doc.txt", cat), log)
        assert artifacts == set(), artifacts
        print("PASS manager branch: purge hook called with (phase, source, cat) "
              "and the artifact was actually removed")

    async def test_manager_run_runs_the_hook_and_maps_the_result():
        calls = []
        log = RecordingLog()
        original_log = hooks.log
        hooks.log = log

        def run(phase, source, completed_phases, cat=None, lizard=None):
            calls.append((phase, source, completed_phases, cat))
            return {"status": "done", "processed": 3}

        manager = FakeManager({"ingestion_phase_run": [FakeHook(run)]})
        cat = types.SimpleNamespace(plugin_manager=manager)
        hook_phase = phases.HookPhase(_spec("legacy_run"), manager=manager)

        try:
            result = await hook_phase.process("doc.txt", cat)
        finally:
            hooks.log = original_log

        _assert_real_arguments(calls, ("legacy_run", "doc.txt", [], cat), log)
        assert result.status == "done", result
        assert result.processed == 3, result
        print("PASS manager branch: run hook called with "
              "(phase, source, completed_phases, cat) and its result is used")

    async def test_fallback_purge_runs_the_hook_and_purges():
        artifacts = {"artifact:A"}
        calls = []
        log = RecordingLog()
        original_log = hooks.log
        hooks.log = log

        def purge(phase, source, cat=None, lizard=None):
            calls.append((phase, source, cat))
            artifacts.discard("artifact:A")
            return {"status": "done"}

        manager = FakeManager(
            {"ingestion_phase_purge": [FakeHook(purge)]}
        )
        cat = types.SimpleNamespace(plugin_manager=manager)
        hook_phase = phases.HookPhase(_spec("legacy_purge_fallback"))

        try:
            await hook_phase.delete("doc.txt", cat)
        finally:
            hooks.log = original_log

        _assert_real_arguments(
            calls, ("legacy_purge_fallback", "doc.txt", cat), log
        )
        assert artifacts == set(), artifacts
        print("PASS fallback branch: purge hook called with "
              "(phase, source, cat) and the artifact was actually removed")

    async def test_fallback_run_runs_the_hook_and_maps_the_result():
        calls = []
        log = RecordingLog()
        original_log = hooks.log
        hooks.log = log

        def run(phase, source, completed_phases, cat=None, lizard=None):
            calls.append((phase, source, completed_phases, cat))
            return {"status": "done"}

        manager = FakeManager({"ingestion_phase_run": [FakeHook(run)]})
        cat = types.SimpleNamespace(plugin_manager=manager)
        hook_phase = phases.HookPhase(_spec("legacy_run_fallback"))

        try:
            result = await hook_phase.process("doc.txt", cat)
        finally:
            hooks.log = original_log

        _assert_real_arguments(
            calls, ("legacy_run_fallback", "doc.txt", [], cat), log
        )
        assert result.status == "done", result
        print("PASS fallback branch: run hook called with "
              "(phase, source, completed_phases, cat) and its result is used")

    async def _main():
        await test_manager_purge_runs_the_hook_and_purges()
        await test_manager_run_runs_the_hook_and_maps_the_result()
        await test_fallback_purge_runs_the_hook_and_purges()
        await test_fallback_run_runs_the_hook_and_maps_the_result()

    try:
        asyncio.run(_main())
    except AssertionError as exc:
        print(f"FAIL {exc}")
        sys.exit(1)
    print("OK HookPhase legacy adapter (four tests)")
