"""Standalone, import-safe tests for the phase ABC, registry and hook adapter.

Covers (todo 1 of the phase-contract v2 plan):
  (a) a subclass without ``delete`` cannot be instantiated (TypeError);
  (b) the registry returns the built-in phases + a fake plugin phase
      contributed through the ``ingestion_phase_objects`` hook;
  (c) PARITY: the same fake phase expressed as a class and as a legacy hook
      produces identical ``process`` results and identical ``delete`` effects;
  (d) ``HookPhase.delete`` raises when the purge hook is missing/no-op;
  (e) ``tag()`` is deterministic for the same config and changes when the
      config changes.

The CAT plugin loader imports EVERY ``.py`` under a plugin folder at
activation, so the whole file is inert on import: every import, class and test
lives inside ``if __name__ == "__main__":``. Run it as a plain script from the
host repository (the plugin is symlinked under ``cat/core_plugins``):

    cd /root/git/grinning-cat-core
    .venv/bin/python /root/git/plugins/cat-efficient-ingestion/test_phase_abc.py
"""

if __name__ == "__main__":
    import asyncio
    import sys
    import types

    from cat.core_plugins.cat_efficient_ingestion import phases

    class FakePhase(phases.IngestionPhase):
        """A fully-implemented fake phase used across the tests."""

        id = "fake_phase"
        depends_on = ()
        settings_category = None

        def __init__(self, artifacts=None, config=None):
            self.artifacts = artifacts if artifacts is not None else set()
            self.config = config if config is not None else {}

        async def applies_to(self, cat, deps_state):
            return True, None

        async def process(self, source, cat):
            return phases.PhaseResult(
                status="done", processed=2, expected=2
            )

        async def delete(self, source, cat):
            self.artifacts.discard("artifact:A")

        async def tag(self, cat, dep_tags):
            own = f"{self.id}:{self.config.get('marker')}"
            return phases._compose_tag(own, dep_tags)

    class NoDeletePhase(phases.IngestionPhase):
        id = "no_delete"

        async def applies_to(self, cat, deps_state):
            return True, None

        async def process(self, source, cat):
            return phases.PhaseResult(status="done")

        async def tag(self, cat, dep_tags):
            return phases._compose_tag(self.id, dep_tags)

    class FakeHook:
        def __init__(self, function, plugin_id="fake"):
            self.function = function
            self.plugin_id = plugin_id

    class FakeManager:
        context_execute_hook = "cat"

        def __init__(self, hooks=None):
            self.hooks = hooks or {}

    def _cat(manager=None):
        if manager is None:
            return types.SimpleNamespace()
        return types.SimpleNamespace(plugin_manager=manager)

    async def test_a_abstract_delete():
        try:
            NoDeletePhase()
        except TypeError as exc:
            assert "delete" in str(exc).lower(), str(exc)
            print("PASS (a) a subclass without delete cannot be instantiated")
            return
        raise AssertionError("NoDeletePhase() unexpectedly instantiated")

    async def test_b_registry_builtins_plus_plugin_phase():
        contributed = FakePhase()
        manager = FakeManager(
            {
                "ingestion_phase_objects": [
                    FakeHook(lambda cat=None: [contributed])
                ]
            }
        )
        result = await phases.get_phases(_cat(manager))
        ids = [phase.id for phase in result]
        assert ids == ["parsing_chunking", "embedding", "fake_phase"], ids
        assert result[2] is contributed, result[2]
        assert all(isinstance(phase, phases.IngestionPhase) for phase in result)
        print("PASS (b) registry returns built-ins + the plugin phase instance")

    async def test_c_parity_class_vs_hook():
        class_artifacts = {"artifact:A"}
        hook_artifacts = {"artifact:A"}

        class_phase = FakePhase(artifacts=class_artifacts, config={"marker": "m1"})
        spec = phases.PhaseSpec("fake_phase", None, ())
        hook_phase = phases.HookPhase(
            spec,
            run_hook=lambda phase, source, completed, cat=None: {
                "status": "done",
                "processed": 2,
                "expected": 2,
            },
            purge_hook=lambda phase, source, cat=None: (
                hook_artifacts.discard("artifact:A") or {"status": "done"}
            ),
            settings_marker=lambda phase: "m1",
        )

        class_result = await class_phase.process("doc.txt", None)
        hook_result = await hook_phase.process("doc.txt", None)
        assert class_result == hook_result, (class_result, hook_result)
        assert class_result.status == "done", class_result
        class_tag = await class_phase.tag(None, {})
        hook_tag = await hook_phase.tag(None, {})
        assert class_tag == hook_tag, (class_tag, hook_tag)

        await class_phase.delete("doc.txt", None)
        await hook_phase.delete("doc.txt", None)
        assert class_artifacts == hook_artifacts == set(), (
            class_artifacts,
            hook_artifacts,
        )
        print("PASS (c) class and hook phases agree on process() and delete()")

    async def test_d_missing_purge_raises():
        spec = phases.PhaseSpec("legacy_no_purge", None, ())
        hook_phase = phases.HookPhase(spec, run_hook=lambda *a, **k: {"status": "done"})
        try:
            await hook_phase.delete("doc.txt", _cat())
        except phases.PhaseDeleteError as exc:
            assert "legacy_no_purge" in str(exc), str(exc)
        else:
            raise AssertionError("delete() with a missing purge hook did not raise")

        noop_phase = phases.HookPhase(
            spec,
            run_hook=lambda *a, **k: {"status": "done"},
            purge_hook=lambda *a, **k: None,
        )
        try:
            await noop_phase.delete("doc.txt", _cat())
        except phases.PhaseDeleteError:
            pass
        else:
            raise AssertionError("delete() with a no-op purge hook did not raise")
        print("PASS (d) HookPhase.delete raises when the purge hook is missing/no-op")

    async def test_e_tag_deterministic_and_config_sensitive():
        spec = phases.PhaseSpec("legacy_tagged", "legacy_cat", ())
        config = {"marker": "settings-v1"}
        hook_phase = phases.HookPhase(
            spec, settings_marker=lambda phase: config["marker"]
        )
        first = await hook_phase.tag(_cat(), {})
        second = await hook_phase.tag(_cat(), {})
        assert first == second, (first, second)

        config["marker"] = "settings-v2"
        third = await hook_phase.tag(_cat(), {})
        assert third != first, (first, third)
        assert first != "legacy_tagged:settings-v1", first
        print("PASS (e) tag() is deterministic and changes with the config")

    async def _main():
        await test_a_abstract_delete()
        await test_b_registry_builtins_plus_plugin_phase()
        await test_c_parity_class_vs_hook()
        await test_d_missing_purge_raises()
        await test_e_tag_deterministic_and_config_sensitive()

    try:
        asyncio.run(_main())
    except AssertionError as exc:
        print(f"FAIL {exc}")
        sys.exit(1)
    print("OK phase ABC + registry + hook adapter (five tests)")
