"""Standalone, import-safe tests for recursive phase tags.

Covers the recursive-tag contract of the phase machine:
  (a) the same configuration yields identical tags across two computations;
  (b) changing a DEEP dependency's config changes the dependent's tag;
  (c) changing an unrelated phase's config does not change the target tag;
  (d) a dependency cycle raises an error naming the cycle;
  (e) a legacy entry without a tag is stale exactly once, then stable after
      the rewrite;
  (f) an async marker hook is awaited and its value participates in the tag.

The CAT plugin loader imports EVERY ``.py`` under a plugin folder at
activation, so the whole file is inert on import: every import, class and test
lives inside ``if __name__ == "__main__":``. Run it as a plain script from the
host repository (the plugin is symlinked under ``cat/core_plugins``):

    cd /root/git/grinning-cat-core
    .venv/bin/python /root/git/plugins/cat-efficient-ingestion/test_phase_tags.py
"""

if __name__ == "__main__":
    import asyncio
    import hashlib
    import sys
    import types

    from cat.core_plugins.cat_efficient_ingestion import phases

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

    def _phase(phase_id, depends_on=(), marker=None, config=None, manager=None):
        spec = phases.PhaseSpec(phase_id, None, tuple(depends_on))
        if manager is not None:
            return phases.HookPhase(spec, cat=_cat(manager), manager=manager)
        return phases.HookPhase(
            spec,
            settings_marker=lambda _pid: config[phase_id] if config is not None else marker,
        )

    async def test_a_same_config_same_tags():
        config = {"a": "A1", "b": "B1", "c": "C1"}
        phases_list = [
            _phase("a", config=config),
            _phase("b", ("a",), config=config),
            _phase("c", ("b",), config=config),
        ]
        first = await phases.compute_tags(_cat(), phases_list)
        second = await phases.compute_tags(_cat(), phases_list)
        assert first == second, (first, second)
        assert set(first) == {"a", "b", "c"}, first
        print("PASS (a) identical config gives identical tags")

    async def test_b_deep_dependency_change_propagates():
        config = {"a": "A1", "b": "B1", "c": "C1"}
        phases_list = [
            _phase("a", config=config),
            _phase("b", ("a",), config=config),
            _phase("c", ("b",), config=config),
        ]
        before = await phases.compute_tags(_cat(), phases_list)
        config["a"] = "A2"
        after = await phases.compute_tags(_cat(), phases_list)
        assert after["a"] != before["a"], (before, after)
        assert after["b"] != before["b"], "direct dependent did not change"
        assert after["c"] != before["c"], "deep dependent did not change"
        print("PASS (b) a deep dependency change invalidates the whole chain")

    async def test_c_unrelated_phase_change_does_not_propagate():
        config = {"a": "A1", "b": "B1", "c": "C1", "x": "X1"}
        phases_list = [
            _phase("a", config=config),
            _phase("b", ("a",), config=config),
            _phase("c", ("b",), config=config),
            _phase("x", config=config),
        ]
        before = await phases.compute_tags(_cat(), phases_list)
        config["x"] = "X2"
        after = await phases.compute_tags(_cat(), phases_list)
        assert after["x"] != before["x"], (before, after)
        assert after["a"] == before["a"], (before, after)
        assert after["b"] == before["b"], (before, after)
        assert after["c"] == before["c"], (before, after)
        print("PASS (c) an unrelated phase change leaves the target tag alone")

    async def test_d_cycle_raises_naming_the_cycle():
        config = {"a": "A1", "b": "B1"}
        phases_list = [
            _phase("a", ("b",), config=config),
            _phase("b", ("a",), config=config),
        ]
        try:
            await phases.compute_tags(_cat(), phases_list)
        except phases.PhaseTagCycleError as exc:
            message = str(exc)
            assert "cycle" in message.lower(), message
            assert "a" in message and "b" in message, message
            assert "a -> b -> a" in message or "b -> a -> b" in message, message
            print("PASS (d) a dependency cycle raises and names the cycle")
            return
        raise AssertionError("a dependency cycle did not raise")

    async def test_e_legacy_entry_is_stale_once_then_stable():
        spec = phases.PhaseSpec("a", None, ())
        current_tag = phases._compose_tag(
            phases._declared_tag("a", None, ()), {}
        )
        legacy = {"state": "complete"}
        assert phases.is_phase_stale(spec, {"a": legacy}, current_tag) is True
        legacy["tag"] = current_tag
        assert phases.is_phase_stale(spec, {"a": legacy}, current_tag) is False
        print("PASS (e) a legacy entry is stale once, then stable after rewrite")

    async def test_f_async_marker_hook_is_awaited():
        awaited = []

        async def marker(phase, cat=None):
            awaited.append(phase)
            return "async-marker-v1"

        manager = FakeManager(
            {"ingestion_phase_settings_marker": [FakeHook(marker)]}
        )
        phase = _phase("a", manager=manager)
        tag = await phase.tag(_cat(manager), {})
        expected = hashlib.sha256(b"a:async-marker-v1|").hexdigest()
        assert awaited == ["a"], awaited
        assert tag == expected, (tag, expected)
        print("PASS (f) an async marker hook is awaited and folds into the tag")

    async def _main():
        await test_a_same_config_same_tags()
        await test_b_deep_dependency_change_propagates()
        await test_c_unrelated_phase_change_does_not_propagate()
        await test_d_cycle_raises_naming_the_cycle()
        await test_e_legacy_entry_is_stale_once_then_stable()
        await test_f_async_marker_hook_is_awaited()

    try:
        asyncio.run(_main())
    except AssertionError as exc:
        print(f"FAIL {exc}")
        sys.exit(1)
    print("OK recursive phase tags (six tests)")
