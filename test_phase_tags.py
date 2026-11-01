"""Standalone, import-safe tests for recursive phase tags.

Covers the recursive-tag contract of the phase machine:
  (a) the same configuration yields identical tags across two resolutions;
  (b) changing a DEEP dependency's config changes the dependent's tag;
  (c) changing an unrelated phase's config does not change the target tag;
  (d) a dependency cycle raises, naming the cycle;
  (e) a legacy entry without a tag is stale exactly once, then stable after
      the rewrite;
  (f) an async ``settings_check`` is awaited and its value participates in
      the tag.

The CAT plugin loader imports EVERY ``.py`` under a plugin folder at
activation, so the whole file is inert on import: every import, class and test
lives inside ``if __name__ == "__main__":``. Run it as a plain script from the
host repository (the plugin is symlinked under ``cat/core_plugins``):

    cd /root/git/grinning-cat-core
    .venv/bin/python /root/git/plugins/cat-efficient-ingestion/test_phase_tags.py
"""

if __name__ == "__main__":
    import asyncio
    import sys

    from cat.core_plugins.cat_efficient_ingestion import phase_tag_resolver, phases

    class TagPhase(phases.IngestionPhase):
        """Minimal phase: inherits the concrete ``tag`` from the ABC."""

        def __init__(self, phase_id, depends_on=(), settings_check=None):
            self.id = phase_id
            self.depends_on = tuple(depends_on)
            self.settings_check = settings_check

        async def applies_to(self, cat, deps_state):
            return True, None

        async def process(self, source, cat):
            return phases.PhaseResult(status="done")

        async def delete(self, source, cat):
            return None

    async def test_a_same_config_same_tags():
        settings = {"a": {"model": "m1"}, "b": {"k": 1}, "c": {"k": 2}}
        phases_list = [
            TagPhase("a"),
            TagPhase("b", ("a",)),
            TagPhase("c", ("b",)),
        ]
        first = await phase_tag_resolver.resolve_tags(settings, phases_list)
        second = await phase_tag_resolver.resolve_tags(settings, phases_list)
        assert first == second, (first, second)
        assert set(first) == {"a", "b", "c"}, first
        print("PASS (a) identical config gives identical tags")

    async def test_b_deep_dependency_change_propagates():
        settings = {"a": {"model": "m1"}, "b": {"k": 1}, "c": {"k": 2}}
        phases_list = [
            TagPhase("a"),
            TagPhase("b", ("a",)),
            TagPhase("c", ("b",)),
        ]
        before = await phase_tag_resolver.resolve_tags(settings, phases_list)
        settings["a"] = {"model": "m2"}
        after = await phase_tag_resolver.resolve_tags(settings, phases_list)
        assert after["a"] != before["a"], (before, after)
        assert after["b"] != before["b"], "direct dependent did not change"
        assert after["c"] != before["c"], "deep dependent did not change"
        print("PASS (b) a deep dependency change invalidates the whole chain")

    async def test_c_unrelated_phase_change_does_not_propagate():
        settings = {
            "a": {"model": "m1"}, "b": {"k": 1}, "c": {"k": 2}, "x": {"k": 9},
        }
        phases_list = [
            TagPhase("a"),
            TagPhase("b", ("a",)),
            TagPhase("c", ("b",)),
            TagPhase("x"),
        ]
        before = await phase_tag_resolver.resolve_tags(settings, phases_list)
        settings["x"] = {"k": 10}
        after = await phase_tag_resolver.resolve_tags(settings, phases_list)
        assert after["x"] != before["x"], (before, after)
        assert after["a"] == before["a"], (before, after)
        assert after["b"] == before["b"], (before, after)
        assert after["c"] == before["c"], (before, after)
        print("PASS (c) an unrelated phase change leaves the target tag alone")

    async def test_d_cycle_raises_naming_the_cycle():
        settings = {"a": {}, "b": {}}
        phases_list = [TagPhase("a", ("b",)), TagPhase("b", ("a",))]
        try:
            await phase_tag_resolver.resolve_tags(settings, phases_list)
        except phase_tag_resolver.PhaseTagCycleError as exc:
            message = str(exc)
            assert "cycle" in message.lower(), message
            assert "a" in message and "b" in message, message
            assert "a -> b -> a" in message or "b -> a -> b" in message, message
            print("PASS (d) a dependency cycle raises and names the cycle")
            return
        raise AssertionError("a dependency cycle did not raise")

    async def test_e_legacy_entry_is_stale_once_then_stable():
        spec = phases.PhaseSpec("a", None, ())
        current_tag = (
            await phase_tag_resolver.resolve_tags(
                {"a": {"model": "m1"}}, [TagPhase("a")]
            )
        )["a"]
        legacy = {"state": "complete"}
        assert phases.is_phase_stale(spec, {"a": legacy}, current_tag) is True
        legacy["tag"] = current_tag
        assert phases.is_phase_stale(spec, {"a": legacy}, current_tag) is False
        print("PASS (e) a legacy entry is stale once, then stable after rewrite")

    async def test_f_async_settings_check_is_awaited():
        awaited = []

        async def check(settings):
            awaited.append(settings.get("model"))
            return {"model": settings.get("model")}

        phase = TagPhase("a", settings_check=check)
        first = (
            await phase_tag_resolver.resolve_tags(
                {"a": {"model": "m1", "noise": 1}}, [phase]
            )
        )["a"]
        assert awaited == ["m1"], awaited
        second = (
            await phase_tag_resolver.resolve_tags(
                {"a": {"model": "m2", "noise": 1}}, [phase]
            )
        )["a"]
        assert first != second, (first, second)
        # An unrelated field is filtered out by settings_check -> same tag.
        third = (
            await phase_tag_resolver.resolve_tags(
                {"a": {"model": "m1", "noise": 2}}, [phase]
            )
        )["a"]
        assert third == first, (first, third)
        print("PASS (f) an async settings_check is awaited and folds into the tag")

    async def _main():
        await test_a_same_config_same_tags()
        await test_b_deep_dependency_change_propagates()
        await test_c_unrelated_phase_change_does_not_propagate()
        await test_d_cycle_raises_naming_the_cycle()
        await test_e_legacy_entry_is_stale_once_then_stable()
        await test_f_async_settings_check_is_awaited()

    try:
        asyncio.run(_main())
    except AssertionError as exc:
        print(f"FAIL {exc}")
        sys.exit(1)
    print("OK recursive phase tags (six tests)")
