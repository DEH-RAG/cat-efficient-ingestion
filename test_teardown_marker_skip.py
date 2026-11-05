"""Standalone, import-safe tests for the agent-teardown skip in the file-delete hook.

The Cat plugin loader imports EVERY ``.py`` under a plugin folder, so the whole
body lives inside ``if __name__ == "__main__":`` and importing this file executes
zero code. Run it as a plain script from the host repository (the plugin is
symlinked under ``cat/core_plugins``)::

    cd /root/git/grinning-cat-core
    .venv/bin/python /root/git/plugins/cat-efficient-ingestion/test_teardown_marker_skip.py

Covers ``plugin.before_file_manager_file_delete``:

  (a) teardown marker present -> ``delete_source`` is NOT called; hook returns
      cleanly and logs an INFO line;
  (b) marker absent -> ``delete_source`` IS called once with the right source;
  (c) a marker-read failure -> falls back to calling ``delete_source``
      (fail-open) and logs a WARNING.
"""

if __name__ == "__main__":
    import asyncio

    from cat.core_plugins.cat_efficient_ingestion import plugin

    class FakeLog:
        def __init__(self):
            self.infos = []
            self.warnings = []
            self.errors = []

        def info(self, msg):
            self.infos.append(str(msg))

        def warning(self, msg):
            self.warnings.append(str(msg))

        def error(self, msg):
            self.errors.append(str(msg))

        def debug(self, msg):
            pass

    class FakeCat:
        agent_key = "agent-1"

    def run_hook(marker_behavior, calls):
        """Run the hook with a fake marker reader; return (log, calls)."""

        async def fake_delete_source(source, cat, scope=None):
            calls.append((source, cat, scope))
            return True

        async def fake_has_delete_marker(agent_id):
            return marker_behavior(agent_id)

        original_delete = plugin.delete_source
        original_marker = plugin.has_delete_marker
        original_log = plugin.log
        fake_log = FakeLog()
        plugin.delete_source = fake_delete_source
        plugin.has_delete_marker = fake_has_delete_marker
        plugin.log = fake_log
        try:
            asyncio.run(
                plugin.before_file_manager_file_delete.function(
                    "doc.txt", "agent", FakeCat()
                )
            )
        finally:
            plugin.delete_source = original_delete
            plugin.has_delete_marker = original_marker
            plugin.log = original_log
        return fake_log, calls

    # (a) marker present -> skip delete_source, clean return, INFO logged.
    calls = []
    log, calls = run_hook(lambda agent_id: True, calls)
    assert calls == [], f"expected no delete_source call, got {calls}"
    assert log.infos, "expected an INFO teardown line"
    assert any("teardown" in msg.lower() for msg in log.infos), log.infos
    assert not log.warnings, log.warnings
    print("OK (a) marker present: delete_source skipped, INFO logged")

    # (b) marker absent -> delete_source called once with the right source.
    calls = []
    log, calls = run_hook(lambda agent_id: False, calls)
    assert len(calls) == 1, f"expected one delete_source call, got {calls}"
    source, cat, scope = calls[0]
    assert source == "doc.txt", source
    assert isinstance(cat, FakeCat), cat
    assert scope == "agent", scope
    assert not log.warnings, log.warnings
    print("OK (b) marker absent: delete_source called once for the source")

    # (c) marker-read failure -> fail-open, delete_source called, WARNING logged.
    def boom(agent_id):
        raise RuntimeError("crud down")

    calls = []
    log, calls = run_hook(boom, calls)
    assert len(calls) == 1, f"expected fail-open delete_source call, got {calls}"
    assert calls[0][0] == "doc.txt", calls[0]
    assert log.warnings, "expected a WARNING on marker-read failure"
    assert any("marker" in msg.lower() for msg in log.warnings), log.warnings
    print("OK (c) marker-read failure: fail-open delete_source called, WARNING logged")

    print("all teardown-marker-skip tests passed")
