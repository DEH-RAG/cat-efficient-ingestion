"""Standalone, import-safe tests for the chunker-name source and log policy.

The Cat plugin loader imports EVERY ``.py`` under a plugin folder, so the whole
body lives inside ``if __name__ == "__main__":`` and importing this file
executes zero code. Run it as a plain script from the host repository::

    cd /root/git/grinning-cat-core
    .venv/bin/python /root/git/plugins/cat-efficient-ingestion/tests/test_log_policy_and_chunker_name.py

Covers:

  (A) the recorded chunker name comes from the SAVED ``chunker`` settings
      (``MathAwareSemanticChunkerSettings`` -> ``MathAwareSemanticChunker``),
      never the old ``default_chunker`` fallback; genuinely unresolvable
      settings record ``None``; and both plugin.py and reembed.py use the same
      helper;
  (B) a revalidation sweep that re-ingested 0 rows logs at DEBUG (not INFO),
      a non-zero sweep logs at INFO, and the expected sweep lock contention
      logs at DEBUG.
"""

if __name__ == "__main__":
    import asyncio

    from cat.core_plugins.cat_efficient_ingestion import (
        hooks,
        plugin,
        reembed,
        resume,
    )

    class FakeLog:
        def __init__(self):
            self.infos = []
            self.warnings = []
            self.errors = []
            self.debugs = []

        def info(self, msg):
            self.infos.append(str(msg))

        def warning(self, msg):
            self.warnings.append(str(msg))

        def error(self, msg):
            self.errors.append(str(msg))

        def debug(self, msg):
            self.debugs.append(str(msg))

    def _async_return(value):
        async def _inner(*args, **kwargs):
            return value

        return _inner

    # ---------------------------------------------------------------- Part A
    def test_chunker_name_from_settings_doc():
        assert (
            hooks.chunker_name_from_settings_doc(
                {"name": "MathAwareSemanticChunkerSettings", "value": {}}
            )
            == "MathAwareSemanticChunker"
        )
        assert (
            hooks.chunker_name_from_settings_doc(
                {"config": "MathAwareSemanticChunkerSettings"}
            )
            == "MathAwareSemanticChunker"
        )
        assert (
            hooks.chunker_name_from_settings_doc({"name": "math-aware-chunker"})
            == "math-aware-chunker"
        )
        assert hooks.chunker_name_from_settings_doc(None) is None
        assert hooks.chunker_name_from_settings_doc({}) is None
        print("PASS chunker name from saved settings (Settings suffix stripped)")

    def _with_settings(doc, coro_factory):
        original = hooks.crud_settings.get_settings_by_category

        async def fake_get(key, category):
            assert category == "chunker", category
            return doc

        hooks.crud_settings.get_settings_by_category = fake_get
        try:
            return asyncio.run(coro_factory())
        finally:
            hooks.crud_settings.get_settings_by_category = original

    def test_resolver_uses_settings_and_never_default_chunker():
        class Cat:
            agent_key = "agent-1"
            chunker = object()  # NO ``name`` attribute, like the real instance

        got = _with_settings(
            {"name": "MathAwareSemanticChunkerSettings", "value": {}},
            lambda: hooks.resolve_chunker_name(Cat()),
        )
        assert got == "MathAwareSemanticChunker", got
        assert got != "default_chunker"
        print("PASS resolver reads the saved settings, not the literal fallback")

    def test_resolver_unresolvable_is_none():
        class Cat:
            agent_key = "agent-2"
            chunker = object()

        got = _with_settings(None, lambda: hooks.resolve_chunker_name(Cat()))
        assert got is None, got
        print("PASS unresolvable chunker settings resolve to None")

    def test_plugin_and_reembed_share_the_helper():
        assert reembed.resolve_chunker_name is hooks.resolve_chunker_name
        print("PASS plugin.py and reembed.py share one chunker-name helper")

    def test_plugin_records_settings_name():
        captured = {}

        async def fake_set_status(*args, **kwargs):
            captured.update(kwargs)

        original = {
            "set_status": plugin.set_status,
            "ingestion_canceled": plugin.ingestion_canceled,
            "caller": plugin._caller,
            "log": plugin.log,
        }
        plugin.set_status = fake_set_status
        plugin.ingestion_canceled = _async_return(False)
        plugin._caller = lambda cat=None, lizard=None: cat
        plugin.log = FakeLog()

        class Cat:
            agent_key = "agent-3"
            chunker = object()  # no name

        try:
            _with_settings(
                {"name": "MathAwareSemanticChunkerSettings", "value": {}},
                lambda: plugin.rabbithole_ingestion_processing.function(
                    "doc.txt", cat=Cat()
                ),
            )
        finally:
            plugin.set_status = original["set_status"]
            plugin.ingestion_canceled = original["ingestion_canceled"]
            plugin._caller = original["caller"]
            plugin.log = original["log"]

        assert captured.get("chunker_name") == "MathAwareSemanticChunker", captured
        assert captured.get("chunker_name") != "default_chunker", captured
        print("PASS the status hook records the real chunker name")

    # ---------------------------------------------------------------- Part B
    def _run_revalidation(entries, pending, reembedded):
        log = FakeLog()

        class Ccat:
            agent_key = "agent-1"

        class SourceObj:
            pass

        async def fake_list_statuses(agent_id):
            return entries

        async def fake_pending(ccat, entry):
            return pending

        async def fake_claim(*args, **kwargs):
            return {"claimed": True}

        async def fake_reembed(*args, **kwargs):
            reembedded.append(kwargs)

        original = {
            "log": resume.log,
            "list_statuses": resume.list_statuses,
            "pending": resume._pending_phases,
            "claim": resume.claim_source_for_resume,
            "source": resume._source_from_entry,
            "reembed": resume.reembed_sources,
            "collection": resume._collection_for_scope,
            "canceled": resume.ingestion_canceled,
            "maintenance": resume._maintenance_marker_active,
            "missing": resume.mark_file_missing,
        }
        resume.log = log
        resume.list_statuses = fake_list_statuses
        resume._pending_phases = fake_pending
        resume.claim_source_for_resume = fake_claim
        resume._source_from_entry = lambda *a, **k: SourceObj()
        resume.reembed_sources = fake_reembed
        resume._collection_for_scope = lambda scope: "declarative"
        resume.ingestion_canceled = _async_return(False)
        resume._maintenance_marker_active = _async_return(False)
        resume.mark_file_missing = _async_return(None)
        try:
            asyncio.run(resume._revalidate_agent(None, "agent-1", ccat=Ccat()))
        finally:
            resume.log = original["log"]
            resume.list_statuses = original["list_statuses"]
            resume._pending_phases = original["pending"]
            resume.claim_source_for_resume = original["claim"]
            resume._source_from_entry = original["source"]
            resume.reembed_sources = original["reembed"]
            resume._collection_for_scope = original["collection"]
            resume.ingestion_canceled = original["canceled"]
            resume._maintenance_marker_active = original["maintenance"]
            resume.mark_file_missing = original["missing"]
        return log

    def test_zero_revalidation_is_debug():
        log = _run_revalidation([], [], [])
        assert any("0 completed row(s) re-ingested" in m for m in log.debugs), log.debugs
        assert not any("0 completed row(s) re-ingested" in m for m in log.infos), log.infos
        print("PASS 0-row revalidation logs at DEBUG, not INFO")

    def test_nonzero_revalidation_is_info():
        entry = {
            "status": "completed",
            "source": "doc.txt",
            "scope": "agent",
            "type": "file",
        }
        log = _run_revalidation([entry], [{"phase": "embedding"}], [])
        assert any("1 completed row(s) re-ingested" in m for m in log.infos), log.infos
        assert not any("1 completed row(s) re-ingested" in m for m in log.debugs), log.debugs
        print("PASS non-zero revalidation logs at INFO")

    def test_sweep_lock_contention_is_debug():
        log = FakeLog()

        class RaisingLock:
            async def __aenter__(self):
                raise resume.crud.LockError("held by another worker")

            async def __aexit__(self, *exc):
                return False

        original = {
            "log": resume.log,
            "canceled": resume.ingestion_canceled,
            "maintenance": resume._maintenance_marker_active,
            "lock": resume.crud.distributed_lock,
        }
        resume.log = log
        resume.ingestion_canceled = _async_return(False)
        resume._maintenance_marker_active = _async_return(False)
        resume.crud.distributed_lock = lambda *a, **k: RaisingLock()
        try:
            asyncio.run(resume._pass_for_agent(None, "agent-1"))
        finally:
            resume.log = original["log"]
            resume.ingestion_canceled = original["canceled"]
            resume._maintenance_marker_active = original["maintenance"]
            resume.crud.distributed_lock = original["lock"]

        assert any("lock held by another worker" in m for m in log.debugs), log.debugs
        assert not any("lock held by another worker" in m for m in log.infos), log.infos
        print("PASS expected sweep lock contention logs at DEBUG")

    test_chunker_name_from_settings_doc()
    test_resolver_uses_settings_and_never_default_chunker()
    test_resolver_unresolvable_is_none()
    test_plugin_and_reembed_share_the_helper()
    test_plugin_records_settings_name()
    test_zero_revalidation_is_debug()
    test_nonzero_revalidation_is_info()
    test_sweep_lock_contention_is_debug()
    print("OK log policy + chunker name")
