"""Standalone, import-safe tests for the phase WARNINGS channel.

A phase that RAN correctly but produced nothing (or less than expected) reports
``done`` + a warning instead of ``error``. The source stays ``completed`` so
dependents are not blocked, but the degradation is persisted in the diary and
aggregated on the source status entry.

Covers:
  (a) done + 2 warnings -> entry carries them, source completed,
      warning_count=2, has_warnings=true;
  (b) error -> unchanged propagation, no warnings;
  (c) warnings persist across a re-read (diary-backed);
  (d) warnings + all phases complete -> completed, never error;
  (e) the aggregate lists each warning with its phase id.

The plugin loader imports EVERY ``.py`` under a plugin folder, so the whole
body lives inside ``if __name__ == "__main__":`` and importing this file
executes zero code.

Run from the host repository (the plugin is symlinked under
``cat/core_plugins``)::

    cd /root/git/grinning-cat-core
    .venv/bin/python /root/git/plugins/cat-efficient-ingestion/tests/test_phase_warnings.py
"""

if __name__ == "__main__":
    import asyncio
    import sys
    import types

    from cat.core_plugins.cat_efficient_ingestion import phases, registry, reembed

    COMPLETE = registry.PHASE_STATE_COMPLETE
    ERROR = registry.PHASE_STATE_ERROR
    WARNING = {"code": "empty-extraction", "message": "the parser returned 0 chunks"}

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

    class MemoryStore:
        """Patch the registry's Redis seams with an in-memory document store."""

        def __init__(self):
            self.docs = {}
            self.saved = {}

        def __enter__(self):
            for name, value in (
                ("_read_doc", registry._read_doc),
                ("_list_status_keys", registry._list_status_keys),
                ("ingestion_canceled", registry.ingestion_canceled),
                ("crud_store", registry.crud.store),
            ):
                self.saved[name] = value

            async def _read_doc(key):
                return self.docs.get(key)

            async def _list_status_keys(agent_id):
                return list(self.docs.keys())

            async def _never_canceled(*args, **kwargs):
                return False

            async def _store(key, doc):
                self.docs[key] = doc

            registry._read_doc = _read_doc
            registry._list_status_keys = _list_status_keys
            registry.ingestion_canceled = _never_canceled
            registry.crud.store = _store
            return self

        def __exit__(self, *exc):
            registry._read_doc = self.saved["_read_doc"]
            registry._list_status_keys = self.saved["_list_status_keys"]
            registry.ingestion_canceled = self.saved["ingestion_canceled"]
            registry.crud.store = self.saved["crud_store"]
            return False

    def _ccat():
        return types.SimpleNamespace(agent_key="warnings_agent", _id="warnings_agent")

    async def _seed(source="doc.txt"):
        await registry.set_status(
            "warnings_agent", "agent", source,
            type_="file", status=registry.IngestionStatus.PROCESSING,
        )

    def _patch_external(result):
        saved = {
            "execute_phase_on_owner": reembed.execute_phase_on_owner,
            "merged_phases": reembed.merged_phases,
            "log": reembed.log,
        }

        async def _hook(*args, **kwargs):
            return result

        async def _merged(*args, **kwargs):
            return {}

        reembed.execute_phase_on_owner = _hook
        reembed.merged_phases = _merged
        reembed.log = FakeLog()
        return saved

    def _restore(saved):
        for name, value in saved.items():
            setattr(reembed, name, value)

    async def _read_doc(source):
        return await registry.get_status("warnings_agent", "agent", source)

    async def _main():
        store = MemoryStore()
        with store:
            # ---- (a) done + 2 warnings -> complete, aggregate, no error ----
            await _seed()
            warnings = [WARNING, {"code": "partial-extraction", "message": "kept 2 of 5"}]
            saved = _patch_external({"status": "done", "warnings": warnings})
            try:
                outcome = {}
                ok = await reembed._run_external_phase(
                    _ccat(), None, "agent", "doc.txt", "graph_concepts", None,
                    counts={}, outcome=outcome,
                )
                assert ok is True, "done must stay success"
                assert outcome.get("warnings") == warnings, outcome
                finished = await reembed._finish_phase(
                    _ccat(), None, "agent", "doc.txt", "graph_concepts", None,
                    True, warnings=outcome.get("warnings"),
                )
                assert finished is True, "a warned phase is complete"
            finally:
                _restore(saved)

            await registry.set_status(
                "warnings_agent", "agent", "doc.txt",
                type_="file", status=registry.IngestionStatus.COMPLETED,
            )
            doc = await _read_doc("doc.txt")
            entry = registry.get_completed_phases(doc)["graph_concepts"]
            assert entry["state"] == COMPLETE, entry
            assert entry.get("warnings") == warnings, entry
            assert doc["status"] == registry.IngestionStatus.COMPLETED.value, doc.get("status")
            assert doc["warning_count"] == 2, doc.get("warning_count")
            assert doc["has_warnings"] is True, doc.get("has_warnings")
            assert [w["code"] for w in doc["warnings"]] == [
                "empty-extraction", "partial-extraction"
            ], doc["warnings"]
            print("PASS (a) done + 2 warnings -> complete entry and source aggregate")

            # ---- (c) warnings persist across a re-read ----
            reread = await _read_doc("doc.txt")
            listed = await registry.list_statuses("warnings_agent")
            assert reread["warning_count"] == 2, reread
            assert listed[0]["warning_count"] == 2, listed[0]
            assert registry.get_completed_phases(reread)["graph_concepts"]["warnings"] == warnings
            # An unrelated status write must not drop the diary-backed warnings.
            await registry.set_status(
                "warnings_agent", "agent", "doc.txt",
                type_="file", status=registry.IngestionStatus.COMPLETED,
            )
            after = await _read_doc("doc.txt")
            assert after["warning_count"] == 2, after
            print("PASS (c) warnings persist across a re-read and an unrelated write")

            # ---- (d) warnings + all complete is completed, never error ----
            diary = registry.get_completed_phases(await _read_doc("doc.txt"))
            assert registry.derive_source_status(diary, ["graph_concepts"]) == COMPLETE, diary
            assert (await _read_doc("doc.txt"))["status"] != registry.IngestionStatus.ERROR.value
            print("PASS (d) warned source derives completed, never error")

            # ---- (b) error -> unchanged propagation, no warnings ----
            await _seed("broken.txt")
            saved = _patch_external({"status": "error", "reason": "transport exploded"})
            try:
                outcome = {}
                ok = await reembed._run_external_phase(
                    _ccat(), None, "agent", "broken.txt", "graph_concepts", None,
                    counts={}, outcome=outcome,
                )
                assert ok is False, "error must fail"
                assert outcome.get("status") == "error", outcome
                assert "warnings" not in outcome, outcome
                await reembed._mark_phase_error(
                    _ccat(), "agent", "broken.txt", "graph_concepts",
                    outcome["reason"], None, "file",
                )
            finally:
                _restore(saved)

            broken = await _read_doc("broken.txt")
            broken_entry = registry.get_completed_phases(broken)["graph_concepts"]
            assert broken_entry["state"] == ERROR, broken_entry
            assert "warnings" not in broken_entry, broken_entry
            assert broken["status"] == registry.IngestionStatus.ERROR.value, broken.get("status")
            assert broken["warnings"] == [], broken.get("warnings")
            assert broken["warning_count"] == 0, broken.get("warning_count")
            assert broken["has_warnings"] is False, broken.get("has_warnings")
            print("PASS (b) error keeps its propagation and carries no warnings")

            # ---- (e) the aggregate lists each warning with its phase id ----
            await _seed("multi.txt")
            await registry.record_phase_completed(
                "warnings_agent", "agent", "multi.txt", "parsing_chunking",
                warnings=[{"code": "empty-extraction", "message": "no chunks"}],
            )
            await registry.record_phase_completed(
                "warnings_agent", "agent", "multi.txt", "embedding",
                warnings=[{"code": "partial-extraction", "message": "3 of 9 vectors"}],
            )
            multi = await _read_doc("multi.txt")
            by_phase = {w["phase"]: w for w in multi["warnings"]}
            assert set(by_phase) == {"parsing_chunking", "embedding"}, multi["warnings"]
            assert by_phase["parsing_chunking"] == {
                "phase": "parsing_chunking",
                "code": "empty-extraction",
                "message": "no chunks",
            }, by_phase
            assert by_phase["embedding"] == {
                "phase": "embedding",
                "code": "partial-extraction",
                "message": "3 of 9 vectors",
            }, by_phase
            assert multi["warning_count"] == 2, multi.get("warning_count")
            print("PASS (e) the aggregate attributes every warning to its phase")

            # ---- PhaseResult / legacy dict carry the optional warnings ----
            direct = phases.PhaseResult(status="done", warnings=[WARNING])
            assert direct.warnings == [WARNING], direct
            coerced = phases._coerce_phase_result(
                "graph_concepts", {"status": "done", "warnings": warnings}
            )
            assert coerced.status == "done", coerced
            assert coerced.warnings == warnings, coerced
            malformed = phases._coerce_phase_result(
                "graph_concepts",
                {"status": "done", "warnings": ["nope", {"code": 1}, {"code": "c", "message": "m"}]},
            )
            assert malformed.warnings == [{"code": "c", "message": "m"}], malformed
            print("PASS (f) PhaseResult and legacy dict carry normalized warnings")

        return True

    try:
        asyncio.run(_main())
    except AssertionError as exc:
        print(f"FAIL {exc}")
        sys.exit(1)
