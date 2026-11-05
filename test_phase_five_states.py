"""Standalone, import-safe tests for the five-state phase diary contract.

Covers:
  (a) a phase body returning done with ZERO items ends ``complete``;
  (b) a phase that raises ends ``error`` with the reason;
  (c) a phase whose ``applies_to`` is false ends ``skipped`` with the reason;
  (d) the DERIVED source status (error / complete / processing) table;
  (e) a legacy entry without ``state`` reads as ``complete`` and is
      normalized on the next write;
  (f) the item-count criterion is gone (a mismatch, and zero items, both
      complete).

The plugin loader imports EVERY ``.py`` under a plugin folder, so the whole
body lives inside ``if __name__ == "__main__":`` and importing this file
executes zero code.

Run from the host repository::

    cd /root/git/grinning-cat-core
    .venv/bin/python /root/git/plugins/cat-efficient-ingestion/test_phase_five_states.py
"""

if __name__ == "__main__":
    import asyncio
    import sys
    import types

    from cat.core_plugins.cat_efficient_ingestion import reembed, registry

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
                ("ingestion_canceled", registry.ingestion_canceled),
                ("crud_store", registry.crud.store),
            ):
                self.saved[name] = value

            async def _read_doc(key):
                return self.docs.get(key)

            async def _never_canceled(*args, **kwargs):
                return False

            async def _store(key, doc):
                self.docs[key] = doc

            registry._read_doc = _read_doc
            registry.ingestion_canceled = _never_canceled
            registry.crud.store = _store
            return self

        def __exit__(self, *exc):
            registry._read_doc = self.saved["_read_doc"]
            registry.ingestion_canceled = self.saved["ingestion_canceled"]
            registry.crud.store = self.saved["crud_store"]
            return False

    def _ccat():
        return types.SimpleNamespace(agent_key="five_agent", _id="five_agent")

    def _doc(store, source="doc.txt", scope="agent", diary=None, status="processing"):
        return store.docs.setdefault(
            registry.status_key("five_agent", scope, source),
            {
                "source": source,
                "scope": scope,
                "type_": "file",
                "status": status,
                "completed_phases": diary or {},
            },
        )

    async def _async_return(value):
        return value

    async def test_a_done_with_zero_items_completes():
        log = FakeLog()
        recorded = []

        async def _record(*args, **kwargs):
            recorded.append((args, kwargs))
            return {}

        saved = (reembed._record_phase, reembed.log)
        reembed._record_phase = _record
        reembed.log = log
        try:
            ok = await reembed._finish_phase(
                _ccat(), None, "agent", "doc.txt", "parsing_chunking", None,
                products=[], expected=0, processed=0,
            )
        finally:
            reembed._record_phase, reembed.log = saved
        assert ok is True, "zero items must still complete"
        assert len(recorded) == 1, recorded
        assert any("telemetry only" in m for m in log.infos), log.infos
        print("PASS (a) done with zero items -> complete")

    async def test_b_raising_phase_ends_error_with_reason():
        log = FakeLog()
        outcome = {}
        set_calls = []

        async def _get_status(*args, **kwargs):
            return _doc(store, status="processing")

        async def _backfill(*args, **kwargs):
            return {}

        async def _merged(*args, **kwargs):
            return {}

        async def _boom(*args, **kwargs):
            raise RuntimeError("parser exploded")

        async def _set_status(*args, **kwargs):
            set_calls.append(kwargs)

        saved = {
            name: getattr(reembed, name)
            for name in (
                "get_status", "backfill_completed_phases", "merged_phases",
                "execute_phase_on_owner", "_set_status", "log",
                "record_phase_error",
            )
        }
        reembed.get_status = _get_status
        reembed.backfill_completed_phases = _backfill
        reembed.merged_phases = _merged
        reembed.execute_phase_on_owner = _boom
        reembed._set_status = _set_status
        reembed.log = log
        try:
            ok = await reembed._run_external_phase(
                _ccat(), None, "agent", "doc.txt", "graph_spacy", None,
                outcome=outcome,
            )
            assert ok is False, "a raising phase must fail"
            assert outcome.get("status") == "error", outcome
            assert outcome.get("reason") == "parser exploded", outcome
            # The machine then writes the matching diary state + derived status.
            await registry.record_phase_error(
                "five_agent", "agent", "doc.txt", "graph_spacy", outcome["reason"]
            )
        finally:
            for name, value in saved.items():
                setattr(reembed, name, value)
        entry = registry.get_completed_phases(_doc(store))["graph_spacy"]
        assert entry["state"] == registry.PHASE_STATE_ERROR, entry
        assert entry["reason"] == "parser exploded", entry
        print("PASS (b) raising phase -> error with the reason")

    async def test_c_not_applicable_ends_skipped_with_reason():
        class NotApplicablePhase:
            id = "graph_spacy"

            async def applies_to(self, cat, deps_state):
                return False, "graph handler is not installed"

            def tag(self, cat):
                return "graph_spacy"

        async def _get_phases(cat):
            return [NotApplicablePhase()]

        saved = reembed.get_phases
        reembed.get_phases = _get_phases
        try:
            applies, reason = await reembed._phase_applies(
                _ccat(), None, "graph_spacy", {}
            )
        finally:
            reembed.get_phases = saved
        assert applies is False, (applies, reason)
        assert reason == "graph handler is not installed", reason
        await reembed._mark_phase_skipped(
            _ccat(), "agent", "doc.txt", "graph_spacy", reason, None, "file"
        )
        entry = registry.get_completed_phases(_doc(store))["graph_spacy"]
        assert entry["state"] == registry.PHASE_STATE_SKIPPED, entry
        assert entry["reason"] == "graph handler is not installed", entry
        print("PASS (c) applies_to false -> skipped with the reason")

    async def test_d_derived_source_status():
        assert registry.derive_source_status({}, ["a"]) == "processing", "absent = pending"
        assert registry.derive_source_status(
            {"a": {"state": "processing"}}, ["a"]
        ) == "processing"
        assert registry.derive_source_status(
            {"a": {"state": "complete"}}, ["a", "b"]
        ) == "processing", "a missing phase keeps the source processing"
        assert registry.derive_source_status(
            {"a": {"state": "complete"}, "b": {"state": "skipped"}}, ["a", "b"]
        ) == "complete"
        assert registry.derive_source_status(
            {"a": {"state": "complete"}, "b": {"state": "error", "reason": "x"}},
            ["a", "b"],
        ) == "error"
        assert registry.derive_source_status(
            {"a": {"marker": "legacy"}}, ["a"]
        ) == "complete", "legacy no-state entry reads complete"
        print("PASS (d) derived source status table")

    async def test_e_legacy_entry_read_complete_and_normalized_on_write():
        key = registry.status_key("five_agent", "agent", "doc.txt")
        store.docs[key] = {
            "source": "doc.txt",
            "scope": "agent",
            "type_": "file",
            "status": "completed",
            "completed_phases": {
                "embedding": {"marker": "m", "settings_version": None, "deps": {}}
            },
        }
        legacy = registry.get_completed_phases(_doc(store))["embedding"]
        assert legacy["state"] == "complete", legacy
        await registry.record_phase_completed(
            "five_agent", "agent", "doc.txt", "parsing_chunking",
            marker="p", tag="parsing_chunking:chunker:",
        )
        stored = _doc(store)["completed_phases"]
        assert stored["embedding"]["state"] == "complete", stored["embedding"]
        for key in ("reason", "started_at", "owner", "tag", "attempt"):
            assert key in stored["embedding"], (key, stored["embedding"])
        assert stored["parsing_chunking"]["state"] == "complete", stored
        print("PASS (e) legacy entry reads complete and normalizes on write")

    async def test_f_count_criterion_is_gone():
        log = FakeLog()
        recorded = []

        async def _record(*args, **kwargs):
            recorded.append(args)
            return {}

        saved = (reembed._record_phase, reembed.log)
        reembed._record_phase = _record
        reembed.log = log
        try:
            # A mismatch (3 of 10) and a zero/zero both complete: no count is
            # consulted, only "the body returned done".
            ok_mismatch = await reembed._finish_phase(
                _ccat(), None, "agent", "doc.txt", "embedding", None,
                products=[], expected=10, processed=3,
            )
            ok_zero = await reembed._finish_phase(
                _ccat(), None, "agent", "doc.txt", "embedding", None,
                products=[], expected=0, processed=0,
            )
        finally:
            reembed._record_phase, reembed.log = saved
        assert ok_mismatch is True and ok_zero is True, (ok_mismatch, ok_zero)
        assert len(recorded) == 2, recorded
        assert not any(
            "NOT marked complete" in m or "processed 3 of 10" in m
            for m in log.warnings
        ), log.warnings
        print("PASS (f) processed/expected no longer decide completion")

    async def _main():
        await test_a_done_with_zero_items_completes()
        await test_b_raising_phase_ends_error_with_reason()
        await test_c_not_applicable_ends_skipped_with_reason()
        await test_d_derived_source_status()
        await test_e_legacy_entry_read_complete_and_normalized_on_write()
        await test_f_count_criterion_is_gone()

    store = MemoryStore()
    with store:
        try:
            asyncio.run(_main())
        except AssertionError as exc:
            print(f"FAIL {exc}")
            sys.exit(1)
    print("OK phase five-state contract (six tests)")
