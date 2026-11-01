"""Standalone, import-safe tests for the reverse-order file deletion.

Covers the file-level ``deleting`` state and EffING's single
``delete_source(source, cat)`` path:

  (a) one source -> EVERY phase's ``delete()`` in REVERSE execution order;
  (b) the status record is removed LAST;
  (c) a crash after the first delete leaves ``deleting`` and the re-run
      completes only the remaining phases;
  (d) a phase without a delete implementation surfaces an error (never a
      silent pass) and leaves the row ``deleting``;
  (e) the legacy purge hook still fires for a hook-declared phase;
  (f) gate FALSE -> the standard deletion runs, NO ``deleting`` is written;
  (g) gate TRUE but only the built-in phases -> standard, no stuck row;
  (h) a diary phase that is no longer installed -> standard, record removed;
  (i) a crash-left ``deleting`` row meets gate FALSE -> standard clears it.

The Cat plugin loader imports EVERY ``.py`` under a plugin folder, so the whole
body lives inside ``if __name__ == "__main__":`` and importing this file
executes zero code. Run it as a plain script from the host repository (the
plugin is symlinked under ``cat/core_plugins``)::

    cd /root/git/grinning-cat-core
    .venv/bin/python /root/git/plugins/cat-efficient-ingestion/test_phase_file_delete.py
"""

if __name__ == "__main__":
    import asyncio
    import sys
    import types

    from cat.core_plugins.cat_efficient_ingestion import phases, plugin

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

    class FakePhase:
        """Duck-typed phase recording its delete calls in a shared event log."""

        def __init__(self, phase_id, events, depends_on=(), fail=False):
            self.id = phase_id
            self.events = events
            self.depends_on = tuple(depends_on)
            self.fail = fail

        async def delete(self, source, cat):
            self.events.append(("delete", self.id))
            if self.fail:
                raise RuntimeError(f"{self.id} delete boom")

    class NoDeletePhase:
        """A phase-shaped object without any ``delete`` implementation."""

        id = "broken_phase"
        depends_on = ()

    class Store:
        """In-memory status registry double recording every state transition."""

        def __init__(self, doc=None):
            self.doc = dict(doc) if doc else None
            self.events = []

        async def get_status(self, agent_id, scope, source):
            return self.doc

        async def set_status(self, agent_id, scope, source, **kwargs):
            doc = dict(self.doc or {})
            doc.update({k: v for k, v in kwargs.items() if k != "extra"})
            extra = kwargs.get("extra")
            if extra:
                doc.update(extra)
            status = kwargs.get("status")
            doc["status"] = getattr(status, "value", status)
            self.doc = doc
            self.events.append(
                ("status", doc["status"], list(doc.get("deleted_phases") or []))
            )
            return doc

        async def delete_status(self, agent_id, scope, source):
            self.events.append(("delete-row", scope))
            self.doc = None

    class Seams:
        """Patch the module-level seams used by ``delete_source``."""

        def __init__(self, gate, phase_list, store, log=None):
            self.gate = gate
            self.phase_list = phase_list
            self.store = store
            self.log = log or FakeLog()
            self.saved = {}

        def __enter__(self):
            for name in (
                "phase_delete_active",
                "get_phases",
                "get_status",
                "set_status",
                "_delete_status_doc",
                "source_lock",
                "log",
            ):
                self.saved[name] = getattr(phases, name)

            async def _gate(cat):
                return self.gate

            async def _phases(cat):
                return list(self.phase_list)

            class _NullLock:
                async def __aenter__(self):
                    return None

                async def __aexit__(self, *exc):
                    return False

            phases.phase_delete_active = _gate
            phases.get_phases = _phases
            phases.get_status = self.store.get_status
            phases.set_status = self.store.set_status
            phases._delete_status_doc = self.store.delete_status
            phases.source_lock = lambda agent_id, source: _NullLock()
            phases.log = self.log
            return self

        def __exit__(self, *exc):
            for name, value in self.saved.items():
                setattr(phases, name, value)
            return False

    def _cat():
        return types.SimpleNamespace(agent_key="phase_agent", _id="phase_agent")

    def _doc(status="completed", diary=None, deleted=None):
        doc = {"source": "doc.txt", "scope": "agent", "status": status}
        if diary is not None:
            doc["completed_phases"] = diary
        if deleted is not None:
            doc["deleted_phases"] = list(deleted)
        return doc

    async def test_a_reverse_order():
        events = []
        alpha = FakePhase("alpha", events)
        beta = FakePhase("beta", events, depends_on=("alpha",))
        gamma = FakePhase("gamma", events, depends_on=("beta",))
        # Deliberately scrambled registration order: the topological order is
        # alpha -> beta -> gamma, so deletion must be gamma -> beta -> alpha.
        with Seams(True, [beta, gamma, alpha], Store()):
            ok = await phases.delete_source("doc.txt", _cat(), scope="agent")
        assert ok is True, ok
        deletes = [event[1] for event in events if event[0] == "delete"]
        assert deletes == ["gamma", "beta", "alpha"], deletes
        print("PASS (a) every phase delete runs in reverse execution order")

    async def test_b_record_removed_last():
        events = []
        phases_list = [
            FakePhase("alpha", events),
            FakePhase("beta", events, depends_on=("alpha",)),
        ]
        store = Store(_doc("completed"))
        store.events = events  # share ONE ordered log
        with Seams(True, phases_list, store):
            ok = await phases.delete_source("doc.txt", _cat(), scope="agent")
        assert ok is True, ok
        kinds = [event[0] for event in events]
        assert kinds[0] == "status" and events[0][1] == "deleting", events
        delete_indexes = [i for i, event in enumerate(events) if event[0] == "delete"]
        row_index = kinds.index("delete-row")
        assert delete_indexes, events
        assert max(delete_indexes) < row_index, events
        assert row_index == len(events) - 1, events
        assert [event[1] for event in events if event[0] == "delete"] == ["beta", "alpha"]
        print("PASS (b) the status record is removed last")

    async def test_c_crash_and_resume():
        events = []
        alpha = FakePhase("alpha", events)
        beta = FakePhase("beta", events, depends_on=("alpha",))
        gamma = FakePhase("gamma", events, depends_on=("beta",))
        store = Store(_doc("completed"))
        store.events = events
        with Seams(True, [alpha, beta, gamma], store):
            beta.fail = True
            try:
                await phases.delete_source("doc.txt", _cat(), scope="agent")
            except phases.PhaseDeleteError:
                pass
            else:
                raise AssertionError("a failing delete must raise PhaseDeleteError")
            assert store.doc is not None, "the record must survive a crash"
            assert store.doc["status"] == "deleting", store.doc
            assert store.doc["deleted_phases"] == ["gamma"], store.doc
            # gamma succeeded, beta was attempted and raised: the failure stops
            # the sweep so alpha (beta's output) is not deleted before its retry.
            assert [e[1] for e in events if e[0] == "delete"] == ["gamma", "beta"], events

            beta.fail = False
            events.clear()
            store.events = events
            ok = await phases.delete_source("doc.txt", _cat(), scope="agent")
        assert ok is True, ok
        deletes = [e[1] for e in events if e[0] == "delete"]
        assert deletes == ["beta", "alpha"], deletes
        assert store.doc is None, store.doc
        print("PASS (c) a crash leaves deleting and the re-run does only the rest")

    async def test_d_missing_delete_is_an_error():
        log = FakeLog()
        store = Store(_doc("completed"))
        with Seams(True, [NoDeletePhase()], store, log=log):
            try:
                await phases.delete_source("doc.txt", _cat(), scope="agent")
            except phases.PhaseDeleteError as exc:
                assert "broken_phase" in str(exc), exc
            else:
                raise AssertionError("a phase without delete must not pass silently")
        assert store.doc is not None and store.doc["status"] == "deleting", store.doc
        assert not any(event[0] == "delete-row" for event in store.events), store.events
        assert any("broken_phase" in message for message in log.errors), log.errors
        print("PASS (d) a phase without delete surfaces an error and does not pass")

    async def test_e_legacy_purge_hook_still_fires():
        calls = []
        spec = phases.PhaseSpec("legacy_phase", None, ())

        def _purge(phase_id, source, cat=None):
            calls.append((phase_id, source))
            return {"status": "done"}

        legacy = phases.HookPhase(spec, purge_hook=_purge)
        with Seams(True, [legacy], Store(_doc("completed"))):
            ok = await phases.delete_source("doc.txt", _cat(), scope="agent")
        assert ok is True, ok
        assert calls == [("legacy_phase", "doc.txt")], calls
        print("PASS (e) the legacy purge hook still fires for a hook-declared phase")

    async def _run_standard_chain(gate, phase_list, store, log=None):
        """before hook (no-op unless the phase path applies) + core wipe + after."""
        with Seams(gate, phase_list, store, log=log):
            await plugin.before_file_manager_file_delete.function(
                "doc.txt", "agent", _cat()
            )
            # the core route deletes the source's vector points between the
            # two hooks; the plugin's standard cleanup is the after hook.
            saved = plugin.delete_status
            plugin.delete_status = store.delete_status
            try:
                await plugin.after_file_manager_file_deleted.function(
                    "doc.txt", "agent", _cat()
                )
            finally:
                plugin.delete_status = saved

    async def test_f_gate_false_standard():
        events = []
        store = Store(_doc("completed"))
        store.events = events
        await _run_standard_chain(False, [FakePhase("alpha", events)], store)
        assert not any(event[0] == "delete" for event in events), events
        assert not any(event[0] == "status" and event[1] == "deleting" for event in events), events
        assert store.doc is None, "the standard deletion must remove the record"
        print("PASS (f) gate false -> standard deletion, no deleting state")

    async def test_g_builtins_only_standard():
        events = []
        store = Store(_doc("completed"))
        store.events = events
        await _run_standard_chain(
            True, [phases.ParsingChunkingPhase(), phases.EmbeddingPhase()], store
        )
        assert not any(event[0] == "delete" for event in events), events
        assert not any(event[0] == "status" and event[1] == "deleting" for event in events), events
        assert store.doc is None, "built-ins-only must not leave a stuck row"
        print("PASS (g) gate true with only built-ins -> standard, no stuck row")

    async def test_h_uninstalled_phase_standard():
        events = []
        store = Store(_doc("completed", diary={"ghost_phase": {"state": "complete"}}))
        store.events = events
        await _run_standard_chain(True, [FakePhase("still_installed", events)], store)
        assert not any(event[0] == "delete" for event in events), events
        assert store.doc is None, "the record of an uninstalled phase must go"
        print("PASS (h) an uninstalled diary phase -> standard, record removed")

    async def test_i_stale_deleting_row_gate_false():
        events = []
        store = Store(_doc("deleting", deleted=["gamma"]))
        store.events = events
        await _run_standard_chain(False, [FakePhase("alpha", events)], store)
        assert store.doc is None, "a stale deleting row must never sit forever"
        assert not any(event[0] == "delete" for event in events), events
        print("PASS (i) a crash-left deleting row is cleared by the standard path")

    async def main():
        await test_a_reverse_order()
        await test_b_record_removed_last()
        await test_c_crash_and_resume()
        await test_d_missing_delete_is_an_error()
        await test_e_legacy_purge_hook_still_fires()
        await test_f_gate_false_standard()
        await test_g_builtins_only_standard()
        await test_h_uninstalled_phase_standard()
        await test_i_stale_deleting_row_gate_false()
        print("OK file-level phase deletion (nine tests)")

    try:
        asyncio.run(main())
    except AssertionError:
        import traceback

        traceback.print_exc()
        sys.exit(1)
