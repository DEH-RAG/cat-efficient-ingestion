"""Standalone, import-safe tests for the ingestion phase locks.

Covers:
  (a) two concurrent workers on the same ``(source, phase)`` run the body ONCE
      (the under-lock diary re-check skips the second one);
  (b) two different phases of the same source never overlap (the per-source
      lock serialises them; enter/exit intervals are asserted disjoint);
  (c) a phase-lock timeout leaves the diary entry unchanged and logs a
      WARNING naming agent/source/phase;
  (d) the delete path takes ONLY the per-source lock (never a phase lock);
  (e) lock keys escape glob metacharacters in the source name;
  (f) ``phases.delete_source`` requests the source lock (and no phase lock);
  (g) a concurrent per-source pass cannot interleave with ``delete_source``;
  (h) the dispatch loop aborts a source whose status is ``deleting``.

The plugin loader imports EVERY ``.py`` under a plugin folder, so the whole
body lives inside ``if __name__ == "__main__":`` and importing this file
executes zero code.

Run from the host repository::

    cd /root/git/grinning-cat-core
    .venv/bin/python /root/git/plugins/cat-efficient-ingestion/test_phase_locks.py
"""

if __name__ == "__main__":
    import asyncio
    import time
    from contextlib import asynccontextmanager

    from redis.exceptions import LockError

    from cat.db import crud as crud_module
    from cat.core_plugins.cat_efficient_ingestion import phases, reembed, registry

    AGENT = "lock_agent"
    SCOPE = "agent"
    SRC = "doc.txt"
    PHASE = "embedding"

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

    class Harness:
        """Replace the Redis seams with an in-process store and async locks."""

        def __init__(self):
            self.docs = {}
            self.locks = {}
            self.requested = []
            self.blocking_override: float | None = None
            self.log = FakeLog()
            self.saved = {}
            # delete_source seams
            self.delete_gate = True
            self.phase_objects = []
            self.status_writes = []
            self.status_deletes = []

        @asynccontextmanager
        async def distributed_lock(self, key_pattern, timeout=10.0, blocking_timeout=15.0):
            self.requested.append(key_pattern)
            lock = self.locks.setdefault(key_pattern, asyncio.Lock())
            wait = (
                self.blocking_override
                if self.blocking_override is not None
                else blocking_timeout
            )
            try:
                await asyncio.wait_for(lock.acquire(), timeout=wait)
            except asyncio.TimeoutError:
                raise LockError(f"Could not acquire lock for pattern '{key_pattern}'")
            try:
                yield lock
            finally:
                if lock.locked():
                    lock.release()

        def __enter__(self):
            self.saved = {
                "distributed_lock": crud_module.distributed_lock,
                "store": crud_module.store,
                "delete": crud_module.delete,
                "read_doc": registry._read_doc,
                "canceled": registry.ingestion_canceled,
                "get_status": reembed.get_status,
                "log": reembed.log,
                "phases_delete_active": phases.phase_delete_active,
                "phases_get_phases": phases.get_phases,
                "phases_get_status": phases.get_status,
                "phases_set_status": phases.set_status,
                "phases_delete_status_doc": phases._delete_status_doc,
                "phases_log": phases.log,
            }

            async def _store(key, doc, *args, **kwargs):
                self.docs[key] = doc
                return doc

            async def _delete(key, *args, **kwargs):
                self.docs.pop(key, None)

            async def _read_doc(key):
                return self.docs.get(key)

            async def _never_canceled(*args, **kwargs):
                return False

            async def _get_status(agent_id, scope, source):
                return self.docs.get(registry.status_key(agent_id, scope, source))

            async def _delete_active(cat):
                return self.delete_gate

            async def _get_phases(cat):
                return list(self.phase_objects)

            async def _phases_set_status(agent_id, scope, source, **kwargs):
                self.status_writes.append((source, kwargs.get("status")))
                doc = self.docs.setdefault(
                    registry.status_key(agent_id, scope, source),
                    {"source": source, "scope": scope, "completed_phases": {}},
                )
                if kwargs.get("status") is not None:
                    doc["status"] = kwargs["status"].value
                return doc

            async def _phases_delete_status_doc(agent_id, scope, source):
                self.status_deletes.append(source)
                self.docs.pop(registry.status_key(agent_id, scope, source), None)

            crud_module.distributed_lock = self.distributed_lock
            crud_module.store = _store
            crud_module.delete = _delete
            registry._read_doc = _read_doc
            registry.ingestion_canceled = _never_canceled
            reembed.get_status = _get_status
            reembed.log = self.log
            phases.phase_delete_active = _delete_active
            phases.get_phases = _get_phases
            phases.get_status = _get_status
            phases.set_status = _phases_set_status
            phases._delete_status_doc = _phases_delete_status_doc
            phases.log = self.log
            return self

        def __exit__(self, *exc):
            crud_module.distributed_lock = self.saved["distributed_lock"]
            crud_module.store = self.saved["store"]
            crud_module.delete = self.saved["delete"]
            registry._read_doc = self.saved["read_doc"]
            registry.ingestion_canceled = self.saved["canceled"]
            reembed.get_status = self.saved["get_status"]
            reembed.log = self.saved["log"]
            phases.phase_delete_active = self.saved["phases_delete_active"]
            phases.get_phases = self.saved["phases_get_phases"]
            phases.get_status = self.saved["phases_get_status"]
            phases.set_status = self.saved["phases_set_status"]
            phases._delete_status_doc = self.saved["phases_delete_status_doc"]
            phases.log = self.saved["phases_log"]
            return False

        def doc(self, diary=None):
            key = registry.status_key(AGENT, SCOPE, SRC)
            return self.docs.setdefault(
                key,
                {
                    "source": SRC,
                    "scope": SCOPE,
                    "type_": "file",
                    "status": "processing",
                    "completed_phases": diary or {},
                },
            )

    def _ccat():
        import types

        return types.SimpleNamespace(agent_key=AGENT, _id=AGENT)

    async def test_a_same_phase_runs_once():
        with Harness() as h:
            h.doc()
            counter = [0]

            async def worker():
                async with reembed._phase_guard(
                    _ccat(), SCOPE, SRC, PHASE
                ) as should_run:
                    if not should_run:
                        return
                    await asyncio.sleep(0.02)
                    counter[0] += 1
                    await registry.record_phase_completed(
                        AGENT, SCOPE, SRC, PHASE
                    )

            await asyncio.gather(worker(), worker())
            assert counter[0] == 1, f"phase body ran {counter[0]} times, expected 1"
            entry = registry.get_completed_phases(h.doc())[PHASE]
            assert entry["state"] == registry.PHASE_STATE_COMPLETE, entry
        print("PASS (a) two workers on the same (source, phase) -> body runs once")

    async def test_b_different_phases_never_overlap():
        with Harness() as h:
            h.doc()
            events = []

            async def worker(phase, delay):
                # ORDERING: source lock FIRST, phase lock INSIDE it.
                async with registry.source_lock(AGENT, SRC):
                    async with reembed._phase_guard(
                        _ccat(), SCOPE, SRC, phase
                    ) as should_run:
                        if not should_run:
                            return
                        events.append(("enter", phase, time.perf_counter()))
                        await asyncio.sleep(delay)
                        events.append(("exit", phase, time.perf_counter()))

            await asyncio.gather(
                worker("parsing_chunking", 0.03),
                worker("embedding", 0.01),
            )

            assert [e[0] for e in events] == [
                "enter",
                "exit",
                "enter",
                "exit",
            ], events
            first_exit = events[1][2]
            second_enter = events[2][2]
            assert first_exit <= second_enter, f"phases overlapped: {events}"
            assert {events[0][1], events[2][1]} == {
                "parsing_chunking",
                "embedding",
            }, events
        print("PASS (b) different phases of one source never overlap")

    async def test_c_timeout_leaves_entry_unchanged_and_warns():
        with Harness() as h:
            diary = {
                PHASE: {
                    "state": registry.PHASE_STATE_PROCESSING,
                    "reason": None,
                    "started_at": 123.0,
                    "owner": None,
                    "tag": "",
                    "attempt": 1,
                }
            }
            h.doc(diary)
            original = dict(h.docs[registry.status_key(AGENT, SCOPE, SRC)])

            key = (
                f"ingestion-phase:{AGENT}:"
                f"{crud_module.escape_glob(SRC)}:{PHASE}"
            )
            held = h.locks.setdefault(key, asyncio.Lock())
            await held.acquire()
            h.blocking_override = 0.05
            try:
                async with reembed._phase_guard(
                    _ccat(), SCOPE, SRC, PHASE
                ) as should_run:
                    assert should_run is False, "a timed-out lock must skip the body"
            finally:
                held.release()

            assert h.docs[registry.status_key(AGENT, SCOPE, SRC)] == original, (
                "a lock timeout must leave the diary entry exactly as it was"
            )
            assert h.log.warnings, "a lock timeout must log a WARNING"
            warning = h.log.warnings[-1]
            for needle in (AGENT, SRC, PHASE):
                assert needle in warning, (needle, warning)
        print("PASS (c) lock timeout -> entry untouched + WARNING (agent/source/phase)")

    async def test_d_delete_path_takes_only_the_source_lock():
        with Harness() as h:
            h.doc()
            await registry.delete_status(AGENT, SCOPE, SRC)

            assert h.requested == [
                f"ingestion-source:{AGENT}:{crud_module.escape_glob(SRC)}"
            ], h.requested
            assert not any(
                key.startswith("ingestion-phase:") for key in h.requested
            ), f"delete path took a phase lock: {h.requested}"
        print("PASS (d) delete path takes ONLY the source lock")

    async def test_e_keys_escape_glob_metacharacters():
        with Harness() as h:
            raw = "a*b?c"
            escaped = crud_module.escape_glob(raw)
            assert escaped == "a\\*b\\?c", escaped

            async with registry.source_lock(AGENT, raw):
                pass
            async with registry.phase_lock(AGENT, raw, PHASE):
                pass

            assert h.requested == [
                f"ingestion-source:{AGENT}:{escaped}",
                f"ingestion-phase:{AGENT}:{escaped}:{PHASE}",
            ], h.requested
            for key in h.requested:
                assert "*" not in key.replace("\\*", ""), key
                assert "?" not in key.replace("\\?", ""), key
        print("PASS (e) lock keys escape source glob metacharacters")

    class FakePhase:
        """A plugin phase whose delete just records the call."""

        depends_on = ()

        def __init__(self, phase_id, events):
            self.id = phase_id
            self.events = events

        async def delete(self, source, cat):
            self.events.append(("delete_enter", source, time.perf_counter()))
            await asyncio.sleep(0.01)
            self.events.append(("delete_exit", source, time.perf_counter()))

    async def test_f_delete_source_takes_only_the_source_lock():
        events = []
        with Harness() as h:
            h.doc()
            h.phase_objects = [FakePhase("fake_phase", events)]
            ok = await phases.delete_source(SRC, _ccat())
            assert ok is True, "the phase path must report success"

            source_key = f"ingestion-source:{AGENT}:{crud_module.escape_glob(SRC)}"
            assert h.requested == [source_key], h.requested
            assert not any(
                key.startswith("ingestion-phase:") for key in h.requested
            ), f"delete path took a phase lock: {h.requested}"
            assert [e[0] for e in events] == ["delete_enter", "delete_exit"], events
            assert [w[1] for w in h.status_writes] == [
                registry.IngestionStatus.DELETING,
                registry.IngestionStatus.DELETING,
            ], h.status_writes
            assert h.status_deletes == [SRC], h.status_deletes
        print("PASS (f) delete_source takes ONLY the source lock, no phase lock")

    async def test_g_delete_cannot_interleave_with_a_pass():
        events = []
        with Harness() as h:
            h.doc()
            h.phase_objects = [FakePhase("fake_phase", events)]

            async def pass_worker():
                # what reembed_sources holds for the whole per-source sequence
                async with registry.source_lock(AGENT, SRC):
                    events.append(("pass_enter", time.perf_counter()))
                    await asyncio.sleep(0.05)
                    events.append(("pass_exit", time.perf_counter()))

            task = asyncio.create_task(pass_worker())
            await asyncio.sleep(0.01)
            ok = await phases.delete_source(SRC, _ccat())
            await task
            assert ok is True, "the delete must wait and then succeed"

            names = [e[0] for e in events]
            assert names == [
                "pass_enter",
                "pass_exit",
                "delete_enter",
                "delete_exit",
            ], events
        print("PASS (g) the pass and delete_source never overlap")

    async def test_h_dispatch_aborts_on_deleting():
        import types

        with Harness() as h:
            key = registry.status_key(AGENT, SCOPE, SRC)
            h.docs[key] = {
                "source": SRC,
                "scope": SCOPE,
                "status": registry.IngestionStatus.DELETING.value,
                "completed_phases": {},
            }
            claimed = []

            async def _claim(*args, **kwargs):
                claimed.append(args)
                return None

            async def _get_phases(*args, **kwargs):
                return []

            async def _rabbit(*args, **kwargs):
                return object()

            names = ("get_phases", "_resolve_rabbit_hole", "claim_source_for_resume")
            saved = {name: getattr(reembed, name) for name in names}
            reembed.get_phases = _get_phases
            reembed._resolve_rabbit_hole = _rabbit
            reembed.claim_source_for_resume = _claim
            try:
                class _Handler:
                    async def get_all_tenant_points(self, *args, **kwargs):
                        return [], None

                async def _embedder():
                    return types.SimpleNamespace(name="stub-embedder")

                ccat = types.SimpleNamespace(
                    agent_key=AGENT,
                    _id=AGENT,
                    vector_memory_handler=_Handler(),
                    chunker=None,
                )
                ccat.embedder = _embedder
                await reembed.reembed_sources(
                    ccat,
                    "declarative",
                    [types.SimpleNamespace(name=SRC, metadata={}, path="/x")],
                    caller_cat=types.SimpleNamespace(id=None),
                )
            finally:
                for name, value in saved.items():
                    setattr(reembed, name, value)

            assert claimed == [], f"a deleting source must not be claimed: {claimed}"
            assert any(
                "deleting" in message for message in h.log.infos
            ), h.log.infos
        print("PASS (h) the dispatch loop aborts on a deleting status")

    async def main():
        await test_a_same_phase_runs_once()
        await test_b_different_phases_never_overlap()
        await test_c_timeout_leaves_entry_unchanged_and_warns()
        await test_d_delete_path_takes_only_the_source_lock()
        await test_e_keys_escape_glob_metacharacters()
        await test_f_delete_source_takes_only_the_source_lock()
        await test_g_delete_cannot_interleave_with_a_pass()
        await test_h_dispatch_aborts_on_deleting()
        print("ALL PHASE-LOCK TESTS PASSED")

    asyncio.run(main())
