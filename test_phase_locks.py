"""Standalone, import-safe tests for the ingestion phase locks.

Covers:
  (a) two concurrent workers on the same ``(source, phase)`` run the body ONCE
      (the under-lock diary re-check skips the second one);
  (b) two different phases of the same source never overlap (the per-source
      lock serialises them; enter/exit intervals are asserted disjoint);
  (c) a phase-lock timeout leaves the diary entry unchanged and logs a
      WARNING naming agent/source/phase;
  (d) the delete path takes ONLY the per-source lock (never a phase lock);
  (e) lock keys escape glob metacharacters in the source name.

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
    from cat.core_plugins.cat_efficient_ingestion import reembed, registry

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

            crud_module.distributed_lock = self.distributed_lock
            crud_module.store = _store
            crud_module.delete = _delete
            registry._read_doc = _read_doc
            registry.ingestion_canceled = _never_canceled
            reembed.get_status = _get_status
            reembed.log = self.log
            return self

        def __exit__(self, *exc):
            crud_module.distributed_lock = self.saved["distributed_lock"]
            crud_module.store = self.saved["store"]
            crud_module.delete = self.saved["delete"]
            registry._read_doc = self.saved["read_doc"]
            registry.ingestion_canceled = self.saved["canceled"]
            reembed.get_status = self.saved["get_status"]
            reembed.log = self.saved["log"]
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

    async def main():
        await test_a_same_phase_runs_once()
        await test_b_different_phases_never_overlap()
        await test_c_timeout_leaves_entry_unchanged_and_warns()
        await test_d_delete_path_takes_only_the_source_lock()
        await test_e_keys_escape_glob_metacharacters()
        print("ALL PHASE-LOCK TESTS PASSED")

    asyncio.run(main())
