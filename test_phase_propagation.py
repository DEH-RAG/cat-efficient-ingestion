"""Standalone, import-safe tests for dependency error/skip propagation.

A phase whose dependency is ``error`` becomes ``error`` with a reason naming
that dependency; a phase whose dependency is ``skipped`` becomes ``skipped``.
Neither ever runs its body, and neither ever becomes ``complete`` behind a
failed dependency. The source status is the existing DERIVED status (any
``error`` -> source ``error``; all terminal-non-error -> ``complete``).

The plugin loader imports EVERY ``.py`` under a plugin folder, so the whole
body lives inside ``if __name__ == "__main__":`` and importing this file
executes zero code.

Run from the host repository (the plugin is symlinked under
``cat/core_plugins``)::

    cd /root/git/grinning-cat-core
    .venv/bin/python /root/git/plugins/cat-efficient-ingestion/test_phase_propagation.py
"""

if __name__ == "__main__":
    import asyncio
    import sys
    import types

    from cat.core_plugins.cat_efficient_ingestion import phases, registry, reembed

    ERROR = registry.PHASE_STATE_ERROR
    SKIPPED = registry.PHASE_STATE_SKIPPED
    COMPLETE = registry.PHASE_STATE_COMPLETE

    class FakeLog:
        def info(self, msg):
            pass

        def warning(self, msg):
            pass

        def error(self, msg):
            pass

        def debug(self, msg):
            pass

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
        return types.SimpleNamespace(agent_key="propagation_agent", _id="propagation_agent")

    def _spec(phase_id, depends_on=()):
        return types.SimpleNamespace(id=phase_id, depends_on=tuple(depends_on))

    def _reset(store, diary=None, status="processing"):
        store.docs.clear()
        store.docs[registry.status_key("propagation_agent", "agent", "doc.txt")] = {
            "source": "doc.txt",
            "scope": "agent",
            "type_": "file",
            "status": status,
            "completed_phases": diary or {},
        }

    async def _diary(store):
        doc = await reembed.get_status("propagation_agent", "agent", "doc.txt")
        return registry.get_completed_phases(doc)

    async def _seed(store, phase_id, state, reason=None):
        if state == ERROR:
            await registry.record_phase_error(
                "propagation_agent", "agent", "doc.txt", phase_id, reason or "boom"
            )
        elif state == SKIPPED:
            await registry.record_phase_skipped(
                "propagation_agent", "agent", "doc.txt", phase_id, reason or "n/a"
            )
        elif state == COMPLETE:
            await registry.record_phase_completed(
                "propagation_agent", "agent", "doc.txt", phase_id, marker="m",
                tag=f"{phase_id}:tag",
            )

    def _patch_dispatch(specs):
        """Patch only the spec lookup the real dispatch gate uses."""
        async def _merged(*args, **kwargs):
            return {spec.id: spec for spec in specs}

        async def _set_status(*args, **kwargs):
            return None

        saved = (reembed.merged_phases, reembed._set_status, reembed.log)
        reembed.merged_phases = _merged
        reembed._set_status = _set_status
        reembed.log = FakeLog()
        return saved

    def _restore(saved):
        reembed.merged_phases, reembed._set_status, reembed.log = saved

    async def _propagate(specs, pending_order, bodies):
        """Mirror the dispatch loop: propagate before any body, else run it.

        Uses the REAL gate (``reembed._apply_dependency_outcome``) and the real
        diary writes, so the assertion "no body ran behind a failed dependency"
        covers the actual dispatch decision. Returns the phases whose body ran.
        """
        pending = list(pending_order)
        guard = 0
        while pending and guard < 100:
            guard += 1
            diary = await _diary(store)
            propagated = False
            for phase_id in list(pending):
                if await reembed._apply_dependency_outcome(
                    _ccat(), None, "agent", "doc.txt", phase_id, diary, None, "file"
                ):
                    pending.remove(phase_id)
                    propagated = True
                    break
            if propagated:
                continue
            phase_id = pending.pop(0)
            bodies.append(phase_id)
            await _seed(store, phase_id, COMPLETE)
        return bodies

    async def test_a_chain_error_propagates():
        _reset(store)
        specs = [
            _spec("A"),
            _spec("B", ("A",)),
            _spec("C", ("B",)),
        ]
        await _seed(store, "A", ERROR, "A exploded")
        saved = _patch_dispatch(specs)
        bodies = []
        try:
            await _propagate(specs, ["B", "C"], bodies)
            diary = await _diary(store)
        finally:
            _restore(saved)
        assert bodies == [], f"no body may run behind an errored dependency: {bodies}"
        assert diary["B"]["state"] == ERROR, diary.get("B")
        assert diary["B"]["reason"] == "dependency A is error", diary.get("B")
        assert diary["C"]["state"] == ERROR, diary.get("C")
        assert diary["C"]["reason"] == "dependency B is error", diary.get("C")
        print("PASS (a) A(error) -> B -> C both error, each naming its dependency")

    async def test_b_skip_propagates():
        _reset(store)
        specs = [_spec("A"), _spec("B", ("A",))]
        await _seed(store, "A", SKIPPED, "no handler")
        saved = _patch_dispatch(specs)
        bodies = []
        try:
            await _propagate(specs, ["B"], bodies)
            diary = await _diary(store)
        finally:
            _restore(saved)
        assert bodies == [], f"no body may run behind a skipped dependency: {bodies}"
        assert diary["B"]["state"] == SKIPPED, diary.get("B")
        assert diary["B"]["reason"] == "dependency A is skipped", diary.get("B")
        print("PASS (b) A(skipped) -> B skipped with the reason")

    async def test_c_never_complete_behind_error():
        _reset(store)
        specs = [_spec("A"), _spec("B", ("A",))]
        await _seed(store, "A", ERROR, "A exploded")
        saved = _patch_dispatch(specs)
        bodies = []
        try:
            await _propagate(specs, ["B"], bodies)
            diary = await _diary(store)
        finally:
            _restore(saved)
        assert bodies == [], "the dependent body must never run"
        assert diary["B"]["state"] != COMPLETE, diary.get("B")
        assert diary["B"]["state"] == ERROR, diary.get("B")
        print("PASS (c) dependent never completes and its body is never called")

    async def test_d_source_status_derived():
        saved = _patch_dispatch([])
        try:
            # (a) any error -> source error
            _reset(store)
            await _seed(store, "A", ERROR, "A exploded")
            await _seed(store, "B", ERROR, "dependency A is error")
            error_derived = registry.derive_source_status(
                await _diary(store), ["A", "B"]
            )
            # (b) all terminal-non-error -> source complete
            _reset(store)
            await _seed(store, "A", SKIPPED, "no handler")
            await _seed(store, "B", SKIPPED, "dependency A is skipped")
            complete_derived = registry.derive_source_status(
                await _diary(store), ["A", "B"]
            )
        finally:
            _restore(saved)
        assert error_derived == ERROR, error_derived
        assert complete_derived == COMPLETE, complete_derived
        print("PASS (d) derived source status error for (a), complete for (b)")

    async def test_e_diamond_deterministic_first_offender():
        # Pure function: FIRST offending dependency in DECLARED order wins.
        diamond_bc = _spec("D", ("B", "C"))
        diamond_cb = _spec("D", ("C", "B"))
        diary = {
            "B": {"state": ERROR, "reason": "B exploded"},
            "C": {"state": ERROR, "reason": "C exploded"},
        }
        assert phases.dependency_outcome(diamond_bc, diary) == (
            ERROR, "dependency B is error"
        )
        assert phases.dependency_outcome(diamond_cb, diary) == (
            ERROR, "dependency C is error"
        )

        # Full dispatch walk on A -> B, A -> C, B+C -> D.
        _reset(store)
        specs = [
            _spec("A"),
            _spec("B", ("A",)),
            _spec("C", ("A",)),
            _spec("D", ("B", "C")),
        ]
        await _seed(store, "A", ERROR, "A exploded")
        saved = _patch_dispatch(specs)
        bodies = []
        try:
            await _propagate(specs, ["B", "C", "D"], bodies)
            diary = await _diary(store)
        finally:
            _restore(saved)
        assert bodies == [], f"no body may run in the diamond: {bodies}"
        assert diary["B"]["reason"] == "dependency A is error", diary.get("B")
        assert diary["C"]["reason"] == "dependency A is error", diary.get("C")
        assert diary["D"]["state"] == ERROR, diary.get("D")
        assert diary["D"]["reason"] == "dependency B is error", diary.get("D")
        print("PASS (e) diamond propagates and reports the first declared offender")

    async def _main():
        await test_a_chain_error_propagates()
        await test_b_skip_propagates()
        await test_c_never_complete_behind_error()
        await test_d_source_status_derived()
        await test_e_diamond_deterministic_first_offender()

    store = MemoryStore()
    with store:
        try:
            asyncio.run(_main())
        except AssertionError as exc:
            print(f"FAIL {exc}")
            sys.exit(1)
