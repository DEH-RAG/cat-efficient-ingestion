#!/usr/bin/env python3
"""Standalone, import-safe tests for the state/data atomicity invariant.

Every path that deletes a phase's products MUST reset that phase's state, the
data deletion MUST happen first, and a failing ``delete()`` MUST leave the entry
``processing`` (never ``complete``) with a WARNING so the next pass retries.

Crash simulation only: in-memory store, fake phases, no real Redis, no real
source. The Cat plugin loader imports EVERY ``.py`` under the plugin folder, so
the whole body lives inside ``if __name__ == "__main__":`` and importing this
file executes zero code.

Run from the host repository (the plugin is symlinked under
``cat/core_plugins``)::

    cd /root/git/grinning-cat-core
    .venv/bin/python /root/git/plugins/cat-efficient-ingestion/test_phase_atomicity.py
"""

if __name__ == "__main__":
    import asyncio
    import sys
    import time
    import types
    from contextlib import asynccontextmanager

    from cat.core_plugins.cat_efficient_ingestion import phases, registry, reembed

    AGENT = "atomic_agent"
    SCOPE = "agent"
    SRC = "doc.txt"
    KEY = registry.status_key(AGENT, SCOPE, SRC)

    GRAPH_PHASES = (
        phases.PhaseSpec(
            "graph_spacy_extraction", None, ("parsing_chunking", "embedding")
        ),
        phases.PhaseSpec(
            "graph_llm_extraction", None, ("parsing_chunking", "embedding")
        ),
        phases.PhaseSpec(
            "catalog_cards", "catalog", ("embedding",)
        ),
    )

    class FakeLog:
        def __init__(self):
            self.warnings = []
            self.errors = []

        def info(self, msg):
            pass

        def warning(self, msg):
            self.warnings.append(str(msg))

        def error(self, msg):
            self.errors.append(str(msg))

        def debug(self, msg):
            pass

    class MemoryStore:
        """In-memory ``cat.db.crud`` double, recording every state write."""

        def __init__(self):
            self.docs = {}
            self.events = []

        async def read(self, key):
            self.events.append(("read", key))
            return self.docs.get(key)

        async def store(self, key, doc):
            self.events.append(("state", key))
            self.docs[key] = doc

        async def delete(self, key):
            self.events.append(("delete-row", key))
            self.docs.pop(key, None)

    @asynccontextmanager
    async def _null_lock(*args, **kwargs):
        yield

    def _old(started_at=None):
        return time.time() - (registry.phase_lease_seconds() + 120)

    def _processing(owner=None):
        return {
            "state": registry.PHASE_STATE_PROCESSING,
            "started_at": _old(),
            "owner": owner,
            "reason": None,
            "tag": "t",
            "attempt": 1,
        }

    def _complete():
        return {
            "state": registry.PHASE_STATE_COMPLETE,
            "started_at": _old(),
            "owner": None,
            "reason": None,
            "tag": "t",
            "attempt": 1,
        }

    def _patch_registry(store):
        """Point the registry at the in-memory store; return a restore callable."""
        saved = {
            "crud": registry.crud,
            "_list_status_keys": registry._list_status_keys,
            "source_lock": registry.source_lock,
        }

        async def _keys(agent_id):
            return [k for k in store.docs if ":ingestion:" in k]

        registry.crud = store
        registry._list_status_keys = _keys
        registry.source_lock = _null_lock
        # A real agent has its master key, otherwise ``set_status`` treats the
        # agent as deleted and every state write becomes a no-op.
        store.docs.setdefault(f"agents:{AGENT}:agent", {})

        def restore():
            for name, value in saved.items():
                setattr(registry, name, value)

        return restore

    def _patch_reembed(store, merged, delete_phase, log):
        saved = {
            name: getattr(reembed, name)
            for name in (
                "merged_phases",
                "_delete_phase",
                "record_phase_purge_failed",
                "log",
            )
        }

        async def _merged(ccat, cat):
            return dict(merged)

        reembed.merged_phases = _merged
        reembed._delete_phase = delete_phase
        reembed.record_phase_purge_failed = (
            lambda *a, **k: registry.record_phase_purge_failed(*a, **k)
        )
        reembed.log = log

        def restore():
            for name, value in saved.items():
                setattr(reembed, name, value)

        return restore

    async def _merged_ids():
        """The MERGED phase list (built-ins + registered), never a literal pair."""
        saved = phases._registered_specs

        async def _registered(cat):
            return [(spec, object()) for spec in GRAPH_PHASES]

        phases._registered_specs = _registered
        try:
            merged = await phases.merged_phases(object(), object())
        finally:
            phases._registered_specs = saved
        return list(merged)

    async def _seed(store, phase_ids):
        store.docs[KEY] = {
            "source": SRC,
            "scope": SCOPE,
            "type": "file",
            "status": "completed",
            "completed_phases": {phase: _complete() for phase in phase_ids},
        }

    def _diary(store):
        doc = store.docs.get(KEY) or {}
        return doc.get("completed_phases") or {}

    async def test_a_cli_wipe_clears_every_merged_phase():
        store = MemoryStore()
        restore = _patch_registry(store)
        try:
            merged = await _merged_ids()
            await _seed(store, merged)
            assert set(_diary(store)) == set(merged), _diary(store)
            cleared = await registry.clear_completed_phases(AGENT, SCOPE, SRC)
            assert cleared is True, cleared
            survivors = set(_diary(store))
            # Assert against the MERGED list: no built-in OR registered phase
            # may survive the wipe (a hardcoded pair would fail here).
            assert survivors == set(), f"survivors after CLI wipe: {survivors}"
            assert not (survivors & set(merged)), survivors & set(merged)
        finally:
            restore()
        print("PASS (a) CLI wipe clears EVERY merged phase (not a hardcoded pair)")

    async def test_b_failing_delete_keeps_processing_and_retries():
        store = MemoryStore()
        restore_registry = _patch_registry(store)
        log = FakeLog()
        attempts = []
        fail_once = {"graph_spacy_extraction"}

        async def _delete_phase(ccat, cat, collection, scope, source, chat_id, phase):
            attempts.append(phase)
            if phase in fail_once:
                fail_once.discard(phase)
                raise RuntimeError("delete boom")
            return True

        merged = {
            "graph_spacy_extraction": GRAPH_PHASES[0],
            "graph_llm_extraction": GRAPH_PHASES[1],
        }
        ccat = types.SimpleNamespace(agent_key=AGENT, _id=AGENT)
        store.docs[KEY] = {
            "source": SRC,
            "scope": SCOPE,
            "type": "file",
            "status": "processing",
            "completed_phases": {
                "graph_spacy_extraction": _processing(),
                "graph_llm_extraction": _complete(),
            },
        }
        restore_reembed = _patch_reembed(store, merged, _delete_phase, log)
        try:
            first = await reembed._recover_abandoned_phases(
                ccat, None, "declarative", SCOPE, SRC, None, store.docs[KEY]
            )
            # A failed delete stops the source and leaves the phase retryable.
            assert first is False, first
            entry = _diary(store)["graph_spacy_extraction"]
            assert entry["state"] == registry.PHASE_STATE_PROCESSING, entry
            assert entry["state"] != registry.PHASE_STATE_COMPLETE
            assert registry.should_abandon(entry) is True, entry
            assert any(
                "graph_spacy_extraction" in m and SRC in m and "raised" in m
                for m in log.warnings
            ), log.warnings
            # The later complete phase WAS deleted before the failure: its
            # state must not survive the deletion of its data.
            assert "graph_llm_extraction" not in _diary(store), _diary(store)

            # NEXT PASS: the retryable entry is abandoned again and now succeeds.
            second = await reembed._recover_abandoned_phases(
                ccat, None, "declarative", SCOPE, SRC, None,
                await registry.get_status(AGENT, SCOPE, SRC),
            )
            assert second is True, second
            assert "graph_spacy_extraction" not in _diary(store), _diary(store)
            assert "graph_llm_extraction" not in _diary(store), _diary(store)
        finally:
            restore_reembed()
            restore_registry()
        print("PASS (b) failing delete() keeps processing + WARNING, retried next pass")

    async def _path_cli(store):
        merged = await _merged_ids()
        deleted = set(merged)
        await _seed(store, merged)

        async def _delete_data():
            store.events.append(("data", SRC))

        await registry.purge_source_for_reingest(
            AGENT, SCOPE, SRC, _delete_data
        )
        return deleted, _diary(store)

    async def _path_recovery(store):
        merged = {
            "parsing_chunking": phases.PHASES["parsing_chunking"],
            "embedding": phases.PHASES["embedding"],
            "graph_spacy_extraction": GRAPH_PHASES[0],
        }
        ccat = types.SimpleNamespace(agent_key=AGENT, _id=AGENT)
        store.docs[KEY] = {
            "source": SRC,
            "scope": SCOPE,
            "type": "file",
            "status": "processing",
            "completed_phases": {
                "parsing_chunking": _processing(),
                "embedding": _complete(),
                "graph_spacy_extraction": _complete(),
            },
        }
        deleted = set()

        async def _delete_phase(ccat, cat, collection, scope, source, chat_id, phase):
            deleted.add(phase)
            store.events.append(("data", phase))
            return True

        restore = _patch_reembed(store, merged, _delete_phase, FakeLog())
        try:
            await reembed._recover_abandoned_phases(
                ccat, None, "declarative", SCOPE, SRC, None, store.docs[KEY]
            )
        finally:
            restore()
        return deleted, _diary(store)

    async def _path_delete_status(store):
        merged = await _merged_ids()
        await _seed(store, merged)
        deleted = set(merged)

        async def _delete_data():
            store.events.append(("data", SRC))

        await _delete_data()
        await registry.delete_status(AGENT, SCOPE, SRC)
        return deleted, _diary(store)

    async def test_c_no_state_survives_its_deleted_data():
        store = MemoryStore()
        restore = _patch_registry(store)
        try:
            for name, path in (
                ("cli-reingest", _path_cli),
                ("recovery-purge", _path_recovery),
                ("delete-status", _path_delete_status),
            ):
                store.docs.clear()
                deleted, diary = await path(store)
                survivors = set(diary) & deleted
                assert not survivors, f"{name}: state survived its deleted data: {survivors}"
            # The graph invalidation resets ONLY the graph phases, for every
            # status doc of the agent (all scopes) after their data is gone.
            graph_ids = [spec.id for spec in GRAPH_PHASES]
            assert graph_ids, "graph phases must be registered"
        finally:
            restore()
        print("PASS (c) property: no state entry survives the deletion of its data")

    async def test_d_state_never_written_before_its_data():
        store = MemoryStore()
        restore = _patch_registry(store)
        try:
            merged = await _merged_ids()
            await _seed(store, merged)
            store.events.clear()

            async def _delete_data():
                store.events.append(("data", SRC))

            await registry.purge_source_for_reingest(
                AGENT, SCOPE, SRC, _delete_data
            )
            kinds = [event[0] for event in store.events]
            first_data = kinds.index("data")
            first_state = min(
                (i for i, kind in enumerate(kinds) if kind in ("state", "delete-row")),
                default=len(kinds),
            )
            # DATA FIRST: every state write (diary clear + row delete) happens
            # after the wipe, never before it.
            assert first_data < first_state, store.events
            assert kinds.count("data") == 1, store.events
        finally:
            restore()
        print("PASS (d) reingest writes state only after the data wipe")

    async def test_e_graph_invalidation_resets_every_graph_phase():
        store = MemoryStore()
        restore = _patch_registry(store)
        try:
            merged = await _merged_ids()
            await _seed(store, merged)
            graph_ids = [spec.id for spec in GRAPH_PHASES]
            changed = await registry.reset_phase_entries_for_agent(AGENT, graph_ids)
            assert changed == 1, changed
            survivors = set(_diary(store))
            # Graph phases are gone; built-ins are untouched (their data was
            # not deleted by the graph wipe).
            assert not (survivors & set(graph_ids)), survivors & set(graph_ids)
            assert {"parsing_chunking", "embedding"} <= survivors, survivors
        finally:
            restore()
        print("PASS (e) --graph invalidation resets only the graph phases")

    class _FakeVectorHandler:
        """Records the data wipe and touches no real vector store."""

        def __init__(self, store):
            self.store = store

        async def delete_tenant_points(self, collection, metadata=None):
            self.store.events.append(("data", (metadata or {}).get("source")))

    async def test_f_orphaned_source_cleanup_resets_the_diary():
        store = MemoryStore()
        restore = _patch_registry(store)
        try:
            merged = await _merged_ids()
            await _seed(store, merged)
            store.events.clear()
            ccat = types.SimpleNamespace(
                agent_key=AGENT,
                _id=AGENT,
                vector_memory_handler=_FakeVectorHandler(store),
            )
            await reembed._cleanup_orphaned_source(
                ccat, "declarative", SRC, None, SCOPE
            )
            assert _diary(store) == {}, _diary(store)
            assert store.docs[KEY]["status"] == "completed", store.docs[KEY]
            kinds = [event[0] for event in store.events]
            assert kinds.index("data") < min(
                i for i, kind in enumerate(kinds) if kind == "state"
            ), store.events
        finally:
            restore()
        print("PASS (f) orphaned-source cleanup wipes data BEFORE resetting the diary")


    async def _main():
        await _merged_ids()
        await test_a_cli_wipe_clears_every_merged_phase()
        await test_b_failing_delete_keeps_processing_and_retries()
        await test_c_no_state_survives_its_deleted_data()
        await test_d_state_never_written_before_its_data()
        await test_e_graph_invalidation_resets_every_graph_phase()
        await test_f_orphaned_source_cleanup_resets_the_diary()

    try:
        asyncio.run(_main())
    except AssertionError as exc:
        print(f"FAIL {exc}")
        sys.exit(1)
    print("OK phase state + data atomicity (six tests)")
