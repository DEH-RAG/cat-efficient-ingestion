"""Standalone, import-safe tests for the ``reindexing`` re-ingest lifecycle.

Covers the split deletion path and the new source-level state:

  (a) the status row is NEVER absent during a full re-ingest cycle
      (``reindexing`` -> ``processing`` -> ``completed``);
  (b) ``delete_artifacts`` runs the phases in REVERSE order and is idempotent;
  (c) derived images are removed by ``delete_artifacts`` and the source file is
      NOT touched;
  (d) ``delete_file`` removes the file record and the status row LAST;
  (e) a crash during ``reindexing`` + a new boot re-runs the reset and reaches
      ``processing`` (a live owner is never touched);
  (f) the dispatch loop skips ``reindexing`` and does not start phases;
  (g) the graph delete runs BEFORE the vector wipe.

The Cat plugin loader imports EVERY ``.py`` under a plugin folder, so the whole
body lives inside ``if __name__ == "__main__":`` and importing this file
executes zero code. Run it as a plain script from the host repository::

    cd /root/git/grinning-cat-core
    .venv/bin/python /root/git/plugins/cat-efficient-ingestion/tests/test_reindexing_state.py
"""

if __name__ == "__main__":
    import asyncio
    import os
    import sys
    import time
    import types
    from contextlib import asynccontextmanager

    from cat.core_plugins.cat_efficient_ingestion import phases, registry, reembed

    AGENT = "reindex_agent"
    SCOPE = "agent"
    SRC = "doc.pdf"
    KEY = registry.status_key(AGENT, SCOPE, SRC)

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

    @asynccontextmanager
    async def _null_lock(*args, **kwargs):
        yield

    class Store:
        """In-memory registry double preserving the status-doc merge semantics."""

        def __init__(self, doc=None, events=None):
            self.docs = {}
            if doc is not None:
                self.docs[KEY] = dict(doc)
            self.events = events if events is not None else []
            self.snapshots = []

        async def get_status(self, agent_id, scope, source):
            key = registry.status_key(agent_id, scope, source)
            return self.docs.get(key)

        async def set_status(
            self,
            agent_id,
            scope,
            source,
            *,
            type_,
            status,
            chat_id=None,
            error=None,
            phase=None,
            embedder_name=None,
            chunker_name=None,
            clear_phase=False,
            extra=None,
            completed_phases=None,
        ):
            key = registry.status_key(agent_id, scope, source)
            doc = dict(self.docs.get(key) or {})
            doc.update(
                {
                    "source": source,
                    "scope": scope,
                    "chat_id": chat_id,
                    "type": type_,
                    "status": getattr(status, "value", status),
                    "error": error,
                    "updated_at": time.time(),
                }
            )
            if clear_phase:
                doc.pop("phase", None)
            if extra:
                doc.update(extra)
            if completed_phases:
                diary = dict(doc.get("completed_phases") or {})
                diary.update(completed_phases)
                doc["completed_phases"] = diary
            self.docs[key] = doc
            self.events.append(("state", doc["status"]))
            self.snapshots.append(dict(doc))
            return doc

        async def delete_status(self, agent_id, scope, source):
            key = registry.status_key(agent_id, scope, source)
            self.events.append(("delete-row", source))
            self.docs.pop(key, None)

        async def clear(self, agent_id, scope, source, phase_ids=None):
            key = registry.status_key(agent_id, scope, source)
            doc = self.docs.get(key)
            self.events.append(("clear-diary", source))
            if isinstance(doc, dict):
                doc = dict(doc)
                doc["completed_phases"] = {}
                self.docs[key] = doc
            return True

    class FakePhase:
        def __init__(self, phase_id, events, depends_on=(), kind=None):
            self.id = phase_id
            self.events = events
            self.depends_on = tuple(depends_on)
            self.kind = kind or phase_id

        async def delete(self, source, cat):
            self.events.append(("delete", self.kind))

    class FileManager:
        def __init__(self, events):
            self.events = events
            self.removed = []

        def remove_file(self, path):
            self.removed.append(path)
            self.events.append(("remove-file", path))
            return True

    class Point:
        def __init__(self, metadata):
            self.payload = {"metadata": metadata}

    class VectorHandler:
        def __init__(self, points):
            self.points = points

        async def get_all_tenant_points(self, *args, **kwargs):
            return list(self.points), None

    class FakeCat:
        def __init__(self, events, points=None):
            self.agent_key = AGENT
            self._id = AGENT
            self.file_manager = FileManager(events)
            self.vector_memory_handler = VectorHandler(points or [])

    class Seams:
        """Patch the module seams of ``registry`` + ``phases``."""

        def __init__(
            self, cat, store, phase_list, *, derived=None, log=None
        ):
            self.cat = cat
            self.store = store
            self.phase_list = phase_list
            self.derived = derived
            self.log = log or FakeLog()
            self.saved = {}

        def __enter__(self):
            saved = self.saved

            async def _phases(cat):
                return list(self.phase_list)

            for module, names in (
                (
                    phases,
                    (
                        "get_phases",
                        "get_status",
                        "set_status",
                        "_delete_status_doc",
                        "source_lock",
                        "clear_completed_phases",
                        "_delete_derived_images",
                        "log",
                    ),
                ),
                (registry, ("set_status", "clear_completed_phases", "source_lock", "current_phase_owner")),
            ):
                for name in names:
                    saved[(module.__name__, name)] = getattr(module, name)

            phases.get_phases = _phases
            phases.get_status = self.store.get_status
            phases.set_status = self.store.set_status
            phases._delete_status_doc = self.store.delete_status
            phases.source_lock = _null_lock
            phases.clear_completed_phases = self.store.clear
            phases.log = self.log

            registry.set_status = self.store.set_status
            registry.clear_completed_phases = self.store.clear
            registry.source_lock = _null_lock
            registry.current_phase_owner = lambda: {
                "pod": "test-pod",
                "pid": os.getpid(),
                "boot": registry.BOOT_EPOCH,
            }

            if self.derived is not None:
                if callable(self.derived):
                    phases._delete_derived_images = self.derived
                else:
                    async def _no_images(source, cat, scope):
                        self.store.events.append(("images", source))

                    phases._delete_derived_images = _no_images
            return self

        def __exit__(self, *exc):
            for (module_name, name), value in self.saved.items():
                module = phases if module_name == phases.__name__ else registry
                setattr(module, name, value)
            return False

    def _four_phases(events):
        # Deliberately scrambled: the topological order is
        # parsing -> embedding -> graph, so deletion is graph -> embedding ->
        # parsing.
        graph = FakePhase("graph", events, ("embedding",), kind="graph-delete")
        parsing = FakePhase("parsing_chunking", events, (), kind="vector-wipe")
        embedding = FakePhase(
            "embedding", events, ("parsing_chunking",), kind="embed"
        )
        return [embedding, graph, parsing]

    async def test_a_row_never_absent_through_full_cycle():
        events = []
        store = Store(events=events)
        cat = FakeCat(events)
        phase_list = [
            FakePhase("parsing_chunking", events),
            FakePhase("embedding", events, ("parsing_chunking",)),
        ]

        async def _delete_data():
            events.append(("data", SRC))

        with Seams(cat, store, phase_list, derived=True):
            await registry.purge_source_for_reingest(
                AGENT, SCOPE, SRC, _delete_data, cat=cat
            )
            # The phase machine then finishes the re-ingest.
            await store.set_status(
                AGENT,
                SCOPE,
                SRC,
                type_="file",
                status=registry.IngestionStatus.COMPLETED,
                clear_phase=True,
            )

        statuses = [snap.get("status") for snap in store.snapshots]
        assert statuses, store.events
        assert "reindexing" in statuses, statuses
        assert "processing" in statuses, statuses
        assert statuses[-1] == "completed", statuses
        assert all(snap is not None for snap in store.snapshots), store.snapshots
        assert "delete-row" not in [event[0] for event in store.events], store.events
        assert not any(event[0] == "delete-row" for event in store.events), store.events
        print("PASS (a) the status row is never absent across the full cycle")

    async def test_b_reverse_order_and_idempotent():
        events = []
        store = Store(events=events)
        cat = FakeCat(events)
        phase_list = _four_phases(events)

        with Seams(cat, store, phase_list, derived=True):
            first = await phases.delete_artifacts(SRC, cat, scope=SCOPE)
            second = await phases.delete_artifacts(SRC, cat, scope=SCOPE)

        assert first is True and second is True, (first, second)
        deletes = [event[1] for event in events if event[0] == "delete"]
        assert deletes == [
            "graph-delete",
            "embed",
            "vector-wipe",
            "graph-delete",
            "embed",
            "vector-wipe",
        ], deletes
        print("PASS (b) delete_artifacts is reverse-ordered and idempotent")

    async def test_c_images_removed_source_not():
        events = []
        image = "doc_img_0_abcd.jpg"
        point = Point({"source": SRC, "image": True, "image_file": image})
        cat = FakeCat(events, points=[point])
        store = Store()
        phase_list = _four_phases(events)

        with Seams(cat, store, phase_list, derived=None):
            await phases.delete_artifacts(SRC, cat, scope=SCOPE)

        assert cat.file_manager.removed, events
        joined = ["/".join(path.split(os.sep)) for path in cat.file_manager.removed]
        assert any(path.endswith(image) for path in joined), cat.file_manager.removed
        assert not any(path.endswith(SRC) for path in joined), cat.file_manager.removed
        print("PASS (c) derived images are removed and the source file is not")

    async def test_d_delete_file_removes_file_then_row():
        events = []
        store = Store(events=events)
        cat = FakeCat(events)
        phase_list = _four_phases(events)

        with Seams(cat, store, phase_list, derived=True):
            ok = await phases.delete_file(SRC, cat, scope=SCOPE)

        assert ok is True, ok
        kinds = [event[0] for event in events]
        delete_indexes = [i for i, kind in enumerate(kinds) if kind == "delete"]
        remove_index = kinds.index("remove-file")
        row_index = kinds.index("delete-row")
        assert max(delete_indexes) < remove_index, events
        assert remove_index < row_index, events
        assert row_index == len(events) - 1, events
        assert any(path.endswith(SRC) for path in cat.file_manager.removed), cat.file_manager.removed
        assert store.docs.get(KEY) is None, store.docs
        print("PASS (d) delete_file removes artifacts, then the file, then the row LAST")

    async def test_e_crash_recovery_reaches_processing():
        events = []
        cat = FakeCat(events)
        phase_list = [
            FakePhase("parsing_chunking", events),
            FakePhase("embedding", events, ("parsing_chunking",)),
        ]
        previous_boot = {
            "state": registry.PHASE_STATE_PROCESSING,
            "started_at": 0,
            "owner": {"pod": "other-pod", "pid": 1, "boot": "previous-boot"},
        }
        crashed = {
            "source": SRC,
            "scope": SCOPE,
            "type": "file",
            "status": "reindexing",
            "completed_phases": {
                "parsing_chunking": {"state": "complete", "tag": "t"},
            },
            "reindex": previous_boot,
        }
        store = Store(crashed, events=events)

        with Seams(cat, store, phase_list, derived=True):
            reset = await phases.recover_reindexing_source(SRC, cat, scope=SCOPE)

        assert reset is True, reset
        doc = store.docs.get(KEY)
        assert doc is not None and doc["status"] == "processing", doc
        assert doc.get("completed_phases") == {}, doc
        assert [e[1] for e in events if e[0] == "delete"] == [
            "embedding",
            "parsing_chunking",
        ], events

        # A LIVE owner of the CURRENT boot is never touched.
        events.clear()
        live = {
            "source": SRC,
            "scope": SCOPE,
            "type": "file",
            "status": "reindexing",
            "reindex": {
                "state": registry.PHASE_STATE_PROCESSING,
                "started_at": time.time(),
                "owner": {
                    "pod": "test-pod",
                    "pid": os.getpid(),
                    "boot": registry.BOOT_EPOCH,
                },
            },
        }
        store2 = Store(live, events=events)
        with Seams(cat, store2, phase_list, derived=True):
            reset2 = await phases.recover_reindexing_source(SRC, cat, scope=SCOPE)
        assert reset2 is False, reset2
        assert not [e for e in events if e[0] == "delete"], events
        assert store2.docs[KEY]["status"] == "reindexing", store2.docs[KEY]
        print("PASS (e) a dead reindexing row is reset to processing; a live one is not")

    async def test_f_dispatch_skips_reindexing():
        started = []

        async def _boom_phase(*args, **kwargs):
            started.append("phase")
            raise AssertionError("a reindexing row must not start a phase body")

        async def _boom_claim(*args, **kwargs):
            started.append("claim")
            raise AssertionError("a reindexing row must not be claimed")

        class Handler:
            async def get_all_tenant_points(self, *args, **kwargs):
                return [], None

        async def _embedder():
            return types.SimpleNamespace(name="embedder")

        ccat = types.SimpleNamespace(
            agent_key=AGENT,
            _id=AGENT,
            vector_memory_handler=Handler(),
            chunker=types.SimpleNamespace(name="chunker"),
            embedder=_embedder,
        )
        source = types.SimpleNamespace(name=SRC, path=SRC, metadata={})

        saved = {
            name: getattr(reembed, name)
            for name in (
                "get_phases",
                "get_status",
                "_resolve_rabbit_hole",
                "claim_source_for_resume",
                "_run_parsing_phase",
                "_embed_phase",
                "_start_heartbeat",
                "_stop_heartbeat",
                "ingestion_canceled",
            )
        }

        async def _phases(cat):
            return []

        async def _status(*args, **kwargs):
            return {"source": SRC, "scope": SCOPE, "status": "reindexing"}

        async def _rabbit(*args, **kwargs):
            return object()

        async def _value(*args, **kwargs):
            return None

        reembed.get_phases = _phases
        reembed.get_status = _status
        reembed._resolve_rabbit_hole = _rabbit
        reembed.claim_source_for_resume = _boom_claim
        reembed._run_parsing_phase = _boom_phase
        reembed._embed_phase = _boom_phase
        reembed._start_heartbeat = _value
        reembed._stop_heartbeat = _value
        reembed.ingestion_canceled = _value
        try:
            await reembed.reembed_sources(
                ccat, "declarative", [source], caller_cat=ccat
            )
        finally:
            for name, value in saved.items():
                setattr(reembed, name, value)

        assert started == [], started
        print("PASS (f) the dispatch loop skips reindexing and starts no phase")

    async def test_g_graph_delete_before_vector_wipe():
        events = []
        cat = FakeCat(events)
        store = Store()
        phase_list = _four_phases(events)

        with Seams(cat, store, phase_list, derived=True):
            await phases.delete_artifacts(SRC, cat, scope=SCOPE)

        deletes = [event[1] for event in events if event[0] == "delete"]
        assert deletes == ["graph-delete", "embed", "vector-wipe"], deletes
        assert deletes.index("graph-delete") < deletes.index("vector-wipe"), deletes
        print("PASS (g) the graph delete runs BEFORE the vector wipe")

    async def _main():
        await test_a_row_never_absent_through_full_cycle()
        await test_b_reverse_order_and_idempotent()
        await test_c_images_removed_source_not()
        await test_d_delete_file_removes_file_then_row()
        await test_e_crash_recovery_reaches_processing()
        await test_f_dispatch_skips_reindexing()
        await test_g_graph_delete_before_vector_wipe()
        print("OK reindexing lifecycle (seven tests)")

    try:
        asyncio.run(_main())
    except AssertionError:
        import traceback

        traceback.print_exc()
        sys.exit(1)
