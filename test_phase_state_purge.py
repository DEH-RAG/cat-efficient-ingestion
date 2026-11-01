"""Standalone, import-safe tests for the per-(source, phase) five-state
diary and its phase-scoped purge on restart.

Crash simulation only: a phase is left ``running`` with an EXPIRED lease in the
diary and the recovery path is exercised against fakes. No real crash, no real
source, no real store: the plugin loader imports EVERY ``.py`` under a plugin
folder, so the whole body lives inside ``if __name__ == "__main__":`` and
importing this file executes zero code.

Run from the host repository (the plugin is symlinked under
``cat/core_plugins``)::

    cd /root/git/grinning-cat-core
    .venv/bin/python /root/git/plugins/cat-efficient-ingestion/test_phase_state_purge.py
"""

if __name__ == "__main__":
    import asyncio
    import sys
    import time
    import types

    from cat.core_plugins.cat_efficient_ingestion import phases, reembed, registry

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

    class Point:
        def __init__(self, point_id, source="doc.txt", vector=None, **meta):
            self.id = point_id
            self.vector = {} if vector is None else vector
            self.payload = {
                "id": point_id,
                "page_content": point_id,
                "metadata": {"source": source, **meta},
            }

    class FakeHandler:
        """In-memory vector store with source-scoped deletes."""

        def __init__(self, points):
            self.points = list(points)
            self.deleted_ids = []

        async def get_all_tenant_points(self, collection_name, metadata=None, with_vectors=False):
            def matches(point):
                meta = (point.payload or {}).get("metadata", {}) or {}
                for key, value in (metadata or {}).items():
                    if meta.get(key) != value:
                        return False
                return True

            return [p for p in self.points if matches(p)], None

        async def delete_tenant_points_by_ids(self, collection_name, ids):
            wanted = set(ids)
            self.deleted_ids.extend(list(ids))
            self.points = [p for p in self.points if p.id not in wanted]

    def _ccat(handler):
        return types.SimpleNamespace(
            agent_key="phase_agent",
            _id="phase_agent",
            vector_memory_handler=handler,
        )

    def _running(started_at, owner=None, attempt=1):
        return {
            "state": registry.PHASE_STATE_PROCESSING,
            "started_at": started_at,
            "attempt": attempt,
            "owner": owner or {"pod": "other-pod", "pid": 999999, "boot": "other-boot"},
        }

    def _complete(marker="m"):
        return {
            "state": registry.PHASE_STATE_COMPLETE,
            "marker": marker,
            "settings_version": None,
            "deps": {},
        }

    class Seams:
        """Patch/restore the reembed seams used by the recovery tests."""

        def __init__(self, merged, purge_phase=None, reset_log=None):
            self.merged = merged
            self.purge_phase = purge_phase
            self.reset_log = [] if reset_log is None else reset_log
            self.saved = {}

        def __enter__(self):
            for name in ("merged_phases", "_purge_phase", "reset_phase_entries", "log"):
                self.saved[name] = getattr(reembed, name)

            async def _merged(*args, **kwargs):
                return self.merged

            async def _reset(agent_id, scope, source, phase_ids):
                self.reset_log.append(list(phase_ids))
                return {}

            reembed.merged_phases = _merged
            reembed.reset_phase_entries = _reset
            if self.purge_phase is not None:
                async def _purge(*args, **kwargs):
                    self.purge_phase.append(args[-1])
                    return True
                reembed._purge_phase = _purge
            return self

        def __exit__(self, *exc):
            for name, value in self.saved.items():
                setattr(reembed, name, value)
            return False

    async def test_a_expired_lease_purges_then_reruns():
        old = time.time() - (registry.phase_lease_seconds() + 60)
        handler = FakeHandler([
            Point("staging"),
            Point("card", vector=[1.0, 0.0], is_catalogue_card=True),
        ])
        doc = {"completed_phases": {"parsing_chunking": _running(old)}}
        reset_log = []
        merged = {"parsing_chunking": object(), "embedding": object()}
        with Seams(merged, reset_log=reset_log):
            result = await reembed._recover_abandoned_phases(
                _ccat(handler), None, "declarative", "agent", "doc.txt", None, doc
            )
        assert result is True, result
        assert handler.deleted_ids == ["staging"], handler.deleted_ids
        assert [p.id for p in handler.points] == ["card"], handler.points
        assert reset_log == [["parsing_chunking"]], reset_log
        # The reset entry is stale again -> the machine re-runs it from scratch
        # only AFTER the purge removed its products above.
        assert phases.is_phase_stale(
            phases.PHASES["parsing_chunking"], {}, None, None
        ) is True
        print("PASS (a) expired running lease purges products, resets and re-runs")

    async def test_b_fresh_lease_is_left_alone():
        fresh = _running(time.time(), owner=registry.current_phase_owner())
        handler = FakeHandler([Point("staging")])
        doc = {"completed_phases": {"parsing_chunking": fresh}}
        reset_log = []
        merged = {"parsing_chunking": object(), "embedding": object()}
        with Seams(merged, reset_log=reset_log):
            result = await reembed._recover_abandoned_phases(
                _ccat(handler), None, "declarative", "agent", "doc.txt", None, doc
            )
        assert result is None, result
        assert handler.deleted_ids == [], handler.deleted_ids
        assert reset_log == [], reset_log
        assert phases.is_phase_stale(
            phases.PHASES["parsing_chunking"],
            doc["completed_phases"], None, None,
        ) is False, "a live running lease must not be stale"
        assert phases.has_live_running_phase(doc["completed_phases"]) is True
        print("PASS (b) a running entry with a live lease is not touched")

    async def test_c_complete_only_after_products():
        log = FakeLog()
        recorded = []

        async def _record(*args, **kwargs):
            recorded.append(args)
            return {}

        saved = {name: getattr(reembed, name) for name in ("_record_phase", "log")}
        reembed.log = log
        reembed._record_phase = _record
        try:
            ok = await reembed._finish_phase(
                _ccat(FakeHandler([])), None, "agent", "doc.txt",
                "parsing_chunking", None, products=[],
            )
        finally:
            for name, value in saved.items():
                setattr(reembed, name, value)
        assert ok is True, "a body that returned done completes even with zero items"
        assert len(recorded) == 1, recorded
        assert not any("produced no products" in m for m in log.warnings), log.warnings
        print("PASS (c) zero items still completes (no count criterion)")

    async def test_d_purge_order_is_reverse():
        old = time.time() - (registry.phase_lease_seconds() + 60)
        diary = {
            "parsing_chunking": _running(old),
            "embedding": _complete(),
            "catalog_cards": _complete(),
        }
        doc = {"completed_phases": diary}
        order = []
        merged = {
            "parsing_chunking": object(),
            "embedding": object(),
            "catalog_cards": object(),
        }
        with Seams(merged, purge_phase=order):
            result = await reembed._recover_abandoned_phases(
                _ccat(FakeHandler([])), None, "declarative", "agent", "doc.txt", None, doc
            )
        assert result is True, result
        assert order == ["catalog_cards", "embedding", "parsing_chunking"], order
        print("PASS (d) purge order is reverse execution order (later phases first)")

    async def test_e_missing_purge_warns_and_leaves_running():
        old = time.time() - (registry.phase_lease_seconds() + 60)
        diary = {"graph_extraction": _running(old)}
        doc = {"completed_phases": diary}
        log = FakeLog()
        reset_log = []
        merged = {"graph_extraction": object()}
        saved = {
            name: getattr(reembed, name)
            for name in ("execute_phase_on_owner", "log")
        }

        async def _no_hook(*args, **kwargs):
            return None

        reembed.execute_phase_on_owner = _no_hook
        reembed.log = log
        try:
            with Seams(merged, reset_log=reset_log):
                result = await reembed._recover_abandoned_phases(
                    _ccat(FakeHandler([])), None, "declarative", "agent",
                    "doc.txt", None, doc,
                )
        finally:
            for name, value in saved.items():
                setattr(reembed, name, value)
        assert result is False, result
        assert reset_log == [], "the entry must stay running (no reset)"
        assert doc["completed_phases"]["graph_extraction"]["state"] == "processing"
        assert any(
            "graph_extraction" in m and "doc.txt" in m and "no purge implementation" in m
            for m in log.warnings
        ), log.warnings
        print("PASS (e) missing purge logs WARNING and leaves the entry running")

    async def test_f_embedding_purge_only_real_vectors():
        handler = FakeHandler([
            Point("staging", vector={}),
            Point("real", vector=[0.5, 0.5]),
            Point("card", vector=[1.0, 0.0], is_catalogue_card=True),
        ])
        removed = await reembed._purge_embedding_phase(
            _ccat(handler), "declarative", "doc.txt", None
        )
        assert removed == 2, removed
        assert sorted(handler.deleted_ids) == ["card", "real"], handler.deleted_ids
        remaining = [p.id for p in handler.points]
        assert remaining == ["staging"], remaining
        print("PASS (f) embedding purge deletes only real-vector points; staging survives")

    async def _main():
        await test_a_expired_lease_purges_then_reruns()
        await test_b_fresh_lease_is_left_alone()
        await test_c_complete_only_after_products()
        await test_d_purge_order_is_reverse()
        await test_e_missing_purge_warns_and_leaves_running()
        await test_f_embedding_purge_only_real_vectors()

    try:
        asyncio.run(_main())
    except AssertionError as exc:
        print(f"FAIL {exc}")
        sys.exit(1)
    print("OK phase state + purge contract (six tests)")
