"""Final-wave findings (N1/N3/N4 + download lease) owned by the plugin.

Moved out of the core suite: these cases exercise the plugin's lease/heartbeat
logic and the generic Redis CRUD layer only, with fake cats. The chunker-change
case that drives the REAL ``PUT /chunking/settings`` route stays in core
(``grinning-cat-core/tests/effing/test_final_wave_findings.py``) because it
needs the full host boot (secure_client / cheshire_cat / lizard).

Standalone and import-safe: the entire body lives under
``if __name__ == "__main__":``. Run from the core checkout::

    cd /root/git/grinning-cat-core
    .venv/bin/python /root/git/plugins/cat-efficient-ingestion/test_final_wave_findings.py
"""
if __name__ == "__main__":
    import asyncio
    import time
    from types import SimpleNamespace
    from unittest.mock import patch

    from _testkit import bootstrap, new_agent_id, run, teardown

    bootstrap()
    from cat.db import crud

    from cat.core_plugins.cat_efficient_ingestion import plugin as ingestion_plugin
    from cat.core_plugins.cat_efficient_ingestion import reembed
    from cat.core_plugins.cat_efficient_ingestion.registry import (
        IngestionStatus,
        claim_source_for_resume,
        get_status,
        set_status,
        status_key,
    )
    from cat.services.memory.models import VectorMemoryType


    def _agent_id() -> str:
        return new_agent_id()


    async def _seed_master_key(agent_id: str) -> None:
        await crud.store(f"agents:{agent_id}:agent", [{"name": "x", "value": 1}])


    async def _drop(agent_id: str) -> None:
        await teardown([agent_id])


    async def _age_doc(agent_id: str, scope: str, source: str, *, age: float, **fields) -> None:
        """Rewrite a status row's ``updated_at`` (and extra fields) to a chosen age."""
        key = status_key(agent_id, scope, source)
        doc = await crud.read(key)
        if isinstance(doc, list):
            doc = doc[0]
        doc = dict(doc)
        doc["updated_at"] = time.time() - age
        doc.update(fields)
        await crud.store(key, doc)


    # ---------------------------------------------------------------------------
    # B2 — active download lease/heartbeat
    # ---------------------------------------------------------------------------


    async def test_active_download_heartbeat_prevents_stealing_but_stale_row_is_claimed():
        """A live download writer keeps the row fresh; a dead one ages out."""
        agent = _agent_id()
        source = "https://example.com/big.pdf"
        await _seed_master_key(agent)
        try:
            await set_status(agent, "agent", source, type_="url", status=IngestionStatus.DOWNLOADING)
            await _age_doc(agent, "agent", source, age=10_000)
            stale_before = (await get_status(agent, "agent", source))["updated_at"]

            task = asyncio.ensure_future(
                ingestion_plugin._heartbeat_status(agent, "agent", source, 0.01)
            )
            try:
                await asyncio.sleep(0.08)
                live = await get_status(agent, "agent", source)
                assert live["status"] == IngestionStatus.DOWNLOADING.value
                assert live["updated_at"] > stale_before, "the writer never refreshed updated_at"

                claimed = await claim_source_for_resume(
                    agent, "agent", source, stale_after=300, owner="live-writer"
                )
                assert claimed is None, "a freshly heartbeat-refreshed download was stolen"
            finally:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # Writer died: the row ages out and is reclaimed.
            await _age_doc(agent, "agent", source, age=10_000)
            reclaimed = await claim_source_for_resume(
                agent, "agent", source, stale_after=300, owner="recovery"
            )
            assert reclaimed is not None
            assert reclaimed["status"] == IngestionStatus.PROCESSING.value
        finally:
            await _drop(agent)


    # ---------------------------------------------------------------------------
    # N1 — lost NX create is detected
    # ---------------------------------------------------------------------------


    async def test_store_nx_detects_a_lost_create():
        """The second ``nx=True`` store must report the lost create as ``None``."""
        agent = _agent_id()
        key = f"agents:{agent}:nx-probe"
        try:
            assert await crud.store(key, {"v": 1}, nx=True) is not None
            assert await crud.store(key, {"v": 2}, nx=True) is None
            stored = await crud.read(key)
            assert (stored[0] if isinstance(stored, list) else stored)["v"] == 1
        finally:
            await _drop(agent)


    # ---------------------------------------------------------------------------
    # N3 — resume_created lease is finite when the claim gate is disabled
    # ---------------------------------------------------------------------------


    async def test_resume_created_row_is_reclaimed_after_its_lease():
        """With ``stale_after=0`` a crashed no-doc claim must not be refused forever."""
        agent = _agent_id()
        source = "doc.pdf"
        await _seed_master_key(agent)
        try:
            await set_status(agent, "agent", source, type_="file", status=IngestionStatus.PROCESSING)
            # A fresh row created by the no-doc path: still owned by its writer.
            await _age_doc(agent, "agent", source, age=0, resume_created=True)
            assert (
                await claim_source_for_resume(agent, "agent", source, stale_after=0, owner="second")
            ) is None

            # The creator crashed long ago: the lease expired, reclaim must succeed.
            await _age_doc(agent, "agent", source, age=10_000)
            claimed = await claim_source_for_resume(
                agent, "agent", source, stale_after=0, owner="second"
            )
            assert claimed is not None
            assert claimed["status"] == IngestionStatus.PROCESSING.value
        finally:
            await _drop(agent)


    # ---------------------------------------------------------------------------
    # N4 — batch/recovery pass starts and stops the heartbeat
    # ---------------------------------------------------------------------------


    class _RecordingPluginManager:
        def __init__(self):
            self.started = []
            self.stopped = []

        async def execute_hook(self, name, *args, caller=None):
            if name == "rabbithole_processing_heartbeat_start":
                self.started.append(args)
            elif name == "rabbithole_processing_heartbeat_stop":
                self.stopped.append(args)
            return None


    async def test_reembed_sources_heartbeats_the_claimed_source():
        """``reembed_sources`` (batch/recovery path) brackets the work with a heartbeat."""
        agent = _agent_id()
        await _seed_master_key(agent)
        plugin_manager = _RecordingPluginManager()

        async def fake_parse(*args, **kwargs):
            return "doc.pdf", []

        async def fake_record(*args, **kwargs):
            return {}

        async def fake_probe(*args, **kwargs):
            return []

        async def fake_complete(*args, **kwargs):
            return None

        async def _empty_points(*args, **kwargs):
            return [], None

        async def embedder():
            return SimpleNamespace(name="test-embedder")

        ccat = SimpleNamespace(
            _id=agent,
            agent_key=agent,
            vector_memory_handler=SimpleNamespace(
                get_all_tenant_points=_empty_points,
                add_points_to_tenant=_empty_points,
                delete_tenant_points=_empty_points,
            ),
            embedder=embedder,
            chunker=SimpleNamespace(name="test-chunker"),
            rabbit_hole=SimpleNamespace(),
            plugin_manager=plugin_manager,
        )
        source = SimpleNamespace(name="doc.pdf", path="doc.pdf", content=None, metadata={})
        try:
            with patch.object(reembed, "_run_parsing_phase", fake_parse), \
                    patch.object(reembed, "_record_phase", fake_record), \
                    patch.object(reembed, "_probe_pending", fake_probe), \
                    patch.object(reembed, "_complete_source", fake_complete):
                await reembed.reembed_sources(
                    ccat, VectorMemoryType.DECLARATIVE, [source], caller_cat=ccat
                )
            assert plugin_manager.started, "batch re-embed never started a heartbeat"
            assert plugin_manager.stopped, "batch re-embed never stopped its heartbeat"
            assert plugin_manager.started[0][0] == "doc.pdf"
        finally:
            await _drop(agent)


    _tests = [
        test_active_download_heartbeat_prevents_stealing_but_stale_row_is_claimed,
        test_store_nx_detects_a_lost_create,
        test_resume_created_row_is_reclaimed_after_its_lease,
        test_reembed_sources_heartbeats_the_claimed_source,
    ]

    run(_tests)
