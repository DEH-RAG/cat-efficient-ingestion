"""Round-2 heartbeat/lease findings (NN1-NN3) owned by the plugin.

Moved out of the core suite: the ``resume_created`` URL lease, the hook-level
download heartbeat bound, heartbeat replacement and tick resilience all run
against the plugin with fake cats and standard Redis CRUD. The concurrent
two-agent case (BN1/NN4) stays in core because it needs the real ``lizard``.

Standalone and import-safe: the entire body lives under
``if __name__ == "__main__":``. Run from the core checkout::

    cd /root/git/grinning-cat-core
    .venv/bin/python /root/git/plugins/cat-efficient-ingestion/test_round2_findings.py
"""
if __name__ == "__main__":
    import asyncio
    import time
    from unittest.mock import patch

    from _testkit import bootstrap, new_agent_id, run, teardown

    bootstrap()
    from cat.db import crud

    from cat.core_plugins.cat_efficient_ingestion import plugin as ingestion_plugin
    from cat.core_plugins.cat_efficient_ingestion.registry import (
        IngestionStatus,
        claim_source_for_resume,
        get_status,
        set_status,
        status_key,
    )


    def _agent_id() -> str:
        return new_agent_id()


    async def _seed_master_key(agent_id: str) -> None:
        await crud.store(f"agents:{agent_id}:agent", [{"name": "x", "value": 1}])


    async def _drop(agent_id: str) -> None:
        await teardown([agent_id])


    async def _age_doc(agent_id: str, scope: str, source: str, *, age: float, **fields) -> None:
        key = status_key(agent_id, scope, source)
        doc = await crud.read(key)
        if isinstance(doc, list):
            doc = doc[0]
        doc = dict(doc)
        doc["updated_at"] = time.time() - age
        doc.update(fields)
        await crud.store(key, doc)


    # ---------------------------------------------------------------------------
    # NN1 — the resume_created lease covers URL rows that flip out of PROCESSING
    # ---------------------------------------------------------------------------


    async def test_resume_created_lease_covers_a_url_row_out_of_processing():
        """A live URL row (DOWNLOADING) created by the no-doc path is not stolen."""
        agent = _agent_id()
        source = "https://example.com/doc.pdf"
        await _seed_master_key(agent)
        try:
            await set_status(agent, "agent", source, type_="url", status=IngestionStatus.PROCESSING)
            await _age_doc(agent, "agent", source, age=0, resume_created=True)

            # The phase machine flips the row to DOWNLOADING (marker is preserved).
            await set_status(agent, "agent", source, type_="url", status=IngestionStatus.DOWNLOADING)
            assert (await get_status(agent, "agent", source))["resume_created"] is True

            # Live writer, claim gate disabled: the lease must still refuse.
            assert (
                await claim_source_for_resume(agent, "agent", source, stale_after=0, owner="second")
            ) is None

            # The writer crashed long ago: the lease expired, reclaim succeeds.
            await _age_doc(agent, "agent", source, age=10_000)
            claimed = await claim_source_for_resume(
                agent, "agent", source, stale_after=0, owner="second"
            )
            assert claimed is not None
            assert claimed["status"] == IngestionStatus.PROCESSING.value
        finally:
            await _drop(agent)


    # ---------------------------------------------------------------------------
    # NN2 — the download heartbeat bound at the HOOK level
    # ---------------------------------------------------------------------------


    class _HookCat:
        """Minimal agent-scoped hook caller (no ``id`` -> agent scope)."""

        def __init__(self, agent_key: str):
            self.agent_key = agent_key


    async def test_download_hook_heartbeat_stops_at_its_bound():
        """``rabbithole_url_downloading`` bounds the heartbeat so a dead download ages out."""
        agent = _agent_id()
        source = "https://example.com/big.pdf"
        await _seed_master_key(agent)
        cat = _HookCat(agent)
        try:
            with patch.object(ingestion_plugin, "_heartbeat_interval", lambda: 0.01), \
                    patch.object(ingestion_plugin, "_download_heartbeat_max_seconds", lambda: 0.05):
                await ingestion_plugin.rabbithole_url_downloading.function(source, source, cat)
                key = ingestion_plugin._heartbeat_key(agent, "agent", source)
                task = ingestion_plugin._heartbeat_tasks[key]

                # While inside the bound the task is alive and refreshing the row.
                await asyncio.sleep(0.02)
                assert not task.done()
                first = (await get_status(agent, "agent", source))["updated_at"]

                # Past the bound the task returns by itself (no completed hook needed).
                await asyncio.wait_for(task, timeout=1.0)
                stopped_at = (await get_status(agent, "agent", source))["updated_at"]
                await asyncio.sleep(0.05)
                after = (await get_status(agent, "agent", source))["updated_at"]
                assert after == stopped_at, "the bounded download heartbeat kept refreshing after its bound"
                assert stopped_at >= first

                # The row now ages out and becomes resumable.
                await _age_doc(agent, "agent", source, age=10_000)
                claimed = await claim_source_for_resume(
                    agent, "agent", source, stale_after=300, owner="recovery"
                )
                assert claimed is not None
        finally:
            ingestion_plugin._heartbeat_tasks.pop(
                ingestion_plugin._heartbeat_key(agent, "agent", source), None
            )
            await _drop(agent)


    # ---------------------------------------------------------------------------
    # NN3 — replacing a heartbeat awaits the cancelled task; a raising tick is safe
    # ---------------------------------------------------------------------------


    async def test_start_heartbeat_awaits_the_replaced_task():
        """The old heartbeat must be fully unwound before the replacement is installed."""
        agent = _agent_id()
        source = "doc.pdf"
        state = {"unwound": False}

        async def fake_heartbeat(agent_id, scope, src, interval, *, max_seconds=None):
            try:
                await asyncio.sleep(3600)
            finally:
                state["unwound"] = True

        try:
            with patch.object(ingestion_plugin, "_heartbeat_status", fake_heartbeat):
                await ingestion_plugin._start_heartbeat(agent, "agent", source, 0.01)
                first = ingestion_plugin._heartbeat_tasks[
                    ingestion_plugin._heartbeat_key(agent, "agent", source)
                ]
                # Let the first task actually start so its ``finally`` unwind is real.
                await asyncio.sleep(0)
                await ingestion_plugin._start_heartbeat(agent, "agent", source, 0.01)

                assert first.cancelled() or first.done()
                assert state["unwound"] is True, "the replaced heartbeat was not awaited"
        finally:
            await ingestion_plugin._cancel_heartbeat(agent, "agent", source)


    async def test_heartbeat_tick_never_raises():
        """A transient store error in one tick must not kill the heartbeat."""
        agent = _agent_id()
        source = "doc.pdf"
        calls = {"n": 0}

        async def flaky_get_status(*args, **kwargs):
            calls["n"] += 1
            raise RuntimeError("redis blip")

        with patch.object(ingestion_plugin, "get_status", flaky_get_status):
            task = asyncio.ensure_future(
                ingestion_plugin._heartbeat_status(agent, "agent", source, 0.01)
            )
            try:
                await asyncio.sleep(0.06)
                assert not task.done(), "a raising tick killed the heartbeat"
                assert calls["n"] >= 2, "the heartbeat did not retry after a failed tick"
            finally:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass


    run([
        test_resume_created_lease_covers_a_url_row_out_of_processing,
        test_download_hook_heartbeat_stops_at_its_bound,
        test_start_heartbeat_awaits_the_replaced_task,
        test_heartbeat_tick_never_raises,
    ])
