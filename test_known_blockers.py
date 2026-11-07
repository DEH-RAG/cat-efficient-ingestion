"""Registry blockers (B1/B3) owned by the plugin.

Moved out of the core suite: B1 (status listing enumerates the right keys), B3
(a stale ``downloaded`` source is resumable) and B3b (a fresh ``downloading``
source is not stolen) exercise the registry + Redis CRUD only, with no host
boot.

The marker cases B2/B2b (real ``CheshireCat`` settings) and the veto case B4
(real dispatcher via ``StrayCat``) stay in core
(``grinning-cat-core/tests/effing/test_known_blockers.py``).

Standalone and import-safe: the entire body lives under
``if __name__ == "__main__":``. Run from the core checkout::

    cd /root/git/grinning-cat-core
    .venv/bin/python /root/git/plugins/cat-efficient-ingestion/test_known_blockers.py
"""
if __name__ == "__main__":
    import time

    from _testkit import bootstrap, new_agent_id, run, teardown

    bootstrap()
    from cat.db import crud

    from cat.core_plugins.cat_efficient_ingestion.registry import (
        IngestionStatus,
        claim_source_for_resume,
        get_status,
        list_statuses,
        set_status,
        status_key,
    )


    def _agent_id() -> str:
        return new_agent_id()


    async def _drop(agent_id: str) -> None:
        await teardown([agent_id])


    async def _seed_master_key(agent_id: str) -> None:
        await crud.store(f"agents:{agent_id}:agent", [{"name": "x", "value": 1}])


    async def test_b1_status_listing_returns_the_seeded_rows():
        """B1: ``list_statuses`` must return the agent's own status docs."""
        agent = _agent_id()
        try:
            await _seed_master_key(agent)
            await set_status(agent, "agent", "doc.pdf", type_="file", status=IngestionStatus.COMPLETED)

            docs = await list_statuses(agent)

            assert [d["source"] for d in docs] == ["doc.pdf"], (
                "B1: list_statuses returned no rows although a status doc exists "
                "(get_agents_main_keys yields agent ids, not full keys)"
            )
        finally:
            await _drop(agent)


    async def test_b3_downloaded_source_can_be_resumed():
        """B3: a stale ``downloaded`` source is claimable for a retry.

        The claim guard accepts ``downloading``/``downloaded`` too, so the resume
        sweep reclaims a download that crashed before processing instead of leaving
        the source a permanent dead end.
        """
        agent = _agent_id()
        source = "https://example.com/stuck.pdf"
        try:
            await _seed_master_key(agent)
            await set_status(
                agent, "agent", source, type_="url", status=IngestionStatus.DOWNLOADED
            )
            # make the row look old enough for the staleness gate
            key = status_key(agent, "agent", source)
            stored = await crud.read(key)
            if isinstance(stored, list):
                stored = stored[0]
            stored["updated_at"] = time.time() - 10_000
            await crud.store(key, stored)

            claimed = await claim_source_for_resume(
                agent, "agent", source, stale_after=0, owner="b3-test"
            )

            assert claimed is not None, (
                "B3: a stale 'downloaded' source is never claimed -> permanent dead end"
            )
            assert claimed["status"] == IngestionStatus.PROCESSING.value
        finally:
            await _drop(agent)


    async def test_b3b_active_download_is_not_stolen():
        """B3 (guard): a fresh ``downloading`` row is never stolen.

        The resumable policy must not reclaim a row that is still actively
        downloading: the existing ``updated_at`` staleness gate protects it.
        """
        agent = _agent_id()
        source = "https://example.com/active.pdf"
        try:
            await _seed_master_key(agent)
            await set_status(
                agent, "agent", source, type_="url", status=IngestionStatus.DOWNLOADING
            )

            claimed = await claim_source_for_resume(
                agent, "agent", source, stale_after=300, owner="b3b-test"
            )

            assert claimed is None, (
                "B3: a fresh 'downloading' source was stolen by the resume sweep"
            )
            doc = await get_status(agent, "agent", source)
            assert doc is not None
            assert doc["status"] == IngestionStatus.DOWNLOADING.value
        finally:
            await _drop(agent)


    run([
        test_b1_status_listing_returns_the_seeded_rows,
        test_b3_downloaded_source_can_be_resumed,
        test_b3b_active_download_is_not_stolen,
    ])
