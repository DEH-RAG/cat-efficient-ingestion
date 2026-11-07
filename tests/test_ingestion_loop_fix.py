"""Regression tests for the ingestion no-op loop (resume <-> phase machine).

A source whose phase diary is already complete but whose status was left
``processing`` (e.g. a run interrupted between the last diary write and the
terminal write) used to be re-handed by every resume sweep forever:
``reembed_sources`` took the "up to date" ``continue`` without ever writing a
terminal status. These tests lock the fix:

- a fully-processed ``processing`` row is terminalized to ``completed`` in ONE
  pass and is not handed again on the next sweep;
- a row with a genuinely stale phase still runs the stale phase;
- an ``error`` row is never resurrected (nor re-handed when its diary is fresh).

Standalone and import-safe (the CAT loader imports every ``.py`` under the
plugin folder): the body lives under ``if __name__ == "__main__":``. Run it
from the host repository::

    cd /root/git/grinning-cat-core
    .venv/bin/python /root/git/plugins/cat-efficient-ingestion/tests/test_ingestion_loop_fix.py

Redis isolation: ``CAT_REDIS_DB=15`` is forced BEFORE importing ``cat``; every
key lives under ``agents:effingloop_<uuid>:`` and is destroyed in ``finally``.
"""
if __name__ == "__main__":
    import asyncio
    import os
    import sys
    import traceback
    import uuid
    from types import SimpleNamespace
    from unittest.mock import patch

    os.environ.setdefault("CAT_REDIS_HOST", "localhost")
    os.environ.setdefault("CAT_REDIS_PORT", "6379")
    os.environ.setdefault("CAT_JWT_SECRET", "test_only_jwt_secret_0123456789_abcdefghij")
    os.environ["CAT_REDIS_DB"] = "15"

    from cat.core_plugins.cat_efficient_ingestion import reembed, resume
    from cat.core_plugins.cat_efficient_ingestion.registry import (
        IngestionStatus,
        get_status,
        set_status,
    )
    from cat.db import crud
    from cat.looking_glass.models import StoredSourceWithMetadata
    from cat.services.memory.models import VectorMemoryType

    _DIARY = {
        "parsing_chunking": {"marker": "c", "settings_version": None, "deps": {}},
        "embedding": {
            "marker": "e",
            "settings_version": None,
            "deps": {"parsing_chunking": "c"},
        },
    }

    def _agent_id() -> str:
        return f"effingloop_{uuid.uuid4().hex}"

    async def _seed_master_key(agent_id: str) -> None:
        await crud.store(f"agents:{agent_id}:agent", [{"name": "x", "value": 1}])

    async def _drop(agent_id: str) -> None:
        await crud.destroy(f"agents:{agent_id}:*")

    async def _seed(agent: str, source: str, status: IngestionStatus) -> None:
        await set_status(
            agent, "agent", source, type_="file", status=status, completed_phases=_DIARY,
        )

    def _stored_chunk_point(source_name: str = "doc.pdf"):
        """One genuinely stored chunk point (what the store-grounded guard inspects)."""
        return SimpleNamespace(
            id="stored-chunk-0",
            payload={
                "page_content": "stored chunk",
                "metadata": {"source": source_name},
            },
            vector={},
        )

    class _FileManager:
        """Minimal file manager so a forced parse can never hit an AttributeError."""

        def read_file(self, *args, **kwargs):
            return b"%PDF-1.4 stub"

        def write_file(self, *args, **kwargs):
            return True

        def remove_file(self, *args, **kwargs):
            return None

        def file_exists(self, *args, **kwargs):
            return True

        def list_files(self, *args, **kwargs):
            return ["doc.pdf"]

    def _ccat(agent: str, points=None):
        chunks = [_stored_chunk_point()] if points is None else points

        async def _get_points(*args, **kwargs):
            return list(chunks), None

        async def _write_points(*args, **kwargs):
            return [], None

        async def embedder():
            return SimpleNamespace(name="test-embedder")

        async def file_handlers():
            return {}

        async def save_file(*args, **kwargs):
            return True

        return SimpleNamespace(
            _id=agent,
            agent_key=agent,
            vector_memory_handler=SimpleNamespace(
                get_all_tenant_points=_get_points,
                add_points_to_tenant=_write_points,
                delete_tenant_points=_write_points,
            ),
            file_manager=_FileManager(),
            file_handlers=file_handlers,
            save_file=save_file,
            embedder=embedder,
            chunker=SimpleNamespace(name="test-chunker"),
            rabbit_hole=SimpleNamespace(),
        )

    def _source(name: str = "doc.pdf"):
        return StoredSourceWithMetadata(name=name, path=name, content=None, metadata={})

    async def test_up_to_date_processing_row_terminalizes_once():
        """All phases fresh + status processing -> completed in one pass, then no-op."""
        agent = _agent_id()
        await _seed_master_key(agent)
        source_name = "doc.pdf"
        await _seed(agent, source_name, IngestionStatus.PROCESSING)

        async def fake_probe(*args, **kwargs):
            return []

        ccat = _ccat(agent)
        try:
            with patch.object(reembed, "_probe_pending", fake_probe):
                await reembed.reembed_sources(
                    ccat, VectorMemoryType.DECLARATIVE, [_source(source_name)],
                    stale_after=0.0, caller_cat=ccat,
                )
            doc = await get_status(agent, "agent", source_name)
            assert doc is not None
            assert doc["status"] == IngestionStatus.COMPLETED.value
            assert "phase" not in doc
            first_updated = doc["updated_at"]

            # Second pass: terminal row, no stale phase -> no write at all.
            with patch.object(reembed, "_probe_pending", fake_probe):
                await reembed.reembed_sources(
                    ccat, VectorMemoryType.DECLARATIVE, [_source(source_name)],
                    stale_after=0.0, caller_cat=ccat,
                )
            doc = await get_status(agent, "agent", source_name)
            assert doc is not None
            assert doc["status"] == IngestionStatus.COMPLETED.value
            assert doc["updated_at"] == first_updated, "terminal row was rewritten (write storm)"
        finally:
            await _drop(agent)

    async def test_terminal_source_is_not_handed_again_by_resume():
        """A completed row is not a resume candidate, so it is never handed again."""
        agent = _agent_id()
        await _seed_master_key(agent)
        source_name = "doc.pdf"
        await _seed(agent, source_name, IngestionStatus.COMPLETED)

        handed = []

        async def fake_reembed(*args, **kwargs):
            handed.append(args)

        class _Lizard:
            async def get_cheshire_cat(self, agent_id):
                return _ccat(agent_id)

        try:
            with patch.object(resume, "reembed_sources", fake_reembed):
                await resume._resume_agent(_Lizard(), agent)
                await resume._resume_agent(_Lizard(), agent)
            assert handed == [], "a terminal completed row was re-handed by the resume sweep"
        finally:
            await _drop(agent)

    async def test_stale_phase_still_runs():
        """A row with a genuinely stale phase is still claimed and processed."""
        agent = _agent_id()
        await _seed_master_key(agent)
        source_name = "doc.pdf"
        await _seed(agent, source_name, IngestionStatus.PROCESSING)

        probes = iter([[{"phase": "embedding"}], []])

        async def fake_probe(*args, **kwargs):
            return next(probes, [])

        embedded = []

        async def fake_embed(*args, **kwargs):
            embedded.append(args)

        point = SimpleNamespace(payload={"metadata": {"source": source_name}})
        ccat = _ccat(agent, points=[point])
        try:
            with patch.object(reembed, "_probe_pending", fake_probe):
                with patch.object(reembed, "_embed_phase", fake_embed):
                    await reembed.reembed_sources(
                        ccat, VectorMemoryType.DECLARATIVE, [_source(source_name)],
                        stale_after=0.0, caller_cat=ccat,
                    )
            assert embedded, "a stale phase was skipped"
            doc = await get_status(agent, "agent", source_name)
            assert doc is not None
            assert doc["status"] == IngestionStatus.COMPLETED.value
        finally:
            await _drop(agent)

    async def test_error_row_is_not_resurrected_by_up_to_date_branch():
        """An up-to-date ERROR row is never rewritten to completed."""
        agent = _agent_id()
        await _seed_master_key(agent)
        source_name = "doc.pdf"
        await _seed(agent, source_name, IngestionStatus.ERROR)
        seeded = await get_status(agent, "agent", source_name)
        assert seeded is not None
        before = seeded["updated_at"]

        async def fake_probe(*args, **kwargs):
            return []

        ccat = _ccat(agent)
        try:
            with patch.object(reembed, "_probe_pending", fake_probe):
                await reembed.reembed_sources(
                    ccat, VectorMemoryType.DECLARATIVE, [_source(source_name)],
                    stale_after=0.0, caller_cat=ccat,
                )
            doc = await get_status(agent, "agent", source_name)
            assert doc is not None
            assert doc["status"] == IngestionStatus.ERROR.value
            assert doc["updated_at"] == before
        finally:
            await _drop(agent)

    async def test_fresh_error_row_is_not_handed_by_resume():
        """An ERROR row whose diary is fresh is terminal: never re-handed nor retried."""
        agent = _agent_id()
        await _seed_master_key(agent)
        source_name = "doc.pdf"
        await _seed(agent, source_name, IngestionStatus.ERROR)

        async def fresh_probe(ccat, entry):
            return []

        handed = []

        async def fake_reembed(*args, **kwargs):
            handed.append(args)

        class _Lizard:
            async def get_cheshire_cat(self, agent_id):
                return _ccat(agent_id)

        try:
            with patch.object(resume, "_pending_phases", fresh_probe):
                with patch.object(resume, "reembed_sources", fake_reembed):
                    await resume._resume_agent(_Lizard(), agent)
            doc = await get_status(agent, "agent", source_name)
            assert doc is not None
            assert doc["status"] == IngestionStatus.ERROR.value
            assert handed == [], "a fresh error row was resurrected/re-handed"
        finally:
            await _drop(agent)

    async def test_fresh_diary_with_empty_store_forces_reparse():
        """A fresh diary is NOT trusted when the store holds zero chunk points."""
        agent = _agent_id()
        await _seed_master_key(agent)
        source_name = "doc.pdf"
        await _seed(agent, source_name, IngestionStatus.PROCESSING)

        async def fresh_probe(*args, **kwargs):
            return []

        parsed = []
        embedded = []

        async def fake_parse(*args, **kwargs):
            parsed.append(args[3])
            return args[3], [_stored_chunk_point(args[3])], 1, 1

        async def fake_embed(*args, **kwargs):
            embedded.append(args[2])

        ccat = _ccat(agent, points=[])
        try:
            with patch.object(reembed, "_probe_pending", fresh_probe):
                with patch.object(reembed, "_run_parsing_phase", fake_parse):
                    with patch.object(reembed, "_embed_phase", fake_embed):
                        await reembed.reembed_sources(
                            ccat, VectorMemoryType.DECLARATIVE, [_source(source_name)],
                            stale_after=0.0, caller_cat=ccat,
                        )
            assert parsed, "fresh diary + EMPTY store did not force parsing_chunking"
            assert embedded, "the forced re-parse was not followed by the forced re-embed"
            doc = await get_status(agent, "agent", source_name)
            assert doc is not None
            assert doc["status"] == IngestionStatus.COMPLETED.value
        finally:
            await _drop(agent)

    _tests = [
        test_up_to_date_processing_row_terminalizes_once,
        test_terminal_source_is_not_handed_again_by_resume,
        test_stale_phase_still_runs,
        test_error_row_is_not_resurrected_by_up_to_date_branch,
        test_fresh_error_row_is_not_handed_by_resume,
        test_fresh_diary_with_empty_store_forces_reparse,
    ]

    async def _main():
        # A SINGLE event loop: the async Redis client is a process singleton.
        passed = 0
        failures = []
        for test in _tests:
            try:
                await test()
            except BaseException as exc:  # noqa: BLE001 - report every failure
                failures.append((test.__name__, exc, traceback.format_exc()))
                print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}")
            else:
                passed += 1
                print(f"PASS {test.__name__}")

        print(f"{passed}/{len(_tests)} passed")
        if failures:
            for name, _exc, tb in failures:
                print(f"\n--- {name} ---\n{tb}")
            return 1
        return 0

    sys.exit(asyncio.run(_main()))
