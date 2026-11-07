"""URL source persistence for the efficient-ingestion phase machine (plugin-owned).

A URL is fetched only once: the resolved bytes are written through the file
manager under a deterministic sanitized name, and a later parse reads that copy
back instead of re-downloading. When the persisted copy is missing the core
resolver is used as a logged fallback.

The tests drive ``_run_parsing_phase`` directly (the single place where URL
bytes are resolved) with a real CheshireCat/file manager and a stubbed source
resolver that counts how many times it is called.

Standalone and import-safe: the entire body lives under
``if __name__ == "__main__":``. Run from the core checkout::

    cd /root/git/grinning-cat-core
    .venv/bin/python /root/git/plugins/cat-efficient-ingestion/tests/test_url_source_persistence.py
"""
if __name__ == "__main__":
    import os
    import shutil
    import tempfile
    from types import SimpleNamespace
    from unittest.mock import patch

    from _testkit import new_agent_id, run_with_host

    def build(host):
        from cat.core_plugins.base_plugin.file_managers.custom import LocalFileManager
        from cat.core_plugins.cat_efficient_ingestion import reembed
        from cat.db import crud
        from cat.services.factory.embedder import embedding_size
        from cat.services.memory.models import VectorMemoryType

        _URL = "https://example.test/docs/notes.txt"
        _TEXT = (
            "Efficient ingestion persists URL bytes so re-ingest is deterministic. " * 20
        ).strip()
        _PAYLOAD = _TEXT.encode("utf-8")

        def _agent_id() -> str:
            return new_agent_id()

        async def _seed_master_key(agent_id: str) -> None:
            await crud.store(f"agents:{agent_id}:agent", [{"name": "x", "value": 1}])

        async def _drop(agent_id: str) -> None:
            await crud.destroy(f"agents:{agent_id}:*")

        def _install_counting_resolver(ccat, calls: list):
            async def _resolve_source_bytes(file, filename, content_type):
                calls.append(file)
                return file, _PAYLOAD, "text/plain", True

            ccat.rabbit_hole._resolve_source_bytes = _resolve_source_bytes

        async def _prepare(agent: str, storage_root):
            await host.lizard.create_cheshire_cat(agent)
            ccat = await host.lizard.get_cheshire_cat(agent)
            embedder = await host.lizard.embedder()
            await ccat.vector_memory_handler.initialize(
                embedder.name, await embedding_size(embedder)
            )
            # the suite boots a DummyFileManager: use a real local one rooted in
            # the per-test temp dir so the write/read round-trip is exercised.
            ccat.file_manager = LocalFileManager()
            ccat.file_manager._root_dir = str(storage_root)
            return ccat

        async def _run(ccat):
            source = SimpleNamespace(name=_URL, path=_URL, content=None, metadata={})
            return await reembed._run_parsing_phase(
                ccat, VectorMemoryType.DECLARATIVE, source, _URL, None, ccat,
                ccat.rabbit_hole, "default_chunker",
            )

        def _texts(points) -> list[str]:
            return sorted(
                p.payload["page_content"]
                for p in points
                if not p.payload.get("metadata", {}).get("image")
            )

        async def test_url_bytes_persisted_and_reused_on_second_run():
            agent = _agent_id()
            storage_root = tempfile.mkdtemp(prefix="effing_url_")
            await _seed_master_key(agent)
            calls: list = []
            ccat = None
            try:
                ccat = await _prepare(agent, storage_root)
                _install_counting_resolver(ccat, calls)
                path = ccat.agent_key
                persisted = reembed._persisted_url_file_name(_URL, path)

                _, first_points, _, _ = await _run(ccat)
                assert calls == [_URL], "the first run must fetch the URL"
                assert ccat.file_manager.read_file(persisted, path) == _PAYLOAD

                _, second_points, _, _ = await _run(ccat)
                assert calls == [_URL], "the second run must read the persisted bytes, not refetch"
                assert ccat.file_manager.read_file(persisted, path) == _PAYLOAD
                assert _texts(second_points) == _texts(first_points), "chunk text must be identical"
                assert _texts(first_points)
            finally:
                if ccat is not None:
                    ccat.file_manager.remove_folder(ccat.agent_key)
                shutil.rmtree(storage_root, ignore_errors=True)
                await _drop(agent)

        async def test_missing_persisted_file_falls_back_to_resolver_and_logs():
            agent = _agent_id()
            storage_root = tempfile.mkdtemp(prefix="effing_url_")
            await _seed_master_key(agent)
            calls: list = []
            warnings: list[str] = []
            ccat = None
            try:
                ccat = await _prepare(agent, storage_root)
                _install_counting_resolver(ccat, calls)
                path = ccat.agent_key
                persisted = reembed._persisted_url_file_name(_URL, path)

                await _run(ccat)
                assert calls == [_URL]
                assert ccat.file_manager.remove_file(os.path.join(path, persisted))
                assert ccat.file_manager.read_file(persisted, path) is None

                with patch.object(reembed.log, "warning", warnings.append):
                    _, points, _, _ = await _run(ccat)

                assert calls == [_URL, _URL], "a missing persisted copy must refetch the URL"
                assert any("No persisted bytes" in w and _URL in w for w in warnings), (
                    f"the fallback must be logged, got: {warnings}"
                )
                assert ccat.file_manager.read_file(persisted, path) == _PAYLOAD, (
                    "the fallback must persist the refetched bytes again"
                )
                assert _texts(points)
            finally:
                if ccat is not None:
                    ccat.file_manager.remove_folder(ccat.agent_key)
                shutil.rmtree(storage_root, ignore_errors=True)
                await _drop(agent)

        return [
            test_url_bytes_persisted_and_reused_on_second_run,
            test_missing_persisted_file_falls_back_to_resolver_and_logs,
        ]

    run_with_host(build)
