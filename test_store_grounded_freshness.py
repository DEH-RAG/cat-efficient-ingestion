"""Standalone, import-safe harness for store-grounded phase freshness.

Covers:
  1. diary FRESH + store EMPTY -> the forced ``parsing_chunking`` + ``embedding``
     run and the embedder IS invoked (the incident: 0 chunk points but a fresh
     diary silently skipped the embed phase forever);
  2. diary FRESH + store POPULATED -> no re-parse, no re-embed (no write, no
     embedder call);
  3. the cheap store probe uses ONE bounded query per source and treats a
     probe error as "populated" (never triggers a destructive re-parse).

The Cat plugin loader imports EVERY ``.py`` under a plugin folder at
activation, so this whole file is inert on import: every import, class and
test lives inside ``if __name__ == "__main__":``. Run it as a plain script
from the host repository (the plugin is symlinked under ``cat/core_plugins``):

    cd /root/git/grinning-cat-core
    .venv/bin/python /root/git/plugins/cat-efficient-ingestion/test_store_grounded_freshness.py
"""

if __name__ == "__main__":
    import asyncio
    import sys
    import types

    from cat.core_plugins.cat_efficient_ingestion import reembed
    from cat.services.memory.models import PointStruct

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

    class FakeEmbedder:
        name = "fake"
        size = 2

        def __init__(self):
            self.calls = 0
            self.docs = []

        def embed_documents(self, docs):
            self.calls += 1
            self.docs = list(docs)
            return [[1.0, 0.0] for _ in docs]

    class FakeHandler:
        """Vector handler with the cheap store probe + write/delete counters."""

        def __init__(self, chunk_count=0):
            self.chunk_count = chunk_count
            self.probes = []
            self.deletes = 0
            self.adds = 0

        async def count_source_chunk_points(self, collection_name, source, chat_id=None):
            self.probes.append((collection_name, source, chat_id))
            return self.chunk_count

        async def get_all_tenant_points(self, *args, **kwargs):
            return [], None

        async def delete_tenant_points(self, *args, **kwargs):
            self.deletes += 1

        async def add_points_to_tenant(self, collection_name, points):
            self.adds += 1

    def _ccat(handler, embedder):
        async def _embedder():
            return embedder

        return types.SimpleNamespace(
            agent_key="fresh_agent", _id="fresh_agent",
            vector_memory_handler=handler,
            embedder=_embedder,
            chunker=types.SimpleNamespace(name="chunker"),
        )

    def _source(name="doc.txt"):
        return types.SimpleNamespace(name=name, metadata={}, path=name)

    def _points(n, source="doc.txt"):
        return [
            PointStruct(
                id=f"c{i}",
                payload={"page_content": f"chunk {i}", "metadata": {"source": source}},
                vector={},
            )
            for i in range(n)
        ]

    class _Hooks:
        """Patch/restore the reembed module seams used by the tests."""

        NAMES = (
            "get_status", "set_phase", "set_status", "_set_status",
            "record_phase_completed", "_record_phase", "_complete_source",
            "_probe_pending", "claim_source_for_resume",
            "_start_heartbeat", "_stop_heartbeat", "_resolve_rabbit_hole",
            "_run_parsing_phase", "ingestion_canceled", "execute_shared_hook",
            "split_oversized", "log",
        )

        def __init__(self):
            self.saved = {}

        def __enter__(self):
            for name in self.NAMES:
                self.saved[name] = getattr(reembed, name)
            return self

        def __exit__(self, *exc):
            for name, value in self.saved.items():
                setattr(reembed, name, value)
            return False

    def _async_return(value=None):
        async def _inner(*args, **kwargs):
            return value
        return _inner

    def _install_seams(log, parse_points):
        reembed.log = log
        reembed.get_status = _async_return(
            {"status": "completed", "phase": None, "source": "doc.txt"}
        )
        reembed.set_phase = _async_return(None)
        reembed.set_status = _async_return(None)
        reembed._set_status = _async_return(None)
        reembed.record_phase_completed = _async_return(None)
        reembed._record_phase = _async_return(None)
        reembed._complete_source = _async_return(None)
        reembed._probe_pending = _async_return([])  # diary says FRESH
        reembed.claim_source_for_resume = _async_return({"claimed": True})
        reembed._start_heartbeat = _async_return(None)
        reembed._stop_heartbeat = _async_return(None)
        reembed._resolve_rabbit_hole = _async_return(object())
        reembed.ingestion_canceled = _async_return(False)
        reembed.execute_shared_hook = _async_return(None)
        reembed.split_oversized = lambda docs, embedder: docs

        async def _parse(ccat, collection_name, source, source_name, chat_id, cat,
                         rabbit_hole, chunker_name):
            return source_name, list(parse_points)

        reembed._run_parsing_phase = _parse

    async def test_fresh_diary_empty_store_reembeds():
        handler = FakeHandler(chunk_count=0)
        embedder = FakeEmbedder()
        ccat = _ccat(handler, embedder)
        log = FakeLog()
        with _Hooks():
            _install_seams(log, _points(3))
            await reembed.reembed_sources(
                ccat, "declarative", [_source()], caller_cat=ccat
            )
        assert handler.probes == [("declarative", "doc.txt", None)], handler.probes
        assert embedder.calls == 1, f"embedder was not invoked (calls={embedder.calls})"
        assert len(embedder.docs) == 3, embedder.docs
        assert handler.adds == 1 and handler.deletes == 1, (
            f"points were not re-embedded: adds={handler.adds} deletes={handler.deletes}"
        )
        assert [m for m in log.infos if "diary-fresh-store-empty" in m], log.infos
        print("PASS fresh diary + empty store forces re-parse and invokes the embedder")

    async def test_fresh_diary_populated_store_skips():
        handler = FakeHandler(chunk_count=5)
        embedder = FakeEmbedder()
        ccat = _ccat(handler, embedder)
        log = FakeLog()
        with _Hooks():
            _install_seams(log, _points(3))
            await reembed.reembed_sources(
                ccat, "declarative", [_source()], caller_cat=ccat
            )
        assert handler.probes == [("declarative", "doc.txt", None)], handler.probes
        assert embedder.calls == 0, f"unexpected re-embed (calls={embedder.calls})"
        assert handler.adds == 0 and handler.deletes == 0, (
            f"unexpected write: adds={handler.adds} deletes={handler.deletes}"
        )
        assert not [m for m in log.infos if "diary-fresh-store-empty" in m], log.infos
        print("PASS fresh diary + populated store skips with no write and no embed call")

    async def test_probe_error_is_conservative():
        class BrokenHandler:
            async def count_source_chunk_points(self, *args, **kwargs):
                raise RuntimeError("neo4j unavailable")

        log = FakeLog()
        with _Hooks():
            reembed.log = log
            populated = await reembed._source_has_stored_points(
                _ccat(BrokenHandler(), FakeEmbedder()), BrokenHandler(),
                "declarative", "doc.txt", None,
            )
        assert populated is True, "a probe error must never trigger a destructive re-parse"
        assert [m for m in log.warnings if "freshness probe failed" in m], log.warnings
        print("PASS store probe errors are conservative (assume populated)")

    async def _main():
        await test_fresh_diary_empty_store_reembeds()
        await test_fresh_diary_populated_store_skips()
        await test_probe_error_is_conservative()

    try:
        asyncio.run(_main())
    except AssertionError as exc:
        print(f"FAIL {exc}")
        sys.exit(1)
    print("OK store-grounded freshness")
