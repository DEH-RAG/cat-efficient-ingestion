"""Standalone, import-safe harness for the built-in empty-output INVARIANTS.

``parsing_chunking`` and ``embedding`` must report ``error`` — never a warning
on a ``done`` phase — when they RAN but produced nothing:

  (a) parsing ran on a source with bytes and produced 0 chunks -> error,
      with no warning channel;
  (b) parsing ran on a source with bytes and produced chunks -> no error
      (the body proceeds, so the dispatch records ``done``);
  (c) embedding had points to embed and stored 0 -> error naming the count
      (the old zero-L2-norm retry is gone: nothing was persisted);
  (d) embedding with nothing to embed -> unchanged (no error result);
  (e) the graph phases STILL report a legitimate empty as ``done`` + a
      warning: the policy did not leak to them.

The plugin loader imports EVERY ``.py`` under a plugin folder, so the whole
body lives inside ``if __name__ == "__main__":`` and importing this file
executes zero code. Run it from the host repository::

    cd /root/git/grinning-cat-core
    .venv/bin/python /root/git/plugins/cat-efficient-ingestion/tests/test_empty_output_errors.py
"""

if __name__ == "__main__":
    import asyncio
    import sys
    import types

    from cat.core_plugins.cat_efficient_ingestion import phases, reembed
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

    class _EmbedSeams:
        def __init__(self):
            self.saved = {}

        def __enter__(self):
            for name in ("log", "ingestion_canceled", "execute_shared_hook", "split_oversized"):
                self.saved[name] = getattr(reembed, name)
            reembed.log = FakeLog()

            async def _false(*args, **kwargs):
                return False

            async def _none(*args, **kwargs):
                return None

            reembed.ingestion_canceled = _false
            reembed.execute_shared_hook = _none
            reembed.split_oversized = lambda docs, embedder: docs
            return self

        def __exit__(self, *exc):
            for name, value in self.saved.items():
                setattr(reembed, name, value)
            return False

    class FakeEmbedder:
        def embed_documents(self, docs):
            return [[0.0, 0.0, 0.0] for _ in docs]

    class FakeHandler:
        def __init__(self):
            self.deletes = 0
            self.adds = 0

        async def delete_tenant_points(self, *args, **kwargs):
            self.deletes += 1

        async def add_points_to_tenant(self, collection_name, points):
            self.adds += 1

    def _ccat(handler=None):
        return types.SimpleNamespace(
            agent_key="empty_agent", _id="empty_agent",
            vector_memory_handler=handler,
        )

    def _points(n):
        return [
            PointStruct(
                id=f"c{i}",
                payload={"page_content": f"chunk {i}", "metadata": {"source": "doc.txt"}},
                vector={},
            )
            for i in range(n)
        ]

    class _ExternalSeams:
        """Minimal seams for ``_run_external_phase`` (no DB, no status writes)."""

        def __init__(self, result):
            self.saved = {}
            self.result = result
            self.doc = None

        def __enter__(self):
            for name in (
                "get_status", "backfill_completed_phases", "merged_phases",
                "execute_phase_on_owner", "log",
            ):
                self.saved[name] = getattr(reembed, name)

            async def _get_status(*args, **kwargs):
                return self.doc

            async def _empty(*args, **kwargs):
                return {}

            async def _hook(*args, **kwargs):
                return self.result

            reembed.get_status = _get_status
            reembed.backfill_completed_phases = _empty
            reembed.merged_phases = _empty
            reembed.execute_phase_on_owner = _hook
            reembed.log = FakeLog()
            return self

        def __exit__(self, *exc):
            for name, value in self.saved.items():
                setattr(reembed, name, value)
            return False

    def _test_a_parsing_content_zero_chunks_is_error():
        failure = phases.ParsingChunkingPhase.empty_output_error("doc.txt", 1234, 0)
        assert failure is not None, "a non-empty source with 0 chunks must be an error"
        assert failure.status == "error", failure
        assert "1234 bytes" in (failure.reason or ""), failure
        assert "doc.txt" in (failure.reason or ""), failure
        assert failure.warnings is None, failure
        print("PASS (a) parsing a non-empty source with 0 chunks is an error, no warning")

    def _test_b_parsing_content_with_chunks_is_not_error():
        assert phases.ParsingChunkingPhase.empty_output_error("doc.txt", 1234, 3) is None
        print("PASS (b) parsing a non-empty source with chunks proceeds (done)")

    async def _test_c_embedding_points_zero_stored_is_error():
        handler = FakeHandler()
        with _EmbedSeams():
            raised = None
            try:
                await reembed._embed_phase(
                    _ccat(handler), "declarative", "doc.txt", _points(2),
                    FakeEmbedder(), None, None, None,
                )
            except Exception as exc:  # noqa: BLE001 - the invariant raises
                raised = exc
        assert raised is not None, "points to embed + 0 stored must be an error"
        assert "stored no points for 2 point(s)" in str(raised), raised
        assert handler.deletes == 0 and handler.adds == 0, (
            f"destructive write ran: deletes={handler.deletes} adds={handler.adds}"
        )
        print("PASS (c) embedding with points and 0 stored is an error naming the count")

    def _test_d_embedding_nothing_to_embed_is_unchanged():
        assert phases.EmbeddingPhase.empty_output_error("doc.txt", 0, 0) is None
        assert phases.EmbeddingPhase.empty_output_error("doc.txt", None, 0) is None
        assert phases.EmbeddingPhase.empty_output_error("doc.txt", 0, 5) is None
        print("PASS (d) embedding with nothing to embed keeps its unchanged path")

    async def _test_e_graph_phases_still_warn():
        warning = {"code": "empty-extraction", "message": "the LLM found no concept"}
        outcome = {}
        with _ExternalSeams({"status": "done", "warnings": [warning]}):
            ok = await reembed._run_external_phase(
                _ccat(), None, "agent", "doc.txt", "graph_concepts", None,
                outcome=outcome,
            )
        assert ok is True, "an external phase that ran stays successful"
        assert outcome.get("status") == "done", outcome
        assert outcome.get("warnings") == [warning], outcome
        print("PASS (e) graph phases still report a legitimate empty as a warning")

    async def _main():
        _test_a_parsing_content_zero_chunks_is_error()
        _test_b_parsing_content_with_chunks_is_not_error()
        await _test_c_embedding_points_zero_stored_is_error()
        _test_d_embedding_nothing_to_embed_is_unchanged()
        await _test_e_graph_phases_still_warn()
        return True

    try:
        asyncio.run(_main())
    except AssertionError as exc:
        print(f"FAIL {exc}")
        sys.exit(1)
