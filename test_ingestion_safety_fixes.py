"""Standalone, import-safe harness for the ingestion-safety fixes.

Covers:
  1. an external ``not_ready`` phase response keeps the row NON-TERMINAL and
     retryable well beyond 5 attempts, and only an exceeded wall-clock budget
     turns it into ``error`` with the phase + elapsed time in the reason;
  2. terminal rows are never rewritten by the not-ready path;
  3. the embedding phase detects zero/non-finite L2-norm vectors: a source
     whose chunks are ALL invalid is NOT completed (it raises the not-ready
     signal BEFORE the destructive replace), while a source with at least one
     valid chunk proceeds normally; counts are reported per source.

The Cat plugin loader imports EVERY ``.py`` under a plugin folder at
activation, so this whole file is inert on import: every import, class and
test lives inside ``if __name__ == "__main__":``. Run it as a plain script
from the host repository (the plugin is symlinked under ``cat/core_plugins``):

    cd /root/git/grinning-cat-core
    .venv/bin/python /root/git/plugins/cat-efficient-ingestion/test_ingestion_safety_fixes.py
"""

if __name__ == "__main__":
    import asyncio
    import copy
    import os
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

    class FakeStatus:
        """In-memory stand-in for the registry status doc."""

        def __init__(self, status="processing", phase="catalog_cards"):
            self.doc = {
                "source": "doc.txt",
                "scope": "agent",
                "status": status,
                "phase": phase,
                "error": None,
            }
            self.set_calls = []

        async def get_status(self, agent_id, scope, source):
            return copy.deepcopy(self.doc)

        async def set_status(
            self, agent_id, scope, source, *, type_, status, chat_id=None,
            error=None, phase=None, extra=None, **kwargs,
        ):
            value = getattr(status, "value", status)
            call = {"status": value, "error": error, "phase": phase, "extra": extra}
            self.set_calls.append(call)
            self.doc["status"] = value
            if error is not None:
                self.doc["error"] = error
            if phase is not None:
                self.doc["phase"] = phase
            if extra:
                self.doc.update(copy.deepcopy(extra))
            return self.doc

        async def set_status_error(self, ccat, source, status, error=None, chat_id=None):
            await self.set_status(
                ccat.agent_key, "agent", source, type_="file",
                status=status, error=error, chat_id=chat_id,
            )

    class _Hooks:
        """Patch/restore the reembed module seams used by the tests."""

        def __init__(self):
            self.saved = {}

        def __enter__(self):
            for name in (
                "get_status", "set_status", "_set_status",
                "backfill_completed_phases", "merged_phases",
                "execute_phase_on_owner", "asyncio", "log",
                "split_oversized", "ingestion_canceled", "execute_shared_hook",
            ):
                self.saved[name] = getattr(reembed, name)
            return self

        def __exit__(self, *exc):
            for name, value in self.saved.items():
                setattr(reembed, name, value)
            return False

    def _ccat(handler=None):
        return types.SimpleNamespace(
            agent_key="safety_agent", _id="safety_agent",
            vector_memory_handler=handler,
        )

    def _install_phase_seams(state, hook, sleep_log=None):
        reembed.get_status = state.get_status
        reembed.set_status = state.set_status
        reembed._set_status = state.set_status_error
        reembed.backfill_completed_phases = _async_return({})
        reembed.merged_phases = _async_return({})

        async def _call_phase(*args, **kwargs):
            return hook(*args, **kwargs)

        reembed.execute_phase_on_owner = _call_phase

        async def _sleep(seconds):
            if sleep_log is not None:
                sleep_log.append(seconds)

        reembed.asyncio = types.SimpleNamespace(sleep=_sleep)

    def _async_return(value=None):
        async def _inner(*args, **kwargs):
            return value
        return _inner

    async def test_not_ready_is_time_bounded_and_retryable():
        os.environ["CAT_INGESTION_NOT_READY_MAX_SECONDS"] = "3600"
        state = FakeStatus()
        calls = {"n": 0}

        def hook(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] <= 8:
                return {"status": "not_ready", "retry_after": 0}
            return {"status": "done"}

        log = FakeLog()
        sleeps = []
        with _Hooks():
            reembed.log = log
            _install_phase_seams(state, hook, sleeps)
            ok = await reembed._run_external_phase(
                _ccat(), None, "agent", "doc.txt", "catalog_cards", None
            )
        assert ok is True, "a long not_ready streak must eventually succeed"
        assert state.doc["status"] == "processing", state.doc
        assert state.doc["phase"] == "catalog_cards", state.doc
        assert not any(c["status"] == "error" for c in state.set_calls), state.set_calls
        hint = state.doc.get("phase_retry") or {}
        assert hint.get("attempts") == 8, hint
        assert hint.get("phase") == "catalog_cards", hint
        assert len(sleeps) == 8, sleeps
        first_wait_logs = [m for m in log.infos if "is not ready" in m]
        assert len(first_wait_logs) == 1, first_wait_logs
        assert not [m for m in log.warnings if "not ready after" in m], log.warnings
        print("PASS not_ready keeps the row pending and retryable (8 attempts > 5)")

    async def test_not_ready_exceeds_budget_marks_error():
        os.environ["CAT_INGESTION_NOT_READY_MAX_SECONDS"] = "0"
        state = FakeStatus()
        log = FakeLog()
        with _Hooks():
            reembed.log = log
            _install_phase_seams(state, lambda *a, **k: {"status": "not_ready", "retry_after": 0})
            ok = await reembed._run_external_phase(
                _ccat(), None, "agent", "doc.txt", "catalog_cards", None
            )
        assert ok is False
        assert state.doc["status"] == "error", state.doc
        error = state.doc.get("error") or ""
        assert "catalog_cards" in error, error
        assert "not ready after" in error, error
        assert [m for m in log.warnings if "not ready after" in m], log.warnings
        print("PASS exceeded budget marks error naming the phase and elapsed time")

    async def test_terminal_row_is_never_rewritten():
        state = FakeStatus(status="error", phase="catalog_cards")
        before = copy.deepcopy(state.doc)
        with _Hooks():
            _install_phase_seams(state, lambda *a, **k: {"status": "not_ready", "retry_after": 0})
            ok = await reembed._run_external_phase(
                _ccat(), None, "agent", "doc.txt", "catalog_cards", None
            )
        assert ok is False
        assert state.doc == before, state.doc
        assert state.set_calls == [], state.set_calls
        print("PASS terminal rows are not rewritten")

    class FakeEmbedder:
        name = "fake"
        size = 3

        def __init__(self, vectors):
            self.vectors = vectors

        def embed_documents(self, docs):
            return [list(v) for v in self.vectors]

    class FakeHandler:
        def __init__(self):
            self.deletes = 0
            self.adds = 0
            self.last_points = []

        async def delete_tenant_points(self, *args, **kwargs):
            self.deletes += 1

        async def add_points_to_tenant(self, collection_name, points):
            self.adds += 1
            self.last_points = list(points)

    def _points(n):
        return [
            PointStruct(
                id=f"c{i}",
                payload={"page_content": f"chunk {i}", "metadata": {"source": "doc.txt"}},
                vector={},
            )
            for i in range(n)
        ]

    def _install_embed_seams(log):
        reembed.log = log
        reembed.ingestion_canceled = _async_return(False)
        reembed.execute_shared_hook = _async_return(None)
        reembed.split_oversized = lambda docs, embedder: docs

    async def test_all_zero_norm_is_not_completed():
        async def _run():
            handler = FakeHandler()
            summary = {}
            log = FakeLog()
            with _Hooks():
                _install_embed_seams(log)
                raised = None
                try:
                    await reembed._embed_phase(
                        _ccat(handler), "declarative", "doc.txt", _points(3),
                        FakeEmbedder([[0.0, 0.0, 0.0]] * 3), None, None, None,
                        zero_norm_summary=summary,
                    )
                except reembed._NothingStored as exc:
                    raised = exc
                assert raised is not None, "all-zero source must raise _NothingStored"
                assert handler.deletes == 0 and handler.adds == 0, (
                    f"destructive replace ran: deletes={handler.deletes} adds={handler.adds}"
                )
                assert summary["doc.txt"]["count"] == 3, summary
                assert summary["doc.txt"]["valid"] == 0, summary
                warnings = [m for m in log.warnings if "zero/non-finite" in m]
                assert warnings and "doc.txt" in warnings[0], log.warnings
                assert "c0" in warnings[0] and "c2" in warnings[0], warnings[0]
                print("PASS all-zero-norm source stays non-terminal and logs ids/count")

        await _run()

    async def test_partial_zero_norm_proceeds():
        handler = FakeHandler()
        summary = {}
        log = FakeLog()
        with _Hooks():
            _install_embed_seams(log)
            points = await reembed._embed_phase(
                _ccat(handler), "declarative", "doc.txt", _points(2),
                FakeEmbedder([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), None, None, None,
                zero_norm_summary=summary,
            )
        assert len(points) == 2, points
        assert handler.deletes == 1 and handler.adds == 1
        assert summary["doc.txt"]["count"] == 1, summary
        assert summary["doc.txt"]["valid"] == 1, summary
        assert summary["doc.txt"]["intended"] == 2, summary
        assert [m for m in log.warnings if "1 chunk(s)" in m], log.warnings
        print("PASS a source with some valid chunks proceeds normally")

    async def test_stale_phase_still_runs():
        state = FakeStatus(status="processing", phase="graph_spacy_extraction")
        calls = {"n": 0}

        def hook(*args, **kwargs):
            calls["n"] += 1
            return {"status": "done"}

        with _Hooks():
            _install_phase_seams(state, hook)
            ok = await reembed._run_external_phase(
                _ccat(), None, "agent", "doc.txt", "graph_spacy_extraction", None
            )
        assert ok is True
        assert calls["n"] == 1
        assert state.set_calls == []
        print("PASS a stale external phase still runs and completes")

    async def _main():
        await test_not_ready_is_time_bounded_and_retryable()
        await test_not_ready_exceeds_budget_marks_error()
        await test_terminal_row_is_never_rewritten()
        await test_all_zero_norm_is_not_completed()
        await test_partial_zero_norm_proceeds()
        await test_stale_phase_still_runs()

    try:
        asyncio.run(_main())
    except AssertionError as exc:
        print(f"FAIL {exc}")
        sys.exit(1)
    print("OK ingestion safety fixes (Fix 1 + Fix 3)")
