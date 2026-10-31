"""Standalone, import-safe tests for the phase completion predicate.

Covers the expected/processed contract of ``_finish_phase``:
  (a) 0 expected / 0 processed -> complete and NO warning;
  (b) 3 / 3 -> complete;
  (c) 2 / 3 -> stays running with a warning naming both numbers;
  (d) unknown counts -> legacy product-presence behaviour (+ DEBUG note);
  (e) an external phase returning ``processed`` / ``expected`` is respected.

The Cat plugin loader imports EVERY ``.py`` under a plugin folder at
activation, so this whole file is inert on import: every import, class and
test lives inside ``if __name__ == "__main__":``. Run it as a plain script
from the host repository (the plugin is symlinked under ``cat/core_plugins``):

    cd /root/git/grinning-cat-core
    .venv/bin/python /root/git/plugins/cat-efficient-ingestion/test_phase_completion_counts.py
"""

if __name__ == "__main__":
    import asyncio
    import types

    from cat.core_plugins.cat_efficient_ingestion import reembed

    class FakeLog:
        def __init__(self):
            self.infos = []
            self.warnings = []
            self.errors = []
            self.debugs = []

        def info(self, msg):
            self.infos.append(str(msg))

        def warning(self, msg):
            self.warnings.append(str(msg))

        def error(self, msg):
            self.errors.append(str(msg))

        def debug(self, msg):
            self.debugs.append(str(msg))

    class _Hooks:
        """Patch/restore the reembed module seams used by these tests."""

        NAMES = (
            "_record_phase", "log", "get_status", "backfill_completed_phases",
            "merged_phases", "execute_phase_on_owner",
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

    def _ccat():
        return types.SimpleNamespace(agent_key="counts_agent", _id="counts_agent")

    def _async_return(value=None):
        async def _inner(*args, **kwargs):
            return value
        return _inner

    def _install_finish_seams(log):
        recorded = []

        async def _record(*args, **kwargs):
            recorded.append((args, kwargs))
            return {}

        reembed.log = log
        reembed._record_phase = _record
        return recorded

    async def _finish(log, products, expected, processed):
        return await reembed._finish_phase(
            _ccat(), None, "agent", "doc.txt", "embedding", None, products,
            expected=expected, processed=processed,
        )

    async def test_a_zero_expected_zero_processed_completes():
        log = FakeLog()
        with _Hooks():
            recorded = _install_finish_seams(log)
            ok = await _finish(log, [], 0, 0)
        assert ok is True, "0 == 0 is a legitimate completion"
        assert len(recorded) == 1, recorded
        assert log.warnings == [], log.warnings
        print("PASS (a) 0 expected / 0 processed -> complete, no warning")

    async def test_b_three_of_three_completes():
        log = FakeLog()
        with _Hooks():
            recorded = _install_finish_seams(log)
            ok = await _finish(log, [object()] * 3, 3, 3)
        assert ok is True, "processed == expected must complete"
        assert len(recorded) == 1, recorded
        assert log.warnings == [], log.warnings
        print("PASS (b) 3 expected / 3 processed -> complete")

    async def test_c_two_of_three_stays_running_with_warning():
        log = FakeLog()
        with _Hooks():
            recorded = _install_finish_seams(log)
            ok = await _finish(log, [object()] * 2, 3, 2)
        assert ok is False, "processed < expected must not complete"
        assert recorded == [], recorded
        assert any("2" in m and "3" in m and "embedding" in m for m in log.warnings), log.warnings
        print("PASS (c) 2 of 3 -> stays running, warning names 2 and 3")

    async def test_d_unknown_counts_keep_legacy_behaviour():
        # non-empty products -> complete, with a DEBUG note about the counts
        log = FakeLog()
        with _Hooks():
            recorded = _install_finish_seams(log)
            ok = await _finish(log, [object()], None, None)
        assert ok is True, "unknown counts + truthy products -> legacy complete"
        assert len(recorded) == 1, recorded
        assert any("unavailable" in m for m in log.debugs), log.debugs
        assert log.warnings == [], log.warnings

        # empty products -> stays running with the legacy warning
        log2 = FakeLog()
        with _Hooks():
            recorded2 = _install_finish_seams(log2)
            ok2 = await _finish(log2, [], None, None)
        assert ok2 is False, "unknown counts + empty products -> legacy running"
        assert recorded2 == [], recorded2
        assert any("produced no products" in m for m in log2.warnings), log2.warnings
        print("PASS (d) unknown counts -> legacy product-presence behaviour")

    async def test_e_external_phase_counts_are_respected():
        log = FakeLog()
        counts = {}

        async def _done(*args, **kwargs):
            return {"status": "done", "processed": 3, "expected": 3}

        with _Hooks():
            reembed.get_status = _async_return({})
            reembed.backfill_completed_phases = _async_return({})
            reembed.merged_phases = _async_return({})
            reembed.execute_phase_on_owner = _done
            ok = await reembed._run_external_phase(
                _ccat(), None, "agent", "doc.txt", "graph_extraction", None,
                counts=counts,
            )
        assert ok is True, ok
        assert counts == {"expected": 3, "processed": 3}, counts

        # the reported pair drives the predicate: 3/3 completes, 2/3 does not
        with _Hooks():
            recorded = _install_finish_seams(log)
            ok_complete = await reembed._finish_phase(
                _ccat(), None, "agent", "doc.txt", "graph_extraction", None, True,
                expected=counts["expected"], processed=counts["processed"],
            )
        assert ok_complete is True, ok_complete
        assert len(recorded) == 1, recorded

        log_mismatch = FakeLog()
        mismatch_counts = {}

        async def _done_mismatch(*args, **kwargs):
            return {"status": "done", "processed": 2, "expected": 3}

        with _Hooks():
            reembed.get_status = _async_return({})
            reembed.backfill_completed_phases = _async_return({})
            reembed.merged_phases = _async_return({})
            reembed.execute_phase_on_owner = _done_mismatch
            ok2 = await reembed._run_external_phase(
                _ccat(), None, "agent", "doc.txt", "graph_extraction", None,
                counts=mismatch_counts,
            )
        assert ok2 is True, ok2
        assert mismatch_counts == {"expected": 3, "processed": 2}, mismatch_counts
        with _Hooks():
            recorded2 = _install_finish_seams(log_mismatch)
            ok_not = await reembed._finish_phase(
                _ccat(), None, "agent", "doc.txt", "graph_extraction", None, True,
                expected=mismatch_counts["expected"],
                processed=mismatch_counts["processed"],
            )
        assert ok_not is False, "an external 2/3 must not complete"
        assert recorded2 == [], recorded2
        assert any("2" in m and "3" in m for m in log_mismatch.warnings), log_mismatch.warnings
        print("PASS (e) external processed/expected is respected (3/3 completes, 2/3 not)")

    async def _main():
        await test_a_zero_expected_zero_processed_completes()
        await test_b_three_of_three_completes()
        await test_c_two_of_three_stays_running_with_warning()
        await test_d_unknown_counts_keep_legacy_behaviour()
        await test_e_external_phase_counts_are_respected()
        print("ALL PHASE COMPLETION COUNT TESTS PASSED")

    try:
        asyncio.run(_main())
    except Exception as exc:  # noqa: BLE001 - plain-script test harness
        print(f"FAILED: {exc}")
        raise
