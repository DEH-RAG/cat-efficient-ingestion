"""Standalone, import-safe tests for restart recovery of ``processing`` phases.

Recovery is ownership-based: only an entry whose ``owner`` belongs to a PREVIOUS
boot (or whose owner is provably dead / has no owner and an expired lease) is
abandoned. A live owner — the current ``BOOT_EPOCH`` or a live pod/pid — is
never touched. An abandoned phase is deleted (``IngestionPhase.delete``) BEFORE
the entry is reset to ``pending``; recovery never writes ``complete``.

Crash simulation only: no real store, no real source, no real process. The
plugin loader imports EVERY ``.py`` under a plugin folder, so the whole body
lives inside ``if __name__ == "__main__":`` and importing this file executes
zero code.

Run from the host repository (the plugin is symlinked under
``cat/core_plugins``)::

    cd /root/git/grinning-cat-core
    .venv/bin/python /root/git/plugins/cat-efficient-ingestion/test_phase_recovery.py
"""

if __name__ == "__main__":
    import asyncio
    import os
    import time
    import types

    from cat.core_plugins.cat_efficient_ingestion import registry, reembed

    PROCESSING = registry.PHASE_STATE_PROCESSING
    PENDING = registry.PHASE_STATE_PENDING
    COMPLETE = registry.PHASE_STATE_COMPLETE

    def _old_started_at():
        return time.time() - (registry.phase_lease_seconds() + 60)

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

    class FakePhase:
        """External-style phase exposing delete/process with ordered calls."""

        def __init__(self, order, phase_id="fake_phase", fail_delete=False):
            self.id = phase_id
            self.order = order
            self.fail_delete = fail_delete

        async def delete(self, source, cat):
            self.order.append("delete")
            if self.fail_delete:
                raise RuntimeError("delete boom")

        async def process(self, source, cat):
            self.order.append("process")

    def _ccat():
        return types.SimpleNamespace(agent_key="phase_agent", _id="phase_agent")

    def _processing(started_at, owner):
        return {
            "state": PROCESSING,
            "started_at": started_at,
            "reason": None,
            "owner": owner,
            "tag": "t",
            "attempt": 1,
        }

    def _previous_boot_owner():
        # Same pod (so the local pid probe applies), a DIFFERENT boot, dead pid.
        return {"pod": registry._pod_name(), "pid": 999999, "boot": "previous-boot"}

    def _patch(doc, phase_obj, log):
        """Install the recovery seams on ``reembed``; return a restore callable."""
        saved = {}
        for name in (
            "get_phases",
            "merged_phases",
            "reset_phase_entries",
            "record_phase_purge_failed",
            "_purge_phase",
            "log",
        ):
            saved[name] = getattr(reembed, name)

        async def _get_phases(cat=None):
            return [phase_obj]

        async def _merged(ccat, cat):
            return {phase_obj.id: object()}

        async def _reset(agent_id, scope, source, phase_ids):
            phase_obj.order.append(("reset", list(phase_ids)))
            diary = doc.get("completed_phases", {})
            for phase in phase_ids:
                diary.pop(phase, None)
            return {}

        async def _purge_failed(agent_id, scope, source, phase, **kwargs):
            doc.setdefault("completed_phases", {}).setdefault(phase, {})
            doc["completed_phases"][phase]["state"] = PROCESSING
            phase_obj.order.append(("purge-failed", phase))
            return {}

        async def _legacy_purge(*args, **kwargs):
            raise AssertionError("legacy purge must not run when delete() exists")

        reembed.get_phases = _get_phases
        reembed.merged_phases = _merged
        reembed.reset_phase_entries = _reset
        reembed.record_phase_purge_failed = _purge_failed
        reembed._purge_phase = _legacy_purge
        reembed.log = log

        def _restore():
            for name, value in saved.items():
                setattr(reembed, name, value)

        return _restore

    async def _recover(doc, phase_obj, log):
        restore = _patch(doc, phase_obj, log)
        try:
            return await reembed._recover_abandoned_phases(
                _ccat(), None, "collection", "agent", "doc.txt", None, doc
            )
        finally:
            restore()

    async def test_should_abandon_precedence():
        now = time.time()
        # 1. not processing -> never abandoned
        assert registry.should_abandon({"state": COMPLETE}) is False
        assert registry.should_abandon(None) is False
        # 2. missing owner -> lease fallback
        assert registry.should_abandon({"state": PROCESSING, "started_at": now}) is False
        assert registry.should_abandon(
            {"state": PROCESSING, "started_at": _old_started_at()}
        ) is True
        # 3. current boot -> never touched, even past the lease
        assert registry.should_abandon(
            _processing(_old_started_at(), registry.current_phase_owner())
        ) is False
        # 4. live same-pod pid (different boot) -> never touched
        live_local = {"pod": registry._pod_name(), "pid": os.getpid(), "boot": "other"}
        assert registry.should_abandon(_processing(now, live_local)) is False
        # 5. previous boot, same pod, dead pid -> abandoned immediately
        assert registry.should_abandon(_processing(now, _previous_boot_owner())) is True
        # 6. foreign pod: the lease decides (live lease untouched, expired dead)
        foreign_live = {"pod": "foreign-pod", "pid": os.getpid(), "boot": "foreign"}
        assert registry.should_abandon(_processing(now, foreign_live)) is False
        assert registry.should_abandon(
            _processing(_old_started_at(), foreign_live)
        ) is True
        print("PASS (precedence) should_abandon follows processing>owner>boot>pod>lease")

    async def test_a_previous_boot_deleted_before_process():
        order = []
        phase = FakePhase(order)
        doc = {"completed_phases": {"fake_phase": _processing(time.time(), _previous_boot_owner())}}
        result = await _recover(doc, phase, FakeLog())
        assert result is True, result
        # delete FIRST, reset only after, and never complete
        assert order == ["delete", ("reset", ["fake_phase"])], order
        # the normal dispatch then re-runs the phase from zero
        await phase.process("doc.txt", None)
        assert [call for call in order if call in ("delete", "process")] == ["delete", "process"]
        print("PASS (a) previous-boot entry: delete() before process(), then re-run")

    async def test_b_current_boot_untouched():
        order = []
        phase = FakePhase(order)
        doc = {
            "completed_phases": {
                "fake_phase": _processing(_old_started_at(), registry.current_phase_owner())
            }
        }
        result = await _recover(doc, phase, FakeLog())
        assert result is None, result
        assert order == [], order
        assert doc["completed_phases"]["fake_phase"]["state"] == PROCESSING
        assert registry.should_abandon(doc["completed_phases"]["fake_phase"]) is False
        print("PASS (b) current BOOT_EPOCH is never touched, even past the lease")

    async def test_c_live_foreign_pod_untouched():
        order = []
        phase = FakePhase(order)
        foreign_live = {"pod": "foreign-pod", "pid": os.getpid(), "boot": "foreign"}
        doc = {"completed_phases": {"fake_phase": _processing(time.time(), foreign_live)}}
        result = await _recover(doc, phase, FakeLog())
        assert result is None, result
        assert order == [], order
        assert doc["completed_phases"]["fake_phase"]["state"] == PROCESSING
        print("PASS (c) a live foreign pod is left untouched")

    async def test_d_missing_owner_lease_fallback():
        # fresh, no owner -> lease still valid -> untouched
        order = []
        phase = FakePhase(order)
        doc = {"completed_phases": {"fake_phase": _processing(time.time(), None)}}
        result = await _recover(doc, phase, FakeLog())
        assert result is None, result
        assert order == [], order

        # expired, no owner -> lease fallback abandons it
        order = []
        phase = FakePhase(order)
        doc = {"completed_phases": {"fake_phase": _processing(_old_started_at(), None)}}
        result = await _recover(doc, phase, FakeLog())
        assert result is True, result
        assert order[0] == "delete", order
        # a missing owner with a missing started_at is also abandoned (no clock)
        assert registry.should_abandon({"state": PROCESSING}) is True
        print("PASS (d) missing owner falls back to the lease")

    async def test_e_delete_raises_keeps_processing_and_warns():
        order = []
        phase = FakePhase(order, fail_delete=True)
        log = FakeLog()
        doc = {"completed_phases": {"fake_phase": _processing(time.time(), _previous_boot_owner())}}
        result = await _recover(doc, phase, log)
        assert result is False, result
        assert order == ["delete", ("purge-failed", "fake_phase")], order
        assert doc["completed_phases"]["fake_phase"]["state"] == PROCESSING
        assert any(
            "fake_phase" in msg and "doc.txt" in msg and "raised" in msg
            for msg in log.warnings
        ), log.warnings
        print("PASS (e) delete() raising keeps processing, logs WARNING, sweep stops")

    async def test_f_recovered_phase_is_pending_never_complete():
        order = []
        phase = FakePhase(order)
        doc = {"completed_phases": {"fake_phase": _processing(time.time(), _previous_boot_owner())}}
        result = await _recover(doc, phase, FakeLog())
        assert result is True, result
        state = registry.phase_state(doc["completed_phases"].get("fake_phase"))
        assert state == PENDING, state
        assert state != COMPLETE
        derived = registry.derive_source_status(doc, ["fake_phase"])
        assert derived == PROCESSING, derived
        print("PASS (f) recovery resets to pending (absent entry), never complete")

    async def main():
        await test_should_abandon_precedence()
        await test_a_previous_boot_deleted_before_process()
        await test_b_current_boot_untouched()
        await test_c_live_foreign_pod_untouched()
        await test_d_missing_owner_lease_fallback()
        await test_e_delete_raises_keeps_processing_and_warns()
        await test_f_recovered_phase_is_pending_never_complete()
        print("ALL PASS")

    asyncio.run(main())
