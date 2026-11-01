"""Standalone, import-safe tests for the per-phase not-ready budget.

Covers:
  (a) a phase declaring 300s becomes ``error`` after 300s of not-ready
      (the clock is driven, nothing sleeps);
  (b) a phase with no declaration still uses the generic 3600s;
  (c) the declared env override wins over the declared value;
  (d) the reason string (and the WARNING log) names the budget actually used;
  (e) a non-finite env override (``inf``) is rejected, falling back to the
      declared value and then to the generic 3600s;
  (f) a malformed generic env value falls back to 3600s with a WARNING and
      no exception;
  (g) ``nan`` and negative values are rejected at both levels;
  (h) a declared numeric string (``"300"``) is coerced and used;
  (i) a declared value of the wrong type (``None``, object, list, bool,
      ``nan``, ``inf``, negative, garbage) never raises: it is ignored with a
      WARNING and resolution falls through to the generic 3600s.

The plugin loader imports EVERY ``.py`` under a plugin folder, so the whole
body lives inside ``if __name__ == "__main__":`` and importing this file
executes zero code.

Run from the host repository::

    cd /root/git/grinning-cat-core
    .venv/bin/python /root/git/plugins/cat-efficient-ingestion/test_phase_not_ready_budget.py
"""

if __name__ == "__main__":
    import asyncio
    import copy
    import os
    import sys
    import types

    from cat.core_plugins.cat_efficient_ingestion import phases as phases_module
    from cat.core_plugins.cat_efficient_ingestion import reembed
    from cat.core_plugins.cat_efficient_ingestion.phases import (
        IngestionPhase,
        PhaseResult,
    )

    _LLM_ENV = "CAT_INGESTION_LLM_NOT_READY_MAX_SECONDS"
    _GENERIC_ENV = "CAT_INGESTION_NOT_READY_MAX_SECONDS"

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

        def __init__(self):
            self.doc = {
                "source": "doc.txt",
                "scope": "agent",
                "status": "processing",
                "phase": "graph_llm_extraction",
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

    class FakeClock:
        def __init__(self, start=1000.0):
            self.now = start

        def time(self):
            return self.now

    class FakePhase(IngestionPhase):
        """A phase carrying (or not) a declared not-ready budget."""

        def __init__(self, phase_id, budget=None, env=None):
            self.id = phase_id
            self.not_ready_budget_seconds = budget
            self.not_ready_budget_env = env

        async def applies_to(self, cat, deps_state):
            return True, None

        async def process(self, source, cat):
            return PhaseResult(status="done")

        async def delete(self, source, cat):
            return None

    class Harness:
        """Patch/restore the reembed seams used by ``_run_external_phase``."""

        def __init__(self, phases):
            self.phases = phases
            self.saved = {}

        def __enter__(self):
            for name in (
                "get_status", "set_status", "_set_status",
                "backfill_completed_phases", "merged_phases",
                "execute_phase_on_owner", "asyncio", "log", "get_phases", "time",
            ):
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

    def _ccat():
        return types.SimpleNamespace(agent_key="budget_agent", _id="budget_agent")

    def _install(harness, state, clock, sleeps, retry_after=0):
        async def _get_phases(cat=None):
            return harness.phases

        async def _call_phase(*args, **kwargs):
            return {"status": "not_ready", "retry_after": retry_after}

        async def _sleep(seconds):
            sleeps.append(seconds)
            clock.now += sleeps_step[0]

        sleeps_step = [0.0]

        reembed.get_status = state.get_status
        reembed.set_status = state.set_status
        reembed._set_status = state.set_status_error
        reembed.backfill_completed_phases = _async_return({})
        reembed.merged_phases = _async_return({})
        reembed.execute_phase_on_owner = _call_phase
        reembed.asyncio = types.SimpleNamespace(sleep=_sleep)
        reembed.get_phases = _get_phases
        reembed.time = types.SimpleNamespace(time=clock.time)
        return sleeps_step

    def _clear_env():
        os.environ.pop(_LLM_ENV, None)
        os.environ.pop(_GENERIC_ENV, None)

    async def _run(phases, step):
        state = FakeStatus()
        clock = FakeClock()
        sleeps = []
        log = FakeLog()
        with Harness(phases) as harness:
            reembed.log = log
            step_box = _install(harness, state, clock, sleeps)
            step_box[0] = step
            ok = await reembed._run_external_phase(
                _ccat(), None, "agent", "doc.txt", "graph_llm_extraction", None
            )
        return ok, state, log, sleeps

    async def test_a_declared_budget_expires_at_300():
        _clear_env()
        phase = FakePhase(
            "graph_llm_extraction",
            budget=300.0,
            env=_LLM_ENV,
        )
        ok, state, log, sleeps = await _run([phase], step=100.0)
        assert ok is False, "a not-ready phase past its budget must fail"
        assert state.doc["status"] == "error", state.doc
        assert len(sleeps) == 3, sleeps  # 0s, 100s, 200s -> 300s expires
        error = state.doc.get("error") or ""
        assert "graph_llm_extraction" in error, error
        assert "max 300s" in error, error
        print("PASS (a) declared 300s budget expires at 300s -> error")

    async def test_b_undeclared_phase_uses_generic_3600():
        _clear_env()
        phase = FakePhase("graph_llm_extraction")
        ok, state, log, sleeps = await _run([phase], step=1000.0)
        assert ok is False
        assert state.doc["status"] == "error", state.doc
        assert len(sleeps) == 4, sleeps  # 0s, 1000s, 2000s, 3000s -> 4000s
        error = state.doc.get("error") or ""
        assert "max 3600s" in error, error
        print("PASS (b) undeclared phase keeps the generic 3600s")

    async def test_c_env_override_wins_over_declared_value():
        _clear_env()
        os.environ[_LLM_ENV] = "120"
        phase = FakePhase(
            "graph_llm_extraction",
            budget=300.0,
            env=_LLM_ENV,
        )
        ok, state, log, sleeps = await _run([phase], step=60.0)
        assert ok is False
        assert state.doc["status"] == "error", state.doc
        assert len(sleeps) == 2, sleeps  # 0s, 60s -> 120s expires
        error = state.doc.get("error") or ""
        assert "max 120s" in error, error
        assert "max 300s" not in error, error
        print("PASS (c) env override 120s wins over declared 300s")

    async def test_d_reason_names_the_budget_actually_used():
        _clear_env()
        phase = FakePhase(
            "graph_llm_extraction",
            budget=300.0,
            env=_LLM_ENV,
        )
        ok, state, log, sleeps = await _run([phase], step=300.0)
        assert ok is False
        error = state.doc.get("error") or ""
        assert "not ready after" in error, error
        assert "max 300s" in error, error
        assert "max 3600s" not in error, error
        warnings = [m for m in log.warnings if "not ready after" in m]
        assert warnings and "max 300s" in warnings[0], log.warnings
        print("PASS (d) reason and WARNING name the 300s budget used")

    async def test_e_inf_env_falls_back_to_declared_then_generic():
        _clear_env()
        os.environ[_LLM_ENV] = "inf"
        declared = FakePhase(
            "graph_llm_extraction",
            budget=300.0,
            env=_LLM_ENV,
        )
        ok, state, log, sleeps = await _run([declared], step=100.0)
        assert ok is False
        assert state.doc["status"] == "error", state.doc
        assert len(sleeps) == 3, sleeps  # 0s, 100s, 200s -> 300s expires
        error = state.doc.get("error") or ""
        assert "max 300s" in error, error

        undeclared = FakePhase("graph_llm_extraction", env=_LLM_ENV)
        ok, state, log, sleeps = await _run([undeclared], step=1000.0)
        assert ok is False
        assert state.doc["status"] == "error", state.doc
        assert len(sleeps) == 4, sleeps  # 0s..3000s -> 4000s
        error = state.doc.get("error") or ""
        assert "max 3600s" in error, error
        print(
            "PASS (e) env inf rejected -> declared 300s, then generic 3600s"
        )

    async def test_f_malformed_generic_env_falls_back_to_3600():
        _clear_env()
        os.environ[_GENERIC_ENV] = "not-a-number"
        try:
            phase = FakePhase("graph_llm_extraction")
            ok, state, log, sleeps = await _run([phase], step=1000.0)
            assert ok is False
            assert state.doc["status"] == "error", state.doc
            assert len(sleeps) == 4, sleeps  # 0s..3000s -> 4000s
            error = state.doc.get("error") or ""
            assert "max 3600s" in error, error
            warnings = [
                m for m in log.warnings if "CAT_INGESTION_NOT_READY_MAX_SECONDS" in m
            ]
            assert warnings, log.warnings
            assert "not-a-number" in warnings[0], warnings[0]
        finally:
            _clear_env()
        print("PASS (f) malformed generic env -> 3600s WARNING, no exception")

    async def test_g_nan_and_negative_values_are_rejected():
        _clear_env()
        os.environ[_LLM_ENV] = "nan"
        assert FakePhase("graph_llm_extraction", env=_LLM_ENV).not_ready_budget() is None
        assert (
            FakePhase(
                "graph_llm_extraction", budget=300.0, env=_LLM_ENV
            ).not_ready_budget()
            == 300.0
        )
        os.environ[_LLM_ENV] = "-5"
        assert FakePhase("graph_llm_extraction", env=_LLM_ENV).not_ready_budget() is None
        assert (
            FakePhase(
                "graph_llm_extraction", budget=float("nan")
            ).not_ready_budget()
            is None
        )
        assert (
            FakePhase(
                "graph_llm_extraction", budget=-1.0
            ).not_ready_budget()
            is None
        )
        _clear_env()
        print("PASS (g) nan and negative values rejected at env and declared levels")

    async def test_h_declared_string_is_coerced():
        _clear_env()
        phase = FakePhase("graph_llm_extraction", budget="300")
        assert phase.not_ready_budget() == 300.0
        ok, state, log, sleeps = await _run([phase], step=100.0)
        assert ok is False
        assert state.doc["status"] == "error", state.doc
        assert len(sleeps) == 3, sleeps  # 0s, 100s, 200s -> 300s expires
        error = state.doc.get("error") or ""
        assert "max 300s" in error, error
        print("PASS (h) declared string '300' coerced to 300.0 and used")

    async def test_i_declared_garbage_never_raises():
        _clear_env()
        original_log = phases_module.log
        phases_module.log = FakeLog()
        bad_values = (
            object(), "", "abc", [300], {"seconds": 300}, True,
            float("nan"), float("inf"), -1.0,
        )
        try:
            for bad in bad_values:
                phase = FakePhase("graph_llm_extraction", budget=bad)
                assert phase.not_ready_budget() is None, bad
            warnings = phases_module.log.warnings
            assert len(warnings) >= len(bad_values), warnings
            assert all("not_ready_budget_seconds" in w for w in warnings), warnings
            # A declared None is "not declared", not a bad value: no warning.
            assert (
                FakePhase("graph_llm_extraction", budget=None).not_ready_budget()
                is None
            )
            assert len(phases_module.log.warnings) == len(bad_values)
        finally:
            phases_module.log = original_log

        # In a real run a garbage declared value falls back to the generic
        # 3600s instead of raising out of the resolver.
        phase = FakePhase("graph_llm_extraction", budget="garbage")
        ok, state, log, sleeps = await _run([phase], step=1000.0)
        assert ok is False
        assert state.doc["status"] == "error", state.doc
        assert len(sleeps) == 4, sleeps  # 0s..3000s -> 4000s
        assert "max 3600s" in (state.doc.get("error") or ""), state.doc
        print(
            "PASS (i) declared garbage/None never raises; "
            "garbage falls back to generic 3600s"
        )

    async def _main():
        await test_a_declared_budget_expires_at_300()
        await test_b_undeclared_phase_uses_generic_3600()
        await test_c_env_override_wins_over_declared_value()
        await test_d_reason_names_the_budget_actually_used()
        await test_e_inf_env_falls_back_to_declared_then_generic()
        await test_f_malformed_generic_env_falls_back_to_3600()
        await test_g_nan_and_negative_values_are_rejected()
        await test_h_declared_string_is_coerced()
        await test_i_declared_garbage_never_raises()

    try:
        asyncio.run(_main())
    except AssertionError as exc:
        print(f"FAIL {exc}")
        sys.exit(1)
    finally:
        _clear_env()
    print("OK per-phase not-ready budget (nine tests)")
