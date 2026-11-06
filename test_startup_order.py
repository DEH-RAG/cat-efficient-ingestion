"""Regression tests for the Efficient Ingestion startup ordering.

The revalidation sweep must be *deferred* to the lifecycle hook (never run at
import/construction time, and never synchronously from the hook), and it must
degrade cleanly when the calling agent's plugin manager has not registered the
``ingestion_phase_pending`` hook — for example an agent whose stored
``active_plugins`` predates this plugin. Before the fix the bare
``execute_hook`` raised ``Hook ingestion_phase_pending not present in any
plugin`` in a burst at boot.

Standalone and import-safe (the CAT loader imports every ``.py`` under the
plugin folder): the body lives under ``if __name__ == "__main__":``. Run it
from the host repository::

    cd /root/git/grinning-cat-core
    .venv/bin/python /root/git/plugins/cat-efficient-ingestion/test_startup_order.py
"""
if __name__ == "__main__":
    import asyncio
    import os
    import sys
    import traceback
    from unittest.mock import AsyncMock, patch

    os.environ.setdefault("CAT_REDIS_HOST", "localhost")
    os.environ.setdefault("CAT_REDIS_PORT", "6379")
    os.environ.setdefault("CAT_JWT_SECRET", "test_only_jwt_secret_0123456789_abcdefghij")
    os.environ["CAT_REDIS_DB"] = "1"

    from cat.core_plugins.cat_efficient_ingestion import resume

    async def test_first_revalidation_is_deferred_to_the_lifecycle_hook():
        """``after_lizard_bootstrap`` only schedules the pass: nothing runs inline."""
        scheduled = []

        def fake_ensure_future(coro):
            scheduled.append(coro)
            return coro

        with patch.object(asyncio, "ensure_future", fake_ensure_future):
            with patch.dict(os.environ, {"CAT_INGESTION_RESUME_INTERVAL_SECONDS": "0"}):
                # returning normally proves the hook itself never revalidates inline
                await resume.after_lizard_bootstrap.function(None)

        assert len(scheduled) == 1
        assert scheduled[0].cr_code.co_name == "_startup_pass"
        # the coroutine is only scheduled, not executed: the first revalidation
        # cannot run before the plugin manager has registered the hooks
        assert scheduled[0].cr_frame is not None
        scheduled[0].close()

    async def test_revalidation_without_registered_phase_hook_does_not_raise():
        """A missing phase hook degrades to "no pending phases", never an exception."""

        class _Manager:
            hooks = {}

            async def execute_hook(self, *args, **kwargs):  # pragma: no cover - must not run
                raise AssertionError("the bare dispatcher must not be called for a missing hook")

        class _Cat:
            agent_key = "agent_1"
            plugin_manager = _Manager()

        reembed = AsyncMock()
        with patch.object(resume, "ingestion_canceled", AsyncMock(return_value=False)):
            with patch.object(
                resume,
                "list_statuses",
                AsyncMock(return_value=[{
                    "source": "doc.pdf",
                    "scope": "agent",
                    "type": "file",
                    "status": "completed",
                }]),
            ):
                with patch.object(
                    resume,
                    "backfill_completed_phases",
                    AsyncMock(return_value={"parsing_chunking": {"marker": "m"}}),
                ):
                    with patch.object(resume, "reembed_sources", reembed):
                        await resume._revalidate_agent(object(), "agent_1", ccat=_Cat())

        reembed.assert_not_awaited()

    _tests = [
        test_first_revalidation_is_deferred_to_the_lifecycle_hook,
        test_revalidation_without_registered_phase_hook_does_not_raise,
    ]

    _passed = 0
    _failures = []
    for _test in _tests:
        try:
            asyncio.run(_test())
        except BaseException as exc:  # noqa: BLE001 - report every failure
            _failures.append((_test.__name__, exc, traceback.format_exc()))
            print(f"FAIL {_test.__name__}: {type(exc).__name__}: {exc}")
        else:
            _passed += 1
            print(f"PASS {_test.__name__}")

    print(f"{_passed}/{len(_tests)} passed")
    if _failures:
        for _name, _exc, _tb in _failures:
            print(f"\n--- {_name} ---\n{_tb}")
        sys.exit(1)
