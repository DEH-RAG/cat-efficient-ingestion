"""Standalone, import-safe regression tests for the legacy (v1) diary livelock.

A v1 diary entry has ``marker`` / ``settings_version`` / ``deps`` but NO
``state`` and NO ``tag``. It used to be an infinite loop:

- the probe (``is_phase_stale``) treated "no tag" as always stale, so the
  phase stayed ``pending``;
- the guard (``phase_state``) treated "no state" as ``complete``, so it
  skipped the body — and the body is the ONLY writer of the v2 ``tag``.

The checks must agree: a legacy entry whose recorded inputs still match the
current configuration is FRESH (no re-parse, backfilled to v2); a legacy entry
whose configuration changed is STALE (the guard runs the body).

Covers:
  (a) v1 matching marker + no tag -> NOT stale, the guard SKIPS the body;
  (b) v1 changed marker -> STALE, the guard RUNS the body;
  (c) a current v1 entry gets its ``tag`` backfilled (v2) without re-running;
  (d) the REAL ``reembed_sources`` loop terminates on a v1 diary instead of
      spinning forever on ``parsing_chunking``.

The plugin loader imports EVERY ``.py`` under a plugin folder, so the whole
body lives inside ``if __name__ == "__main__":`` and importing this file
executes zero code. Run it as a plain script from the host repository::

    cd /root/git/grinning-cat-core
    .venv/bin/python /root/git/plugins/cat-efficient-ingestion/tests/test_legacy_diary_livelock.py
"""

if __name__ == "__main__":
    import asyncio
    import sys
    import types
    from contextlib import asynccontextmanager

    from cat.db.cruds import settings as crud_settings
    from cat.core_plugins.cat_efficient_ingestion import phases, reembed, registry

    AGENT = "legacy_agent"
    SCOPE = "agent"
    SRC = "ud2_2026_a.pdf"
    MATCHING_MARKER = '{"config":"chunker","value":{"chunk_size":400}}'

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

    def _ccat():
        return types.SimpleNamespace(agent_key=AGENT, _id=AGENT)

    def _legacy_entry(marker=MATCHING_MARKER, deps=None):
        return {
            "marker": marker,
            "settings_version": None,
            "deps": deps or {},
        }

    def _diary(marker=MATCHING_MARKER):
        return {
            "parsing_chunking": _legacy_entry(marker),
            "embedding": _legacy_entry(
                marker, {"parsing_chunking": marker}
            ),
        }

    class Harness:
        """Patch the store, hooks and locks used by the guard / dispatch."""

        def __init__(self, diary, marker=MATCHING_MARKER):
            self.diary = diary
            self.marker = marker
            self.entry_writes = []
            self.log = FakeLog()
            self.saved = {}

        def _doc(self):
            return {
                "source": SRC,
                "scope": SCOPE,
                "type_": "file",
                "status": "processing",
                "completed_phases": self.diary,
            }

        def __enter__(self):
            self.saved = {
                "get_status": reembed.get_status,
                "registry_get_status": registry.get_status,
                "registry_write": registry._write_phase_entry,
                "phase_lock": reembed.phase_lock,
                "execute_phase_on_owner": phases.execute_phase_on_owner,
                "settings_by_category": crud_settings.get_settings_by_category,
                "reembed_log": reembed.log,
                "phases_log": phases.log,
            }

            async def _get_status(*args, **kwargs):
                return self._doc()

            async def _write_entry(agent_id, scope, source, phase, entry, **kwargs):
                self.entry_writes.append((phase, dict(entry)))
                self.diary[phase] = dict(entry)
                return self._doc()

            async def _marker(*args, **kwargs):
                return self.marker

            async def _no_settings(*args, **kwargs):
                return None

            @asynccontextmanager
            async def _noop_lock(*args, **kwargs):
                yield

            reembed.get_status = _get_status
            registry.get_status = _get_status
            registry._write_phase_entry = _write_entry
            reembed.phase_lock = _noop_lock
            phases.execute_phase_on_owner = _marker
            crud_settings.get_settings_by_category = _no_settings
            reembed.log = self.log
            phases.log = self.log
            return self

        def __exit__(self, *exc):
            reembed.get_status = self.saved["get_status"]
            registry.get_status = self.saved["registry_get_status"]
            registry._write_phase_entry = self.saved["registry_write"]
            reembed.phase_lock = self.saved["phase_lock"]
            phases.execute_phase_on_owner = self.saved["execute_phase_on_owner"]
            crud_settings.get_settings_by_category = self.saved["settings_by_category"]
            reembed.log = self.saved["reembed_log"]
            phases.log = self.saved["phases_log"]
            return False

    async def test_a_matching_legacy_entry_is_fresh_and_body_skipped():
        with Harness(_diary()) as h:
            async with reembed._phase_guard(
                _ccat(), _ccat(), SCOPE, SRC, "parsing_chunking"
            ) as should_run:
                assert should_run is False, (
                    "a legacy entry with an unchanged configuration must NOT re-run"
                )
            # migration: the skipped entry was backfilled to v2 with a tag
            entry = h.diary["parsing_chunking"]
            assert entry.get("tag"), f"tag was not backfilled: {entry}"
            assert entry.get("state") == registry.PHASE_STATE_COMPLETE, entry
            assert any("skipping its body" in m for m in h.log.infos), h.log.infos
        print("PASS (a) v1 matching marker -> fresh, guard skips body, tag backfilled")

    async def test_b_changed_legacy_marker_is_stale_and_body_runs():
        with Harness(_diary(), marker='{"config":"chunker","value":{"chunk_size":999}}'):
            async with reembed._phase_guard(
                _ccat(), _ccat(), SCOPE, SRC, "parsing_chunking"
            ) as should_run:
                assert should_run is True, (
                    "a legacy entry whose configuration changed MUST re-run"
                )
        print("PASS (b) v1 changed marker -> stale, guard runs the body")

    async def test_c_backfill_is_idempotent_and_preserves_inputs():
        with Harness(_diary()) as h:
            current_tag = "tag-123"
            wrote = await registry.backfill_phase_tag(
                AGENT, SCOPE, SRC, "parsing_chunking", current_tag,
                type_="file", chat_id=None,
            )
            assert wrote is True, "a current v1 entry must be backfilled"
            entry = h.diary["parsing_chunking"]
            assert entry["tag"] == current_tag, entry
            assert entry["state"] == registry.PHASE_STATE_COMPLETE, entry
            assert entry["marker"] == MATCHING_MARKER, entry
            assert entry["deps"] == {}, entry
            # A second call is a no-op (already v2).
            again = await registry.backfill_phase_tag(
                AGENT, SCOPE, SRC, "parsing_chunking", current_tag,
                type_="file", chat_id=None,
            )
            assert again is False, "a v2 entry must never be rewritten"
        print("PASS (c) v1 entry backfilled to v2 once, inputs preserved")

    class _DispatchHarness(Harness):
        """Drive the real reembed_sources loop, stubbing only the phase bodies."""

        def __enter__(self):
            super().__enter__()
            self.saved.update({
                "resolve_rabbit_hole": reembed._resolve_rabbit_hole,
                "source_has_points": reembed._source_has_stored_points,
                "start_heartbeat": reembed._start_heartbeat,
                "stop_heartbeat": reembed._stop_heartbeat,
                "claim": reembed.claim_source_for_resume,
                "complete_source": reembed._complete_source,
                "run_parsing": reembed._run_parsing_phase,
                "embed": reembed._embed_phase,
                "canceled": reembed.ingestion_canceled,
                "shared_hook": reembed.execute_shared_hook,
                "split": reembed.split_oversized,
                "set_status": reembed.set_status,
                "set_phase": reembed.set_phase,
            })

            def _value(value):
                async def _inner(*args, **kwargs):
                    return value
                return _inner

            async def _claim(*args, **kwargs):
                return {"claimed": True}

            reembed._resolve_rabbit_hole = _value(object())
            reembed._source_has_stored_points = _value(True)
            reembed._start_heartbeat = _value(None)
            reembed._stop_heartbeat = _value(None)
            reembed.claim_source_for_resume = _claim
            reembed._complete_source = _value(None)
            reembed.ingestion_canceled = _value(False)
            reembed.execute_shared_hook = _value(None)
            reembed.split_oversized = lambda docs, embedder: docs
            reembed.set_status = _value(None)
            reembed.set_phase = _value(None)

            async def _boom(*args, **kwargs):
                raise AssertionError("a fresh v1 diary must NOT re-run a phase body")

            reembed._run_parsing_phase = _boom
            reembed._embed_phase = _boom
            return self

        def __exit__(self, *exc):
            for name in (
                "resolve_rabbit_hole", "source_has_points", "start_heartbeat",
                "stop_heartbeat", "claim", "complete_source", "run_parsing",
                "embed", "canceled", "shared_hook", "split", "set_status",
                "set_phase",
            ):
                setattr(reembed, name, self.saved[name])
            return super().__exit__(*exc)

    async def test_d_reembed_sources_terminates_on_v1_diary():
        class Handler:
            async def get_all_tenant_points(self, *args, **kwargs):
                return [], None

        ccat = types.SimpleNamespace(
            agent_key=AGENT,
            _id=AGENT,
            vector_memory_handler=Handler(),
            chunker=types.SimpleNamespace(name="chunker"),
        )

        async def _embedder():
            return types.SimpleNamespace(name="embedder")

        ccat.embedder = _embedder
        source = types.SimpleNamespace(
            name=SRC, path=SRC, metadata={}
        )
        with _DispatchHarness(_diary()) as h:
            # The regression: without the fix this loop never returns (the
            # probe keeps re-reporting parsing_chunking). A timeout is a FAIL.
            await asyncio.wait_for(
                reembed.reembed_sources(
                    ccat, "declarative", [source], caller_cat=ccat
                ),
                timeout=5.0,
            )
            assert h.entry_writes, "the v1 diary was not migrated at completion"
            assert h.diary["parsing_chunking"].get("tag"), h.diary
            assert h.diary["embedding"].get("tag"), h.diary
        print("PASS (d) real reembed_sources loop terminates on a v1 diary")

    async def _main():
        await test_a_matching_legacy_entry_is_fresh_and_body_skipped()
        await test_b_changed_legacy_marker_is_stale_and_body_runs()
        await test_c_backfill_is_idempotent_and_preserves_inputs()
        await test_d_reembed_sources_terminates_on_v1_diary()

    try:
        asyncio.run(_main())
    except AssertionError as exc:
        print(f"FAIL {exc}")
        sys.exit(1)
    except asyncio.TimeoutError:
        print("FAIL reembed_sources did not terminate (legacy livelock)")
        sys.exit(1)
    print("OK legacy diary livelock")
