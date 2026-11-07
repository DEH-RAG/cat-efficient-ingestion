"""Standalone, import-safe harness for the CLI/maintenance coordination contract.

Moved here from ``cat-graphrag.my/test_ingestion_safety_fixes.py`` (renamed so
it does not clash with EffING's own Fix 1+3 harness of the same name): the
assertions are about EffING's resume/revalidation sweep, so EffING owns the
test, while the CLI half is still the real MyGRAPH ``maintenance_agent``.

Proves the maintenance marker contract between the GraphRAG maintenance CLI
(MyGRAPH ``maintenance_agent.py``) and the EffING resume/revalidation sweep
(``resume.py``):

  - writing the marker (through ``cat.db.crud``) makes the sweep see the agent
    as busy: ``_resume_agent`` / ``_revalidate_agent`` / ``_pass_for_agent``
    skip it WITHOUT enumerating or claiming its rows;
  - deleting the marker makes the sweep resume normally;
  - the marker carries a generous TTL (2h) so a crashed CLI expires on its own;
  - a marker read failure is fail-safe (treated as "no marker", never blocks
    the sweep forever).

The Cat plugin loader imports EVERY ``.py`` under a plugin folder at
activation, so this whole file is inert on import: every import, class and
test lives inside ``if __name__ == "__main__":``. ``cat.db.crud`` is replaced
with an in-memory fake, so it uses no Redis and no network at all.

The EffING package is guaranteed importable through ``effing_testkit`` (the
real installation when the host provides it, a synthetic one in a bare
checkout). The sibling MyGRAPH checkout owns ``maintenance_agent.py``; point
``MYGRAPH_PLUGIN_DIR`` at it if it is not next to this plugin. Run it as a
plain script from the host repository:

    cd /root/git/grinning-cat-core
    .venv/bin/python /root/git/plugins/cat-efficient-ingestion/tests/test_maintenance_coordination.py
"""

if __name__ == "__main__":
    import asyncio
    import copy
    import os
    import sys
    import time
    import types
    import uuid

    _HERE = os.path.dirname(os.path.abspath(__file__))
    _MYGRAPH_DIR = os.environ.get("MYGRAPH_PLUGIN_DIR") or os.path.join(
        os.path.dirname(os.path.dirname(_HERE)), "cat-graphrag.my"
    )
    if os.path.isdir(_MYGRAPH_DIR) and _MYGRAPH_DIR not in sys.path:
        sys.path.insert(0, _MYGRAPH_DIR)

    sys.path.insert(0, _HERE)
    import effing_testkit

    effing_testkit.install_stub_package()
    try:
        import maintenance_agent
        from cat.core_plugins.cat_efficient_ingestion import resume
        from cat.db import crud

        class FakeCrud:
            """In-memory ``cat.db.crud`` stand-in (no Redis, no network)."""

            def __init__(self):
                self.data = {}
                self.expires = {}
                self.last_store_kwargs = {}

            def _expire(self, key):
                deadline = self.expires.get(key)
                if deadline is not None and time.time() >= deadline:
                    self.data.pop(key, None)
                    self.expires.pop(key, None)

            async def store(
                self, key, value, path="$", nx=False, xx=False, expire=None
            ):
                self.last_store_kwargs = {"expire": expire}
                self.data[key] = copy.deepcopy(value)
                if expire is not None:
                    self.expires[key] = time.time() + expire
                return value

            async def read(self, key, path="$"):
                self._expire(key)
                return copy.deepcopy(self.data.get(key))

            async def delete(self, key, path="$"):
                self.data.pop(key, None)
                self.expires.pop(key, None)

        def _use_fake_crud(fake):
            crud.store = fake.store
            crud.read = fake.read
            crud.delete = fake.delete

        def _restore_crud(original):
            crud.store, crud.read, crud.delete = original

        def _agent_id() -> str:
            return f"safetyfix_{uuid.uuid4().hex}"

        def _assert_test_agent(agent_id: str) -> None:
            # Hard guard: only ever clean up our own uniquely-prefixed key.
            assert agent_id.startswith("safetyfix_"), agent_id

        async def _always(value):
            async def _inner(*args, **kwargs):
                return value

            return _inner

        async def _must_not_run(*args, **kwargs):
            raise AssertionError(
                "the sweep enumerated/claimed rows while the maintenance marker existed"
            )

        async def test_marker_key_and_ttl():
            agent_id = _agent_id()
            _assert_test_agent(agent_id)
            key = maintenance_agent._maintenance_marker_key(agent_id)
            assert key == f"agents:{agent_id}:ingestion:maintenance", key
            assert resume._maintenance_marker_key(agent_id) == key

            original_crud = (crud.store, crud.read, crud.delete)
            fake = FakeCrud()
            _use_fake_crud(fake)
            try:
                await maintenance_agent._set_maintenance_marker(agent_id)

                assert (
                    fake.last_store_kwargs.get("expire")
                    == maintenance_agent._MAINTENANCE_MARKER_TTL_SECONDS
                )
                assert fake.last_store_kwargs["expire"] == 2 * 60 * 60, (
                    fake.last_store_kwargs
                )

                read_back = await crud.read(key)
                assert isinstance(read_back, dict), read_back
                assert read_back.get("owner") == "maintenance", read_back
                assert read_back.get("started_at"), read_back
                assert await resume._maintenance_marker_active(agent_id) is True

                # Simulate the TTL expiring (no sleep): the marker disappears and
                # the sweep treats the agent as free again.
                fake.expires[key] = time.time() - 1
                assert await resume._maintenance_marker_active(agent_id) is False

                # Re-set then delete: same result.
                await maintenance_agent._set_maintenance_marker(agent_id)
                assert await resume._maintenance_marker_active(agent_id) is True
                await maintenance_agent._clear_maintenance_marker(agent_id)
                assert await resume._maintenance_marker_active(agent_id) is False
            finally:
                _restore_crud(original_crud)
            print(
                "PASS marker key/TTL is shared by the CLI and the sweep "
                "(expiry + delete)"
            )

        async def test_sweep_skips_then_resumes():
            agent_id = _agent_id()
            _assert_test_agent(agent_id)
            ccat = types.SimpleNamespace(
                agent_key=agent_id, _id=agent_id, plugin_manager=None
            )
            lizard = types.SimpleNamespace()

            original_canceled = resume.ingestion_canceled
            original_list = resume.list_statuses
            original_crud = (crud.store, crud.read, crud.delete)
            fake = FakeCrud()
            _use_fake_crud(fake)
            resume.ingestion_canceled = await _always(False)
            resume.list_statuses = _must_not_run
            try:
                await maintenance_agent._set_maintenance_marker(agent_id)
                # All three entry points must short-circuit while the marker exists.
                await resume._resume_agent(lizard, agent_id, ccat=ccat)
                await resume._revalidate_agent(lizard, agent_id, ccat=ccat)
                await resume._pass_for_agent(lizard, agent_id)
                print("PASS agent skipped while the maintenance marker is present")

                # Marker deleted -> the sweep enumerates its rows again.
                await maintenance_agent._clear_maintenance_marker(agent_id)
                resume.list_statuses = await _always([])
                await resume._resume_agent(lizard, agent_id, ccat=ccat)
                await resume._revalidate_agent(lizard, agent_id, ccat=ccat)
                print("PASS sweep resumes after the marker is deleted")
            finally:
                resume.ingestion_canceled = original_canceled
                resume.list_statuses = original_list
                await maintenance_agent._clear_maintenance_marker(agent_id)
                _restore_crud(original_crud)

        async def test_read_failure_is_fail_safe():
            agent_id = _agent_id()
            _assert_test_agent(agent_id)
            original_read = crud.read

            async def _boom(*args, **kwargs):
                raise RuntimeError("redis down")

            crud.read = _boom
            try:
                assert await resume._maintenance_marker_active(agent_id) is False
            finally:
                crud.read = original_read
            print("PASS a marker read failure does not block the sweep")

        async def _main():
            await test_marker_key_and_ttl()
            await test_sweep_skips_then_resumes()
            await test_read_failure_is_fail_safe()

        try:
            asyncio.run(_main())
        except AssertionError as exc:
            print(f"FAIL {exc}")
            sys.exit(1)
        print("OK maintenance coordination (3 cases)")
    finally:
        effing_testkit.uninstall_stub_package()
