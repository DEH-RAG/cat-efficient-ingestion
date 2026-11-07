"""Standalone, import-safe test: EffING uses the unified delete-marker contract.

The plugin loader imports EVERY ``.py`` under a plugin folder, so the whole body
lives inside ``if __name__ == "__main__":`` and importing this file executes
zero code. Run it from the host repository (the plugin is symlinked under
``cat/core_plugins``)::

    cd /root/git/grinning-cat-core
    .venv/bin/python /root/git/plugins/cat-efficient-ingestion/test_delete_marker_schema.py

Covers the one-schema contract owned by ``cat.looking_glass.agent_deletion``:

  (a) ``set_delete_marker`` writes EXACTLY
      ``{"delete_marker": True, "started_at": <unix float>}``;
  (b) a legacy marker ``{"delete_marker": True}`` (no ``started_at``) is still
      treated as present (presence-only reads), so old data keeps working;
  (c) the registry helpers ARE the core helpers (imported, not mirrored), so a
      second schema implementation cannot appear.

Redis isolation: ``CAT_REDIS_DB=1`` is forced BEFORE importing ``cat``; every
key lives under a unique ``agents:effingtest_<uuid>:`` namespace and is
destroyed after each test.
"""
if __name__ == "__main__":
    import asyncio
    import os
    import sys
    import traceback
    import uuid

    os.environ.setdefault("CAT_REDIS_HOST", "localhost")
    os.environ.setdefault("CAT_REDIS_PORT", "6379")
    os.environ.setdefault("CAT_JWT_SECRET", "test_only_jwt_secret_0123456789_abcdefghij")
    os.environ["CAT_REDIS_DB"] = "1"

    from cat.core_plugins.cat_efficient_ingestion import registry
    from cat.db import crud
    from cat.looking_glass import agent_deletion

    agent_id = f"effingtest_{uuid.uuid4().hex}"
    key = registry.delete_marker_key(agent_id)

    def _unwrap(doc):
        """Unwrap the RedisJSON one-element list wrapper."""
        while isinstance(doc, list) and len(doc) == 1:
            doc = doc[0]
        return doc

    async def test_written_schema_is_the_core_schema():
        await registry.set_delete_marker(agent_id)

        doc = _unwrap(await crud.read(key))
        assert isinstance(doc, dict), doc
        assert set(doc) == {"delete_marker", "started_at"}, doc
        assert doc["delete_marker"] is True
        assert isinstance(doc["started_at"], float), doc
        assert doc["started_at"] > 0

    async def test_legacy_marker_without_started_at_is_present():
        await registry.clear_delete_marker(agent_id)
        await crud.store(key, {"delete_marker": True})

        assert await registry.has_delete_marker(agent_id) is True
        assert await registry.ingestion_canceled(agent_id) is True
        # The core helper agrees: same key, same presence-only read.
        assert await agent_deletion.has_delete_marker(agent_id) is True

    async def test_registry_helpers_are_the_core_helpers():
        assert registry.set_delete_marker is agent_deletion.set_delete_marker
        assert registry.has_delete_marker is agent_deletion.has_delete_marker
        assert registry.clear_delete_marker is agent_deletion.clear_delete_marker
        assert registry.delete_marker_key is agent_deletion.delete_marker_key
        assert registry.delete_marker_key(agent_id) == agent_deletion.delete_marker_key(agent_id)

    _tests = [
        test_written_schema_is_the_core_schema,
        test_legacy_marker_without_started_at_is_present,
        test_registry_helpers_are_the_core_helpers,
    ]

    async def _main():
        # A SINGLE event loop: the async Redis client is a process singleton.
        passed = 0
        failures = []
        for test in _tests:
            await crud.store(f"agents:{agent_id}:agent", [{"name": "x", "value": 1}])
            try:
                await test()
            except BaseException as exc:  # noqa: BLE001 - report every failure
                failures.append((test.__name__, exc, traceback.format_exc()))
                print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}")
            else:
                passed += 1
                print(f"PASS {test.__name__}")
            finally:
                await crud.destroy(f"agents:{agent_id}:*")

        print(f"{passed}/{len(_tests)} passed")
        if failures:
            for name, _exc, tb in failures:
                print(f"\n--- {name} ---\n{tb}")
            return 1
        return 0

    sys.exit(asyncio.run(_main()))
