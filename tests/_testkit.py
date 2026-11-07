"""Shared, import-safe harness for the Efficient Ingestion plugin test suite.

Why import-safe
---------------
The CAT host imports **every** ``.py`` file under a plugin folder (recursively)
when it activates the plugin. Importing this module must therefore execute no
code that touches the host, the network, Redis or the filesystem: at module
level it only defines constants, functions and classes. The host is touched
exclusively inside :func:`bootstrap`, :func:`teardown` and :func:`run`, which
the standalone test scripts call from under ``if __name__ == "__main__":``.

Redis safety
------------
Production uses Redis db ``0``, ``1`` and ``2``; the plugin test suite uses the
dedicated **db 15**. ``bootstrap()`` defaults ``CAT_REDIS_DB`` to 15 **before
importing anything from ``cat``**. The ``Database`` singleton binds its
connection the first time it is used, so a bind to a production db would make
scoped cleanup dangerous. After importing the host, ``bootstrap()`` does not
merely trust the db number: it rejects production/too-low dbs and then *proves*
db15 is ours by scanning it for keys, aborting if any key does not start with
``agents:effingtest`` (emptiness is proved at bootstrap, not assumed). It also
re-checks the live client's connection pool, if one already exists.

Cleanup is always scoped (``agents:<id>:*`` for the ids a test created); this
module never calls ``flushdb``/``flushall``.

Running a suite
---------------
::

    host = bootstrap()
    from cat.db import crud
    ...
    run([test_one, test_two])

``run`` executes all tests in ONE event loop (the async Redis client is a
process singleton), prints ``PASS``/``FAIL`` per test and ``N/M passed``, and
exits non-zero on any failure.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import traceback
from types import SimpleNamespace

#: The only Redis db a plugin test may touch. ``db=0``, ``1`` and ``2`` are
#: production.
DEDICATED_REDIS_DB = 15

#: Every key a plugin test owns must start with this prefix; the guard treats
#: anything else in the dedicated db as a foreign key.
OWNED_KEY_PREFIX = "agents:effingtest"

#: Prefix per test *file*; individual tests add a uuid suffix (e.g.
#: ``effingtest_<uuid>``). Kept here so cleanup helpers can reason about it.
DEFAULT_AGENT_PREFIX = "effingtest"

_bootstrapped = False


def _resolve_redis_host() -> str | None:
    """Pick the RedisJSON host, preferring an explicit env override.

    A plain ``redis-server`` may be listening on the host's ``localhost:6379``
    without the RedisJSON module, which the registry needs. In the dev stack
    the module lives in the ``redis-stack-server`` container, so resolve its
    address via docker before falling back to ``localhost``.
    """
    host = os.environ.get("CAT_REDIS_HOST")
    if host:
        return host
    try:
        result = subprocess.run(
            [
                "docker",
                "inspect",
                "dev-grinning-cat-redis-1",
                "--format",
                "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        ip = result.stdout.strip()
        if result.returncode == 0 and ip:
            return ip
    except Exception:  # noqa: BLE001 - docker is optional; fall back below
        pass
    return "localhost"


def _force_test_env() -> None:
    """Default the isolated Redis db and the minimal host env BEFORE importing cat.

    ``CAT_REDIS_DB`` is only defaulted (not overwritten) so an explicitly forced
    production db reaches :func:`_guard_dedicated_db` and aborts the run instead
    of being silently replaced.
    """
    os.environ.setdefault("CAT_REDIS_HOST", _resolve_redis_host() or "localhost")
    os.environ.setdefault("CAT_REDIS_PORT", "6379")
    os.environ.setdefault(
        "CAT_JWT_SECRET", "test_only_jwt_secret_0123456789_abcdefghij"
    )
    # MUST happen before any ``cat`` import: the Database singleton binds here.
    os.environ.setdefault("CAT_REDIS_DB", str(DEDICATED_REDIS_DB))


def _guard_db_is_ours(db: int) -> None:
    """Prove ``db`` belongs to this suite, not production.

    Opens a short-lived **synchronous** Redis client (the async host client is a
    singleton that may not be usable yet), scans every key and aborts unless all
    of them start with :data:`OWNED_KEY_PREFIX`. Read-only: it never flushes.
    """
    from redis import Redis

    from cat.db.database import get_redis_kwargs

    kwargs = get_redis_kwargs()
    client = Redis(
        host=kwargs["host"],
        port=kwargs["port"],
        db=db,
        password=kwargs.get("password"),
        ssl=kwargs.get("ssl", False),
        decode_responses=True,
    )
    try:
        foreign = [
            key
            for key in client.scan_iter(match="*")
            if not str(key).startswith(OWNED_KEY_PREFIX)
        ]
    finally:
        client.close()

    if foreign:
        examples = ", ".join(repr(key) for key in foreign[:3])
        raise RuntimeError(
            f"Refusing to run plugin tests: Redis db {db} holds "
            f"{len(foreign)} key(s) that do not start with "
            f"{OWNED_KEY_PREFIX!r} (e.g. {examples}). The db is not proven to be "
            "ours; a destructive op could touch another suite's or production "
            "data."
        )


def _guard_dedicated_db() -> None:
    """Abort unless the resolved Redis db is the dedicated test db and is ours."""
    from cat.db.database import Database, get_redis_kwargs

    env_db = get_redis_kwargs().get("db")
    if env_db in (0, 1, 2):
        raise RuntimeError(
            "Refusing to run plugin tests: resolved CAT_REDIS_DB is "
            f"{env_db!r}, a PRODUCTION Redis db (0/1/2). A destructive op could "
            "wipe production data."
        )
    if env_db is None or env_db < 10:
        raise RuntimeError(
            "Refusing to run plugin tests: resolved CAT_REDIS_DB is "
            f"{env_db!r}; the dedicated test db must be >= 10 "
            f"(expected {DEDICATED_REDIS_DB})."
        )
    if env_db != DEDICATED_REDIS_DB:
        raise RuntimeError(
            "Refusing to run plugin tests: resolved CAT_REDIS_DB is "
            f"{env_db!r}, not the dedicated test db {DEDICATED_REDIS_DB}. "
            "A destructive op against db=0 could wipe production data."
        )

    database = Database()
    live = getattr(database, "_async_db", None)
    pool = getattr(live, "connection_pool", None)
    if pool is not None:
        live_db = pool.connection_kwargs.get("db")
        if live_db != DEDICATED_REDIS_DB:
            raise RuntimeError(
                "Refusing to run plugin tests: the live Redis client is bound to "
                f"db={live_db!r}, not the dedicated test db "
                f"{DEDICATED_REDIS_DB}. It was likely created before "
                "bootstrap() defaulted CAT_REDIS_DB."
            )

    _guard_db_is_ours(DEDICATED_REDIS_DB)


class FakePluginManager:
    """Minimal hook dispatcher: records calls, runs registered handlers."""

    def __init__(self, hooks: dict | None = None):
        self.hooks = {} if hooks is None else hooks
        self.calls: list[str] = []

    async def execute_hook(self, name, *args, caller=None, **kwargs):
        self.calls.append(name)
        for hook in self.hooks.get(name, []):
            function = getattr(hook, "function", hook)
            result = function(*args)
            if hasattr(result, "__await__"):
                result = await result
            if isinstance(result, list):
                return result
        return None


class FakeCheshireCat:
    """Minimal agent-scoped object sufficient for EffING's hooks."""

    def __init__(self, agent_key: str, plugin_manager=None, **services):
        self.agent_key = agent_key
        self._id = agent_key
        self.plugin_manager = plugin_manager or FakePluginManager()
        for name, value in services.items():
            setattr(self, name, value)

    def has_pending_source_tasks(self, source) -> bool:  # noqa: ARG002
        return False


#: A StrayCat is the agent-scoped view of a CheshireCat; reuse the fake.
FakeStrayCat = FakeCheshireCat


class FakeLizard:
    """Minimal BillTheLizard stand-in: hands out fake cats by agent key."""

    def __init__(self, plugin_manager=None):
        self.plugin_manager = plugin_manager or FakePluginManager()
        self._cats: dict[str, FakeCheshireCat] = {}

    async def create_cheshire_cat(self, agent_key: str) -> FakeCheshireCat:
        cat = FakeCheshireCat(agent_key, self.plugin_manager)
        self._cats[agent_key] = cat
        return cat

    async def get_cheshire_cat(self, agent_key: str) -> FakeCheshireCat:
        return self._cats.get(agent_key) or await self.create_cheshire_cat(agent_key)


class FakeHandler:
    """A fake plugin hook handler that records its invocations."""

    def __init__(self, result=None):
        self.result = result
        self.calls: list[tuple] = []

    async def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


def new_agent_id(prefix: str = DEFAULT_AGENT_PREFIX) -> str:
    """Return a fresh, unique, test-scoped agent id."""
    import uuid

    return f"{prefix}_{uuid.uuid4().hex}"


def _build_test_client():
    """Build a ``TestClient`` over the plugin's own ``/ingestion`` routes.

    The plugin declares its routes through ``@endpoint`` (CatEndpoint objects,
    not an APIRouter). Activate just those on a throwaway FastAPI app. Returns
    ``None`` if the host's test client stack is unavailable.
    """
    try:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from cat.core_plugins.cat_efficient_ingestion import endpoints as effing_endpoints
        from cat.looking_glass.mad_hatter.decorators.endpoint import CatEndpoint

        app = FastAPI()
        for value in vars(effing_endpoints).values():
            if isinstance(value, CatEndpoint):
                value.activate(app)
        return TestClient(app)
    except Exception:  # noqa: BLE001 - the client is a convenience, not required
        return None


def bootstrap() -> SimpleNamespace:
    """Force the isolated Redis db, build a minimal host and return handles.

    Import-safe: nothing from ``cat`` is imported until this function is
    called. Returns a namespace exposing the host modules and fakes the test
    scripts need.
    """
    global _bootstrapped
    _force_test_env()

    # Deferred host imports: safe now that CAT_REDIS_DB is forced.
    from cat.db import crud
    from cat.core_plugins.cat_efficient_ingestion import plugin, reembed, registry
    from cat.core_plugins.cat_efficient_ingestion import endpoints as effing_endpoints

    _guard_dedicated_db()

    host = SimpleNamespace(
        crud=crud,
        plugin=plugin,
        reembed=reembed,
        registry=registry,
        endpoints=effing_endpoints,
        redis_db=DEDICATED_REDIS_DB,
        new_agent_id=new_agent_id,
        FakePluginManager=FakePluginManager,
        FakeCheshireCat=FakeCheshireCat,
        FakeStrayCat=FakeStrayCat,
        FakeLizard=FakeLizard,
        FakeHandler=FakeHandler,
        client=_build_test_client(),
    )
    _bootstrapped = True
    return host


async def teardown(agent_ids) -> None:
    """Scoped cleanup: delete ``agents:<id>:*`` for each id a test created.

    Never touches another key and never flushes a database.
    """
    if isinstance(agent_ids, str):
        agent_ids = [agent_ids]
    from cat.db import crud

    for agent_id in agent_ids:
        if not agent_id:
            continue
        await crud.destroy(f"agents:{agent_id}:*")


async def _execute_tests(tests) -> tuple[int, int, list[tuple[str, BaseException, str]]]:
    """Run ``tests`` in the *current* event loop, printing one line per test."""
    passed = 0
    failures: list[tuple[str, BaseException, str]] = []
    for test in tests:
        name = getattr(test, "__name__", repr(test))
        try:
            result = test()
            if hasattr(result, "__await__"):
                await result
        except BaseException as exc:  # noqa: BLE001 - report every failure
            failures.append((name, exc, traceback.format_exc()))
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            passed += 1
            print(f"PASS {name}")
    return passed, len(list(tests)), failures


def _report(passed: int, total: int, failures) -> None:
    for name, _exc, tb in failures:
        print(f"\n--- {name} ---\n{tb}")
    print(f"{passed}/{total} passed")
    sys.exit(0 if passed == total else 1)


def run(main) -> None:
    """Run ``main`` (a list/tuple of async test callables) in ONE event loop.

    Prints ``PASS``/``FAIL`` per test and ``N/M passed``; exits non-zero when
    any test failed. A single loop is required because the async Redis client
    is a process singleton.
    """
    tests = list(main)
    passed, total, failures = asyncio.run(_execute_tests(tests))
    _report(passed, total, failures)


# ---------------------------------------------------------------------------
# Real-host runner
# ---------------------------------------------------------------------------


def _redis_client():
    """A short-lived synchronous client on the dedicated db (read/delete only)."""
    from redis import Redis

    from cat.db.database import get_redis_kwargs

    kwargs = get_redis_kwargs()
    return Redis(
        host=kwargs["host"],
        port=kwargs["port"],
        db=DEDICATED_REDIS_DB,
        password=kwargs.get("password"),
        ssl=kwargs.get("ssl", False),
        decode_responses=True,
    )


def _snapshot_dedicated_db() -> set[str]:
    """Return the key set currently living in the dedicated db."""
    client = _redis_client()
    try:
        return {str(key) for key in client.scan_iter(match="*")}
    finally:
        client.close()


def _restore_dedicated_db(before: set[str]) -> None:
    """Delete only the keys a host run created, leaving the prior state intact.

    Booting the real app writes ``system:*`` and ``agents:<id>:*`` keys. They
    are removed here by diffing against :func:`_snapshot_dedicated_db`, so the
    db the strict :func:`_guard_db_is_ours` proves at the next bootstrap stays
    exactly as this run found it. Scoped deletion — never ``flushdb``.
    """
    client = _redis_client()
    try:
        current = {str(key) for key in client.scan_iter(match="*")}
        created = sorted(current - before)
        if created:
            client.delete(*created)
    finally:
        client.close()


def run_with_host(build, *, agent_id: str = "agent_test") -> None:
    """Boot the real host app once, then run the tests ``build(host)`` returns.

    For the suites that genuinely need the live ``CheshireCat``/``BillTheLizard``
    boot, the real routes and dispatcher, and a real (in-memory) vector store.
    The same Redis-isolation contract as :func:`bootstrap` applies: ``CAT_REDIS_DB``
    is forced to :data:`DEDICATED_REDIS_DB` **before** importing ``cat``, the db is
    guarded, and no ``flushdb`` is ever issued — the run's own keys are removed by
    diffing the key set.

    ``build(host)`` is called while the app is live and must return the list of
    zero-argument test callables (they may close over ``host``). ``host`` exposes
    ``lizard``, ``client`` (async httpx over the ASGI app), ``headers``,
    ``agent_id``, ``core_root``, ``memory_client`` and ``new_agent_id``.
    """
    _force_test_env()
    os.environ.setdefault("CAT_API_KEY", "meow_http")
    _guard_dedicated_db()
    before = _snapshot_dedicated_db()

    async def _amain():
        from contextlib import ExitStack
        from inspect import iscoroutinefunction
        from pathlib import Path
        from unittest.mock import patch

        from asgi_lifespan import LifespanManager
        from httpx import ASGITransport, AsyncClient
        from qdrant_client import AsyncQdrantClient

        from cat import utils
        from cat.auth import auth_utils
        from cat.looking_glass.mad_hatter.plugin import Plugin
        from cat.routes import routes_utils
        from cat.services.factory.vector_db import QdrantHandler
        from cat.startup import create_app

        core_root = Path(utils.__file__).resolve().parent.parent
        memory_client = AsyncQdrantClient(":memory:")
        stack = ExitStack()
        try:

            def _mock_qdrant_init(self, *args, **kwargs):
                super(QdrantHandler, self).__init__()
                self._client = memory_client
                self.save_memory_snapshots = kwargs.get("save_memory_snapshots", False)

            stack.enter_context(patch.object(QdrantHandler, "__init__", _mock_qdrant_init))
            stack.enter_context(patch.object(QdrantHandler, "close", lambda self: None))
            stack.enter_context(
                patch.object(
                    utils,
                    "get_plugins_path",
                    lambda: str(core_root / "tests/mocks/mock_plugin_folder") + "/",
                )
            )
            stack.enter_context(
                patch.object(
                    utils,
                    "get_file_manager_root_storage_path",
                    lambda: str(core_root / "tests/data/storage"),
                )
            )

            async def _noop_install_requirements(self):
                return None

            stack.enter_context(
                patch.object(Plugin, "_install_requirements", _noop_install_requirements)
            )
            stack.enter_context(
                patch.object(
                    auth_utils, "extract_agent_id_from_request", lambda *a, **k: agent_id
                )
            )

            async def _immediate_task(background_tasks, func, *args, **kwargs):
                if iscoroutinefunction(func):
                    await func(*args, **kwargs)
                else:
                    func(*args, **kwargs)

            stack.enter_context(
                patch.object(routes_utils, "run_background_task", _immediate_task)
            )

            utils.singleton.instances.clear()
            app = create_app()
            async with LifespanManager(app):
                async with AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://test"
                ) as client:
                    client._fastapi_test_app = app
                    host = SimpleNamespace(
                        app=app,
                        lizard=app.state.lizard,
                        client=client,
                        memory_client=memory_client,
                        agent_id=agent_id,
                        headers={
                            "X-Agent-ID": agent_id,
                            "Authorization": "Bearer meow_http",
                        },
                        core_root=core_root,
                        new_agent_id=new_agent_id,
                    )
                    tests = build(host)
                    if hasattr(tests, "__await__"):
                        tests = await tests
                    return await _execute_tests(list(tests))
        finally:
            stack.close()
            utils.singleton.instances.clear()
            try:
                collections = await memory_client.get_collections()
                for collection in collections.collections:
                    await memory_client.delete_collection(collection.name)
            except Exception:  # noqa: BLE001 - cleanup must not hide the outcome
                pass
            await memory_client.close()

    try:
        passed, total, failures = asyncio.run(_amain())
    except BaseException as exc:  # noqa: BLE001 - report boot/teardown failures
        traceback.print_exc()
        _restore_dedicated_db(before)
        print(f"FAIL host boot: {type(exc).__name__}: {exc}")
        print("0/1 passed")
        sys.exit(1)

    _restore_dedicated_db(before)
    _report(passed, total, failures)
