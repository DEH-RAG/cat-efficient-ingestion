"""Tests for the ``/ingestion/status`` endpoint (read-time reconcile).

Uses the repo's route-test conventions: httpx AsyncClient + ASGITransport via
the ``secure_client`` fixture, agent header from ``tests.utils.agent_id``,
Redis db=1 (flushed by the autouse ``encapsulate_each_test`` fixture).

The endpoint builds a *fresh* ``CheshireCat`` per request (via
``lizard.get_cheshire_cat``), so the file manager is monkeypatched at the
``ServiceProvider`` level and files are written straight to the mocked storage
root (``tests/data/storage``) rather than through the fixture's cat.

The engine-configuration endpoints (``/ingestion/settings*``) are NOT tested
here as plugin routes anymore: they are served by the CAT core route
(``cat/routes/ingestion.py``, ``AuthResource.INGESTION``) and the plugin must
not register them. The settings tests below assert the CORE route's behavior.
"""
import importlib
import os
import urllib.parse

import pytest

from cat.core_plugins.base_plugin.file_managers.custom import LocalFileManager
from cat.plugins.cat_efficient_ingestion.registry import (
    IngestionStatus,
    get_status,
    set_status,
)
from cat.db import crud
from cat.db.database import DEFAULT_AGENTS_KEY, DEFAULT_CONVERSATIONS_KEY
from cat.services.service_provider import ServiceProvider
from tests.utils import agent_id, chat_id, create_new_user, new_user_password

STORAGE_ROOT = "tests/data/storage"


@pytest.fixture(autouse=True)
async def _upstream_enumeration_shim(monkeypatch, client):
    """Stand-in for the deferred ``get_agents_main_keys`` port (plan todo 2).

    Upstream CAT's ``cat.db.cruds.settings.get_agents_main_keys`` returns the
    *agent ids* of the matching keys (``k.split(":")[1]``), while the registry
    relies on MyCAT's semantics (keys with the ``agents:`` prefix and the
    ``:agent`` suffix stripped). Until the MyCAT version is ported to the core
    (plan todo 2), patch the registry's reference with the MyCAT semantics so
    the real plugin code (enumeration -> read -> reconcile -> purge) is
    exercised end-to-end. Both module instances are patched: the plugin may be
    loaded as a core plugin (``cat.core_plugins.*``) or installed into the
    plugins folder (``cat.plugins.*``).

    Depends on ``client`` so the patch is applied AFTER the app is created:
    plugin activation ``importlib.reload``s every plugin module, which would
    otherwise overwrite the patch.
    """

    async def _mycat_semantics(pattern):
        from cat.db.database import get_async_db

        keys = [k async for k in get_async_db().scan_iter(pattern)]
        return sorted({k.removeprefix("agents:").removesuffix(":agent") for k in keys})

    for mod_name in (
        "cat.plugins.cat_efficient_ingestion.registry",
        "cat.core_plugins.cat_efficient_ingestion.registry",
    ):
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        monkeypatch.setattr(mod, "get_agents_main_keys", _mycat_semantics)


def _write_storage_file(rel_path: str, content: str = "hello") -> None:
    """Write a file directly into the mocked file-manager storage root."""
    full = os.path.join(STORAGE_ROOT, rel_path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w") as f:
        f.write(content)


def _use_local_file_manager(monkeypatch) -> None:
    """Make freshly-created CheshireCats use the on-disk LocalFileManager."""

    async def fake_get_file_manager(self, agent_key, plugin_manager):
        return LocalFileManager()

    monkeypatch.setattr(ServiceProvider._class, "get_file_manager", fake_get_file_manager)


async def test_ingestion_status_returns_seeded(secure_client, secure_client_headers, cheshire_cat, monkeypatch):
    _use_local_file_manager(monkeypatch)
    _write_storage_file(os.path.join(agent_id, "doc.pdf"))

    await set_status(agent_id, "agent", "doc.pdf", type_="file", status=IngestionStatus.COMPLETED)

    response = await secure_client.get("/ingestion/status", headers=secure_client_headers)
    assert response.status_code == 200
    json = response.json()
    assert isinstance(json, list)
    assert len(json) == 1
    entry = json[0]
    assert entry["source"] == "doc.pdf"
    assert entry["scope"] == "agent"
    assert entry["chat_id"] is None
    assert entry["type"] == "file"
    assert entry["status"] == "completed"
    assert "created_at" in entry
    assert "updated_at" in entry


async def test_chat_scope_returns_only_chat_entries(secure_client, secure_client_headers, cheshire_cat, monkeypatch):
    _use_local_file_manager(monkeypatch)
    _write_storage_file(os.path.join(agent_id, "agent_doc.pdf"))
    _write_storage_file(os.path.join(agent_id, chat_id, "chat_doc.pdf"))

    # create the conversation so the chat-scope reconcile keeps the entry
    await crud.store(
        f"{DEFAULT_AGENTS_KEY}:{agent_id}:{DEFAULT_CONVERSATIONS_KEY}:user1:{chat_id}",
        {"history": []},
    )

    await set_status(agent_id, "agent", "agent_doc.pdf", type_="file", status=IngestionStatus.COMPLETED)
    await set_status(
        agent_id, chat_id, "chat_doc.pdf", type_="file", status=IngestionStatus.COMPLETED, chat_id=chat_id,
    )

    # agent scope returns only the agent entry
    response = await secure_client.get("/ingestion/status", headers=secure_client_headers)
    assert response.status_code == 200
    assert [e["source"] for e in response.json()] == ["agent_doc.pdf"]

    # chat scope returns only that chat's entry
    response = await secure_client.get(f"/ingestion/status?chat_id={chat_id}", headers=secure_client_headers)
    assert response.status_code == 200
    assert [e["source"] for e in response.json()] == ["chat_doc.pdf"]


async def test_file_status_purged_when_file_missing(secure_client, secure_client_headers, cheshire_cat, monkeypatch):
    _use_local_file_manager(monkeypatch)
    # seed a status for a file that does not exist in the file manager
    await set_status(agent_id, "agent", "ghost.pdf", type_="file", status=IngestionStatus.COMPLETED)

    response = await secure_client.get("/ingestion/status", headers=secure_client_headers)
    assert response.status_code == 200
    assert response.json() == []

    # the Redis key is gone
    assert await get_status(agent_id, "agent", "ghost.pdf") is None


async def test_url_status_purged_when_no_web_points(secure_client, secure_client_headers, cheshire_cat):
    # seed a URL status with no corresponding web point in the vector store
    await set_status(agent_id, "agent", "https://example.com/doc", type_="url", status=IngestionStatus.COMPLETED)

    response = await secure_client.get("/ingestion/status", headers=secure_client_headers)
    assert response.status_code == 200
    assert response.json() == []

    # the Redis key is gone
    assert await get_status(agent_id, "agent", "https://example.com/doc") is None


async def test_url_status_survives_when_web_point_exists(secure_client, secure_client_headers, cheshire_cat, monkeypatch):
    """B1: ``_canonical_urls`` passes ``str(collection)`` to the vector handler.

    A fake handler that raises on a non-str collection proves the endpoint no
    longer crashes and URL statuses with live web points survive the reconcile.
    """
    class FakeVectorMemoryHandler:
        def __init__(self):
            self.calls = []

        async def get_all_tenant_points_from_web(self, collection):
            self.calls.append(collection)
            if not isinstance(collection, str):
                raise TypeError("bad argument type for built-in operation")
            point = type("P", (), {"payload": {"metadata": {"source": "https://example.com/doc"}}})()
            return [point], None

    handler = FakeVectorMemoryHandler()

    async def fake_get_vector_memory_handler(self, *args, **kwargs):
        return handler

    monkeypatch.setattr(ServiceProvider._class, "get_vector_memory_handler", fake_get_vector_memory_handler)

    await set_status(agent_id, "agent", "https://example.com/doc", type_="url", status=IngestionStatus.COMPLETED)

    response = await secure_client.get("/ingestion/status", headers=secure_client_headers)
    assert response.status_code == 200
    json = response.json()
    assert len(json) == 1
    assert json[0]["source"] == "https://example.com/doc"
    assert json[0]["status"] == "completed"
    # the handler was called with a str, never the VectorMemoryType enum
    assert handler.calls, "vector handler was never called"
    assert all(isinstance(c, str) for c in handler.calls)


async def test_error_status_not_purged_when_source_absent(secure_client, secure_client_headers, cheshire_cat, monkeypatch):
    """M3: an ``error`` entry for an absent source survives the reconcile.

    A failed upload never lands in the file manager, so without the carve-out
    it would be purged on first read and the error badge could never appear.
    A ``completed`` entry for an absent source is still purged.
    """
    _use_local_file_manager(monkeypatch)
    await set_status(agent_id, "agent", "failed.pdf", type_="file", status=IngestionStatus.ERROR, error="boom")
    await set_status(agent_id, "agent", "ghost.pdf", type_="file", status=IngestionStatus.COMPLETED)

    response = await secure_client.get("/ingestion/status", headers=secure_client_headers)
    assert response.status_code == 200
    assert [e["source"] for e in response.json()] == ["failed.pdf"]

    # the completed entry is gone; the error entry survives
    assert await get_status(agent_id, "agent", "ghost.pdf") is None
    assert await get_status(agent_id, "agent", "failed.pdf") is not None


async def test_empty_registry_returns_empty_list(secure_client, secure_client_headers, cheshire_cat):
    response = await secure_client.get("/ingestion/status", headers=secure_client_headers)
    assert response.status_code == 200
    assert response.json() == []


async def test_forbidden_without_memory_read(secure_client, secure_client_headers, client, cheshire_cat):
    # default user has only CHAT:WRITE, no MEMORY.READ
    data = await create_new_user(secure_client, headers=secure_client_headers)
    res = await client.post("/auth/token", json={"username": data["username"], "password": new_user_password})
    received_token = res.json()["access_token"]

    response = await client.get(
        "/ingestion/status",
        headers={"Authorization": f"Bearer {received_token}", "X-Agent-ID": agent_id},
    )
    assert response.status_code == 403
    assert response.json()["detail"] == "Forbidden"


async def test_inflight_status_survives_when_source_absent(
    secure_client, secure_client_headers, cheshire_cat, monkeypatch
):
    """N2: in-flight entries (uploaded/downloading/downloaded/processing) are
    NEVER purged by the read-time reconcile, even when the source is not (yet)
    in the canonical lists — the processing queue must stay visible across
    sessions/workers. Only completed (vanished) and error (kept) are special.
    """
    _use_local_file_manager(monkeypatch)
    for status in ("uploaded", "downloading", "downloaded", "processing"):
        await set_status(agent_id, "agent", f"inflight_{status}.pdf", type_="file", status=status)

    response = await secure_client.get("/ingestion/status", headers=secure_client_headers)
    assert response.status_code == 200
    sources = {e["source"] for e in response.json()}
    assert sources == {
        "inflight_uploaded.pdf",
        "inflight_downloading.pdf",
        "inflight_downloaded.pdf",
        "inflight_processing.pdf",
    }


async def test_delete_status_error_survives(secure_client, secure_client_headers, cheshire_cat, monkeypatch):
    """DELETE /ingestion/status removes an error row (dismissal)."""
    _use_local_file_manager(monkeypatch)
    await set_status(agent_id, "agent", "failed.pdf", type_="file", status=IngestionStatus.ERROR, error="boom")

    response = await secure_client.delete(
        f"/ingestion/status?source={urllib.parse.quote('failed.pdf')}",
        headers=secure_client_headers,
    )
    assert response.status_code == 200
    assert response.json() == {"deleted": True}
    assert await get_status(agent_id, "agent", "failed.pdf") is None


async def test_delete_inflight_status_refused(secure_client, secure_client_headers, cheshire_cat, monkeypatch):
    """DELETE /ingestion/status refuses in-flight sources: deleting the row
    while work is running would orphan the pipeline (the hooks re-create a
    completed row) — the teacher must remove the file instead.
    """
    _use_local_file_manager(monkeypatch)
    await set_status(agent_id, "agent", "busy.pdf", type_="file", status=IngestionStatus.PROCESSING)

    response = await secure_client.delete(
        f"/ingestion/status?source={urllib.parse.quote('busy.pdf')}",
        headers=secure_client_headers,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["deleted"] is False
    assert body["reason"] == "in_flight"
    assert await get_status(agent_id, "agent", "busy.pdf") is not None


async def test_delete_status_not_found(secure_client, secure_client_headers, cheshire_cat):
    response = await secure_client.delete(
        f"/ingestion/status?source={urllib.parse.quote('nope.pdf')}",
        headers=secure_client_headers,
    )
    assert response.status_code == 200
    assert response.json() == {"deleted": False, "reason": "not_found"}


# ---------------------------------------------------------------------------
# /ingestion/settings (engine configuration) — CAT CORE route
# ---------------------------------------------------------------------------
# These endpoints are served by the CAT core (cat/routes/ingestion.py,
# AuthResource.INGESTION); the plugin must NOT register them. The tests below
# assert the CORE route's behavior against the plugin's config class. They are
# skipped on hosts without the core route (MyCAT has no cat/routes/ingestion.py
# and the plugin no longer registers the settings endpoints there).

CONFIG_NAME = "EfficientIngestionConfiguration"
CORE_DEFAULT_NAME = "CoreIngestionConfiguration"

_CORE_INGESTION_ROUTE = importlib.util.find_spec("cat.routes.ingestion") is not None

_CORE_ROUTE_ONLY = pytest.mark.skipif(
    not _CORE_INGESTION_ROUTE,
    reason="CAT core /ingestion/settings route not present (MyCAT); the plugin endpoints were removed",
)


@_CORE_ROUTE_ONLY
async def test_settings_list_returns_defaults(secure_client, secure_client_headers, cheshire_cat):
    """GET /ingestion/settings (core route) lists the allowed engines with
    scheme and the effective selection; a never-saved engine has an empty
    value. With nothing saved the core default engine is selected."""
    response = await secure_client.get("/ingestion/settings", headers=secure_client_headers)
    assert response.status_code == 200
    body = response.json()
    assert "settings" in body
    assert "selected_configuration" in body
    # nothing saved -> the core default engine is the effective choice
    assert body["selected_configuration"] == CORE_DEFAULT_NAME
    names = [s["name"] for s in body["settings"]]
    assert CONFIG_NAME in names
    entry = next(s for s in body["settings"] if s["name"] == CONFIG_NAME)
    assert entry["value"] == {}  # never saved -> defaults
    assert "scheme" in entry
    assert entry["scheme"]["title"] == CONFIG_NAME


@_CORE_ROUTE_ONLY
async def test_put_setting_persists_and_round_trips(secure_client, secure_client_headers, cheshire_cat):
    """PUT persists the config (and selects it); GET after PUT round-trips."""
    payload = {"ingestion_max_concurrency": 7}
    response = await secure_client.put(
        f"/ingestion/settings/{CONFIG_NAME}", json=payload, headers=secure_client_headers
    )
    assert response.status_code == 200
    assert response.json() == {"name": CONFIG_NAME, "value": payload}

    # list reflects the saved value and the selection
    response = await secure_client.get("/ingestion/settings", headers=secure_client_headers)
    assert response.status_code == 200
    body = response.json()
    assert body["selected_configuration"] == CONFIG_NAME
    entry = next(s for s in body["settings"] if s["name"] == CONFIG_NAME)
    assert entry["value"] == payload

    # single-setting GET round-trips the same value
    response = await secure_client.get(f"/ingestion/settings/{CONFIG_NAME}", headers=secure_client_headers)
    assert response.status_code == 200
    single = response.json()
    assert single["name"] == CONFIG_NAME
    assert single["value"] == payload
    assert "scheme" in single


@_CORE_ROUTE_ONLY
async def test_get_setting_unknown_name_rejected(secure_client, secure_client_headers, cheshire_cat):
    """GET of an unknown engine configuration is rejected (reference behavior:
    CustomValidationException -> 400)."""
    response = await secure_client.get("/ingestion/settings/NoSuchEngine", headers=secure_client_headers)
    assert response.status_code == 400
    assert "NoSuchEngine" in response.json()["detail"]


@_CORE_ROUTE_ONLY
async def test_put_setting_unknown_name_400(secure_client, secure_client_headers, cheshire_cat):
    """PUT of an unknown engine configuration -> 400 (core route raises
    CustomValidationException, unlike the old plugin route's 404)."""
    response = await secure_client.put(
        "/ingestion/settings/NoSuchEngine", json={}, headers=secure_client_headers
    )
    assert response.status_code == 400
    assert "NoSuchEngine" in response.json()["detail"]


@_CORE_ROUTE_ONLY
async def test_put_setting_invalid_body_accepted(secure_client, secure_client_headers, cheshire_cat):
    """PUT with a type-mismatched field is ACCEPTED by the core route (200):
    upstream's ``upsert_service`` stores the payload without model validation
    (the old plugin route rejected it with 400). The value round-trips as-is."""
    response = await secure_client.put(
        f"/ingestion/settings/{CONFIG_NAME}",
        json={"ingestion_max_concurrency": "not-an-int"},
        headers=secure_client_headers,
    )
    assert response.status_code == 200
    assert response.json()["value"] == {"ingestion_max_concurrency": "not-an-int"}


@_CORE_ROUTE_ONLY
async def test_put_setting_non_dict_body_400(secure_client, secure_client_headers, cheshire_cat):
    """PUT with a non-object body is rejected with 400 (request validation)."""
    response = await secure_client.put(
        f"/ingestion/settings/{CONFIG_NAME}",
        content='"just a string"',
        headers=secure_client_headers,
    )
    assert response.status_code == 400


@_CORE_ROUTE_ONLY
async def test_settings_forbidden_without_ingestion_permission(
    secure_client, secure_client_headers, client, cheshire_cat
):
    """Default user has only CHAT:WRITE: the core route's INGESTION READ/WRITE
    endpoints are 403 (the old plugin route required SYSTEM instead)."""
    data = await create_new_user(secure_client, headers=secure_client_headers)
    res = await client.post("/auth/token", json={"username": data["username"], "password": new_user_password})
    received_token = res.json()["access_token"]
    headers = {"Authorization": f"Bearer {received_token}", "X-Agent-ID": agent_id}

    response = await client.get("/ingestion/settings", headers=headers)
    assert response.status_code == 403
    assert response.json()["detail"] == "Forbidden"

    response = await client.put(
        f"/ingestion/settings/{CONFIG_NAME}", json={}, headers=headers
    )
    assert response.status_code == 403
    assert response.json()["detail"] == "Forbidden"
