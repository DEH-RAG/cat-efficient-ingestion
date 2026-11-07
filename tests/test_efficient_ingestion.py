"""Efficient-ingestion engine selection through the ServiceFactory (plugin-owned).

The engine is exposed through the ``ServiceFactory`` (``ingestion`` category):
configs are stored in the global ``system:agent`` settings list, the plugin
registers ``EfficientIngestionConfiguration`` via ``factory_allowed_ingestions``
and, when selected, it is the active engine. The category ``ingestion`` holds a
SINGLE setting whose ``name`` is the active configuration class (no separate
selection entry).

Runs against UPSTREAM CAT and the real host app: the config inherits the
upstream ``BaseIngestionConfiguration`` and the engine is resolved through the
upstream ``ServiceFactory`` API. The tests that need the
``factory_allowed_ingestions`` hook install the plugin (and its MmodING
dependency) into the mock plugin folder when it is not already a core plugin.

Standalone and import-safe: the entire body lives under
``if __name__ == "__main__":``. Run from the core checkout::

    cd /root/git/grinning-cat-core
    .venv/bin/python /root/git/plugins/cat-efficient-ingestion/tests/test_efficient_ingestion.py
"""
if __name__ == "__main__":
    import os
    from contextlib import asynccontextmanager
    from pathlib import Path

    from _testkit import run_with_host

    PLUGIN_ID = "cat_efficient_ingestion"
    MMODING_ID = "cat_multimodal_ingestion"
    PLUGIN_ROOT = Path(__file__).resolve().parent.parent
    MMODING_SRC = PLUGIN_ROOT.parent / MMODING_ID

    def build(host):
        from cat.core_plugins.cat_efficient_ingestion.configs import (
            EfficientIngestionConfiguration,
        )
        from cat.core_plugins.cat_efficient_ingestion.reembed import EfficientIngestionEngine
        from cat.db.cruds import settings as crud_settings
        from cat.db.database import DEFAULT_SYSTEM_KEY, get_sync_db
        from cat.db.models import Setting
        from cat.services.factory.ingestion import (
            BaseIngestionConfiguration,
            BaseIngestionEngine,
        )
        from cat.services.service_factory import ServiceFactory

        MOCK_PLUGIN_FOLDER = host.core_root / "tests/mocks/mock_plugin_folder"

        def _cleanup():
            db = get_sync_db()
            db.json().delete("system:agent", '$[?(@.category == "ingestion")]')

        def _build_factory(lizard) -> ServiceFactory:
            return ServiceFactory(
                agent_key=DEFAULT_SYSTEM_KEY,
                hook_manager=lizard.plugin_manager,
                factory_allowed_handler_name="factory_allowed_ingestions",
                setting_category="ingestion",
                schema_name="ingestionName",
            )

        @asynccontextmanager
        async def effing_plugin_installed(h):
            """Ensure EffING is loaded so ``factory_allowed_ingestions`` fires.

            On this CAT checkout the plugin is a core plugin (symlinked in
            ``cat/core_plugins/``, loaded at bootstrap) — nothing to do.
            Otherwise install it (and its MmodING dependency) into the mock
            plugin folder through the standard MadHatter activation path.
            """
            pm = h.lizard.plugin_manager
            if PLUGIN_ID in pm.get_core_plugins_ids:
                yield
                return
            links = []
            for plugin_id, src in (
                (MMODING_ID, str(MMODING_SRC)),
                (PLUGIN_ID, str(PLUGIN_ROOT)),
            ):
                link = os.path.join(str(MOCK_PLUGIN_FOLDER), plugin_id)
                if not os.path.exists(link):
                    os.symlink(src, link)
                links.append(link)
                await pm.install_extracted_plugin(plugin_id)
            try:
                yield
            finally:
                # deactivate EffING first: MmodING cannot be deactivated while
                # EffING (which depends on it) is still active
                for plugin_id in (PLUGIN_ID, MMODING_ID):
                    if plugin_id in pm.active_plugins:
                        await pm.deactivate_plugin(plugin_id)
                for link in links:
                    if os.path.islink(link):
                        os.remove(link)

        async def test_configuration_defaults():
            cfg = EfficientIngestionConfiguration()
            assert cfg.model_dump() == {"ingestion_max_concurrency": 5}
            assert cfg.pyclass() is EfficientIngestionEngine

        async def test_configuration_inherits_upstream_base():
            """The config must inherit the upstream ``BaseIngestionConfiguration``
            so the factory resolves it as a replaceable class on upstream CAT.
            Instantiating the engine also proves every abstract method of
            ``BaseIngestionEngine`` is implemented."""
            assert issubclass(EfficientIngestionConfiguration, BaseIngestionConfiguration)
            assert EfficientIngestionConfiguration.base_class() is BaseIngestionEngine
            assert EfficientIngestionConfiguration.pyclass() is EfficientIngestionEngine

            cfg = EfficientIngestionConfiguration(ingestion_max_concurrency=3)
            engine = cfg.pyclass()(ingestion_max_concurrency=cfg.ingestion_max_concurrency)
            assert isinstance(engine, EfficientIngestionEngine)
            assert engine.ingestion_max_concurrency == 3

        async def test_factory_allows_plugin_config_and_instantiates_engine():
            async with effing_plugin_installed(host):
                _cleanup()
                sf = _build_factory(host.lizard)

                schemas = await sf.get_schemas()
                assert sf.default_config_class.__name__ in schemas
                assert "EfficientIngestionConfiguration" in schemas

                await sf.upsert_service(
                    "EfficientIngestionConfiguration", {"ingestion_max_concurrency": 5}
                )
                engine = await sf.get_from_config_name("EfficientIngestionConfiguration")
                assert type(engine).__name__ == "EfficientIngestionEngine"
                assert engine.ingestion_max_concurrency == 5
                _cleanup()

        async def test_upsert_stores_category_ingestion_and_value():
            async with effing_plugin_installed(host):
                _cleanup()
                sf = _build_factory(host.lizard)
                await sf.upsert_service(
                    "EfficientIngestionConfiguration", {"ingestion_max_concurrency": 3}
                )

                entry = await crud_settings.get_setting_by_name(
                    DEFAULT_SYSTEM_KEY, "EfficientIngestionConfiguration"
                )
                assert entry is not None
                assert entry["category"] == "ingestion"
                assert entry["value"] == {"ingestion_max_concurrency": 3}

                engine = await sf.get_from_config_name("EfficientIngestionConfiguration")
                assert type(engine).__name__ == "EfficientIngestionEngine"
                assert engine.ingestion_max_concurrency == 3
                _cleanup()

        async def test_upsert_does_not_duplicate_category_entry():
            """Editing the ingestion settings must replace the single category
            entry, NOT append a new dict to the ``system:agent`` list."""
            async with effing_plugin_installed(host):
                _cleanup()
                sf = _build_factory(host.lizard)
                await sf.upsert_service(
                    "EfficientIngestionConfiguration", {"ingestion_max_concurrency": 3}
                )
                await sf.upsert_service(
                    "EfficientIngestionConfiguration", {"ingestion_max_concurrency": 7}
                )

                db = get_sync_db()
                matches = (
                    db.json().get("system:agent", '$[?(@.category == "ingestion")]') or []
                )
                assert len(matches) == 1
                assert matches[0]["value"] == {"ingestion_max_concurrency": 7}

                engine = await sf.get_from_config_name("EfficientIngestionConfiguration")
                assert engine.ingestion_max_concurrency == 7
                _cleanup()

        async def test_endpoints_list_and_select():
            async with effing_plugin_installed(host):
                await host.lizard.create_cheshire_cat(host.agent_id)

                # select the plugin engine explicitly (creates the single entry)
                res = await host.client.put(
                    "/ingestion/settings/EfficientIngestionConfiguration",
                    headers=host.headers,
                    json={},
                )
                assert res.status_code == 200

                listing = await host.client.get(
                    "/ingestion/settings", headers=host.headers
                )
                assert listing.status_code == 200
                body = listing.json()
                names = [s["name"] for s in body["settings"]]
                assert "EfficientIngestionConfiguration" in names
                assert body["selected_configuration"] == "EfficientIngestionConfiguration"

                # select the core default engine explicitly (name-agnostic)
                core_name = _build_factory(host.lizard).default_config_class.__name__
                assert core_name in names
                res = await host.client.put(
                    f"/ingestion/settings/{core_name}",
                    headers=host.headers,
                    json={},
                )
                assert res.status_code == 200

                db = get_sync_db()
                matches = (
                    db.json().get("system:agent", '$[?(@.category == "ingestion")]') or []
                )
                assert len(matches) == 1

                listing2 = await host.client.get(
                    "/ingestion/settings", headers=host.headers
                )
                assert listing2.json()["selected_configuration"] == core_name
                _cleanup()

        async def test_legacy_reembed_max_concurrency_is_migrated():
            """A config previously saved with ``reembed_max_concurrency`` (the
            pre-rename field) keeps working: the value maps onto
            ``ingestion_max_concurrency``."""
            async with effing_plugin_installed(host):
                _cleanup()

                await crud_settings.upsert_setting_by_category(
                    DEFAULT_SYSTEM_KEY,
                    Setting(
                        name="EfficientIngestionConfiguration",
                        value={"reembed_max_concurrency": 8},
                        category="ingestion",
                    ),
                )

                sf = _build_factory(host.lizard)
                engine = await sf.get_from_config_name("EfficientIngestionConfiguration")
                assert type(engine).__name__ == "EfficientIngestionEngine"
                assert engine.ingestion_max_concurrency == 8
                _cleanup()

        return [
            test_configuration_defaults,
            test_configuration_inherits_upstream_base,
            test_factory_allows_plugin_config_and_instantiates_engine,
            test_upsert_stores_category_ingestion_and_value,
            test_upsert_does_not_duplicate_category_entry,
            test_endpoints_list_and_select,
            test_legacy_reembed_max_concurrency_is_migrated,
        ]

    run_with_host(build)
