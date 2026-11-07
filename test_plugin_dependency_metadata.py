"""Plugin dependency/manifest metadata and host-loader contract (plugin-owned).

The manifest and pyproject assertions are pure statements about THIS plugin's
files. The host-side loader checks (``Plugin.missing_dependencies`` resolved
against ``cat/core_plugins`` and the end-to-end ``lizard`` plugin-manager load)
and the ``min_cat_version`` pin against the host version actually running need
the real host. All of them live here together.

Standalone and import-safe: the entire body lives under
``if __name__ == "__main__":``. Run from the core checkout::

    cd /root/git/grinning-cat-core
    .venv/bin/python /root/git/plugins/cat-efficient-ingestion/test_plugin_dependency_metadata.py
"""
if __name__ == "__main__":
    import json
    import tomllib
    from pathlib import Path
    from unittest.mock import patch

    from _testkit import bootstrap, run_with_host

    bootstrap()

    PLUGIN_ID = "cat_efficient_ingestion"
    MMODING_ID = "cat_multimodal_ingestion"

    def _real_plugins_path() -> Path:
        """The real ``cat/core_plugins`` folder, immune to the test harness mock.

        The canonical plugin path: the host loads ``cat/core_plugins/`` in
        preference to ``cat/plugins/`` (``mad_hatter.py``), so the harness must
        resolve EffING and its MmodING dependency there.
        """
        from cat import utils

        return Path(utils.__file__).resolve().parent / "core_plugins"

    def _plugin_dir() -> Path:
        here = Path(__file__).resolve().parent
        for candidate in (here, here.parent, here.parent.parent):
            if (candidate / "plugin.json").is_file():
                return candidate
        return _real_plugins_path() / PLUGIN_ID

    def _manifest() -> dict:
        return json.loads((_plugin_dir() / "plugin.json").read_text())

    def test_plugin_json_dependency_matches_folder_id():
        """The declared dependency is the folder id of the multimodal plugin.

        Upstream matches dependencies against plugin folder ids exactly; the
        hyphenated repo name (``cat-multimodal-ingestion``) is not a folder id and
        would be reported missing, blocking the load.
        """
        assert _manifest()["dependencies"] == [MMODING_ID]
        assert (_real_plugins_path() / MMODING_ID).is_dir(), (
            f"multimodal plugin folder {MMODING_ID!r} not found in the plugins path"
        )

    def test_pyproject_declares_project_dependencies():
        """CAT installs plugin pip deps from ``pyproject.toml [project].dependencies`` (uv)."""
        data = tomllib.loads((_plugin_dir() / "pyproject.toml").read_text())
        deps = data["project"]["dependencies"]
        assert deps, "pyproject.toml must declare [project].dependencies for CAT's uv-based install"

    def test_manifest_version_matches_pyproject():
        """``plugin.json`` and ``pyproject.toml`` must advertise the same version."""
        data = tomllib.loads((_plugin_dir() / "pyproject.toml").read_text())
        assert _manifest()["version"] == data["project"]["version"]

    _tests = [
        test_plugin_json_dependency_matches_folder_id,
        test_pyproject_declares_project_dependencies,
        test_manifest_version_matches_pyproject,
    ]

    def build(host):
        from cat.looking_glass.mad_hatter.plugin import Plugin

        def test_loads_on_cat_with_dependency_resolved():
            """The plugin loads on CAT: no missing dependencies when the
            multimodal plugin is present."""
            plugin = Plugin(str(_plugin_dir()))
            assert plugin.missing_dependencies([PLUGIN_ID, MMODING_ID]) == []

        def test_missing_dependency_is_reported_not_silent():
            """Failure mode: a missing dependency is reported clearly, never
            silently skipped."""
            plugin = Plugin(str(_plugin_dir()))
            assert plugin.missing_dependencies([PLUGIN_ID]) == [MMODING_ID]

        async def test_plugin_manager_loads_effing_with_dependency_resolved():
            """End-to-end: the CAT plugin manager loads EffING with its dependency
            resolved. The harness redirects ``get_plugins_path`` to a mock folder;
            restore the real one for this test."""
            from cat import utils as cat_utils

            with patch.object(
                cat_utils, "get_plugins_path", lambda: str(_real_plugins_path())
            ):
                loaded = await host.lizard.plugin_manager.load_plugin(PLUGIN_ID)
            assert loaded.plugin is not None, (
                f"plugin failed to load; missing dependencies: {loaded.missing_dependencies}"
            )
            assert loaded.missing_dependencies == []

        def _host_version() -> str:
            """The host CAT version this plugin is running against."""
            from cat import utils

            root = Path(utils.__file__).resolve().parent.parent
            return tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]

        def test_manifest_min_cat_version_matches_the_host_it_needs():
            """The manifest pins the minimum host version the plugin actually needs."""
            assert _manifest()["min_cat_version"] == _host_version()

        return list(_tests) + [
            test_loads_on_cat_with_dependency_resolved,
            test_missing_dependency_is_reported_not_silent,
            test_plugin_manager_loads_effing_with_dependency_resolved,
            test_manifest_min_cat_version_matches_the_host_it_needs,
        ]

    run_with_host(build)
