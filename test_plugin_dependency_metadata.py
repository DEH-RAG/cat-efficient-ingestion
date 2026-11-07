"""Plugin dependency/manifest metadata owned by the plugin.

Moved out of the core suite: the manifest and pyproject assertions are pure
statements about THIS plugin's files. The host-side loader checks
(``Plugin.missing_dependencies`` against ``cat/core_plugins`` and the
``lizard`` end-to-end load) and the host-version pin stay in core
(``grinning-cat-core/tests/effing/test_plugin_dependency_metadata.py``),
because they exercise the host loader.

Standalone and import-safe: the entire body lives under
``if __name__ == "__main__":``. Run from the core checkout::

    cd /root/git/grinning-cat-core
    .venv/bin/python /root/git/plugins/cat-efficient-ingestion/test_plugin_dependency_metadata.py
"""
if __name__ == "__main__":
    import json
    import tomllib
    from pathlib import Path

    from _testkit import bootstrap, run

    bootstrap()

    PLUGIN_ID = "cat_efficient_ingestion"
    MMODING_ID = "cat_multimodal_ingestion"


    def _real_plugins_path() -> Path:
        """The real ``cat/core_plugins`` folder, immune to the test conftest's mock.

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


    run([
        test_plugin_json_dependency_matches_folder_id,
        test_pyproject_declares_project_dependencies,
        test_manifest_version_matches_pyproject,
    ])
