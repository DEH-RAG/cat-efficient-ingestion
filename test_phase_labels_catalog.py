"""Standalone, import-safe tests for phase/status display labels.

Covers:
  (a) ``IngestionPhase.display_label()`` returns ``label`` when set and ``id``
      otherwise, and every built-in phase declares a non-empty label;
  (b) the phase catalog contains every registered phase in dependency order
      with non-empty labels and a 1-based ``order``;
  (c) a status entry carries ``phase_label`` matching the phase's label and
      ``status_label`` matching the lifecycle label, while ``phase``/``status``
      stay untouched;
  (d) the lifecycle catalog exposes every state with a non-empty label.

The CAT plugin loader imports EVERY ``.py`` under a plugin folder at
activation, so the whole file is inert on import: every import, class and test
lives inside ``if __name__ == "__main__":``. Run it as a plain script from the
host repository::

    cd /root/git/grinning-cat-core
    .venv/bin/python /root/git/plugins/cat-efficient-ingestion/test_phase_labels_catalog.py
"""

if __name__ == "__main__":
    import asyncio
    import sys
    import types

    from cat.core_plugins.cat_efficient_ingestion import endpoints, phases, registry

    class FakePhase(phases.IngestionPhase):
        id = "fake_phase"
        label = None
        depends_on = ("embedding",)
        settings_category = None

        async def applies_to(self, cat, deps_state):
            return True, None

        async def process(self, source, cat):
            return phases.PhaseResult(status="done")

        async def delete(self, source, cat):
            return None

    class LabelledPhase(phases.IngestionPhase):
        id = "labelled_phase"
        label = "A labelled phase"
        depends_on = ()

        async def applies_to(self, cat, deps_state):
            return True, None

        async def process(self, source, cat):
            return phases.PhaseResult(status="done")

        async def delete(self, source, cat):
            return None

    class FakeHook:
        def __init__(self, function, plugin_id="fake"):
            self.function = function
            self.plugin_id = plugin_id

    class FakeManager:
        context_execute_hook = "cat"

        def __init__(self, hooks=None):
            self.hooks = hooks or {}

    def _cat(manager=None):
        if manager is None:
            return types.SimpleNamespace()
        return types.SimpleNamespace(plugin_manager=manager)

    async def test_a_display_label():
        parsing = phases.ParsingChunkingPhase()
        embedding = phases.EmbeddingPhase()
        assert parsing.display_label() == "Parsing and chunking", parsing.display_label()
        assert embedding.display_label() == "Computing embeddings", embedding.display_label()
        assert FakePhase().display_label() == "fake_phase"
        assert LabelledPhase().display_label() == "A labelled phase"
        print("PASS (a) display_label() falls back to id and built-ins are labelled")

    async def test_b_catalog_dependency_order():
        contributed = FakePhase()
        manager = FakeManager(
            {"ingestion_phase_objects": [FakeHook(lambda cat=None: [contributed])]}
        )
        catalog = await phases.phase_catalog(_cat(manager))
        ids = [row["id"] for row in catalog]
        assert set(ids) == {"parsing_chunking", "embedding", "fake_phase"}, ids
        assert ids.index("parsing_chunking") < ids.index("embedding"), ids
        assert ids.index("embedding") < ids.index("fake_phase"), ids
        assert [row["order"] for row in catalog] == [1, 2, 3], catalog
        for row in catalog:
            assert row["label"], row
            assert isinstance(row["depends_on"], list), row
        print("PASS (b) catalog lists every phase in dependency order with labels")

    async def test_c_status_entry_labels():
        catalog = await phases.phase_catalog(None)
        phase_labels = {row["id"]: row["label"] for row in catalog}
        state_labels = {row["id"]: row["label"] for row in registry.status_catalog()}

        entries = [
            {"source": "doc.pdf", "phase": "embedding", "status": "processing"},
            {"source": "old.pdf", "phase": None, "status": "completed"},
            {"source": "https://example.com", "phase": "downloading", "status": "downloading"},
        ]
        decorated = endpoints._decorate_entries(entries, phase_labels, state_labels)

        assert decorated[0]["phase"] == "embedding", decorated[0]
        assert decorated[0]["phase_label"] == "Computing embeddings", decorated[0]
        assert decorated[0]["status_label"] == "Processing", decorated[0]

        assert decorated[1]["phase"] is None, decorated[1]
        assert decorated[1]["phase_label"] is None, decorated[1]
        assert decorated[1]["status_label"] == "Completed", decorated[1]

        assert decorated[2]["phase_label"] == "Downloading", decorated[2]
        assert decorated[2]["status_label"] == "Downloading", decorated[2]
        assert all("status" in row for row in decorated), decorated
        print("PASS (c) status entries carry phase_label and status_label")

    async def test_d_lifecycle_labels():
        states = registry.status_catalog()
        ids = [row["id"] for row in states]
        expected = {
            "uploaded",
            "downloading",
            "downloaded",
            "processing",
            "reindexing",
            "deleting",
            "completed",
            "error",
        }
        assert set(ids) == expected, ids
        by_id = {row["id"]: row["label"] for row in states}
        assert by_id["reindexing"] == "Re-indexing", by_id
        assert by_id["deleting"] == "Deleting", by_id
        assert by_id["processing"] == "Processing", by_id
        assert by_id["completed"] == "Completed", by_id
        assert by_id["error"] == "Error", by_id
        assert all(row["label"] for row in states), states
        print("PASS (d) lifecycle states expose non-empty labels")

    async def _main():
        await test_a_display_label()
        await test_b_catalog_dependency_order()
        await test_c_status_entry_labels()
        await test_d_lifecycle_labels()

    try:
        asyncio.run(_main())
    except AssertionError as exc:
        print(f"FAIL {exc}")
        sys.exit(1)
    print("OK phase/status labels + catalog (four tests)")
