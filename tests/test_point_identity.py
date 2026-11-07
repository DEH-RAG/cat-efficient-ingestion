"""Point identity through the efficient-ingestion store/embed phases.

A Document-level id (e.g. the deterministic catalogue-card uuid5) must survive
the phase machine: the parsing/chunking store and the embedding re-store both
use it as the point id, so a re-ingest upserts instead of duplicating. Plain
chunks (no Document id) keep the historical random uuid4.

Standalone and import-safe (the CAT loader imports every ``.py`` under the
plugin folder): the body lives under ``if __name__ == "__main__":``. Run it
from the host repository::

    cd /root/git/grinning-cat-core
    .venv/bin/python /root/git/plugins/cat-efficient-ingestion/tests/test_point_identity.py
"""
if __name__ == "__main__":
    import asyncio
    import os
    import sys
    import traceback
    import uuid
    from types import SimpleNamespace
    from unittest.mock import patch

    os.environ.setdefault("CAT_REDIS_HOST", "localhost")
    os.environ.setdefault("CAT_REDIS_PORT", "6379")
    os.environ.setdefault("CAT_JWT_SECRET", "test_only_jwt_secret_0123456789_abcdefghij")
    os.environ["CAT_REDIS_DB"] = "15"

    from langchain_core.documents import Document

    from cat.core_plugins.cat_efficient_ingestion import reembed
    from cat.services.memory.models import PointStruct, VectorMemoryType

    class _FakeEmbedder:
        name = "test-embedder"
        size = 384

        def embed_documents(self, texts):
            return [[0.1] * 4 for _ in texts]

    def _fake_cat():
        stored: dict = {}

        async def add_points(collection_name, points):
            stored.setdefault("added", []).extend(points)

        async def delete_points(collection_name, metadata=None):
            stored["deleted"] = metadata

        cat = SimpleNamespace(
            agent_key="agent",
            vector_memory_handler=SimpleNamespace(
                add_points_to_tenant=add_points,
                delete_tenant_points=delete_points,
            ),
            plugin_manager=SimpleNamespace(execute_hook=_noop_hook),
        )
        return cat, stored

    async def _noop_hook(*args, **kwargs):
        return None

    async def _not_canceled(agent_key):
        return False

    async def test_store_empty_vectors_honours_document_id():
        cat, stored = _fake_cat()
        card_id = uuid.uuid5(uuid.NAMESPACE_URL, "catalogue_card:test.txt").hex
        docs = [Document(page_content="card", metadata={"source": "test.txt"}, id=card_id)]

        points = await reembed._store_empty_vectors(
            cat, VectorMemoryType.DECLARATIVE, "test.txt", None, docs, [],
        )

        assert [p.id for p in points] == [card_id]
        assert [p.id for p in stored["added"]] == [card_id]

    async def test_store_empty_vectors_generates_uuid4_without_document_id():
        cat, stored = _fake_cat()
        docs = [Document(page_content="chunk", metadata={"source": "test.txt"})]

        points = await reembed._store_empty_vectors(
            cat, VectorMemoryType.DECLARATIVE, "test.txt", None, docs, [],
        )

        assert len(points) == 1
        assert len(points[0].id) == 32
        assert uuid.UUID(points[0].id).version == 4
        assert stored["added"][0].id == points[0].id

    async def test_embed_phase_preserves_stored_point_id():
        cat, stored = _fake_cat()
        card_id = uuid.uuid5(uuid.NAMESPACE_URL, "catalogue_card:test.txt").hex
        source_points = [
            PointStruct(
                id=card_id,
                payload={"id": card_id, "page_content": "card", "metadata": {"source": "test.txt"}},
                vector={},
            ),
        ]

        points = await reembed._embed_phase(
            cat, VectorMemoryType.DECLARATIVE, "test.txt", source_points,
            _FakeEmbedder(), None, cat, rabbit_hole=None,
        )

        assert [p.id for p in points] == [card_id]
        assert [p.id for p in stored["added"]] == [card_id]

    async def test_embed_phase_reuses_existing_point_id():
        """Re-embedding keeps the id already stored on the point (idempotent pass)."""
        cat, stored = _fake_cat()
        source_points = [
            PointStruct(
                id="existing-point-id",
                payload={"page_content": "chunk", "metadata": {"source": "test.txt"}},
                vector={},
            ),
        ]

        points = await reembed._embed_phase(
            cat, VectorMemoryType.DECLARATIVE, "test.txt", source_points,
            _FakeEmbedder(), None, cat, rabbit_hole=None,
        )

        assert [p.id for p in points] == ["existing-point-id"]
        assert [p.id for p in stored["added"]] == ["existing-point-id"]

    _tests = [
        test_store_empty_vectors_honours_document_id,
        test_store_empty_vectors_generates_uuid4_without_document_id,
        test_embed_phase_preserves_stored_point_id,
        test_embed_phase_reuses_existing_point_id,
    ]

    _passed = 0
    _failures = []
    for _test in _tests:
        try:
            with patch.object(reembed, "ingestion_canceled", _not_canceled):
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
