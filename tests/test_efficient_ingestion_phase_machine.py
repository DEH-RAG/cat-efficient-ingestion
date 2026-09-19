"""Tests for the serial phase-dispatch loop in ``reembed_sources`` (Todo 6).

Pins the clock-free ATOMIC COMPLETION invariant and the terminal COMPLETED
write of the phase machine:

  (a) an embedder-only change records ONLY ``embedding`` in the diary (the
      parsing marker is preserved unchanged);
  (b) a chunker change records ``parsing_chunking`` THEN ``embedding`` (the
      chain: embedding's ``deps`` reference the NEW parsing marker);
  (c) a crash after parsing (the embedding body raises) -> the row is ERROR and
      the interrupted embedding has NO diary entry; the next pass resumes at
      ``embedding`` WITHOUT re-parsing;
  (d) an ``error`` row that fails again stays ERROR (never resurrected to
      COMPLETED, diary unchanged);
  (e) a double COMPLETED write is harmless: after a successful pass the source
      is COMPLETED with a full diary, and a second pass on the fresh diary is
      skipped WITHOUT any status write (idempotent).

Every test also asserts the ATOMIC COMPLETION INVARIANT: a phase's diary entry
exists only after that phase completed; an interrupted (error) phase has NO
entry (-> stale on the next pass, so a restart re-runs it).
"""
import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from cat.plugins.cat_efficient_ingestion import reembed
from cat.plugins.cat_efficient_ingestion.registry import (
    IngestionStatus,
    get_completed_phases,
    get_status,
    set_status,
)

# ---------------------------------------------------------------------------
# helpers (mirror of the resume-file stubs, self-contained on purpose)
# ---------------------------------------------------------------------------


class _FakeEmbedder:
    name = "test-embedder"
    size = 384


def _stored_source(name):
    from cat.looking_glass.models import StoredSourceWithMetadata

    return StoredSourceWithMetadata(name=name, path=name, content=None, metadata={})


def _point(source_name):
    from types import SimpleNamespace

    return SimpleNamespace(payload={"metadata": {"source": source_name}})


def _stub_reembed_env(monkeypatch, cheshire_cat, existing_points, pending_result):
    """Point ``reembed_sources`` at a deterministic environment.

    The ``ingestion_phase_pending`` probe is DIARY-AWARE: a phase in
    ``pending_result`` is reported stale until its diary entry carries marker
    ``"test-marker"`` (the fixed ``ingestion_phase_settings_marker`` result) —
    exactly like the real clock-free registrant, so the loop re-probe after a
    recorded phase returns empty without a hand-rolled script. Phase bodies are
    faked and counted, so the machine never touches vectors.
    """
    from types import SimpleNamespace

    from langchain_core.documents import Document

    monkeypatch.setattr(cheshire_cat, "embedder", AsyncMock(return_value=_FakeEmbedder()))
    monkeypatch.setattr(cheshire_cat, "chunker", SimpleNamespace(name="test-chunker"))
    monkeypatch.setattr(
        cheshire_cat.vector_memory_handler, "get_all_tenant_points",
        AsyncMock(return_value=(existing_points, None)),
    )
    monkeypatch.setattr(
        cheshire_cat.vector_memory_handler, "delete_tenant_points",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        cheshire_cat.vector_memory_handler, "add_points_to_tenant",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(cheshire_cat.file_manager, "read_file", lambda source, remote: b"fake content")
    monkeypatch.setattr(cheshire_cat.file_manager, "remove_file", lambda path: True)

    calls: dict[str, Any] = {"probe": [], "parse_calls": 0, "embed_calls": 0}
    base_pending = list(pending_result or [])

    async def fake_parse_and_chunk(ccat, rabbit_hole, source, file_bytes, content_type, cat):
        calls["parse_calls"] += 1
        return [Document(page_content="parsed chunk")], []

    async def fake_embed_phase(*args, **kwargs):
        calls["embed_calls"] += 1
        return []

    async def fake_execute_hook(name, *args, **kwargs):
        if name == "ingestion_phase_pending":
            calls["probe"].append((args, kwargs))
            completed = args[2]
            diary = {e["phase"]: e for e in completed if isinstance(e, dict)}
            return [
                p for p in base_pending
                if not isinstance(diary.get(p["phase"]), dict)
                or diary[p["phase"]].get("marker") != "test-marker"
            ]
        if name == "ingestion_phase_settings_marker":
            return "test-marker"
        return []

    monkeypatch.setattr(cheshire_cat.plugin_manager, "execute_hook", fake_execute_hook)
    monkeypatch.setattr(reembed, "_parse_and_chunk", fake_parse_and_chunk)
    monkeypatch.setattr(reembed, "_embed_phase", fake_embed_phase)
    return calls


def _completed_diary(parsing_marker="pc1", embedding_marker="em1"):
    """A completed row's diary: parsing recorded, embedding depends on it."""
    return {
        "parsing_chunking": {"marker": parsing_marker, "settings_version": "v1", "deps": {}},
        "embedding": {
            "marker": embedding_marker, "settings_version": "v1",
            "deps": {"parsing_chunking": parsing_marker},
        },
    }


# ---------------------------------------------------------------------------
# (a) embedder-only change records ONLY embedding (parse marker preserved)
# ---------------------------------------------------------------------------


async def test_machine_embedder_change_records_only_embedding(cheshire_cat, monkeypatch):
    agent_key = cheshire_cat.agent_key
    source = "embedder_only.pdf"
    await set_status(
        agent_key, "agent", source, type_="file", status=IngestionStatus.COMPLETED,
        completed_phases=_completed_diary(parsing_marker="pc1", embedding_marker="em_old"),
    )

    calls = _stub_reembed_env(
        monkeypatch, cheshire_cat,
        existing_points=[_point(source)],
        pending_result=[{"phase": "embedding"}],
    )
    await reembed.reembed_sources(cheshire_cat, "declarative", [_stored_source(source)])

    doc = await get_status(agent_key, "agent", source)
    assert doc is not None
    assert doc["status"] == "completed"
    assert "phase" not in doc  # terminal write cleared the work phase

    diary = get_completed_phases(doc)
    # ONLY embedding was re-recorded; parsing was NOT re-run (its marker is
    # preserved unchanged)
    assert set(diary) == {"parsing_chunking", "embedding"}
    assert diary["parsing_chunking"]["marker"] == "pc1"
    assert calls["parse_calls"] == 0
    assert calls["embed_calls"] == 1

    # embedding recorded AT COMPLETION with the current marker + deps copied
    # from the current diary
    emb = diary["embedding"]
    assert emb["marker"] == "test-marker"
    assert emb["deps"] == {"parsing_chunking": "pc1"}
    assert "settings_version" in emb

    # decision probe + post-embedding re-probe (now empty)
    assert len(calls["probe"]) == 2


# ---------------------------------------------------------------------------
# (b) chunker change records parsing THEN embedding (chain)
# ---------------------------------------------------------------------------


async def test_machine_chunker_change_records_parsing_then_embedding(cheshire_cat, monkeypatch):
    agent_key = cheshire_cat.agent_key
    source = "chunker_change.pdf"
    await set_status(
        agent_key, "agent", source, type_="file", status=IngestionStatus.COMPLETED,
        completed_phases=_completed_diary(parsing_marker="pc_old", embedding_marker="em_old"),
    )

    calls = _stub_reembed_env(
        monkeypatch, cheshire_cat,
        existing_points=[_point(source)],
        pending_result=[{"phase": "parsing_chunking"}, {"phase": "embedding"}],
    )
    await reembed.reembed_sources(cheshire_cat, "declarative", [_stored_source(source)])

    doc = await get_status(agent_key, "agent", source)
    assert doc is not None
    assert doc["status"] == "completed"

    diary = get_completed_phases(doc)
    assert set(diary) == {"parsing_chunking", "embedding"}
    # parsing re-ran (new marker) and embedding followed the chain: its deps
    # reference the NEW parsing marker, proving the serial order
    assert diary["parsing_chunking"]["marker"] == "test-marker"
    assert diary["embedding"]["marker"] == "test-marker"
    assert diary["embedding"]["deps"] == {"parsing_chunking": "test-marker"}
    assert calls["parse_calls"] == 1
    assert calls["embed_calls"] == 1

    # decision probe + after-parsing re-probe + after-embedding re-probe
    assert len(calls["probe"]) == 3


# ---------------------------------------------------------------------------
# (c) crash after parsing -> ERROR; next pass resumes at embedding (no re-parse)
# ---------------------------------------------------------------------------


async def test_machine_crash_after_parsing_is_error_and_resumes_at_embedding(cheshire_cat, monkeypatch):
    agent_key = cheshire_cat.agent_key
    source = "crash_after_parsing.pdf"
    await set_status(
        agent_key, "agent", source, type_="file", status=IngestionStatus.COMPLETED,
        completed_phases=_completed_diary(parsing_marker="pc_old", embedding_marker="em_old"),
    )

    calls = _stub_reembed_env(
        monkeypatch, cheshire_cat,
        existing_points=[_point(source)],
        pending_result=[{"phase": "parsing_chunking"}, {"phase": "embedding"}],
    )
    # the embedding body crashes exactly once (the interrupted operation)
    boom = {"left": True}

    async def flaky_embed(*args, **kwargs):
        calls["embed_calls"] += 1
        if boom["left"]:
            boom["left"] = False
            raise RuntimeError("embedding boom")
        return []

    monkeypatch.setattr(reembed, "_embed_phase", flaky_embed)

    # ---- PASS 1: parsing completes and records; embedding crashes -> ERROR ----
    await reembed.reembed_sources(cheshire_cat, "declarative", [_stored_source(source)], stale_after=0.0)
    doc = await get_status(agent_key, "agent", source)
    assert doc is not None
    assert doc["status"] == "error"
    assert doc["error"] == "embedding boom"

    # ATOMIC COMPLETION INVARIANT: the interrupted embedding was NOT recorded
    # with the CURRENT inputs — its entry still carries the OLD marker (from
    # the pre-existing completion), so it remains stale on the next pass and is
    # re-run. Only the completed parsing was re-recorded.
    diary = get_completed_phases(doc)
    assert diary["parsing_chunking"]["marker"] == "test-marker"
    assert diary["embedding"]["marker"] == "em_old"
    assert calls["embed_calls"] == 1  # embedding was attempted, then crashed

    # ---- PASS 2: the error row resumes at embedding (no re-parse) ----
    parse_before = calls["parse_calls"]
    await reembed.reembed_sources(cheshire_cat, "declarative", [_stored_source(source)], stale_after=0.0)
    doc = await get_status(agent_key, "agent", source)
    assert doc["status"] == "completed"
    diary = get_completed_phases(doc)
    assert set(diary) == {"parsing_chunking", "embedding"}
    # resumed at embedding WITHOUT re-running parsing
    assert calls["parse_calls"] == parse_before
    assert calls["embed_calls"] == 2  # the crash attempt + the successful resume


# ---------------------------------------------------------------------------
# (d) an error row that fails again stays ERROR (never resurrected)
# ---------------------------------------------------------------------------


async def test_machine_error_row_failing_again_stays_error(cheshire_cat, monkeypatch):
    agent_key = cheshire_cat.agent_key
    source = "stubborn_error.pdf"
    # a pre-existing ERROR row (embedding previously crashed after parsing)
    await set_status(
        agent_key, "agent", source, type_="file", status=IngestionStatus.ERROR,
        error="previous boom",
        completed_phases={
            "parsing_chunking": {"marker": "pc1", "settings_version": "v1", "deps": {}},
        },
    )

    calls = _stub_reembed_env(
        monkeypatch, cheshire_cat,
        existing_points=[_point(source)],
        pending_result=[{"phase": "embedding"}],
    )

    async def always_raising_embed(*args, **kwargs):
        calls["embed_calls"] += 1
        raise RuntimeError("still broken")

    monkeypatch.setattr(reembed, "_embed_phase", always_raising_embed)

    await reembed.reembed_sources(cheshire_cat, "declarative", [_stored_source(source)], stale_after=0.0)

    doc = await get_status(agent_key, "agent", source)
    # the row is NOT resurrected to COMPLETED: it stays ERROR with the NEW error
    assert doc is not None
    assert doc["status"] == "error"
    assert doc["error"] == "still broken"

    # the interrupted embedding has NO diary entry (atomic-completion invariant)
    diary = get_completed_phases(doc)
    assert set(diary) == {"parsing_chunking"}
    assert "embedding" not in diary


# ---------------------------------------------------------------------------
# (e) a double COMPLETED write is harmless (idempotent terminal state)
# ---------------------------------------------------------------------------


async def test_machine_double_completed_write_is_harmless(cheshire_cat, monkeypatch):
    agent_key = cheshire_cat.agent_key
    source = "idempotent.pdf"
    await set_status(
        agent_key, "agent", source, type_="file", status=IngestionStatus.COMPLETED,
        completed_phases=_completed_diary(parsing_marker="pc1", embedding_marker="em_old"),
    )

    calls = _stub_reembed_env(
        monkeypatch, cheshire_cat,
        existing_points=[_point(source)],
        pending_result=[{"phase": "embedding"}],
    )

    # PASS 1: full pass -> COMPLETED with a complete diary. The embedding body
    # fires after_rabbithole_stored_documents which also writes COMPLETED; the
    # dispatcher re-probes and writes COMPLETED again — the double write must
    # be harmless (the row ends COMPLETED, not ERROR/PROCESSING).
    await reembed.reembed_sources(cheshire_cat, "declarative", [_stored_source(source)])
    doc = await get_status(agent_key, "agent", source)
    assert doc is not None
    assert doc["status"] == "completed"
    diary = get_completed_phases(doc)
    assert set(diary) == {"parsing_chunking", "embedding"}

    # PASS 2: the diary is fresh -> the probe is empty and the machine SKIPS
    # the source without ANY status write (idempotent terminal state)
    from cat.db import crud

    writes = []
    real_store = crud.store

    async def recording_store(key, value, *args, **kwargs):
        writes.append((key, value))
        return await real_store(key, value, *args, **kwargs)

    monkeypatch.setattr(crud, "store", recording_store)
    await reembed.reembed_sources(cheshire_cat, "declarative", [_stored_source(source)])

    assert writes == []
    doc = await get_status(agent_key, "agent", source)
    assert doc["status"] == "completed"
