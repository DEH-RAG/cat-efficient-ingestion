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


def _stub_reembed_env(
    monkeypatch, cheshire_cat, existing_points, pending_result,
    run_result=None, diary_fresh_phases=None, specs_result=None,
):
    """Point ``reembed_sources`` at a deterministic environment.

    The ``ingestion_phase_pending`` probe is DIARY-AWARE: a phase in
    ``pending_result`` is reported stale until its diary entry carries marker
    ``"test-marker"`` (the fixed ``ingestion_phase_settings_marker`` result) —
    exactly like the real clock-free registrant, so the loop re-probe after a
    recorded phase returns empty without a hand-rolled script. Phases listed in
    ``diary_fresh_phases`` are treated as EXTERNAL (no marker): fresh as soon as
    ANY diary entry exists for them. ``specs_result`` (a list of ``PhaseSpec``)
    is returned by the ``ingestion_phase_specs`` hook, so ``merged_phases``
    treats those phases as REGISTERED (real marker + declared deps) instead of
    unknown. Phase bodies are faked and counted, so the machine never touches
    vectors. ``run_result`` (value or callable) is the ``ingestion_phase_run``
    result; ``None`` (default) means the hook is unimplemented, exactly like the
    no-op registrant.
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

    calls: dict[str, Any] = {"probe": [], "parse_calls": 0, "embed_calls": 0, "run": []}
    base_pending = list(pending_result or [])
    diary_fresh = set(diary_fresh_phases or [])
    specs = list(specs_result or [])

    async def fake_parse_and_chunk(ccat, rabbit_hole, source, file_bytes, content_type, cat):
        calls["parse_calls"] += 1
        return [Document(page_content="parsed chunk")], []

    async def fake_embed_phase(*args, **kwargs):
        calls["embed_calls"] += 1
        return []

    async def fake_execute_hook(name, *args, **kwargs):
        if name == "ingestion_phase_specs":
            return specs
        if name == "ingestion_phase_pending":
            calls["probe"].append((args, kwargs))
            completed = args[2]
            diary = {e["phase"]: e for e in completed if isinstance(e, dict)}
            out = []
            for p in base_pending:
                entry = diary.get(p["phase"])
                if p["phase"] in diary_fresh:
                    # external phases carry no marker: fresh once recorded
                    if not isinstance(entry, dict):
                        out.append(p)
                elif not isinstance(entry, dict) or entry.get("marker") != "test-marker":
                    out.append(p)
            return out
        if name == "ingestion_phase_settings_marker":
            return "test-marker"
        if name == "ingestion_phase_run":
            calls["run"].append((args, kwargs))
            if callable(run_result):
                return run_result(*args, **kwargs)
            return run_result
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


# ---------------------------------------------------------------------------
# Todo 7: EXTERNAL phase dispatch through the ``ingestion_phase_run`` hook
# ---------------------------------------------------------------------------
# (a) a fake ``ingestion_phase_run`` returning ``{"status": "done"}`` records
#     the external phase in the diary and the machine proceeds to COMPLETED;
# (b) ``{"status": "not_ready", "retry_after": 0}`` retries then succeeds;
# (c) ``not_ready`` beyond the retry bound -> row ERROR (no infinite loop);
# (d) a raise -> row ERROR (fail-hard, never silent success);
# (e) an unimplemented external phase (hook returns None) -> ERROR.


def _external_pending(phase="graphrag_index"):
    return [{"phase": phase}]


async def test_machine_external_phase_done_records_and_completes(cheshire_cat, monkeypatch):
    agent_key = cheshire_cat.agent_key
    source = "external_done.pdf"
    await set_status(
        agent_key, "agent", source, type_="file", status=IngestionStatus.COMPLETED,
        completed_phases=_completed_diary(parsing_marker="pc1", embedding_marker="em1"),
    )

    calls = _stub_reembed_env(
        monkeypatch, cheshire_cat,
        existing_points=[_point(source)],
        pending_result=_external_pending(),
        run_result={"status": "done"},
        diary_fresh_phases={"graphrag_index"},
    )
    await reembed.reembed_sources(cheshire_cat, "declarative", [_stored_source(source)])

    doc = await get_status(agent_key, "agent", source)
    assert doc is not None
    assert doc["status"] == "completed"
    assert "phase" not in doc  # terminal write cleared the work phase

    # the external phase is recorded in the diary (minimal entry) and the
    # re-probe emptied -> COMPLETED (stale-state protection)
    diary = get_completed_phases(doc)
    assert set(diary) == {"parsing_chunking", "embedding", "graphrag_index"}
    assert diary["graphrag_index"] == {"marker": None, "settings_version": None, "deps": {}}
    assert diary["parsing_chunking"]["marker"] == "pc1"  # untouched

    # the run hook was invoked exactly once with (phase, source, completed_list)
    assert len(calls["run"]) == 1
    phase_arg, source_arg, completed_list = calls["run"][0][0]
    assert phase_arg == "graphrag_index"
    assert source_arg == source
    assert {e["phase"] for e in completed_list} == {"parsing_chunking", "embedding"}

    # decision probe + re-probe after recording
    assert len(calls["probe"]) == 2


async def test_machine_external_phase_not_ready_retries_then_succeeds(cheshire_cat, monkeypatch):
    agent_key = cheshire_cat.agent_key
    source = "external_retry.pdf"
    await set_status(
        agent_key, "agent", source, type_="file", status=IngestionStatus.COMPLETED,
        completed_phases=_completed_diary(parsing_marker="pc1", embedding_marker="em1"),
    )

    attempts = {"n": 0}

    def flaky_run(*args, **kwargs):
        attempts["n"] += 1
        if attempts["n"] < 3:
            return {"status": "not_ready", "retry_after": 0}
        return {"status": "done"}

    calls = _stub_reembed_env(
        monkeypatch, cheshire_cat,
        existing_points=[_point(source)],
        pending_result=_external_pending(),
        run_result=flaky_run,
        diary_fresh_phases={"graphrag_index"},
    )
    await reembed.reembed_sources(cheshire_cat, "declarative", [_stored_source(source)])

    doc = await get_status(agent_key, "agent", source)
    assert doc is not None
    assert doc["status"] == "completed"
    assert attempts["n"] == 3  # not_ready twice, done on the third attempt
    assert "graphrag_index" in get_completed_phases(doc)


async def test_machine_external_phase_not_ready_beyond_bound_is_error(cheshire_cat, monkeypatch):
    agent_key = cheshire_cat.agent_key
    source = "external_stuck.pdf"
    await set_status(
        agent_key, "agent", source, type_="file", status=IngestionStatus.COMPLETED,
        completed_phases=_completed_diary(parsing_marker="pc1", embedding_marker="em1"),
    )

    calls = _stub_reembed_env(
        monkeypatch, cheshire_cat,
        existing_points=[_point(source)],
        pending_result=_external_pending(),
        run_result={"status": "not_ready", "retry_after": 0},
        diary_fresh_phases={"graphrag_index"},
    )
    await reembed.reembed_sources(cheshire_cat, "declarative", [_stored_source(source)])

    doc = await get_status(agent_key, "agent", source)
    assert doc is not None
    assert doc["status"] == "error"
    assert "not ready" in doc["error"]

    # bounded: exactly MAX_RETRIES + 1 attempts, no infinite loop
    assert len(calls["run"]) == reembed._PHASE_RUN_MAX_RETRIES + 1

    # atomic-completion invariant: the never-completed external phase has NO
    # diary entry
    assert "graphrag_index" not in get_completed_phases(doc)


async def test_machine_external_phase_raise_is_error(cheshire_cat, monkeypatch):
    agent_key = cheshire_cat.agent_key
    source = "external_raise.pdf"
    await set_status(
        agent_key, "agent", source, type_="file", status=IngestionStatus.COMPLETED,
        completed_phases=_completed_diary(parsing_marker="pc1", embedding_marker="em1"),
    )

    def boom(*args, **kwargs):
        raise RuntimeError("external boom")

    calls = _stub_reembed_env(
        monkeypatch, cheshire_cat,
        existing_points=[_point(source)],
        pending_result=_external_pending(),
        run_result=boom,
        diary_fresh_phases={"graphrag_index"},
    )
    await reembed.reembed_sources(cheshire_cat, "declarative", [_stored_source(source)])

    doc = await get_status(agent_key, "agent", source)
    assert doc is not None
    assert doc["status"] == "error"
    assert doc["error"] == "external boom"
    # fail-hard: the phase is NOT recorded, the row is NOT resurrected
    assert "graphrag_index" not in get_completed_phases(doc)
    assert len(calls["run"]) == 1  # a raise is a permanent failure, no retry


async def test_machine_external_phase_unimplemented_is_error(cheshire_cat, monkeypatch):
    agent_key = cheshire_cat.agent_key
    source = "external_none.pdf"
    await set_status(
        agent_key, "agent", source, type_="file", status=IngestionStatus.COMPLETED,
        completed_phases=_completed_diary(parsing_marker="pc1", embedding_marker="em1"),
    )

    # run_result defaults to None: the hook is unimplemented (no registrant)
    calls = _stub_reembed_env(
        monkeypatch, cheshire_cat,
        existing_points=[_point(source)],
        pending_result=_external_pending(),
        diary_fresh_phases={"graphrag_index"},
    )
    await reembed.reembed_sources(cheshire_cat, "declarative", [_stored_source(source)])

    doc = await get_status(agent_key, "agent", source)
    assert doc is not None
    assert doc["status"] == "error"
    assert "expected a status dict" in doc["error"]
    # misleading-success guard: None is NEVER treated as success
    assert "graphrag_index" not in get_completed_phases(doc)
    assert len(calls["run"]) == 1


# ---------------------------------------------------------------------------
# (f) REGISTERED phases (ingestion_phase_specs) get a REAL diary entry
# ---------------------------------------------------------------------------


def _registered_specs(phase="fake_phase", settings_category="fake_settings",
                      depends_on=("parsing_chunking", "embedding")):
    from cat.plugins.cat_efficient_ingestion.phases import PhaseSpec

    return [PhaseSpec(phase, settings_category, depends_on)]


async def test_machine_registered_phase_records_real_marker_and_deps(cheshire_cat, monkeypatch):
    """A phase registered via ``ingestion_phase_specs`` is recorded like a
    built-in: REAL marker from ``ingestion_phase_settings_marker`` (not None)
    and ``deps`` copied from the CURRENT diary upstream markers."""
    agent_key = cheshire_cat.agent_key
    source = "registered_done.pdf"
    await set_status(
        agent_key, "agent", source, type_="file", status=IngestionStatus.COMPLETED,
        completed_phases=_completed_diary(parsing_marker="pc1", embedding_marker="em1"),
    )

    calls = _stub_reembed_env(
        monkeypatch, cheshire_cat,
        existing_points=[_point(source)],
        pending_result=[{"phase": "fake_phase"}],
        run_result={"status": "done"},
        specs_result=_registered_specs(),
    )
    await reembed.reembed_sources(cheshire_cat, "declarative", [_stored_source(source)])

    doc = await get_status(agent_key, "agent", source)
    assert doc is not None
    assert doc["status"] == "completed"

    diary = get_completed_phases(doc)
    assert set(diary) == {"parsing_chunking", "embedding", "fake_phase"}
    entry = diary["fake_phase"]
    # REAL marker from the settings-marker hook, NOT the None fallback
    assert entry["marker"] == "test-marker"
    # deps reference the upstream markers from the CURRENT diary
    assert entry["deps"] == {"parsing_chunking": "pc1", "embedding": "em1"}
    assert "settings_version" in entry
    # upstream entries untouched
    assert diary["parsing_chunking"]["marker"] == "pc1"
    assert diary["embedding"]["marker"] == "em1"

    # the run hook was invoked exactly once with (phase, source, completed_list)
    assert len(calls["run"]) == 1
    phase_arg, source_arg, completed_list = calls["run"][0][0]
    assert phase_arg == "fake_phase"
    assert source_arg == source
    assert {e["phase"] for e in completed_list} == {"parsing_chunking", "embedding"}

    # decision probe + re-probe after recording (now empty: marker matches)
    assert len(calls["probe"]) == 2


async def test_machine_registered_phase_without_settings_category_records_none_version(cheshire_cat, monkeypatch):
    """A registered phase with ``settings_category=None`` records
    ``settings_version=None`` (no settings entry to fingerprint) but still a
    REAL marker — the marker-only invalidation path."""
    agent_key = cheshire_cat.agent_key
    source = "registered_nocat.pdf"
    await set_status(
        agent_key, "agent", source, type_="file", status=IngestionStatus.COMPLETED,
        completed_phases=_completed_diary(parsing_marker="pc1", embedding_marker="em1"),
    )

    calls = _stub_reembed_env(
        monkeypatch, cheshire_cat,
        existing_points=[_point(source)],
        pending_result=[{"phase": "fake_phase"}],
        run_result={"status": "done"},
        specs_result=_registered_specs(settings_category=None),
    )
    await reembed.reembed_sources(cheshire_cat, "declarative", [_stored_source(source)])

    doc = await get_status(agent_key, "agent", source)
    assert doc is not None
    assert doc["status"] == "completed"

    entry = get_completed_phases(doc)["fake_phase"]
    assert entry["marker"] == "test-marker"  # REAL marker, not None
    assert entry["settings_version"] is None  # no settings category -> None
    assert entry["deps"] == {"parsing_chunking": "pc1", "embedding": "em1"}
    assert len(calls["run"]) == 1
