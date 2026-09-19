"""End-to-end integration tests for the EffING two-phase machine (Todo 16).

Drives ``reembed_sources`` (``reembed.py``) through the FULL serial dispatch
loop with a fake ``cat.plugin_manager.execute_hook`` that simulates the REAL
registrant behavior — the ``ingestion_phase_pending`` accumulator runs the
actual ``phases.is_phase_stale`` logic over the ``PHASES`` DAG (dependency
check + settings-version fast-path + marker slow-path), the
``ingestion_phase_settings_marker`` hook returns a CONTROLLABLE per-phase
marker, and ``ingestion_phase_run`` returns ``{"status": "done"}`` for
external phases. The recorded diary ``settings_version`` is driven by a
controllable ``crud_settings.get_settings_by_category`` so the opaque
fast-path token round-trips coherently with the probe.

The end-to-end coverage:

  (a) a fresh ingest records ``parsing_chunking`` then ``embedding`` in the
      diary (serial order, dependency chain);
  (b) an embedder-settings change (marker differs) re-runs ONLY ``embedding``;
  (c) a chunker-settings change (marker differs) re-runs ``parsing_chunking``
      then ``embedding`` (embedding follows the new parsing marker via deps);
  (d) re-saving IDENTICAL settings (version changed, marker equal) re-runs
      NOTHING (the no-op-save guard);
  (e) a crash between phases (the embedding body raises) -> the row is ERROR
      and the next pass resumes at ``embedding`` WITHOUT re-parsing;
  (f) a delete-marked agent aborts the pass WITHOUT writing any zombie
      ``agents:*:ingestion:*`` row;
  (g) an external fake phase dispatches via ``ingestion_phase_run`` and is
      recorded in the diary ONLY on ``done`` (misleading-success guard).

These run against the same real-cat / redis-stack test harness as the Todo 5-7
unit tests (conftest forces ``CAT_REDIS_DB=1`` + flush guard). Import-safe:
no side effects at import time.
"""
from typing import Any
from unittest.mock import AsyncMock

import pytest

from cat.plugins.cat_efficient_ingestion import reembed
from cat.plugins.cat_efficient_ingestion.phases import (
    PHASES,
    _diary_from_completed_phases,
    is_phase_stale,
)
from cat.plugins.cat_efficient_ingestion.registry import (
    IngestionStatus,
    get_completed_phases,
    get_status,
    set_delete_marker,
    set_status,
)

# ---------------------------------------------------------------------------
# helpers (self-contained, mirror of the phase-machine unit-test stubs)
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


def _diary(parsing_marker="pc1", embedding_marker="em1", parsing_sv="v1", embedding_sv="v1"):
    """A completed row's diary: parsing recorded, embedding depends on it."""
    return {
        "parsing_chunking": {"marker": parsing_marker, "settings_version": parsing_sv, "deps": {}},
        "embedding": {
            "marker": embedding_marker, "settings_version": embedding_sv,
            "deps": {"parsing_chunking": parsing_marker},
        },
    }


def _integration_env(
    monkeypatch, cheshire_cat, existing_points,
    markers=None, settings_versions=None,
    external_pending=None, external_run=None,
):
    """Point ``reembed_sources`` at a deterministic, REAL-registrant environment.

    The fake ``execute_hook`` implements the actual ``phases.py`` contract:

      - ``ingestion_phase_pending``: runs ``is_phase_stale`` over ``PHASES``
        (dependency + settings-version fast-path + marker slow-path) with the
        CONTROLLABLE ``markers`` / ``settings_versions``, then appends each
        pending EXTERNAL phase from ``external_pending`` while its diary entry
        is still absent;
      - ``ingestion_phase_settings_marker``: returns ``markers[phase]``;
      - ``ingestion_phase_run``: returns ``external_run[phase]`` (value or
        callable) for external phases;
      - ``before_ingestion_status_completed``: no-op (None).

    ``crud_settings.get_settings_by_category`` is monkeypatched (the official
    CRUD API surface, not raw Redis) so the diary ``settings_version`` recorded
    at phase completion equals the probe's ``settings_versions[category]`` —
    making the settings-version fast-path round-trip end-to-end.

    Phase bodies are faked and counted; the machine never touches real vectors.
    """
    from types import SimpleNamespace

    from langchain_core.documents import Document

    markers = dict(markers or {})
    settings_versions = dict(settings_versions or {})
    external_pending = dict(external_pending or {})
    external_run = dict(external_run or {})

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

    # make the recorded settings_version match the probe's current version
    real_get_settings = reembed.crud_settings.get_settings_by_category

    async def fake_get_settings(cat_key, category):
        if category in settings_versions:
            return {"updated_at": settings_versions[category]}
        return None

    monkeypatch.setattr(reembed.crud_settings, "get_settings_by_category", fake_get_settings)

    calls: dict[str, Any] = {"probe": [], "probe_results": [], "parse_calls": 0, "embed_calls": 0, "run": []}

    async def fake_parse_and_chunk(ccat, rabbit_hole, source, file_bytes, content_type, cat):
        calls["parse_calls"] += 1
        return [Document(page_content="parsed chunk")], []

    async def fake_embed_phase(*args, **kwargs):
        calls["embed_calls"] += 1
        return []

    async def fake_execute_hook(name, *args, **kwargs):
        if name == "ingestion_phase_pending":
            pending = list(args[0]) if args and args[0] else []
            completed = args[2] if len(args) > 2 else []
            diary = _diary_from_completed_phases(completed)
            # REAL registrant logic: PHASES DAG + two-level staleness
            for spec in PHASES.values():
                current_sv = settings_versions.get(spec.settings_category)
                current_marker = markers.get(spec.id)
                if is_phase_stale(spec, diary, current_sv, current_marker):
                    pending.append({"phase": spec.id})
            # external phases: pending while their diary entry is absent
            for phase, wanted in external_pending.items():
                entry = diary.get(phase)
                if wanted and not isinstance(entry, dict):
                    pending.append({"phase": phase})
            calls["probe"].append((args, kwargs))
            calls["probe_results"].append(list(pending))
            return pending
        if name == "ingestion_phase_settings_marker":
            phase = args[0] if args else kwargs.get("phase")
            return markers.get(phase)
        if name == "ingestion_phase_run":
            calls["run"].append((args, kwargs))
            phase = args[0] if args else kwargs.get("phase")
            result = external_run.get(phase)
            if callable(result):
                return result(*args, **kwargs)
            return result
        if name == "before_ingestion_status_completed":
            return None
        return []

    monkeypatch.setattr(cheshire_cat.plugin_manager, "execute_hook", fake_execute_hook)
    monkeypatch.setattr(reembed, "_parse_and_chunk", fake_parse_and_chunk)
    monkeypatch.setattr(reembed, "_embed_phase", fake_embed_phase)
    return calls


# ---------------------------------------------------------------------------
# (a) fresh ingest records parsing_chunking THEN embedding in the diary
# ---------------------------------------------------------------------------


async def test_integration_fresh_ingest_records_both_phases(cheshire_cat, monkeypatch):
    agent_key = cheshire_cat.agent_key
    source = "fresh_ingest.pdf"

    calls = _integration_env(
        monkeypatch, cheshire_cat,
        existing_points=[],
        markers={"parsing_chunking": "m1", "embedding": "m1"},
        settings_versions={"chunker": "v1", "embedder": "v1"},
    )
    # no status doc, no points -> the machine decides parsing_chunking first
    await reembed.reembed_sources(cheshire_cat, "declarative", [_stored_source(source)])

    doc = await get_status(agent_key, "agent", source)
    assert doc is not None
    assert doc["status"] == "completed"
    assert "phase" not in doc  # terminal write cleared the work phase

    # BOTH phases were recorded in the diary with the current markers, and
    # embedding deps reference the parsing marker (serial order: parsing ran
    # and recorded BEFORE embedding started).
    diary = get_completed_phases(doc)
    assert set(diary) == {"parsing_chunking", "embedding"}
    assert diary["parsing_chunking"]["marker"] == "m1"
    assert diary["parsing_chunking"]["settings_version"] == "v1"
    assert diary["embedding"]["marker"] == "m1"
    assert diary["embedding"]["settings_version"] == "v1"
    assert diary["embedding"]["deps"] == {"parsing_chunking": "m1"}
    assert calls["parse_calls"] == 1
    assert calls["embed_calls"] == 1

    # the machine probed BETWEEN the phases: after parsing recorded, embedding
    # was still pending; after embedding recorded, nothing was pending anymore
    assert len(calls["probe"]) == 2
    assert [p["phase"] for p in calls["probe_results"][0]] == ["embedding"]
    assert calls["probe_results"][1] == []


# ---------------------------------------------------------------------------
# (b) embedder-settings change (marker differs) re-runs ONLY embedding
# ---------------------------------------------------------------------------


async def test_integration_embedder_change_reruns_only_embedding(cheshire_cat, monkeypatch):
    agent_key = cheshire_cat.agent_key
    source = "embedder_change.pdf"
    await set_status(
        agent_key, "agent", source, type_="file", status=IngestionStatus.COMPLETED,
        completed_phases=_diary(parsing_marker="m1", embedding_marker="m_old"),
    )

    calls = _integration_env(
        monkeypatch, cheshire_cat,
        existing_points=[_point(source)],
        markers={"parsing_chunking": "m1", "embedding": "m_new"},
        settings_versions={"chunker": "v1", "embedder": "v2"},
    )
    await reembed.reembed_sources(cheshire_cat, "declarative", [_stored_source(source)])

    doc = await get_status(agent_key, "agent", source)
    assert doc is not None
    assert doc["status"] == "completed"

    diary = get_completed_phases(doc)
    assert set(diary) == {"parsing_chunking", "embedding"}
    # ONLY embedding was re-recorded; parsing was NOT re-run (chunker settings
    # version v1 == v1 -> fast-path fresh; its marker is preserved unchanged)
    assert diary["parsing_chunking"]["marker"] == "m1"
    assert diary["parsing_chunking"]["settings_version"] == "v1"
    assert diary["embedding"]["marker"] == "m_new"
    assert diary["embedding"]["settings_version"] == "v2"
    assert diary["embedding"]["deps"] == {"parsing_chunking": "m1"}
    assert calls["parse_calls"] == 0
    assert calls["embed_calls"] == 1

    # decision probe + post-embedding re-probe (now empty)
    assert len(calls["probe"]) == 2
    assert [p["phase"] for p in calls["probe_results"][0]] == ["embedding"]
    assert calls["probe_results"][1] == []


# ---------------------------------------------------------------------------
# (c) chunker-settings change re-runs parsing_chunking THEN embedding (chain)
# ---------------------------------------------------------------------------


async def test_integration_chunker_change_reruns_parsing_then_embedding(cheshire_cat, monkeypatch):
    agent_key = cheshire_cat.agent_key
    source = "chunker_change.pdf"
    await set_status(
        agent_key, "agent", source, type_="file", status=IngestionStatus.COMPLETED,
        completed_phases=_diary(parsing_marker="pc_old", embedding_marker="m1"),
    )

    calls = _integration_env(
        monkeypatch, cheshire_cat,
        existing_points=[_point(source)],
        markers={"parsing_chunking": "pc_new", "embedding": "m1"},
        settings_versions={"chunker": "v2", "embedder": "v1"},
    )
    await reembed.reembed_sources(cheshire_cat, "declarative", [_stored_source(source)])

    doc = await get_status(agent_key, "agent", source)
    assert doc is not None
    assert doc["status"] == "completed"

    diary = get_completed_phases(doc)
    assert set(diary) == {"parsing_chunking", "embedding"}
    # parsing re-ran (new marker) and embedding FOLLOWED the chain: its deps
    # reference the NEW parsing marker, proving the serial order
    assert diary["parsing_chunking"]["marker"] == "pc_new"
    assert diary["embedding"]["marker"] == "m1"
    assert diary["embedding"]["settings_version"] == "v1"
    assert diary["embedding"]["deps"] == {"parsing_chunking": "pc_new"}
    assert calls["parse_calls"] == 1
    assert calls["embed_calls"] == 1

    # decision probe (parsing only) + after-parsing re-probe (embedding now
    # stale by DEPENDENCY) + after-embedding re-probe (empty)
    assert len(calls["probe"]) == 3
    assert [p["phase"] for p in calls["probe_results"][0]] == ["parsing_chunking"]
    assert [p["phase"] for p in calls["probe_results"][1]] == ["embedding"]
    assert calls["probe_results"][2] == []


# ---------------------------------------------------------------------------
# (d) re-saving IDENTICAL settings (version changed, marker equal) re-runs NOTHING
# ---------------------------------------------------------------------------


async def test_integration_noop_save_reruns_nothing(cheshire_cat, monkeypatch):
    agent_key = cheshire_cat.agent_key
    source = "noop_save.pdf"
    await set_status(
        agent_key, "agent", source, type_="file", status=IngestionStatus.COMPLETED,
        completed_phases=_diary(parsing_marker="m1", embedding_marker="m1"),
    )

    calls = _integration_env(
        monkeypatch, cheshire_cat,
        existing_points=[_point(source)],
        # IDENTICAL markers (re-saved settings with the same content)...
        markers={"parsing_chunking": "m1", "embedding": "m1"},
        # ...but the settings entries were rewritten (versions changed, so the
        # fast-path does NOT apply and the marker slow-path decides: equal -> FRESH)
        settings_versions={"chunker": "v2", "embedder": "v2"},
    )

    from cat.db import crud

    writes = []
    real_store = crud.store

    async def recording_store(key, value, *args, **kwargs):
        writes.append((key, value))
        return await real_store(key, value, *args, **kwargs)

    monkeypatch.setattr(crud, "store", recording_store)

    await reembed.reembed_sources(cheshire_cat, "declarative", [_stored_source(source)])

    # the no-op-save guard: NOTHING re-ran and NO status write happened
    assert calls["parse_calls"] == 0
    assert calls["embed_calls"] == 0
    assert writes == []

    doc = await get_status(agent_key, "agent", source)
    assert doc is not None
    assert doc["status"] == "completed"
    # the diary is byte-for-byte untouched (still the original markers/versions)
    diary = get_completed_phases(doc)
    assert diary == _diary(parsing_marker="m1", embedding_marker="m1")


# ---------------------------------------------------------------------------
# (e) a crash between phases resumes at embedding on the next pass (no re-parse)
# ---------------------------------------------------------------------------


async def test_integration_crash_between_phases_resumes_at_embedding(cheshire_cat, monkeypatch):
    agent_key = cheshire_cat.agent_key
    source = "crash_between.pdf"
    await set_status(
        agent_key, "agent", source, type_="file", status=IngestionStatus.COMPLETED,
        completed_phases=_diary(parsing_marker="pc_old", embedding_marker="em_old"),
    )

    calls = _integration_env(
        monkeypatch, cheshire_cat,
        existing_points=[_point(source)],
        markers={"parsing_chunking": "pc1", "embedding": "em1"},
        settings_versions={"chunker": "v2", "embedder": "v2"},
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

    # ATOMIC COMPLETION INVARIANT: the interrupted embedding was NOT re-recorded
    # with the current inputs — its entry still carries the OLD marker (from the
    # pre-existing completion), so it stays stale on the next pass.
    diary = get_completed_phases(doc)
    assert diary["parsing_chunking"]["marker"] == "pc1"
    assert diary["embedding"]["marker"] == "em_old"
    assert calls["parse_calls"] == 1
    assert calls["embed_calls"] == 1  # embedding was attempted, then crashed

    # ---- PASS 2: the error row resumes at embedding (no re-parse) ----
    parse_before = calls["parse_calls"]
    await reembed.reembed_sources(cheshire_cat, "declarative", [_stored_source(source)], stale_after=0.0)
    doc = await get_status(agent_key, "agent", source)
    assert doc["status"] == "completed"
    diary = get_completed_phases(doc)
    assert set(diary) == {"parsing_chunking", "embedding"}
    assert diary["embedding"]["marker"] == "em1"
    assert diary["embedding"]["deps"] == {"parsing_chunking": "pc1"}
    # resumed at embedding WITHOUT re-running parsing
    assert calls["parse_calls"] == parse_before
    assert calls["embed_calls"] == 2  # the crash attempt + the successful resume


# ---------------------------------------------------------------------------
# (f) a delete-marked agent aborts without zombie rows
# ---------------------------------------------------------------------------


async def test_integration_canceled_agent_aborts_no_zombie_rows(cheshire_cat, monkeypatch):
    agent_key = cheshire_cat.agent_key
    source = "canceled_agent.pdf"
    await set_status(
        agent_key, "agent", source, type_="file", status=IngestionStatus.COMPLETED,
        completed_phases=_diary(parsing_marker="m1", embedding_marker="m_old"),
    )
    # the agent is being deleted: the marker is present -> ingestion_canceled
    await set_delete_marker(agent_key)

    calls = _integration_env(
        monkeypatch, cheshire_cat,
        existing_points=[_point(source)],
        markers={"parsing_chunking": "m1", "embedding": "m_new"},
        settings_versions={"chunker": "v1", "embedder": "v2"},
    )

    from cat.db import crud

    writes = []
    real_store = crud.store

    async def recording_store(key, value, *args, **kwargs):
        writes.append((key, value))
        return await real_store(key, value, *args, **kwargs)

    monkeypatch.setattr(crud, "store", recording_store)

    # the pass would re-embed this source (embedding is stale)...
    await reembed.reembed_sources(cheshire_cat, "declarative", [_stored_source(source)], stale_after=0.0)

    # ...but the delete marker aborts the claim: NOT a single
    # agents:*:ingestion:* row is written after cancellation (no zombie rows).
    assert writes == []
    assert calls["parse_calls"] == 0
    assert calls["embed_calls"] == 0

    # the pre-existing COMPLETED row is untouched (no resurrection, no phase)
    doc = await get_status(agent_key, "agent", source)
    assert doc is not None
    assert doc["status"] == "completed"
    assert "phase" not in doc


# ---------------------------------------------------------------------------
# (g) external fake phase dispatches via ingestion_phase_run, recorded on done
# ---------------------------------------------------------------------------


async def test_integration_external_phase_recorded_on_done(cheshire_cat, monkeypatch):
    agent_key = cheshire_cat.agent_key
    source = "external_done.pdf"
    await set_status(
        agent_key, "agent", source, type_="file", status=IngestionStatus.COMPLETED,
        completed_phases=_diary(parsing_marker="m1", embedding_marker="m1"),
    )

    calls = _integration_env(
        monkeypatch, cheshire_cat,
        existing_points=[_point(source)],
        markers={"parsing_chunking": "m1", "embedding": "m1"},
        settings_versions={"chunker": "v1", "embedder": "v1"},
        external_pending={"graphrag_index": True},
        external_run={"graphrag_index": {"status": "done"}},
    )
    await reembed.reembed_sources(cheshire_cat, "declarative", [_stored_source(source)])

    doc = await get_status(agent_key, "agent", source)
    assert doc is not None
    assert doc["status"] == "completed"

    # the external phase was dispatched and RECORDED in the diary (minimal
    # entry: no marker/settings version/deps — the registrant owns the inputs)
    diary = get_completed_phases(doc)
    assert set(diary) == {"parsing_chunking", "embedding", "graphrag_index"}
    assert diary["graphrag_index"] == {"marker": None, "settings_version": None, "deps": {}}
    assert diary["parsing_chunking"]["marker"] == "m1"  # untouched

    # the run hook was invoked exactly once with (phase, source, completed_list)
    assert len(calls["run"]) == 1
    phase_arg, source_arg, completed_list = calls["run"][0][0]
    assert phase_arg == "graphrag_index"
    assert source_arg == source
    assert {e["phase"] for e in completed_list} == {"parsing_chunking", "embedding"}

    # decision probe + re-probe after the external recording (now empty)
    assert len(calls["probe"]) == 2
    assert [p["phase"] for p in calls["probe_results"][0]] == ["graphrag_index"]
    assert calls["probe_results"][1] == []


async def test_integration_external_phase_not_recorded_on_non_done(cheshire_cat, monkeypatch):
    agent_key = cheshire_cat.agent_key
    source = "external_none.pdf"
    await set_status(
        agent_key, "agent", source, type_="file", status=IngestionStatus.COMPLETED,
        completed_phases=_diary(parsing_marker="m1", embedding_marker="m1"),
    )

    # external_run defaults to None: the registrant is unimplemented for the
    # phase -> ingestion_phase_run returns None -> fail-hard (misleading-success
    # guard: a phase that did NOT report done is NEVER recorded in the diary)
    calls = _integration_env(
        monkeypatch, cheshire_cat,
        existing_points=[_point(source)],
        markers={"parsing_chunking": "m1", "embedding": "m1"},
        settings_versions={"chunker": "v1", "embedder": "v1"},
        external_pending={"graphrag_index": True},
    )
    await reembed.reembed_sources(cheshire_cat, "declarative", [_stored_source(source)])

    doc = await get_status(agent_key, "agent", source)
    assert doc is not None
    assert doc["status"] == "error"
    assert "expected a status dict" in doc["error"]

    # misleading-success guard: the external phase is NOT in the diary
    assert "graphrag_index" not in get_completed_phases(doc)
    assert len(calls["run"]) == 1
