"""Integration tests for a fake REGISTERED phase end-to-end (Todo 11).

Drives ``reembed_sources`` (``reembed.py``) through the FULL serial dispatch
loop with a fake ``ingestion_phase_specs`` registrant declaring ``fake_phase``
(``depends_on=("parsing_chunking", "embedding")``) — exactly like the Todo 4
registration tests, but across the whole machine against the isolated test DB
(conftest forces ``CAT_REDIS_DB=1`` + flush guard):

  (a) a fresh ingest records ``parsing_chunking`` -> ``embedding`` ->
      ``fake_phase`` in the diary (serial order, dependency chain, terminal
      COMPLETED);
  (b) a marker change re-runs ONLY ``fake_phase`` (parse/embed calls stay 0,
      the fake run is called once, the diary marker updates);
  (c) a crash mid-``fake_phase`` (the run hook raises) -> row ERROR and
      ``fake_phase`` is NOT in the diary (atomic-completion invariant); the
      second pass resumes and completes;
  (d) a delete-marked agent aborts the pass WITHOUT writing any zombie
      ``agents:*:ingestion:*`` row (the stale completed row is untouched).

The harness is the ``_stub_reembed_env`` from the Todo 4 registration tests
(self-contained on purpose): the real ``cheshire_cat`` fixture, the fake
``ingestion_phase_specs`` / ``ingestion_phase_settings_marker`` /
``ingestion_phase_run`` hooks and the counted phase bodies. Import-safe: no
side effects at import time.
"""
from typing import Any
from unittest.mock import AsyncMock

from cat.plugins.cat_efficient_ingestion import reembed
from cat.plugins.cat_efficient_ingestion.phases import PhaseSpec
from cat.plugins.cat_efficient_ingestion.registry import (
    IngestionStatus,
    get_completed_phases,
    get_status,
    set_delete_marker,
    set_status,
)

# ---------------------------------------------------------------------------
# helpers (mirror of the phase-machine test harness, self-contained on purpose)
# ---------------------------------------------------------------------------

FAKE_SPEC = PhaseSpec("fake_phase", None, ("parsing_chunking", "embedding"))


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


def _entry(marker="m1", settings_version: str | None = "v1", deps=None):
    """A diary entry: ``{marker, settings_version, deps}``."""
    return {"marker": marker, "settings_version": settings_version, "deps": deps or {}}


def _fresh_diary():
    """A fully fresh diary: parsing and embedding both completed, consistent."""
    return {
        "parsing_chunking": _entry(marker="pc1", settings_version="v1"),
        "embedding": _entry(
            marker="em1", settings_version="v1",
            deps={"parsing_chunking": "pc1"},
        ),
    }


def _fake_diary():
    """A fresh diary including a completed fake_phase with matching deps."""
    diary = _fresh_diary()
    diary["fake_phase"] = _entry(
        marker="fp1", settings_version=None,
        deps={"parsing_chunking": "pc1", "embedding": "em1"},
    )
    return diary


# ---------------------------------------------------------------------------
# (a) fresh ingest records parsing_chunking -> embedding -> fake_phase
# ---------------------------------------------------------------------------


async def test_integration_fresh_ingest_records_parsing_embedding_fake_phase(cheshire_cat, monkeypatch):
    """A fake ``ingestion_phase_specs`` registrant declares ``fake_phase``; a
    fresh ingest (no status doc, no points) records ``parsing_chunking`` ->
    ``embedding`` -> ``fake_phase`` in the diary and terminates COMPLETED."""
    agent_key = cheshire_cat.agent_key
    source = "fresh_registered.pdf"

    calls = _stub_reembed_env(
        monkeypatch, cheshire_cat,
        existing_points=[],
        pending_result=[
            {"phase": "parsing_chunking"},
            {"phase": "embedding"},
            {"phase": "fake_phase"},
        ],
        run_result={"status": "done"},
        specs_result=[FAKE_SPEC],
    )
    # no status doc, no points -> the machine decides parsing_chunking first
    await reembed.reembed_sources(cheshire_cat, "declarative", [_stored_source(source)])

    doc = await get_status(agent_key, "agent", source)
    assert doc is not None
    assert doc["status"] == "completed"
    assert "phase" not in doc  # terminal write cleared the work phase

    # ALL THREE phases recorded in the diary (serial order)
    diary = get_completed_phases(doc)
    assert set(diary) == {"parsing_chunking", "embedding", "fake_phase"}

    # fake_phase is a REGISTERED phase: REAL marker + deps referencing the
    # upstream markers from the CURRENT diary (settings_category=None -> no
    # settings fast-path token)
    entry = diary["fake_phase"]
    assert entry["marker"] == "test-marker"
    assert entry["settings_version"] is None
    assert entry["deps"] == {
        "parsing_chunking": diary["parsing_chunking"]["marker"],
        "embedding": diary["embedding"]["marker"],
    }
    # embedding deps reference the parsing marker (serial order: parsing ran
    # and recorded BEFORE embedding started)
    assert diary["embedding"]["deps"] == {
        "parsing_chunking": diary["parsing_chunking"]["marker"],
    }

    # each phase body ran exactly once
    assert calls["parse_calls"] == 1
    assert calls["embed_calls"] == 1
    assert len(calls["run"]) == 1
    phase_arg, source_arg, completed_list = calls["run"][0][0]
    assert phase_arg == "fake_phase"
    assert source_arg == source
    # the fake_phase run saw the diary with parsing + embedding recorded
    assert {e["phase"] for e in completed_list} == {"parsing_chunking", "embedding"}

    # the machine re-probed BETWEEN the phases: after parsing (embedding +
    # fake_phase pending), after embedding (fake_phase pending), after
    # fake_phase (nothing pending anymore)
    assert len(calls["probe"]) == 3


# ---------------------------------------------------------------------------
# (b) a marker change re-runs ONLY fake_phase
# ---------------------------------------------------------------------------


async def test_integration_marker_change_reruns_only_fake_phase(cheshire_cat, monkeypatch):
    """A completed diary with all three phases; the fake
    ``ingestion_phase_settings_marker`` for ``fake_phase`` changed (the
    recorded marker differs from the current one) -> ONLY ``fake_phase`` is
    re-run: parse/embed calls stay 0, the fake run is called once, and the
    diary marker updates."""
    agent_key = cheshire_cat.agent_key
    source = "marker_change.pdf"
    await set_status(
        agent_key, "agent", source, type_="file", status=IngestionStatus.COMPLETED,
        completed_phases=_fake_diary(),  # fake_phase recorded with marker "fp1"
    )

    calls = _stub_reembed_env(
        monkeypatch, cheshire_cat,
        existing_points=[_point(source)],
        pending_result=[{"phase": "fake_phase"}],
        run_result={"status": "done"},
        specs_result=[FAKE_SPEC],
    )
    await reembed.reembed_sources(cheshire_cat, "declarative", [_stored_source(source)])

    doc = await get_status(agent_key, "agent", source)
    assert doc is not None
    assert doc["status"] == "completed"

    diary = get_completed_phases(doc)
    assert set(diary) == {"parsing_chunking", "embedding", "fake_phase"}
    # ONLY fake_phase was re-run: the built-in phase bodies were NOT touched
    assert calls["parse_calls"] == 0
    assert calls["embed_calls"] == 0
    assert len(calls["run"]) == 1
    # the diary marker updated to the CURRENT marker
    assert diary["fake_phase"]["marker"] == "test-marker"
    # upstream entries untouched
    assert diary["parsing_chunking"]["marker"] == "pc1"
    assert diary["embedding"]["marker"] == "em1"
    # deps still reference the upstream markers from the CURRENT diary
    assert diary["fake_phase"]["deps"] == {"parsing_chunking": "pc1", "embedding": "em1"}

    # decision probe + re-probe after the fake_phase recording (now empty)
    assert len(calls["probe"]) == 2


# ---------------------------------------------------------------------------
# (c) a crash mid-fake_phase resumes it (atomic-completion invariant)
# ---------------------------------------------------------------------------


async def test_integration_crash_mid_fake_phase_resumes(cheshire_cat, monkeypatch):
    """The fake ``ingestion_phase_run`` raises on the first pass -> row ERROR
    and ``fake_phase`` is NOT in the diary (atomic completion: the entry is
    written only on success); the second pass resumes and completes."""
    agent_key = cheshire_cat.agent_key
    source = "crash_fake_phase.pdf"
    # a completed diary with parsing + embedding (fresh markers); fake_phase
    # has NOT run yet
    await set_status(
        agent_key, "agent", source, type_="file", status=IngestionStatus.COMPLETED,
        completed_phases=_fresh_diary(),
    )

    boom = {"left": True}

    def flaky_run(*args, **kwargs):
        if boom["left"]:
            boom["left"] = False
            raise RuntimeError("fake_phase boom")
        return {"status": "done"}

    calls = _stub_reembed_env(
        monkeypatch, cheshire_cat,
        existing_points=[_point(source)],
        pending_result=[{"phase": "fake_phase"}],
        run_result=flaky_run,
        specs_result=[FAKE_SPEC],
    )

    # ---- PASS 1: the run hook raises -> row ERROR, fake_phase NOT recorded ----
    await reembed.reembed_sources(cheshire_cat, "declarative", [_stored_source(source)], stale_after=0.0)
    doc = await get_status(agent_key, "agent", source)
    assert doc is not None
    assert doc["status"] == "error"
    assert doc["error"] == "fake_phase boom"

    # ATOMIC COMPLETION INVARIANT: the interrupted fake_phase has NO diary
    # entry (it is written ONLY on successful completion)
    diary = get_completed_phases(doc)
    assert set(diary) == {"parsing_chunking", "embedding"}
    assert "fake_phase" not in diary
    assert len(calls["run"]) == 1

    # ---- PASS 2: the error row resumes and completes ----
    await reembed.reembed_sources(cheshire_cat, "declarative", [_stored_source(source)], stale_after=0.0)
    doc = await get_status(agent_key, "agent", source)
    assert doc["status"] == "completed"
    diary = get_completed_phases(doc)
    assert set(diary) == {"parsing_chunking", "embedding", "fake_phase"}
    assert diary["fake_phase"]["marker"] == "test-marker"
    assert len(calls["run"]) == 2


# ---------------------------------------------------------------------------
# (d) a delete-marked agent aborts without zombie rows
# ---------------------------------------------------------------------------


async def test_integration_delete_marked_agent_aborts_no_zombie_rows(cheshire_cat, monkeypatch):
    """A delete-marked agent aborts the pass: the stale completed row (whose
    ``fake_phase`` marker differs) is NOT re-run and NO ``agents:*:ingestion:*``
    row is written after cancellation (no zombie rows); the row is untouched."""
    agent_key = cheshire_cat.agent_key
    source = "canceled_registered.pdf"
    # a stale completed row: fake_phase marker differs -> the pass WOULD re-run it
    diary = _fake_diary()
    await set_status(
        agent_key, "agent", source, type_="file", status=IngestionStatus.COMPLETED,
        completed_phases=diary,
    )
    # the agent is being deleted: the marker is present -> ingestion_canceled
    await set_delete_marker(agent_key)

    calls = _stub_reembed_env(
        monkeypatch, cheshire_cat,
        existing_points=[_point(source)],
        pending_result=[{"phase": "fake_phase"}],
        run_result={"status": "done"},
        specs_result=[FAKE_SPEC],
    )

    from cat.db import crud

    writes = []
    real_store = crud.store

    async def recording_store(key, value, *args, **kwargs):
        writes.append((key, value))
        return await real_store(key, value, *args, **kwargs)

    monkeypatch.setattr(crud, "store", recording_store)

    # the pass would re-embed this source (fake_phase is stale)...
    await reembed.reembed_sources(cheshire_cat, "declarative", [_stored_source(source)], stale_after=0.0)

    # ...but the delete marker aborts the claim: NOT a single
    # agents:*:ingestion:* row is written after cancellation (no zombie rows).
    zombie = [k for k, _ in writes if k.startswith(f"agents:{agent_key}:ingestion:")]
    assert zombie == []
    assert calls["parse_calls"] == 0
    assert calls["embed_calls"] == 0
    assert len(calls["run"]) == 0

    # the pre-existing COMPLETED row is untouched (no resurrection, no phase)
    doc = await get_status(agent_key, "agent", source)
    assert doc is not None
    assert doc["status"] == "completed"
    assert "phase" not in doc
    assert get_completed_phases(doc) == diary
