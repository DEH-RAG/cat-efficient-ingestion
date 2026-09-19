"""Tests for the EffING phase-registration API end-to-end (Todo 4).

Covers the full registration surface with a FAKE plugin declared through the
``ingestion_phase_specs`` accumulator hook:

Section 1 — ``merged_phases`` (phases.py):
  (a) a fake ``ingestion_phase_specs`` registrant returning
      ``[PhaseSpec("fake_phase", None, ("parsing_chunking", "embedding"))]``
      -> merged = built-ins + ``fake_phase``;
  (b) a registered spec with a duplicate built-in id is IGNORED (built-in
      wins, ``PHASES`` itself is never mutated);
  (c) malformed hook output (None / non-PhaseSpec entries) is skipped without
      crash;
  (d) ``ccat=None`` -> built-ins only.

Section 2 — ``ingestion_phase_pending`` over the registered phases:
  (e) ``fake_phase`` with a matching marker (via a fake
      ``ingestion_phase_settings_marker``) -> NOT stale;
  (f) ``fake_phase`` with a differing marker -> stale, appended AFTER the
      built-in phases;
  (g) ``fake_phase`` with a missing upstream -> stale.

Section 3 — dispatcher end-to-end (``reembed_sources`` + ``_stub_reembed_env``):
  (h) a fake ``ingestion_phase_specs`` registrant declares ``fake_phase``; the
      dispatcher runs it via ``ingestion_phase_run`` (``{"status": "done"}``)
      and records a REAL diary entry (marker from
      ``ingestion_phase_settings_marker`` + deps referencing the upstream
      markers);
  (i) a registered phase whose ``ingestion_phase_run`` returns None -> row
      ERROR (fail-hard, no silent success);
  (j) a registered phase with a missing upstream is reported stale and re-run
      (the re-run repairs the deps against the CURRENT diary).

The harness is the one from ``test_efficient_ingestion_phase_machine.py``:
the real ``cheshire_cat`` fixture, ``_stub_reembed_env``, ``_completed_diary``,
``_stored_source`` and ``_point`` helpers (self-contained on purpose).
"""
from typing import Any
from unittest.mock import AsyncMock

from cat.plugins.cat_efficient_ingestion import phases, reembed
from cat.plugins.cat_efficient_ingestion.phases import PHASES, PhaseSpec
from cat.plugins.cat_efficient_ingestion.registry import (
    IngestionStatus,
    get_completed_phases,
    get_status,
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


def _completed_diary(parsing_marker="pc1", embedding_marker="em1"):
    """A completed row's diary: parsing recorded, embedding depends on it."""
    return {
        "parsing_chunking": {"marker": parsing_marker, "settings_version": "v1", "deps": {}},
        "embedding": {
            "marker": embedding_marker, "settings_version": "v1",
            "deps": {"parsing_chunking": parsing_marker},
        },
    }


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


def _diary_list(diary):
    """Convert a diary dict to the hook's list[dict] shape."""
    return [{"phase": phase, **entry} for phase, entry in diary.items()]


def _stub_settings(monkeypatch, settings):
    """Point phases.crud_settings.get_settings_by_category at a fake store."""
    async def fake_get(key_id, category):
        return settings.get(category)

    monkeypatch.setattr(phases.crud_settings, "get_settings_by_category", fake_get)


def _stub_registration_hooks(monkeypatch, cheshire_cat, markers, specs):
    """Point the real plugin_manager's execute_hook at a FAKE registrant.

    ``ingestion_phase_specs`` returns ``specs`` (the fake plugin's declared
    phases); ``ingestion_phase_settings_marker`` returns ``markers[phase]``.
    Everything else returns None (never called by the registrant).
    """
    async def fake_execute_hook(name, *args, **kwargs):
        if name == "ingestion_phase_specs":
            return list(specs)
        if name == "ingestion_phase_settings_marker":
            return markers.get(args[0])
        return None

    monkeypatch.setattr(cheshire_cat.plugin_manager, "execute_hook", fake_execute_hook)


# ---------------------------------------------------------------------------
# Section 1: merged_phases (registration merge)
# ---------------------------------------------------------------------------


async def test_merged_phases_appends_registered_spec(cheshire_cat, monkeypatch):
    """(a) A fake ``ingestion_phase_specs`` registrant returning
    ``[PhaseSpec("fake_phase", None, ("parsing_chunking", "embedding"))]`` ->
    merged = built-ins + ``fake_phase``."""
    monkeypatch.setattr(
        cheshire_cat.plugin_manager, "execute_hook",
        AsyncMock(return_value=[FAKE_SPEC]),
    )
    merged = await phases.merged_phases(cheshire_cat, cheshire_cat)
    assert set(merged) == {"parsing_chunking", "embedding", "fake_phase"}
    assert merged["fake_phase"] == FAKE_SPEC
    # built-ins are the SAME objects, never replaced
    assert merged["parsing_chunking"] is PHASES["parsing_chunking"]
    assert merged["embedding"] is PHASES["embedding"]


async def test_merged_phases_duplicate_builtin_id_ignored(cheshire_cat, monkeypatch):
    """(b) A registered spec with a duplicate built-in id is IGNORED: the
    built-in wins and ``PHASES`` itself is never mutated."""
    monkeypatch.setattr(
        cheshire_cat.plugin_manager, "execute_hook",
        AsyncMock(return_value=[FAKE_SPEC, PhaseSpec("embedding", None, ())]),
    )
    merged = await phases.merged_phases(cheshire_cat, cheshire_cat)
    assert set(merged) == {"parsing_chunking", "embedding", "fake_phase"}
    assert merged["embedding"] is PHASES["embedding"]  # built-in wins
    assert set(PHASES) == {"parsing_chunking", "embedding"}  # never mutated


async def test_merged_phases_malformed_output_skipped(cheshire_cat, monkeypatch):
    """(c) Malformed hook output (None / non-PhaseSpec entries) is skipped
    without crash; valid entries are still appended."""
    for malformed in (None, "junk", [None, "junk", 42]):
        monkeypatch.setattr(
            cheshire_cat.plugin_manager, "execute_hook",
            AsyncMock(return_value=malformed),
        )
        merged = await phases.merged_phases(cheshire_cat, cheshire_cat)
        assert set(merged) == {"parsing_chunking", "embedding"}
    # mixed: malformed entries skipped, the valid one appended
    monkeypatch.setattr(
        cheshire_cat.plugin_manager, "execute_hook",
        AsyncMock(return_value=[None, "junk", FAKE_SPEC]),
    )
    merged = await phases.merged_phases(cheshire_cat, cheshire_cat)
    assert set(merged) == {"parsing_chunking", "embedding", "fake_phase"}


async def test_merged_phases_ccat_none_builtins_only():
    """(d) ``ccat=None`` (unit-test contract) -> built-ins only, fresh dict."""
    merged = await phases.merged_phases(None, None)
    assert merged == PHASES
    assert merged is not PHASES


# ---------------------------------------------------------------------------
# Section 2: ingestion_phase_pending over the registered phases
# ---------------------------------------------------------------------------


async def test_pending_registered_phase_fresh_when_marker_matches(cheshire_cat, monkeypatch):
    """(e) ``fake_phase`` with a matching marker (via a fake
    ``ingestion_phase_settings_marker``) -> NOT stale: nothing pending."""
    _stub_settings(monkeypatch, {"chunker": {"updated_at": "v1"},
                                 "embedder": {"updated_at": "v1"}})
    _stub_registration_hooks(
        monkeypatch, cheshire_cat,
        markers={"parsing_chunking": "pc1", "embedding": "em1", "fake_phase": "fp1"},
        specs=[FAKE_SPEC],
    )
    pending = await phases.ingestion_phase_pending.function(
        [], "doc.pdf", _diary_list(_fake_diary()), cheshire_cat
    )
    assert pending == []


async def test_pending_registered_phase_stale_when_marker_differs(cheshire_cat, monkeypatch):
    """(f) ``fake_phase`` with a differing marker -> stale (built-ins stay
    fresh via the settings fast-path)."""
    _stub_settings(monkeypatch, {"chunker": {"updated_at": "v1"},
                                 "embedder": {"updated_at": "v1"}})
    _stub_registration_hooks(
        monkeypatch, cheshire_cat,
        markers={"parsing_chunking": "pc1", "embedding": "em1", "fake_phase": "fp2"},
        specs=[FAKE_SPEC],
    )
    pending = await phases.ingestion_phase_pending.function(
        [], "doc.pdf", _diary_list(_fake_diary()), cheshire_cat
    )
    assert [p["phase"] for p in pending] == ["fake_phase"]


async def test_pending_registered_phase_appended_after_builtins(cheshire_cat, monkeypatch):
    """(f) No diary -> every merged phase is stale; the registered phase is
    appended AFTER the built-in phases."""
    _stub_settings(monkeypatch, {"chunker": {"updated_at": "v1"},
                                 "embedder": {"updated_at": "v1"}})
    _stub_registration_hooks(
        monkeypatch, cheshire_cat,
        markers={"parsing_chunking": "pc1", "embedding": "em1", "fake_phase": "fp1"},
        specs=[FAKE_SPEC],
    )
    pending = await phases.ingestion_phase_pending.function(
        [], "doc.pdf", [], cheshire_cat
    )
    assert [p["phase"] for p in pending] == [
        "parsing_chunking", "embedding", "fake_phase",
    ]


async def test_pending_registered_phase_stale_when_upstream_missing(cheshire_cat, monkeypatch):
    """(g) ``fake_phase`` with a missing upstream (its deps reference upstreams
    absent from the diary) -> stale."""
    _stub_settings(monkeypatch, {"chunker": {"updated_at": "v1"},
                                 "embedder": {"updated_at": "v1"}})
    _stub_registration_hooks(
        monkeypatch, cheshire_cat,
        markers={"parsing_chunking": "pc1", "embedding": "em1", "fake_phase": "fp1"},
        specs=[FAKE_SPEC],
    )
    diary = {"fake_phase": _entry(
        marker="fp1", settings_version=None,
        deps={"parsing_chunking": "pc1", "embedding": "em1"},
    )}
    pending = await phases.ingestion_phase_pending.function(
        [], "doc.pdf", _diary_list(diary), cheshire_cat
    )
    assert [p["phase"] for p in pending] == [
        "parsing_chunking", "embedding", "fake_phase",
    ]


# ---------------------------------------------------------------------------
# Section 3: dispatcher end-to-end (reembed_sources + _stub_reembed_env)
# ---------------------------------------------------------------------------


async def test_dispatcher_registered_phase_records_real_marker_and_deps(cheshire_cat, monkeypatch):
    """(h) A fake ``ingestion_phase_specs`` registrant declares ``fake_phase``;
    the dispatcher runs it via ``ingestion_phase_run`` (``{"status": "done"}``)
    and records a REAL diary entry: marker from
    ``ingestion_phase_settings_marker`` + deps referencing the upstream
    markers from the CURRENT diary."""
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
        specs_result=[FAKE_SPEC],
    )
    await reembed.reembed_sources(cheshire_cat, "declarative", [_stored_source(source)])

    doc = await get_status(agent_key, "agent", source)
    assert doc is not None
    assert doc["status"] == "completed"
    assert "phase" not in doc  # terminal write cleared the work phase

    diary = get_completed_phases(doc)
    assert set(diary) == {"parsing_chunking", "embedding", "fake_phase"}
    entry = diary["fake_phase"]
    # REAL marker from the settings-marker hook, NOT the None fallback
    assert entry["marker"] == "test-marker"
    # settings_category=None -> no settings fast-path token
    assert entry["settings_version"] is None
    # deps reference the upstream markers from the CURRENT diary
    assert entry["deps"] == {"parsing_chunking": "pc1", "embedding": "em1"}
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


async def test_dispatcher_registered_phase_run_none_is_error(cheshire_cat, monkeypatch):
    """(i) A registered phase whose ``ingestion_phase_run`` returns None ->
    row ERROR (fail-hard, no silent success): the phase is NOT recorded and
    the row is not resurrected."""
    agent_key = cheshire_cat.agent_key
    source = "registered_none.pdf"
    await set_status(
        agent_key, "agent", source, type_="file", status=IngestionStatus.COMPLETED,
        completed_phases=_completed_diary(parsing_marker="pc1", embedding_marker="em1"),
    )

    # run_result defaults to None: the fake registrant declares the phase but
    # nobody implements the run hook
    calls = _stub_reembed_env(
        monkeypatch, cheshire_cat,
        existing_points=[_point(source)],
        pending_result=[{"phase": "fake_phase"}],
        specs_result=[FAKE_SPEC],
    )
    await reembed.reembed_sources(cheshire_cat, "declarative", [_stored_source(source)])

    doc = await get_status(agent_key, "agent", source)
    assert doc is not None
    assert doc["status"] == "error"
    assert "expected a status dict" in doc["error"]
    # misleading-success guard: None is NEVER treated as success
    assert "fake_phase" not in get_completed_phases(doc)
    assert len(calls["run"]) == 1


async def test_dispatcher_registered_phase_missing_upstream_rerun(cheshire_cat, monkeypatch):
    """(j) A registered phase with a missing upstream is reported stale and
    re-run: the recorded deps LACK the embedding upstream marker, the
    dispatcher re-runs ``fake_phase`` and the re-run repairs the deps against
    the CURRENT diary."""
    agent_key = cheshire_cat.agent_key
    source = "registered_missing_upstream.pdf"
    diary = _completed_diary(parsing_marker="pc1", embedding_marker="em1")
    diary["fake_phase"] = _entry(
        marker="fp1", settings_version=None,
        deps={"parsing_chunking": "pc1"},  # embedding dep missing -> stale
    )
    await set_status(
        agent_key, "agent", source, type_="file", status=IngestionStatus.COMPLETED,
        completed_phases=diary,
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

    entry = get_completed_phases(doc)["fake_phase"]
    assert entry["marker"] == "test-marker"  # re-run recorded the current marker
    # the re-run repaired the deps against the CURRENT diary
    assert entry["deps"] == {"parsing_chunking": "pc1", "embedding": "em1"}
    assert len(calls["run"]) == 1  # re-run exactly once
    # decision probe + re-probe after recording (now empty: marker matches)
    assert len(calls["probe"]) == 2