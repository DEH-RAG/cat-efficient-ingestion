"""Tests for the EffING phase DAG and clock-free two-level staleness.

Covers ``phases.py``:

- the ``PHASES`` DAG shape (``parsing_chunking`` -> ``embedding``);
- the pure ``is_phase_stale`` decision: no diary -> stale; settings
  fast-path by opaque version equality (the no-op-save guard); slow-path by
  plugin marker; dependency by upstream marker identity;
- the ``ingestion_phase_pending`` accumulator registrant, driven with a fake
  ``cat`` (agent_key + plugin_manager) and a stubbed settings CRUD — no
  Redis needed;
- the phase-registration merge: ``merged_phases`` appends specs declared via
  the ``ingestion_phase_specs`` accumulator hook (built-ins win on duplicate
  ids, malformed entries skipped) and the registrant probes ALL merged
  phases (fresh when markers match, stale when they differ, stale on a
  missing upstream);
- adversarial input: ``pending=None``, ``completed_phases=None`` / non-dict
  entries never crash.

The hook contract is the MyGRAPH-compatible one: ``completed_phases`` is a
``list[dict]`` where every entry carries at least a ``"phase"`` key; extra
keys (e.g. MyGRAPH's ``"gen"``) are ignored.
"""
from unittest.mock import AsyncMock

from cat.plugins.cat_efficient_ingestion import phases
from cat.plugins.cat_efficient_ingestion.phases import (
    PHASES,
    PhaseSpec,
    _diary_from_completed_phases,
    is_phase_stale,
)


def _entry(marker="m1", settings_version="v1", deps=None):
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


# ---------------------------------------------------------------------------
# Phase DAG shape
# ---------------------------------------------------------------------------


def test_phase_dag_shape():
    assert set(PHASES) == {"parsing_chunking", "embedding"}
    assert PHASES["parsing_chunking"] == PhaseSpec("parsing_chunking", "chunker", ())
    assert PHASES["embedding"] == PhaseSpec(
        "embedding", "embedder", ("parsing_chunking",)
    )


# ---------------------------------------------------------------------------
# Pure staleness: is_phase_stale
# ---------------------------------------------------------------------------


def test_no_diary_both_phases_stale():
    """(a) No diary -> every phase is stale (never ran)."""
    assert is_phase_stale(PHASES["parsing_chunking"], {}, None, "pc1") is True
    assert is_phase_stale(PHASES["embedding"], {}, None, "em1") is True


def test_embedding_fresh_when_version_changed_but_marker_equal():
    """(b) Embedder settings re-saved (version bumped) but marker equal ->
    embedding FRESH — the no-op-save guard."""
    diary = _fresh_diary()
    assert is_phase_stale(PHASES["embedding"], diary, "v2", "em1") is False


def test_embedding_stale_when_marker_differs():
    """(c) Version changed AND marker differs -> embedding stale."""
    diary = _fresh_diary()
    assert is_phase_stale(PHASES["embedding"], diary, "v2", "em2") is True


def test_embedding_stale_when_parsing_marker_changed():
    """(d) Upstream (parsing) marker changed -> embedding stale via dep,
    even with identical embedder settings."""
    diary = _fresh_diary()
    diary["parsing_chunking"] = _entry(marker="pc2", settings_version="v1")
    assert is_phase_stale(PHASES["embedding"], diary, "v1", "em1") is True


def test_identical_settings_resave_nothing_stale():
    """(e) Identical settings re-save (versions bumped, markers equal) ->
    nothing stale."""
    diary = _fresh_diary()
    assert is_phase_stale(PHASES["parsing_chunking"], diary, "v2", "pc1") is False
    assert is_phase_stale(PHASES["embedding"], diary, "v2", "em1") is False


def test_missing_upstream_marker_downstream_stale():
    """(f) Recorded deps lack the upstream marker -> downstream stale."""
    diary = _fresh_diary()
    diary["embedding"] = _entry(marker="em1", settings_version="v1", deps={})
    assert is_phase_stale(PHASES["embedding"], diary, "v1", "em1") is True


def test_missing_upstream_entry_downstream_stale():
    """Upstream entry absent from the diary -> downstream stale."""
    diary = {"embedding": _entry(marker="em1", settings_version="v1",
                                 deps={"parsing_chunking": "pc1"})}
    assert is_phase_stale(PHASES["embedding"], diary, "v1", "em1") is True


def test_missing_settings_entry_falls_to_slow_path_stale():
    """No current settings entry (updated_at None) -> fast-path skipped ->
    slow-path with unknown marker -> conservative stale."""
    diary = _fresh_diary()
    assert is_phase_stale(PHASES["embedding"], diary, None, None) is True


def test_parsing_fresh_when_version_equal():
    """Fast-path alone: parsing fresh when its settings version is unchanged."""
    diary = _fresh_diary()
    assert is_phase_stale(PHASES["parsing_chunking"], diary, "v1", "pc1") is False


def test_malformed_entries_never_crash():
    """Adversarial: non-dict / partial entries are stale, never raise."""
    assert is_phase_stale(PHASES["embedding"], {"embedding": "junk"}, "v1", "m1") is True
    assert is_phase_stale(
        PHASES["embedding"], {"embedding": {"marker": "m1"}}, "v1", "m1"
    ) is True  # no deps key -> upstream missing -> stale


# ---------------------------------------------------------------------------
# Diary normalization
# ---------------------------------------------------------------------------


def test_diary_conversion_from_list_and_malformed():
    assert _diary_from_completed_phases(None) == {}
    assert _diary_from_completed_phases("junk") == {}
    assert _diary_from_completed_phases([42, None, "x"]) == {}
    converted = _diary_from_completed_phases(
        [{"phase": "embedding", "gen": "g1"}, "junk", None]
    )
    assert converted == {"embedding": {"phase": "embedding", "gen": "g1"}}
    # MyGRAPH-style entries without marker/version are kept whole (extra keys ignored)
    assert _diary_from_completed_phases(
        [{"phase": "parsing_chunking", "gen": "g0"}]
    ) == {"parsing_chunking": {"phase": "parsing_chunking", "gen": "g0"}}
    # status-doc shape and raw diary dict are accepted too
    assert _diary_from_completed_phases(
        {"completed_phases": {"embedding": {"marker": "m"}}}
    ) == {"embedding": {"marker": "m"}}
    assert _diary_from_completed_phases(
        {"embedding": {"marker": "m"}, "junk": "not-a-dict"}
    ) == {"embedding": {"marker": "m"}}


# ---------------------------------------------------------------------------
# Accumulator hook: ingestion_phase_pending
# ---------------------------------------------------------------------------


class _FakePluginManager:
    """Minimal plugin_manager: execute_hook returns the configured markers
    and the configured registered phase specs."""

    def __init__(self, markers, specs=None):
        self.markers = markers
        self.specs = list(specs or [])
        self.calls = []

    async def execute_hook(self, hook_name, *args, **kwargs):
        self.calls.append((hook_name, args, kwargs))
        if hook_name == "ingestion_phase_settings_marker":
            return self.markers.get(args[0])
        if hook_name == "ingestion_phase_specs":
            return list(self.specs)
        return None


class _FakeCat:
    """Minimal CheshireCat stand-in: agent_key + plugin_manager."""

    def __init__(self, markers=None, agent_key="agent_test", specs=None):
        self.agent_key = agent_key
        self.plugin_manager = _FakePluginManager(markers or {}, specs)


def _stub_settings(monkeypatch, settings):
    """Point phases.crud_settings.get_settings_by_category at a fake store."""
    async def fake_get(key_id, category):
        return settings.get(category)

    monkeypatch.setattr(phases.crud_settings, "get_settings_by_category", fake_get)


def _diary_list(diary):
    """Convert a diary dict to the hook's list[dict] shape."""
    return [{"phase": phase, **entry} for phase, entry in diary.items()]


async def test_hook_no_diary_both_phases_pending(monkeypatch):
    """(a) No diary -> both built-in phases are appended."""
    _stub_settings(monkeypatch, {"chunker": {"updated_at": "v1"},
                                 "embedder": {"updated_at": "v1"}})
    cat = _FakeCat(markers={"parsing_chunking": "pc1", "embedding": "em1"})
    pending = await phases.ingestion_phase_pending.function([], "doc.pdf", [], cat)
    assert {p["phase"] for p in pending} == {"parsing_chunking", "embedding"}


async def test_hook_noop_save_guard_nothing_pending(monkeypatch):
    """(b)+(e) Identical settings re-save (versions bumped, markers equal) ->
    nothing pending."""
    _stub_settings(monkeypatch, {"chunker": {"updated_at": "v2"},
                                 "embedder": {"updated_at": "v2"}})
    cat = _FakeCat(markers={"parsing_chunking": "pc1", "embedding": "em1"})
    pending = await phases.ingestion_phase_pending.function(
        [], "doc.pdf", _diary_list(_fresh_diary()), cat
    )
    assert pending == []


async def test_hook_marker_differs_embedding_pending(monkeypatch):
    """(c) Embedder marker differs -> only embedding is pending."""
    _stub_settings(monkeypatch, {"chunker": {"updated_at": "v2"},
                                 "embedder": {"updated_at": "v2"}})
    cat = _FakeCat(markers={"parsing_chunking": "pc1", "embedding": "em2"})
    pending = await phases.ingestion_phase_pending.function(
        [], "doc.pdf", _diary_list(_fresh_diary()), cat
    )
    assert [p["phase"] for p in pending] == ["embedding"]


async def test_hook_parsing_marker_changed_cascades_to_embedding(monkeypatch):
    """(d) Chunker settings rewritten (version bumped, new marker) -> parsing
    pending; embedding follows on the NEXT probe once parsing re-records its
    new marker in the diary (dep check is against the diary, not the hook)."""
    _stub_settings(monkeypatch, {"chunker": {"updated_at": "v2"},
                                 "embedder": {"updated_at": "v1"}})
    cat = _FakeCat(markers={"parsing_chunking": "pc2", "embedding": "em1"})

    # probe 1: parsing stale (version bumped + marker differs); embedding's
    # dep still matches the OLD diary marker -> fresh this round
    pending = await phases.ingestion_phase_pending.function(
        [], "doc.pdf", _diary_list(_fresh_diary()), cat
    )
    assert [p["phase"] for p in pending] == ["parsing_chunking"]

    # probe 2: parsing re-ran and recorded pc2 -> embedding stale via dep
    diary = _fresh_diary()
    diary["parsing_chunking"] = _entry(marker="pc2", settings_version="v2")
    pending = await phases.ingestion_phase_pending.function(
        [], "doc.pdf", _diary_list(diary), cat
    )
    assert [p["phase"] for p in pending] == ["embedding"]


async def test_hook_missing_upstream_marker_embedding_pending(monkeypatch):
    """(f) Recorded deps lack the upstream marker -> embedding pending."""
    _stub_settings(monkeypatch, {"chunker": {"updated_at": "v1"},
                                 "embedder": {"updated_at": "v1"}})
    diary = _fresh_diary()
    diary["embedding"] = _entry(marker="em1", settings_version="v1", deps={})
    cat = _FakeCat(markers={"parsing_chunking": "pc1", "embedding": "em1"})
    pending = await phases.ingestion_phase_pending.function(
        [], "doc.pdf", _diary_list(diary), cat
    )
    assert [p["phase"] for p in pending] == ["embedding"]


async def test_hook_missing_settings_entry_conservative_stale(monkeypatch):
    """No settings entry for a category -> fast-path skipped -> the marker
    hook cannot compute a marker (None) -> conservative stale."""
    _stub_settings(monkeypatch, {"chunker": {"updated_at": "v1"}})  # embedder missing
    cat = _FakeCat(markers={"parsing_chunking": "pc1", "embedding": None})
    pending = await phases.ingestion_phase_pending.function(
        [], "doc.pdf", _diary_list(_fresh_diary()), cat
    )
    assert [p["phase"] for p in pending] == ["embedding"]


async def test_hook_pending_none_no_crash(monkeypatch):
    """Adversarial: pending=None is normalized to a list, never crashed on."""
    _stub_settings(monkeypatch, {"chunker": {"updated_at": "v1"},
                                 "embedder": {"updated_at": "v1"}})
    cat = _FakeCat(markers={"parsing_chunking": "pc1", "embedding": "em1"})
    pending = await phases.ingestion_phase_pending.function(None, "doc.pdf", [], cat)
    assert isinstance(pending, list)
    assert {p["phase"] for p in pending} == {"parsing_chunking", "embedding"}


async def test_hook_malformed_completed_phases_no_crash(monkeypatch):
    """Adversarial: completed_phases=None / non-dict entries -> no crash."""
    _stub_settings(monkeypatch, {"chunker": {"updated_at": "v1"},
                                 "embedder": {"updated_at": "v1"}})
    cat = _FakeCat(markers={"parsing_chunking": "pc1", "embedding": "em1"})
    for malformed in (None, "junk", [42, None, "x"], [{"phase": "embedding"}]):
        pending = await phases.ingestion_phase_pending.function(
            [], "doc.pdf", malformed, cat
        )
        assert isinstance(pending, list)


async def test_hook_cat_none_no_crash():
    """Adversarial: cat=None (unit-test contract) -> diary alone decides."""
    pending = await phases.ingestion_phase_pending.function([], "doc.pdf", [], None)
    assert {p["phase"] for p in pending} == {"parsing_chunking", "embedding"}


async def test_hook_uses_settings_crud_and_marker_hook(monkeypatch):
    """The hook reads the current updated_at via crud_settings and the marker
    via execute_hook("ingestion_phase_settings_marker", phase, caller=cat)."""
    _stub_settings(monkeypatch, {"chunker": {"updated_at": "v1"},
                                 "embedder": {"updated_at": "v1"}})
    cat = _FakeCat(markers={"parsing_chunking": "pc1", "embedding": "em1"})
    await phases.ingestion_phase_pending.function(
        [], "doc.pdf", _diary_list(_fresh_diary()), cat
    )
    marker_calls = [c for c in cat.plugin_manager.calls
                    if c[0] == "ingestion_phase_settings_marker"]
    assert {c[1][0] for c in marker_calls} == {"parsing_chunking", "embedding"}
    assert all(c[2].get("caller") is cat for c in marker_calls)


# ---------------------------------------------------------------------------
# Phase registration: merged_phases + pending over ALL merged phases
# ---------------------------------------------------------------------------

FAKE_SPEC = PhaseSpec("fake_phase", None, ("parsing_chunking", "embedding"))


def _fake_diary():
    """A fresh diary including a completed fake_phase with matching deps."""
    diary = _fresh_diary()
    diary["fake_phase"] = _entry(
        marker="fp1", settings_version=None,
        deps={"parsing_chunking": "pc1", "embedding": "em1"},
    )
    return diary


async def test_merged_phases_merges_registered_and_builtins_win():
    """merged_phases: registered specs appended; a duplicate built-in id is
    IGNORED (built-in wins); PHASES itself is never mutated."""
    cat = _FakeCat(specs=[
        FAKE_SPEC,
        PhaseSpec("embedding", None, ()),  # duplicate of a built-in
    ])
    merged = await phases.merged_phases(cat, cat)
    assert set(merged) == {"parsing_chunking", "embedding", "fake_phase"}
    assert merged["embedding"] is PHASES["embedding"]
    assert merged["fake_phase"] == FAKE_SPEC
    assert set(PHASES) == {"parsing_chunking", "embedding"}  # untouched


async def test_merged_phases_malformed_specs_no_crash():
    """Adversarial: execute_hook returns None / non-PhaseSpec entries ->
    skipped, never raised on."""
    cat = _FakeCat(specs=[None, "junk", 42, FAKE_SPEC])
    merged = await phases.merged_phases(cat, cat)
    assert set(merged) == {"parsing_chunking", "embedding", "fake_phase"}


async def test_merged_phases_cat_none_builtins_only():
    """Adversarial: cat=None (unit-test contract) -> built-ins only."""
    assert await phases.merged_phases(None, None) == PHASES


async def test_hook_registered_phase_fresh_when_markers_match(monkeypatch):
    """(a) fake_phase declared via ingestion_phase_specs, fresh diary with
    matching markers -> NOT stale (nothing pending)."""
    _stub_settings(monkeypatch, {"chunker": {"updated_at": "v1"},
                                 "embedder": {"updated_at": "v1"}})
    cat = _FakeCat(
        markers={"parsing_chunking": "pc1", "embedding": "em1",
                 "fake_phase": "fp1"},
        specs=[FAKE_SPEC],
    )
    pending = await phases.ingestion_phase_pending.function(
        [], "doc.pdf", _diary_list(_fake_diary()), cat
    )
    assert pending == []


async def test_hook_registered_phase_stale_when_marker_differs(monkeypatch):
    """(b) fake_phase marker differs (via the fake
    ingestion_phase_settings_marker) -> stale."""
    _stub_settings(monkeypatch, {"chunker": {"updated_at": "v1"},
                                 "embedder": {"updated_at": "v1"}})
    cat = _FakeCat(
        markers={"parsing_chunking": "pc1", "embedding": "em1",
                 "fake_phase": "fp2"},
        specs=[FAKE_SPEC],
    )
    pending = await phases.ingestion_phase_pending.function(
        [], "doc.pdf", _diary_list(_fake_diary()), cat
    )
    assert [p["phase"] for p in pending] == ["fake_phase"]


async def test_hook_registered_phase_appended_after_builtins(monkeypatch):
    """(b) no diary -> built-ins are appended FIRST, the registered phase
    AFTER them."""
    _stub_settings(monkeypatch, {"chunker": {"updated_at": "v1"},
                                 "embedder": {"updated_at": "v1"}})
    cat = _FakeCat(
        markers={"parsing_chunking": "pc1", "embedding": "em1",
                 "fake_phase": "fp1"},
        specs=[FAKE_SPEC],
    )
    pending = await phases.ingestion_phase_pending.function([], "doc.pdf", [], cat)
    assert [p["phase"] for p in pending] == [
        "parsing_chunking", "embedding", "fake_phase",
    ]


async def test_hook_registered_phase_missing_upstream_stale(monkeypatch):
    """(c) fake_phase's upstreams missing from the diary -> stale."""
    _stub_settings(monkeypatch, {"chunker": {"updated_at": "v1"},
                                 "embedder": {"updated_at": "v1"}})
    cat = _FakeCat(
        markers={"parsing_chunking": "pc1", "embedding": "em1",
                 "fake_phase": "fp1"},
        specs=[FAKE_SPEC],
    )
    diary = {"fake_phase": _entry(
        marker="fp1", settings_version=None,
        deps={"parsing_chunking": "pc1", "embedding": "em1"},
    )}
    pending = await phases.ingestion_phase_pending.function(
        [], "doc.pdf", _diary_list(diary), cat
    )
    assert [p["phase"] for p in pending] == [
        "parsing_chunking", "embedding", "fake_phase",
    ]


async def test_hook_registered_duplicate_builtin_id_ignored(monkeypatch):
    """(d) a registered spec duplicating a built-in id is IGNORED: the
    built-in spec wins, so its settings fast-path still applies and a changed
    marker for the duplicate does NOT make embedding stale."""
    _stub_settings(monkeypatch, {"chunker": {"updated_at": "v1"},
                                 "embedder": {"updated_at": "v1"}})
    cat = _FakeCat(
        markers={"parsing_chunking": "pc1", "embedding": "em2"},
        specs=[PhaseSpec("embedding", None, ())],  # duplicate, no fast-path
    )
    pending = await phases.ingestion_phase_pending.function(
        [], "doc.pdf", _diary_list(_fresh_diary()), cat
    )
    assert pending == []