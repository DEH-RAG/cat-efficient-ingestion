"""Tests for the EffING phase-machine extension hooks (``hooks.py``).

Covers the five ``@hook(priority=0)`` declarations and their conservative
default contracts:

- ``ingestion_phase_pending`` — accumulator identity (returns ``pending``
  unchanged, even when ``None``);
- ``ingestion_phase_run`` — default ``None`` (fail-hard / unimplemented);
- ``before_ingestion_status_completed`` — default no-op;
- ``ingestion_phase_settings_marker`` — default ``None`` ("unknown" ->
  conservative stale);
- ``ingestion_phase_specs`` — accumulator identity (returns ``specs``
  unchanged, even when ``None``).

Registration is asserted through the exact mechanism MadHatter uses when
loading a plugin (``getmembers(module, isinstance(obj, CatHook))`` in
``Plugin._load_decorated_functions``), plus the ``priority == 0`` contract so
external plugins can override them.

Import safety is asserted statically (AST): the module's top level contains
only the five decorated function definitions and the single
``from cat import hook`` import — no Redis, no network, no side effects.
"""

import ast
from inspect import getmembers
from pathlib import Path

from cat.looking_glass.mad_hatter.decorators.hook import CatHook

from cat.plugins.cat_efficient_ingestion import hooks
from cat.plugins.cat_efficient_ingestion.phases import PhaseSpec

EXPECTED_HOOKS = {
    "ingestion_phase_pending",
    "ingestion_phase_run",
    "before_ingestion_status_completed",
    "ingestion_phase_settings_marker",
    "ingestion_phase_specs",
}

HOOKS_FILE = Path(hooks.__file__).resolve()


def _registered_hooks():
    """Collect CatHook instances exactly like MadHatter does."""
    return {
        name: obj
        for name, obj in getmembers(hooks, lambda o: isinstance(o, CatHook))
    }


def test_all_five_phase_hooks_registered_with_priority_zero():
    """The five hook names are registered as CatHook instances, priority 0."""
    registered = _registered_hooks()
    assert EXPECTED_HOOKS <= set(registered), (
        f"missing hooks: {EXPECTED_HOOKS - set(registered)}"
    )
    for name in EXPECTED_HOOKS:
        assert registered[name].name == name
        assert registered[name].priority == 0, (
            f"{name} must be priority 0 so external plugins can override it"
        )


def test_ingestion_phase_pending_default_is_identity():
    """Default accumulator returns ``pending`` unchanged."""
    assert hooks.ingestion_phase_pending.function([], "doc.pdf", [], None) == []
    pending = [{"phase": "embedding"}]
    assert (
        hooks.ingestion_phase_pending.function(pending, "doc.pdf", [], None)
        is pending
    )


def test_ingestion_phase_pending_handles_none_pending():
    """Adversarial: ``pending=None`` is returned safely, not crashed on."""
    assert hooks.ingestion_phase_pending.function(None, "doc.pdf", [], None) is None


def test_ingestion_phase_run_default_is_none():
    """Default ``None`` = fail-hard / unimplemented."""
    assert hooks.ingestion_phase_run.function("embedding", "doc.pdf", [], None) is None


def test_before_ingestion_status_completed_default_is_noop():
    """Default gate returns ``None`` (source allowed to complete)."""
    assert hooks.before_ingestion_status_completed.function("doc.pdf", None) is None


def test_ingestion_phase_settings_marker_default_is_none():
    """Default marker ``None`` = unknown -> conservative stale."""
    assert hooks.ingestion_phase_settings_marker.function("embedding", None) is None


def test_ingestion_phase_specs_default_is_identity():
    """Default accumulator returns ``specs`` unchanged."""
    assert hooks.ingestion_phase_specs.function([], None) == []
    specs = [PhaseSpec("graph_embedding", "graphrag", ("embedding",))]
    assert hooks.ingestion_phase_specs.function(specs, None) is specs


def test_ingestion_phase_specs_handles_none_specs():
    """Adversarial: ``specs=None`` is returned safely, not crashed on."""
    assert hooks.ingestion_phase_specs.function(None, None) is None


def test_hooks_module_import_has_zero_side_effects():
    """Top level is only the 5 decorated defs + ``from cat import hook``.

    No other imports (no Redis, no network, no ``cat.db``), no assignments,
    no calls, no executable statements at import time.
    """
    tree = ast.parse(HOOKS_FILE.read_text())
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.append((node.lineno, "import " + ", ".join(a.name for a in node.names)))
        elif isinstance(node, ast.ImportFrom):
            imports.append((node.lineno, f"from {node.module} import " + ", ".join(a.name for a in node.names)))

    # exactly one import: `from cat import hook`
    assert imports == [(17, "from cat import hook")], f"unexpected imports: {imports}"

    # top-level statements: module docstring + the single import + 4 decorated
    # FunctionDefs — nothing executable (no assignments, no calls, no other
    # imports, no try/except, no class bodies).
    top_level = [node for node in tree.body]
    assert isinstance(top_level[0], ast.Expr) and isinstance(
        top_level[0].value, ast.Constant
    ), "first statement must be the module docstring"
    assert isinstance(top_level[1], ast.ImportFrom), "second statement must be the import"
    funcs = top_level[2:]
    assert all(isinstance(node, ast.FunctionDef) for node in funcs), (
        f"non-function top-level statements: {[type(n).__name__ for n in funcs]}"
    )
    assert {n.name for n in funcs} == EXPECTED_HOOKS
    # every function is decorated with @hook
    for node in funcs:
        assert node.decorator_list, f"{node.name} is not decorated"