"""Static guards on the plugin's own import graph.

Two defects this package has already hit, both of which only blow up when the
plugin is loaded by a running Cat (an ImportError at load time, or a NameError
on the first hook call) and neither of which any unit test would notice:

- a module importing the registry from the ``ingestion_status`` plugin, the
  one this package absorbed, which no longer exists;
- ``plugin.py`` calling helpers defined in a sibling module without importing
  them.

Pure ``ast`` inspection: it needs neither the ``cat`` package nor a running
Cat, so it stays a cheap regression guard on the package layout.

Standalone, import-safe: the whole body lives under ``if __name__ ==
"__main__":`` because the CAT plugin loader imports every ``.py`` under the
plugin folder. Run it from the host repository::

    cd /root/git/grinning-cat-core
    .venv/bin/python /root/git/plugins/cat-efficient-ingestion/tests/test_plugin_imports.py
"""
if __name__ == "__main__":
    import ast
    import builtins
    import sys
    import traceback
    from pathlib import Path

    def _plugin_dir() -> Path:
        """The plugin folder: walk up from this file to the folder holding plugin.json."""
        here = Path(__file__).resolve().parent
        for candidate in (here, here.parent, here.parent.parent):
            if (candidate / "plugin.json").is_file():
                return candidate
        from cat import utils  # lazy: keeps in-repo runs cat-free

        return Path(utils.__file__).resolve().parent / "core_plugins" / "cat_efficient_ingestion"

    PLUGIN_DIR = _plugin_dir()
    # The guard covers the PRODUCT import graph. Standalone ``test_*.py``
    # scripts co-located with the plugin import the canonical path from inside
    # their ``__main__`` guard (never executed by the CAT loader), so they are
    # not part of the package's import graph.
    MODULES = {
        p.stem: p for p in PLUGIN_DIR.glob("*.py") if not p.name.startswith("test_")
    }
    BUILTINS = set(dir(builtins))

    def _imports(path: Path):
        """Yield every ``(module, level, names)`` imported by a source file."""
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                yield node.module, node.level, [a.name for a in node.names]
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    yield alias.name, 0, []

    def _exported_names(path: Path) -> set:
        """Top-level names a module defines or re-exports."""
        exported = set()
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                exported.add(node.name)
            elif isinstance(node, ast.Assign):
                exported.update(t.id for t in node.targets if isinstance(t, ast.Name))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                exported.add(node.target.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                exported.update(a.asname or a.name.split(".")[0] for a in node.names)
        return exported

    def _bound_names(tree: ast.AST) -> set:
        """Every name bound anywhere in a module (assignment, def, import, arg...)."""
        bound = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                bound.add(node.id)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bound.add(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                bound.update(a.asname or a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.arg):
                bound.add(node.arg)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                bound.add(node.name)
            elif isinstance(node, (ast.Global, ast.Nonlocal)):
                bound.update(node.names)
        return bound

    def _loaded_names(tree: ast.AST) -> set:
        """Every name read anywhere in a module."""
        return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}

    def test_no_import_from_the_former_ingestion_status_plugin():
        """No module imports from the absorbed ``ingestion_status`` plugin (it is gone)."""
        gone = {"ingestion_status", "cat_ingestion_status"}
        offenders = [
            f"{path.name}: {module}"
            for path in sorted(MODULES.values())
            for module, level, _ in _imports(path)
            if level == 0 and module and gone & set(module.split("."))
        ]
        assert offenders == [], f"stale imports of the removed plugin: {offenders}"

    def test_intra_package_imports_are_relative_and_resolvable():
        """Sibling modules are imported relatively, and every target exists."""
        missing = []
        absolute_self = []
        for path in sorted(MODULES.values()):
            for module, level, names in _imports(path):
                if level == 0:
                    if not module or "." not in module:
                        continue
                    sibling = module.rsplit(".", 1)[1]
                    if sibling in MODULES and sibling != path.stem:
                        if set(names) <= _exported_names(MODULES[sibling]):
                            absolute_self.append(f"{path.name}: {module}")
                    continue
                if module is None:
                    # ``from . import sibling`` (lazy, in-function import)
                    missing.extend(
                        f"{path.name}: .{name}" for name in names if name not in MODULES
                    )
                    continue
                if module not in MODULES:
                    missing.append(f"{path.name}: .{module}")
                    continue
                exported = _exported_names(MODULES[module])
                missing.extend(
                    f"{path.name}: .{module}.{name}" for name in names if name not in exported
                )
        assert absolute_self == [], f"intra-package imports must be relative: {absolute_self}"
        assert missing == [], f"unresolvable relative imports: {missing}"

    def test_no_sibling_helper_used_without_importing_it():
        """A name defined in a sibling module and used here must be imported."""
        offenders = []
        trees = {name: ast.parse(path.read_text()) for name, path in MODULES.items()}
        owners = {}
        for name, path in MODULES.items():
            for exported in _exported_names(path):
                owners.setdefault(exported, set()).add(name)
        for name, tree in trees.items():
            for used in sorted(_loaded_names(tree) - _bound_names(tree) - BUILTINS):
                defined_in = sorted(owners.get(used, set()) - {name})
                if defined_in:
                    offenders.append(
                        f"{name}.py: uses {used} (defined in {defined_in}) without importing it"
                    )
        assert offenders == [], f"missing sibling imports: {offenders}"

    _tests = [
        test_no_import_from_the_former_ingestion_status_plugin,
        test_intra_package_imports_are_relative_and_resolvable,
        test_no_sibling_helper_used_without_importing_it,
    ]

    _passed = 0
    _failures = []
    for _test in _tests:
        try:
            _test()
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
