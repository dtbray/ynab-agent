import ast
from collections.abc import Iterable, Mapping
from pathlib import Path


SOURCE_ROOT = Path(__file__).parents[1] / "src" / "ynab_agent"

FORBIDDEN_IMPORTS: Mapping[str, tuple[str, ...]] = {
    "planning": (
        "fastapi",
        "rich",
        "sqlalchemy",
        "typer",
        "ynab_agent.api",
        "ynab_agent.cli",
        "ynab_agent.cli_commands",
        "ynab_agent.cli_support",
        "ynab_agent.config",
        "ynab_agent.db",
        "ynab_agent.http_api",
        "ynab_agent.runtime",
        "ynab_agent.services",
    ),
    "services": (
        "fastapi",
        "rich",
        "sqlalchemy",
        "typer",
        "ynab_agent.api",
        "ynab_agent.cli",
        "ynab_agent.cli_commands",
        "ynab_agent.cli_support",
        "ynab_agent.config",
        "ynab_agent.db",
        "ynab_agent.http_api",
        "ynab_agent.runtime",
    ),
    "cli_commands": ("ynab_agent.http_api",),
    "http_api": (
        "ynab_agent.cli",
        "ynab_agent.cli_commands",
        "ynab_agent.cli_support",
    ),
}


def _imported_modules(path: Path) -> Iterable[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            yield node.module


def _find_forbidden_imports(
    source_root: Path,
    rules: Mapping[str, tuple[str, ...]],
) -> list[str]:
    violations: list[str] = []
    for package, forbidden_prefixes in rules.items():
        package_root = source_root / package
        if not package_root.exists():
            continue
        for path in package_root.rglob("*.py"):
            for module in _imported_modules(path):
                if any(
                    module == prefix or module.startswith(f"{prefix}.")
                    for prefix in forbidden_prefixes
                ):
                    violations.append(
                        f"{path.relative_to(source_root)} imports {module}"
                    )
    return violations


def test_internal_dependency_direction() -> None:
    assert _find_forbidden_imports(SOURCE_ROOT, FORBIDDEN_IMPORTS) == []


def test_architecture_check_detects_recursive_internal_inversions(
    tmp_path: Path,
) -> None:
    package = tmp_path / "services" / "nested"
    package.mkdir(parents=True)
    (package / "bad.py").write_text(
        "from ynab_agent.db.manager import DatabaseManager\n",
        encoding="utf-8",
    )

    assert _find_forbidden_imports(
        tmp_path,
        {"services": FORBIDDEN_IMPORTS["services"]},
    ) == ["services/nested/bad.py imports ynab_agent.db.manager"]
