# Repository Guidelines

## Project Structure & Module Organization
- Core parsing/validation lives in `src/pyshort/core` (tokenizer, parser, validator, symbols; keep zero-dependency).
- Decompiler/formatter are in `src/pyshort/decompiler` and `src/pyshort/formatter`. Analyzer utilities live in `src/pyshort/analyzer`; repository/index tooling in `src/pyshort/indexer`; visualization helpers in `src/pyshort/visualization`; CLI entrypoints in `src/pyshort/cli`; ecosystem tools in `src/pyshort/ecosystem`.
- Tests: `tests/` holds unit/integration/API/regression suites (`critical_bug_fixes_test.py`, compliance fixtures in `tests/compliance/fixtures`).
- Docs in `docs/`; benchmarks/experiments under `benchmarks/` and `experiments/`; historical specs in `archive/`.

## Setup, Build, and Run Commands
- Install with uv: `uv sync --extra dev --extra cli --extra analysis` (add `--extra viz` after installing Graphviz headers `graphviz`/`libgraphviz-dev`), then `source .venv/bin/activate` or use `uv run ...`.
- Lint/format/type-check: `uv run ruff check src tests`, `uv run black --check src tests`, `uv run mypy src`.
- Test suites: `uv run pytest`, `uv run python -m unittest tests.compliance.test_rfc_compliance -v` or `./run_compliance_tests.sh`, and `uv run pytest tests/critical_bug_fixes_test.py -k <name>` for focused regressions.
- CLI smoke tests: `uv run pyshort-parse sample.pys`, `uv run py2short model.py > model.pys`, `uv run pyshort` (main CLI) or `uv run python -m pyshort.ecosystem.server <path>` for the ecosystem server.

## Coding Style & Naming Conventions
- Python ≥3.10, 4-space indent, line length 100. Prefer type hints; keep core modules dependency-free unless absolutely necessary.
- Formatting via Black; lint with Ruff (E/W/F/I/N/UP/B/C4; long-line/except/SIM relaxed in config).
- Tests follow `test_*.py` files, `Test*` classes, `test_*` functions. Use snake_case for functions/vars, PascalCase for classes. Keep PyShorthand token/symbol mappings consistent with `core/symbols.py` (ASCII ↔ Unicode).

## Testing Guidelines
- Use pytest for new unit/integration coverage; keep fixtures near the test file when small, otherwise place reusable PyShorthand samples in `tests/compliance/fixtures` or `tests/integration/fixtures`.
- Coverage runs by default (`--cov=pyshort --cov-report=term-missing --cov-report=html`).
- For protocol changes, add/update compliance tests and fixtures first (TDD), then verify `run_compliance_tests.sh`. Add regression cases to `critical_bug_fixes_test.py` for previously broken paths.
- Prefer focused tests that assert parser/decompiler outputs and analyzer graph shapes rather than broad end-to-end snapshots.

## Commit & Pull Request Guidelines
- Commit messages are short and imperative; conventional prefixes used in history (`feat:`, `fix:`, `chore:`). Reference issues/PRs when relevant (e.g., `feat: add execution flow tracing (#123)`).
- PRs should include a short scope summary, test commands executed, and any docs/CLI flag updates.
- Avoid mixing refactors with behavioral changes; keep commits scoped. Update docs in `docs/` or `README.md` when modifying protocol semantics or CLI surface.
