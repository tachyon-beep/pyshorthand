"""Lint command for PyShorthand CLI."""

import json
import sys
from argparse import Namespace
from pathlib import Path

from pyshort.core.ast_nodes import DiagnosticSeverity
from pyshort.core.validator import Linter


def lint_command(args: Namespace) -> int:
    """Execute the lint command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        input_path = Path(args.input)

        # Collect files to lint
        files_to_lint = []
        if input_path.is_file():
            files_to_lint.append(input_path)
        elif input_path.is_dir():
            files_to_lint.extend(input_path.rglob("*.pys"))
        else:
            print(f"Error: {args.input} is not a file or directory", file=sys.stderr)
            return 1

        if not files_to_lint:
            print(f"No .pys files found in {args.input}", file=sys.stderr)
            return 1

        # Lint each file
        linter = Linter(strict=args.strict)
        all_diagnostics = []
        total_errors = 0
        total_warnings = 0

        for file_path in files_to_lint:
            try:
                diagnostics = linter.check_file(str(file_path))

                if diagnostics:
                    all_diagnostics.append({"file": str(file_path), "diagnostics": diagnostics})

                    errors = sum(1 for d in diagnostics if d.severity == DiagnosticSeverity.ERROR)
                    warnings = sum(
                        1 for d in diagnostics if d.severity == DiagnosticSeverity.WARNING
                    )

                    total_errors += errors
                    total_warnings += warnings

                    if not args.json:
                        # Human-readable output
                        if diagnostics:
                            print(f"\n{file_path}:")
                            for diagnostic in diagnostics:
                                severity_icon = {
                                    DiagnosticSeverity.ERROR: "✗",
                                    DiagnosticSeverity.WARNING: "⚠",
                                    DiagnosticSeverity.INFO: "ℹ",
                                    DiagnosticSeverity.HINT: "💡",
                                }
                                icon = severity_icon.get(diagnostic.severity, "•")
                                print(f"  {icon} {diagnostic}")

            except Exception as e:
                print(f"Error linting {file_path}: {e}", file=sys.stderr)
                total_errors += 1

        # Output results
        if args.json:
            # JSON output
            output = {
                "files": len(files_to_lint),
                "errors": total_errors,
                "warnings": total_warnings,
                "diagnostics": [
                    {
                        "file": item["file"],
                        "issues": [
                            {
                                "severity": d.severity.value,
                                "line": d.line,
                                "column": d.column,
                                "message": d.message,
                                "suggestion": d.suggestion,
                            }
                            for d in item["diagnostics"]
                        ],
                    }
                    for item in all_diagnostics
                ],
            }
            print(json.dumps(output, indent=2))
        else:
            # Human-readable summary
            print(f"\n{'=' * 60}")
            print(f"Linted {len(files_to_lint)} file(s)")
            print(f"Errors: {total_errors}")
            print(f"Warnings: {total_warnings}")

            if total_errors == 0 and total_warnings == 0:
                print("✓ All checks passed!")

        # Return non-zero if errors found
        return 1 if total_errors > 0 else 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


def main() -> int:
    """Main entry point for pyshort-lint command."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate and lint PyShorthand files")
    parser.add_argument("input", help="Input .pys file or directory")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as errors")
    parser.add_argument("--json", action="store_true", help="Output diagnostics as JSON")

    args = parser.parse_args()
    return lint_command(args)


if __name__ == "__main__":
    sys.exit(main())
