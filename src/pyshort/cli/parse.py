"""Parse command for PyShorthand CLI."""

import json
import sys
from argparse import Namespace

from pyshort.core.parser import parse_file


def parse_command(args: Namespace) -> int:
    """Execute the parse command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        # Parse the file
        ast = parse_file(args.input)

        # Check for parse errors
        if ast.has_errors():
            print(f"Parse errors in {args.input}:", file=sys.stderr)
            for diagnostic in ast.diagnostics:
                print(f"  {diagnostic}", file=sys.stderr)
            return 1

        # Convert to dictionary
        ast_dict = ast.to_dict()

        # Output
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                if args.pretty:
                    json.dump(ast_dict, f, indent=2, ensure_ascii=False)
                else:
                    json.dump(ast_dict, f, ensure_ascii=False)
            print(f"Parsed AST written to {args.output}")
        else:
            # Print to stdout
            if args.pretty:
                print(json.dumps(ast_dict, indent=2, ensure_ascii=False))
            else:
                print(json.dumps(ast_dict, ensure_ascii=False))

        # Show summary
        if ast.has_warnings():
            print(f"\nWarnings: {len([d for d in ast.diagnostics if d.severity.value == 'warning'])}", file=sys.stderr)

        return 0

    except FileNotFoundError:
        print(f"Error: File not found: {args.input}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


def main() -> int:
    """Main entry point for pyshort-parse command."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse PyShorthand files into structured AST"
    )
    parser.add_argument("input", help="Input .pys file")
    parser.add_argument("-o", "--output", help="Output JSON file")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")

    args = parser.parse_args()
    return parse_command(args)


if __name__ == "__main__":
    sys.exit(main())
