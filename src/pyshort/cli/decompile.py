"""Decompile command for PyShorthand CLI."""

import sys
from argparse import Namespace
from pathlib import Path

from pyshort.decompiler.py2short import decompile_file


def decompile_command(args: Namespace) -> int:
    """Execute the decompile command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code
    """
    input_path = args.input
    output_path = args.output
    aggressive = args.aggressive

    # If no output path, default to input path with .pys extension
    if not output_path:
        output_path = str(Path(input_path).with_suffix(".pys"))

    try:
        result = decompile_file(input_path, output_path, aggressive=aggressive)

        # If output path was provided, file was already written
        # Otherwise, print to stdout
        if not args.output:
            print(result)
        else:
            print(f"Generated PyShorthand: {output_path}", file=sys.stderr)

        return 0

    except (OSError, SyntaxError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc(file=sys.stderr)
        return 1


def main() -> int:
    """Main entry point for py2short command."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate PyShorthand from Python source code")
    parser.add_argument("input", help="Input Python file")
    parser.add_argument("-o", "--output", help="Output .pys file (defaults to input.pys)")
    parser.add_argument(
        "--aggressive", action="store_true", help="Aggressive inference (fewer TODOs)"
    )

    args = parser.parse_args()
    return decompile_command(args)


if __name__ == "__main__":
    sys.exit(main())
