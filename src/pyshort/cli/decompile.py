"""Decompile command for PyShorthand CLI."""

import sys
from argparse import Namespace


def decompile_command(args: Namespace) -> int:
    """Execute the decompile command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code
    """
    print("Error: Decompiler not yet implemented (Phase 2)", file=sys.stderr)
    print("This will convert Python source to PyShorthand notation", file=sys.stderr)
    return 1


def main() -> int:
    """Main entry point for py2short command."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate PyShorthand from Python source code"
    )
    parser.add_argument("input", help="Input Python file")
    parser.add_argument("-o", "--output", help="Output .pys file")
    parser.add_argument(
        "--aggressive", action="store_true", help="Aggressive inference (fewer TODOs)"
    )

    args = parser.parse_args()
    return decompile_command(args)


if __name__ == "__main__":
    sys.exit(main())
