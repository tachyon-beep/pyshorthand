"""Main CLI entry point for PyShorthand tools."""

import argparse
import sys


def main() -> int:
    """Main entry point for pyshort CLI."""
    parser = argparse.ArgumentParser(
        description="PyShorthand Protocol Toolchain",
        epilog="Use 'pyshort <command> --help' for more information on a specific command.",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Parse command
    parse_parser = subparsers.add_parser("parse", help="Parse PyShorthand files")
    parse_parser.add_argument("input", help="Input .pys file")
    parse_parser.add_argument("-o", "--output", help="Output JSON file")
    parse_parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")

    # Lint command
    lint_parser = subparsers.add_parser("lint", help="Validate and lint PyShorthand files")
    lint_parser.add_argument("input", help="Input .pys file or directory")
    lint_parser.add_argument("--strict", action="store_true", help="Treat warnings as errors")
    lint_parser.add_argument("--json", action="store_true", help="Output diagnostics as JSON")

    # Format command
    fmt_parser = subparsers.add_parser("fmt", help="Auto-format PyShorthand files")
    fmt_parser.add_argument("input", help="Input .pys file or directory")
    fmt_parser.add_argument("-w", "--write", action="store_true", help="Write changes in-place")
    fmt_parser.add_argument("--check", action="store_true", help="Check if files need formatting")
    fmt_parser.add_argument("--diff", action="store_true", help="Show diff of formatting changes")

    # Viz command
    viz_parser = subparsers.add_parser("viz", help="Generate visualizations")
    viz_parser.add_argument("input", help="Input .pys file")
    viz_parser.add_argument("-o", "--output", help="Output file")
    viz_parser.add_argument(
        "-t",
        "--type",
        choices=["flowchart", "classDiagram", "graph"],
        default="flowchart",
        help="Diagram type",
    )
    viz_parser.add_argument(
        "-d",
        "--direction",
        choices=["TB", "LR", "RL", "BT"],
        default="TB",
        help="Diagram direction",
    )

    # Version command
    version_parser = subparsers.add_parser("version", help="Show version information")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "parse":
        from pyshort.cli.parse import parse_command

        return parse_command(args)
    elif args.command == "lint":
        from pyshort.cli.lint import lint_command

        return lint_command(args)
    elif args.command == "fmt":
        from pyshort.cli.format import format_command

        return format_command(args)
    elif args.command == "viz":
        from pyshort.cli.viz import viz_command

        return viz_command(args)
    elif args.command == "version":
        from pyshort import __version__

        print(f"PyShorthand v{__version__}")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
