"""Visualization CLI for PyShorthand.

Generates diagrams from PyShorthand files:
- Mermaid flowcharts for dataflow visualization
- Class diagrams for entity relationships
- Architecture graphs for module organization

Usage:
    pyshort-viz input.pys --format mermaid --output diagram.mmd
    pyshort-viz input.pys --type classDiagram --direction LR
"""

import argparse
import sys
from argparse import Namespace
from pathlib import Path

from ..core.parser import parse_file
from ..visualization.mermaid import MermaidConfig


def viz_command(args: Namespace) -> int:
    """Command handler for pyshort viz subcommand."""

    # Convert args to expected format
    class VizArgs:
        def __init__(self, args):
            self.input = Path(args.input)
            self.output = Path(args.output) if args.output else None
            self.format = "mermaid"  # Default format
            self.type = args.type if hasattr(args, "type") else "flowchart"
            self.direction = args.direction if hasattr(args, "direction") else "TB"
            self.no_state = False
            self.no_methods = False
            self.no_deps = False
            self.no_color = False
            self.no_metadata = False

    return _viz_main(VizArgs(args))


def _viz_main(parsed_args) -> int:
    """Core visualization logic."""
    # Check input exists
    if not parsed_args.input.exists():
        print(f"Error: File not found: {parsed_args.input}", file=sys.stderr)
        return 1

    try:
        # Parse PyShorthand file
        ast = parse_file(str(parsed_args.input))

        # Check for parse errors
        errors = [d for d in ast.diagnostics if d.severity.value == "error"]
        if errors:
            print(
                f"Warning: {len(errors)} parse error(s) in {parsed_args.input}",
                file=sys.stderr,
            )
            for err in errors[:5]:  # Show first 5
                print(f"  Line {err.line}: {err.message}", file=sys.stderr)

        # Generate diagram
        if parsed_args.format == "mermaid":
            config = MermaidConfig(
                diagram_type=parsed_args.type,
                direction=parsed_args.direction,
                show_state_vars=not parsed_args.no_state,
                show_methods=not parsed_args.no_methods,
                show_dependencies=not parsed_args.no_deps,
                color_by_risk=not parsed_args.no_color,
                include_metadata=not parsed_args.no_metadata,
            )

            from ..visualization.mermaid import MermaidGenerator

            generator = MermaidGenerator(config)
            diagram = generator.generate(ast)

            # Output
            if parsed_args.output:
                parsed_args.output.write_text(diagram, encoding="utf-8")
                print(f"✓ Diagram written to {parsed_args.output}")
            else:
                print(diagram)

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


def main(args: list | None = None) -> int:
    """Main entry point for visualization CLI."""
    parser = argparse.ArgumentParser(
        description="Generate visualizations from PyShorthand files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "input",
        type=Path,
        help="PyShorthand file to visualize",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output file (default: print to stdout)",
    )

    parser.add_argument(
        "-f",
        "--format",
        choices=["mermaid"],
        default="mermaid",
        help="Output format (default: mermaid)",
    )

    parser.add_argument(
        "-t",
        "--type",
        choices=["flowchart", "classDiagram", "graph"],
        default="flowchart",
        help="Diagram type (default: flowchart)",
    )

    parser.add_argument(
        "-d",
        "--direction",
        choices=["TB", "LR", "RL", "BT"],
        default="TB",
        help="Diagram direction: TB (top-bottom), LR (left-right), RL, BT (default: TB)",
    )

    parser.add_argument(
        "--no-state",
        action="store_true",
        help="Hide state variables in diagrams",
    )

    parser.add_argument(
        "--no-methods",
        action="store_true",
        help="Hide methods in class diagrams",
    )

    parser.add_argument(
        "--no-deps",
        action="store_true",
        help="Hide dependency edges",
    )

    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable risk-based coloring",
    )

    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Hide metadata in diagrams",
    )

    parsed_args = parser.parse_args(args)
    return _viz_main(parsed_args)


if __name__ == "__main__":
    sys.exit(main())
