"""CLI tool for Python to PyShorthand decompilation."""

import argparse
import sys
from pathlib import Path

from pyshort.decompiler import decompile, decompile_file


def main():
    """Main entry point for py2short CLI."""
    parser = argparse.ArgumentParser(
        description="Decompile Python code to PyShorthand IR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Decompile a Python file
  py2short model.py -o model.pys

  # Decompile and print to stdout
  py2short model.py

  # Decompile with aggressive type inference
  py2short model.py --aggressive -o model.pys

  # Process multiple files
  py2short src/*.py --output-dir out/
        """
    )

    parser.add_argument(
        "input",
        nargs="+",
        help="Python source file(s) to decompile"
    )

    parser.add_argument(
        "-o", "--output",
        help="Output file path (for single input file)"
    )

    parser.add_argument(
        "--output-dir",
        help="Output directory (for multiple input files)"
    )

    parser.add_argument(
        "-a", "--aggressive",
        action="store_true",
        help="Use aggressive type inference"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    # Handle single vs multiple files
    if len(args.input) == 1:
        input_file = args.input[0]

        try:
            if args.verbose:
                print(f"Decompiling {input_file}...", file=sys.stderr)

            result = decompile_file(
                input_file,
                output_path=args.output,
                aggressive=args.aggressive
            )

            # If no output file, print to stdout
            if not args.output:
                print(result)
            elif args.verbose:
                print(f"Wrote {args.output}", file=sys.stderr)

        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    else:
        # Multiple files
        if not args.output_dir:
            print("Error: --output-dir required for multiple input files", file=sys.stderr)
            sys.exit(1)

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for input_file in args.input:
            try:
                input_path = Path(input_file)
                output_path = output_dir / f"{input_path.stem}.pys"

                if args.verbose:
                    print(f"Decompiling {input_file} -> {output_path}...", file=sys.stderr)

                decompile_file(
                    input_file,
                    output_path=str(output_path),
                    aggressive=args.aggressive
                )

            except Exception as e:
                print(f"Error processing {input_file}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
