"""Format command for PyShorthand CLI."""

import sys
from argparse import Namespace
from pathlib import Path

from pyshort.formatter import FormatConfig, format_file
from pyshort.core.config import load_config


def format_command(args: Namespace) -> int:
    """Execute the format command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        input_path = Path(args.input)

        # Load config file (if exists)
        file_config = load_config()
        format_config = file_config.get("format", {})

        # Collect files to format
        files_to_format = []
        if input_path.is_file():
            files_to_format.append(input_path)
        elif input_path.is_dir():
            files_to_format.extend(input_path.rglob("*.pys"))
        else:
            print(f"Error: {args.input} is not a file or directory", file=sys.stderr)
            return 1

        if not files_to_format:
            print(f"No .pys files found in {args.input}", file=sys.stderr)
            return 1

        # Create config (CLI args override config file)
        config = FormatConfig(
            indent=args.indent if (hasattr(args, "indent") and args.indent is not None) else format_config.get("indent", 2),
            align_types=not args.no_align if hasattr(args, "no_align") else format_config.get("align_types", True),
            prefer_unicode=not args.ascii if hasattr(args, "ascii") else format_config.get("prefer_unicode", True),
            sort_state_by=args.sort_state if (hasattr(args, "sort_state") and args.sort_state is not None) else format_config.get("sort_state_by", "location"),
            max_line_length=args.line_length if (hasattr(args, "line_length") and args.line_length is not None) else format_config.get("max_line_length", 100),
        )

        # Format each file
        needs_formatting = []
        formatted_count = 0
        error_count = 0

        for file_path in files_to_format:
            try:
                if args.check:
                    # Check mode: see if file needs formatting
                    with open(file_path, "r", encoding="utf-8") as f:
                        original = f.read()
                    formatted = format_file(str(file_path), config, in_place=False)

                    if original != formatted:
                        needs_formatting.append(file_path)
                        print(f"Would reformat: {file_path}")
                        if args.diff:
                            # Show diff (simple version)
                            print(f"\n--- {file_path} (original)")
                            print(f"+++ {file_path} (formatted)")
                            orig_lines = original.split("\n")
                            fmt_lines = formatted.split("\n")
                            for i, (orig, fmt) in enumerate(zip(orig_lines, fmt_lines), 1):
                                if orig != fmt:
                                    print(f"- {i}: {orig}")
                                    print(f"+ {i}: {fmt}")
                else:
                    # Format mode
                    formatted = format_file(str(file_path), config, in_place=args.write)

                    if args.write:
                        formatted_count += 1
                        print(f"✓ Formatted: {file_path}")
                    else:
                        # Print to stdout
                        if len(files_to_format) > 1:
                            print(f"\n=== {file_path} ===")
                        print(formatted)

            except Exception as e:
                error_count += 1
                print(f"Error formatting {file_path}: {e}", file=sys.stderr)
                if args.verbose:
                    import traceback

                    traceback.print_exc()
                continue  # Continue with other files

        # Summary
        if args.check:
            if needs_formatting:
                print(f"\n✗ {len(needs_formatting)} file(s) need formatting (out of {len(files_to_format)} checked)")
                return 1  # Exit code 1 for CI/CD
            else:
                print(f"\n✓ All {len(files_to_format)} file(s) are formatted correctly")
        elif args.write:
            print(f"\n✓ Formatted {formatted_count} file(s)")
            if error_count > 0:
                print(f"✗ {error_count} file(s) had errors", file=sys.stderr)

        return 1 if error_count > 0 else 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def main() -> int:
    """Main entry point for pyshort-fmt command."""
    import argparse
    from pyshort.core.config import create_default_config

    parser = argparse.ArgumentParser(
        description="Format PyShorthand files for consistency",
        epilog="Example: pyshort-fmt src/ --write",
    )
    parser.add_argument("input", nargs="?", help="Input .pys file or directory")
    parser.add_argument(
        "-w", "--write", action="store_true", help="Write changes in-place (default: print to stdout)"
    )
    parser.add_argument(
        "--check", action="store_true", help="Check if files need formatting (don't modify)"
    )
    parser.add_argument("--diff", action="store_true", help="Show diff when using --check")
    parser.add_argument("--indent", type=int, help="Indentation spaces (default: 2)")
    parser.add_argument("--no-align", action="store_true", help="Don't align type annotations")
    parser.add_argument("--ascii", action="store_true", help="Use ASCII notation instead of Unicode")
    parser.add_argument(
        "--sort-state",
        choices=["location", "name", "none"],
        help="How to sort state variables (default: location)",
    )
    parser.add_argument(
        "--line-length", type=int, help="Maximum line length (default: 100)"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--init-config",
        action="store_true",
        help="Create a default .pyshortrc config file in current directory",
    )

    args = parser.parse_args()

    # Handle --init-config
    if args.init_config:
        config_path = Path.cwd() / ".pyshortrc"
        if config_path.exists():
            print(f"Error: {config_path} already exists", file=sys.stderr)
            return 1
        create_default_config(config_path)
        print(f"✓ Created {config_path}")
        return 0

    # Require input for normal operation
    if not args.input:
        parser.print_help()
        return 1

    return format_command(args)


if __name__ == "__main__":
    sys.exit(main())
