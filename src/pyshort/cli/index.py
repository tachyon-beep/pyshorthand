"""CLI tool for repository indexing."""

import argparse
import sys
from pathlib import Path

from pyshort.indexer import RepositoryIndexer


def main():
    """Main entry point for pyshort-index CLI."""
    parser = argparse.ArgumentParser(
        description="Index Python repositories and generate PyShorthand specs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index current repository
  pyshort-index .

  # Index repository and save results
  pyshort-index /path/to/repo -o repo_index.json

  # Index with report
  pyshort-index . --report

  # Generate PyShorthand for all files
  pyshort-index . --generate-pys --output-dir pys_output/

  # Verbose output
  pyshort-index . -v
        """,
    )

    parser.add_argument("repo_path", help="Path to repository root")

    parser.add_argument("-o", "--output", help="Output JSON file for index")

    parser.add_argument(
        "-r", "--report", action="store_true", help="Generate human-readable report"
    )

    parser.add_argument(
        "--generate-pys",
        action="store_true",
        help="Generate PyShorthand files for all Python files",
    )

    parser.add_argument(
        "--output-dir", help="Output directory for PyShorthand files (with --generate-pys)"
    )

    parser.add_argument(
        "--exclude", nargs="+", help="Additional patterns to exclude (e.g., 'tests' 'docs')"
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    parser.add_argument(
        "--stats-only", action="store_true", help="Only show statistics, don't generate full index"
    )

    parser.add_argument(
        "--dep-graph", action="store_true", help="Generate dependency graph (Mermaid format)"
    )

    parser.add_argument(
        "--entity-map",
        action="store_true",
        help="Show entity map (all classes/functions by module)",
    )

    args = parser.parse_args()

    # Validate inputs
    repo_path = Path(args.repo_path)
    if not repo_path.exists():
        print(f"Error: Repository path '{repo_path}' does not exist", file=sys.stderr)
        sys.exit(1)

    if not repo_path.is_dir():
        print(f"Error: '{repo_path}' is not a directory", file=sys.stderr)
        sys.exit(1)

    if args.generate_pys and not args.output_dir:
        print("Error: --output-dir required when using --generate-pys", file=sys.stderr)
        sys.exit(1)

    try:
        # Create indexer
        exclude_patterns = None
        if args.exclude:
            exclude_patterns = args.exclude

        indexer = RepositoryIndexer(str(repo_path), exclude_patterns=exclude_patterns)

        # Index repository
        if args.verbose:
            print(f"Indexing repository: {repo_path}")
            print()

        index = indexer.index_repository(verbose=args.verbose)

        # Show statistics
        if args.verbose or args.stats_only or args.report:
            print()
            stats = index.statistics
            print("=" * 80)
            print("REPOSITORY STATISTICS")
            print("=" * 80)
            print(f"Repository: {repo_path}")
            print(f"Total Python files: {stats['total_files']}")
            print(f"Total lines of code: {stats['total_lines']:,}")
            print(f"Average lines per file: {stats['avg_lines_per_file']}")
            print(f"Total entities: {stats['total_entities']}")
            print(f"  Classes: {stats['total_classes']}")
            print(f"  Functions: {stats['total_functions']}")
            print("=" * 80)

        if args.stats_only:
            sys.exit(0)

        # Save index to JSON
        if args.output:
            if args.verbose:
                print(f"\nSaving index to {args.output}...")

            indexer.save_index(args.output)

            if args.verbose:
                print(f"✓ Index saved to {args.output}")

        # Generate report
        if args.report:
            print()
            report = indexer.generate_report()
            print(report)

        # Generate PyShorthand files
        if args.generate_pys:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            if args.verbose:
                print(f"\nGenerating PyShorthand files to {output_dir}/...")

            success_count = 0
            error_count = 0

            for module_path, module_info in index.modules.items():
                if module_info.pyshorthand:
                    # Create output path maintaining directory structure
                    output_file = output_dir / f"{module_path.replace('.', '/')}.pys"
                    output_file.parent.mkdir(parents=True, exist_ok=True)

                    try:
                        with open(output_file, "w") as f:
                            f.write(module_info.pyshorthand)
                        success_count += 1

                        if args.verbose:
                            print(f"  ✓ {module_path} -> {output_file}")

                    except Exception as e:
                        error_count += 1
                        if args.verbose:
                            print(f"  ✗ {module_path}: {e}")

            print()
            print(f"Generated {success_count} PyShorthand files")
            if error_count > 0:
                print(f"Failed: {error_count} files")

        # Generate dependency graph
        if args.dep_graph:
            print()
            print("Dependency Graph (Mermaid):")
            print()
            mermaid_graph = indexer.generate_dependency_graph_mermaid()
            print(mermaid_graph)
            print()

        # Generate entity map
        if args.entity_map:
            print()
            entity_map = indexer.generate_entity_map_report()
            print(entity_map)

        # If no output options specified, print summary
        if (
            not args.output
            and not args.report
            and not args.generate_pys
            and not args.verbose
            and not args.dep_graph
            and not args.entity_map
        ):
            print(f"Indexed {index.statistics['total_files']} Python files")
            print(
                f"Found {index.statistics['total_entities']} entities ({index.statistics['total_classes']} classes, {index.statistics['total_functions']} functions)"
            )
            print(f"Total lines: {index.statistics['total_lines']:,}")
            print()
            print("Use --report for detailed report")
            print("Use -o <file> to save index")
            print("Use --generate-pys to generate PyShorthand files")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
