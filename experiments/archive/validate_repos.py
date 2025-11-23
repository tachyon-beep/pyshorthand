#!/usr/bin/env python3
"""
Validation script for py2short decompiler.
Tests against diverse open-source Python repositories.
"""

import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Test repositories with diverse patterns
REPOS = [
    {
        "name": "nanoGPT",
        "url": "https://github.com/karpathy/nanoGPT.git",
        "description": "Minimal PyTorch GPT implementation",
        "test_files": ["model.py", "train.py"],
        "category": "PyTorch/ML",
    },
    {
        "name": "minGPT",
        "url": "https://github.com/karpathy/minGPT.git",
        "description": "Minimal PyTorch GPT with training",
        "test_files": ["mingpt/model.py", "mingpt/trainer.py"],
        "category": "PyTorch/ML",
    },
    {
        "name": "fastapi",
        "url": "https://github.com/tiangolo/fastapi.git",
        "description": "FastAPI framework (web APIs)",
        "test_files": ["fastapi/applications.py", "fastapi/routing.py"],
        "category": "FastAPI/Web",
    },
    {
        "name": "httpx",
        "url": "https://github.com/encode/httpx.git",
        "description": "Async HTTP client",
        "test_files": ["httpx/_client.py", "httpx/_models.py"],
        "category": "Async/HTTP",
    },
    {
        "name": "pydantic",
        "url": "https://github.com/pydantic/pydantic.git",
        "description": "Data validation with type hints",
        "test_files": ["pydantic/main.py", "pydantic/fields.py"],
        "category": "Pydantic/Validation",
    },
    {
        "name": "flask",
        "url": "https://github.com/pallets/flask.git",
        "description": "Flask web framework",
        "test_files": ["src/flask/app.py", "src/flask/views.py"],
        "category": "Flask/Web",
    },
    {
        "name": "transformers",
        "url": "https://github.com/huggingface/transformers.git",
        "description": "HuggingFace Transformers library",
        "test_files": ["src/transformers/models/bert/modeling_bert.py"],
        "category": "PyTorch/ML",
        "depth": 1,  # Shallow clone - it's huge
    },
    {
        "name": "numpy-financial",
        "url": "https://github.com/numpy/numpy-financial.git",
        "description": "NumPy financial functions",
        "test_files": ["numpy_financial/_financial.py"],
        "category": "NumPy/Scientific",
    },
]


@dataclass
class TestResult:
    """Result of testing a single file."""

    file: str
    success: bool
    output_file: str | None = None
    error_message: str | None = None
    parse_time: float = 0.0
    lines_of_code: int = 0
    output_lines: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass
class RepoResult:
    """Results for an entire repository."""

    name: str
    category: str
    description: str
    clone_success: bool
    clone_time: float = 0.0
    files_tested: int = 0
    files_success: int = 0
    files_failed: int = 0
    test_results: list[TestResult] = field(default_factory=list)
    error_message: str | None = None


class RepoValidator:
    """Validates py2short against real-world repositories."""

    def __init__(self, work_dir: Path):
        self.work_dir = work_dir
        self.work_dir.mkdir(exist_ok=True)
        self.results: list[RepoResult] = []

    def clone_repo(self, repo: dict) -> tuple[bool, float, str | None]:
        """Clone a repository."""
        repo_path = self.work_dir / repo["name"]

        # Skip if already cloned
        if repo_path.exists():
            print(f"  ✓ Already cloned: {repo['name']}")
            return True, 0.0, None

        start = time.time()
        try:
            cmd = ["git", "clone"]

            # Add depth flag if specified (for large repos)
            if repo.get("depth"):
                cmd.extend(["--depth", str(repo["depth"])])

            cmd.extend([repo["url"], str(repo_path)])

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300  # 5 min timeout
            )

            elapsed = time.time() - start

            if result.returncode != 0:
                return False, elapsed, result.stderr

            return True, elapsed, None

        except subprocess.TimeoutExpired:
            return False, time.time() - start, "Clone timeout (5 min)"
        except Exception as e:
            return False, time.time() - start, str(e)

    def test_file(self, repo_name: str, file_path: Path) -> TestResult:
        """Test py2short on a single file."""
        if not file_path.exists():
            return TestResult(
                file=str(file_path.name),
                success=False,
                error_message=f"File not found: {file_path}",
            )

        # Count lines
        try:
            lines_of_code = len(file_path.read_text().splitlines())
        except:
            lines_of_code = 0

        # Run py2short
        output_file = file_path.with_suffix(".pys")
        start = time.time()

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pyshort.cli.decompile",
                    str(file_path),
                    "-o",
                    str(output_file),
                ],
                capture_output=True,
                text=True,
                timeout=60,  # 1 min per file
            )

            elapsed = time.time() - start

            # Check for warnings in output
            warnings = []
            if result.stderr:
                warnings = [
                    line for line in result.stderr.splitlines() if "warning" in line.lower()
                ]

            if result.returncode != 0:
                return TestResult(
                    file=file_path.name,
                    success=False,
                    error_message=result.stderr or result.stdout,
                    parse_time=elapsed,
                    lines_of_code=lines_of_code,
                    warnings=warnings,
                )

            # Count output lines
            output_lines = 0
            if output_file.exists():
                try:
                    output_lines = len(output_file.read_text().splitlines())
                except:
                    pass

            return TestResult(
                file=file_path.name,
                success=True,
                output_file=str(output_file),
                parse_time=elapsed,
                lines_of_code=lines_of_code,
                output_lines=output_lines,
                warnings=warnings,
            )

        except subprocess.TimeoutExpired:
            return TestResult(
                file=file_path.name,
                success=False,
                error_message="Timeout (60s)",
                parse_time=60.0,
                lines_of_code=lines_of_code,
            )
        except Exception as e:
            return TestResult(
                file=file_path.name,
                success=False,
                error_message=str(e),
                lines_of_code=lines_of_code,
            )

    def validate_repo(self, repo: dict) -> RepoResult:
        """Validate all test files in a repository."""
        print(f"\n{'='*60}")
        print(f"Testing: {repo['name']} ({repo['category']})")
        print(f"  {repo['description']}")
        print(f"{'='*60}")

        result = RepoResult(
            name=repo["name"],
            category=repo["category"],
            description=repo["description"],
            clone_success=False,
        )

        # Clone repository
        print(f"Cloning {repo['name']}...")
        clone_success, clone_time, error = self.clone_repo(repo)
        result.clone_success = clone_success
        result.clone_time = clone_time

        if not clone_success:
            result.error_message = error
            print(f"  ✗ Clone failed: {error}")
            return result

        print(f"  ✓ Cloned in {clone_time:.2f}s")

        # Test each file
        repo_path = self.work_dir / repo["name"]
        for test_file in repo["test_files"]:
            file_path = repo_path / test_file
            print(f"\n  Testing: {test_file}")

            test_result = self.test_file(repo["name"], file_path)
            result.test_results.append(test_result)
            result.files_tested += 1

            if test_result.success:
                result.files_success += 1
                compression = (
                    (1 - test_result.output_lines / test_result.lines_of_code) * 100
                    if test_result.lines_of_code > 0
                    else 0
                )
                print(
                    f"    ✓ Success ({test_result.lines_of_code} → {test_result.output_lines} lines, {compression:.1f}% compression)"
                )
                print(f"      Time: {test_result.parse_time:.3f}s")
                if test_result.warnings:
                    print(f"      Warnings: {len(test_result.warnings)}")
            else:
                result.files_failed += 1
                print("    ✗ Failed")
                if test_result.error_message:
                    # Print first 3 lines of error
                    error_lines = test_result.error_message.splitlines()[:3]
                    for line in error_lines:
                        print(f"      {line}")

        return result

    def run_validation(self, repos: list[dict]) -> list[RepoResult]:
        """Run validation on all repositories."""
        print("\n🔍 PyShorthand Decompiler Validation")
        print(f"Testing {len(repos)} repositories")
        print(f"Work directory: {self.work_dir}\n")

        for repo in repos:
            result = self.validate_repo(repo)
            self.results.append(result)

        return self.results

    def generate_report(self) -> dict:
        """Generate summary report."""
        total_repos = len(self.results)
        cloned_repos = sum(1 for r in self.results if r.clone_success)
        total_files = sum(r.files_tested for r in self.results)
        success_files = sum(r.files_success for r in self.results)
        failed_files = sum(r.files_failed for r in self.results)

        success_rate = (success_files / total_files * 100) if total_files > 0 else 0

        # Group by category
        by_category = {}
        for result in self.results:
            cat = result.category
            if cat not in by_category:
                by_category[cat] = {"tested": 0, "success": 0, "failed": 0}
            by_category[cat]["tested"] += result.files_tested
            by_category[cat]["success"] += result.files_success
            by_category[cat]["failed"] += result.files_failed

        # Collect all errors
        errors = []
        for result in self.results:
            for test in result.test_results:
                if not test.success:
                    errors.append(
                        {"repo": result.name, "file": test.file, "error": test.error_message}
                    )

        report = {
            "summary": {
                "total_repos": total_repos,
                "cloned_repos": cloned_repos,
                "total_files": total_files,
                "success_files": success_files,
                "failed_files": failed_files,
                "success_rate": f"{success_rate:.1f}%",
            },
            "by_category": by_category,
            "errors": errors,
            "detailed_results": [
                {
                    "repo": r.name,
                    "category": r.category,
                    "description": r.description,
                    "clone_success": r.clone_success,
                    "files_tested": r.files_tested,
                    "files_success": r.files_success,
                    "files_failed": r.files_failed,
                    "test_results": [
                        {
                            "file": t.file,
                            "success": t.success,
                            "lines_of_code": t.lines_of_code,
                            "output_lines": t.output_lines,
                            "parse_time": t.parse_time,
                            "error": t.error_message,
                            "warnings_count": len(t.warnings),
                        }
                        for t in r.test_results
                    ],
                }
                for r in self.results
            ],
        }

        return report

    def print_summary(self):
        """Print human-readable summary."""
        report = self.generate_report()

        print(f"\n{'='*60}")
        print("VALIDATION SUMMARY")
        print(f"{'='*60}\n")

        summary = report["summary"]
        print(f"Repositories: {summary['cloned_repos']}/{summary['total_repos']} cloned")
        print(f"Files tested: {summary['total_files']}")
        print(f"  ✓ Success: {summary['success_files']}")
        print(f"  ✗ Failed:  {summary['failed_files']}")
        print(f"Success rate: {summary['success_rate']}")

        print("\nBy Category:")
        for cat, stats in report["by_category"].items():
            rate = (stats["success"] / stats["tested"] * 100) if stats["tested"] > 0 else 0
            print(f"  {cat:20s} {stats['success']}/{stats['tested']} ({rate:.1f}%)")

        if report["errors"]:
            print(f"\nErrors ({len(report['errors'])}):")
            for err in report["errors"][:10]:  # Show first 10
                print(f"  • {err['repo']}/{err['file']}")
                if err["error"]:
                    error_preview = err["error"].splitlines()[0][:80]
                    print(f"    {error_preview}")

            if len(report["errors"]) > 10:
                print(f"  ... and {len(report['errors']) - 10} more")

        print(f"\n{'='*60}\n")


def main():
    """Main entry point."""
    # Use test_repos directory
    work_dir = Path(__file__).parent / "test_repos"

    validator = RepoValidator(work_dir)

    # Run validation
    validator.run_validation(REPOS)

    # Print summary
    validator.print_summary()

    # Save detailed report
    report = validator.generate_report()
    report_path = Path(__file__).parent / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Detailed report saved to: {report_path}")

    # Exit with failure if success rate < 70%
    success_rate = float(report["summary"]["success_rate"].rstrip("%"))
    if success_rate < 70:
        print(f"\n⚠️  Success rate {success_rate}% is below 70% threshold")
        sys.exit(1)
    else:
        print(f"\n✓ Success rate {success_rate}% meets 70% threshold")
        sys.exit(0)


if __name__ == "__main__":
    main()
