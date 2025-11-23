#!/usr/bin/env python3
"""
Analyze results from A/B testing and generate comprehensive report
"""

import json
import statistics
import sys
from collections import defaultdict


def load_results(results_file: str) -> list[dict]:
    """Load results from JSON file"""
    with open(results_file) as f:
        return json.load(f)


def analyze_results(results: list[dict]):
    """Generate comprehensive analysis"""

    # Group by format
    by_format = defaultdict(list)
    by_question = defaultdict(lambda: {"original": None, "pyshorthand": None})

    for result in results:
        by_format[result["format"]].append(result)
        by_question[result["question_id"]][result["format"]] = result

    original_results = by_format["original"]
    pyshort_results = by_format["pyshorthand"]

    # Calculate overall metrics
    print("\n" + "=" * 80)
    print("OVERALL RESULTS")
    print("=" * 80)

    print("\n📊 ORIGINAL PYTHON CODE")
    print_summary(original_results)

    print("\n📊 PYSHORTHAND CODE")
    print_summary(pyshort_results)

    # Compare
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    compare_metrics(original_results, pyshort_results)

    # By difficulty
    print("\n" + "=" * 80)
    print("RESULTS BY DIFFICULTY")
    print("=" * 80)

    analyze_by_difficulty(results)

    # By category
    print("\n" + "=" * 80)
    print("RESULTS BY CATEGORY")
    print("=" * 80)

    analyze_by_category(results)

    # Question-by-question
    print("\n" + "=" * 80)
    print("QUESTION-BY-QUESTION ANALYSIS")
    print("=" * 80)

    analyze_questions(by_question)

    # Token efficiency
    print("\n" + "=" * 80)
    print("TOKEN EFFICIENCY")
    print("=" * 80)

    analyze_token_efficiency(original_results, pyshort_results)

    # Areas for improvement
    print("\n" + "=" * 80)
    print("AREAS FOR IMPROVEMENT")
    print("=" * 80)

    identify_improvements(by_question)


def print_summary(results: list[dict]):
    """Print summary statistics"""
    total = len(results)
    correct = sum(1 for r in results if r["is_correct"])
    avg_time = statistics.mean(r["response_time_ms"] for r in results)
    avg_tokens = statistics.mean(r["total_tokens"] for r in results)
    avg_completeness = statistics.mean(r["completeness_score"] for r in results)

    print(f"  Accuracy:      {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"  Avg Time:      {avg_time:.0f}ms")
    print(f"  Avg Tokens:    {avg_tokens:.0f}")
    print(f"  Completeness:  {avg_completeness:.2f}")


def compare_metrics(original: list[dict], pyshort: list[dict]):
    """Compare metrics between formats"""

    orig_correct = sum(1 for r in original if r["is_correct"])
    py_correct = sum(1 for r in pyshort if r["is_correct"])

    orig_time = statistics.mean(r["response_time_ms"] for r in original)
    py_time = statistics.mean(r["response_time_ms"] for r in pyshort)

    orig_tokens = sum(r["total_tokens"] for r in original)
    py_tokens = sum(r["total_tokens"] for r in pyshort)

    orig_prompt = sum(r["prompt_tokens"] for r in original)
    py_prompt = sum(r["prompt_tokens"] for r in pyshort)

    print("\n📈 Accuracy:")
    print(f"   Original:     {orig_correct}/20 ({100*orig_correct/20:.1f}%)")
    print(f"   PyShorthand:  {py_correct}/20 ({100*py_correct/20:.1f}%)")
    print(
        f"   Difference:   {py_correct - orig_correct} ({100*(py_correct - orig_correct)/20:+.1f}%)"
    )

    print("\n⏱️  Response Time:")
    print(f"   Original:     {orig_time:.0f}ms")
    print(f"   PyShorthand:  {py_time:.0f}ms")
    print(
        f"   Speedup:      {orig_time/py_time:.2f}x {'faster' if py_time < orig_time else 'slower'}"
    )

    print("\n🎫 Total Tokens:")
    print(f"   Original:     {orig_tokens:,}")
    print(f"   PyShorthand:  {py_tokens:,}")
    print(f"   Reduction:    {100*(orig_tokens - py_tokens)/orig_tokens:.1f}%")

    print("\n📝 Prompt Tokens:")
    print(f"   Original:     {orig_prompt:,}")
    print(f"   PyShorthand:  {py_prompt:,}")
    print(f"   Reduction:    {100*(orig_prompt - py_prompt)/orig_prompt:.1f}%")


def analyze_by_difficulty(results: list[dict]):
    """Analyze results by difficulty level"""

    # Map question IDs to difficulty
    difficulty_map = {
        **dict.fromkeys(range(1, 6), "easy"),
        **dict.fromkeys(range(6, 11), "medium"),
        **dict.fromkeys(range(11, 16), "medium-hard"),
        **dict.fromkeys(range(16, 21), "hard"),
    }

    by_difficulty = defaultdict(lambda: {"original": [], "pyshorthand": []})

    for result in results:
        difficulty = difficulty_map[result["question_id"]]
        by_difficulty[difficulty][result["format"]].append(result)

    for difficulty in ["easy", "medium", "medium-hard", "hard"]:
        print(f"\n{difficulty.upper()}")
        orig = by_difficulty[difficulty]["original"]
        pysh = by_difficulty[difficulty]["pyshorthand"]

        if orig and pysh:
            orig_correct = sum(1 for r in orig if r["is_correct"])
            pysh_correct = sum(1 for r in pysh if r["is_correct"])
            print(f"  Accuracy:   Original {orig_correct}/5  |  PyShorthand {pysh_correct}/5")

            if orig:
                orig_time = statistics.mean(r["response_time_ms"] for r in orig)
                pysh_time = statistics.mean(r["response_time_ms"] for r in pysh)
                print(
                    f"  Avg Time:   Original {orig_time:.0f}ms  |  PyShorthand {pysh_time:.0f}ms  ({orig_time/pysh_time:.2f}x)"
                )


def analyze_by_category(results: list[dict]):
    """Analyze results by category"""

    # Map question IDs to categories
    category_map = {
        1: "structure",
        2: "structure",
        3: "architecture",
        4: "structure",
        5: "structure",
        6: "signature",
        7: "signature",
        8: "signature",
        9: "signature",
        10: "structure",
        11: "architecture",
        12: "signature",
        13: "architecture",
        14: "implementation",
        15: "architecture",
        16: "implementation",
        17: "implementation",
        18: "implementation",
        19: "implementation",
        20: "implementation",
    }

    by_category = defaultdict(lambda: {"original": [], "pyshorthand": []})

    for result in results:
        category = category_map[result["question_id"]]
        by_category[category][result["format"]].append(result)

    for category in ["structure", "signature", "architecture", "implementation"]:
        print(f"\n{category.upper()}")
        orig = by_category[category]["original"]
        pysh = by_category[category]["pyshorthand"]

        if orig and pysh:
            orig_correct = sum(1 for r in orig if r["is_correct"])
            pysh_correct = sum(1 for r in pysh if r["is_correct"])
            total = len(orig)
            print(
                f"  Accuracy:   Original {orig_correct}/{total}  |  PyShorthand {pysh_correct}/{total}"
            )


def analyze_questions(by_question: dict):
    """Analyze question by question"""

    print("\n| Q# | Original | PyShort | Time Δ | Tokens Δ | Notes |")
    print("|----|----------|---------|--------|----------|-------|")

    for qid in sorted(by_question.keys()):
        orig = by_question[qid]["original"]
        pysh = by_question[qid]["pyshorthand"]

        if not orig or not pysh:
            continue

        orig_status = "✅" if orig["is_correct"] else "❌"
        pysh_status = "✅" if pysh["is_correct"] else "❌"

        time_delta = orig["response_time_ms"] - pysh["response_time_ms"]
        time_ratio = (
            orig["response_time_ms"] / pysh["response_time_ms"]
            if pysh["response_time_ms"] > 0
            else 0
        )

        token_delta = orig["total_tokens"] - pysh["total_tokens"]

        notes = ""
        if orig["is_correct"] and not pysh["is_correct"]:
            notes = "⚠️ PyShort failed"
        elif not orig["is_correct"] and pysh["is_correct"]:
            notes = "🎉 PyShort better"
        elif time_ratio > 2:
            notes = f"🚀 {time_ratio:.1f}x faster"

        print(
            f"| {qid:2d} | {orig_status:^8} | {pysh_status:^7} | {time_delta:+5d}ms | {token_delta:+7d} | {notes} |"
        )


def analyze_token_efficiency(original: list[dict], pyshort: list[dict]):
    """Analyze token efficiency"""

    orig_prompt_total = sum(r["prompt_tokens"] for r in original)
    pysh_prompt_total = sum(r["prompt_tokens"] for r in pyshort)

    orig_completion_total = sum(r["completion_tokens"] for r in original)
    pysh_completion_total = sum(r["completion_tokens"] for r in pyshort)

    print("\n📊 Token Breakdown:")
    print("\n  Prompt Tokens:")
    print(f"    Original:     {orig_prompt_total:,}")
    print(f"    PyShorthand:  {pysh_prompt_total:,}")
    print(f"    Reduction:    {100*(orig_prompt_total - pysh_prompt_total)/orig_prompt_total:.1f}%")

    print("\n  Completion Tokens:")
    print(f"    Original:     {orig_completion_total:,}")
    print(f"    PyShorthand:  {pysh_completion_total:,}")
    print(
        f"    Difference:   {100*(pysh_completion_total - orig_completion_total)/orig_completion_total:+.1f}%"
    )

    print("\n💰 Cost Efficiency (assuming $3/1M prompt, $15/1M completion):")
    orig_cost = (orig_prompt_total * 3 + orig_completion_total * 15) / 1_000_000
    pysh_cost = (pysh_prompt_total * 3 + pysh_completion_total * 15) / 1_000_000

    print(f"    Original:     ${orig_cost:.4f}")
    print(f"    PyShorthand:  ${pysh_cost:.4f}")
    print(
        f"    Savings:      ${orig_cost - pysh_cost:.4f} ({100*(orig_cost - pysh_cost)/orig_cost:.1f}%)"
    )


def identify_improvements(by_question: dict):
    """Identify areas where PyShorthand could improve"""

    print("\n🔍 Questions where PyShorthand failed but Original succeeded:\n")

    failures = []
    for qid in sorted(by_question.keys()):
        orig = by_question[qid]["original"]
        pysh = by_question[qid]["pyshorthand"]

        if orig and pysh and orig["is_correct"] and not pysh["is_correct"]:
            failures.append((qid, pysh))

    if not failures:
        print("  None! 🎉 PyShorthand matched or exceeded original on all questions.")
    else:
        for qid, result in failures:
            print(f"  Q{qid}: {result['answer'][:100]}...")

    print("\n💡 Recommendations based on failures:")

    if failures:
        # Map questions to what they test
        if any(qid in [14, 15] for qid, _ in failures):
            print("  - Consider capturing more nested structure information")
        if any(qid in [17, 18, 19, 20] for qid, _ in failures):
            print("  - Implementation details are expected to be missing (by design)")
        if any(qid in [1, 2, 3, 4, 5] for qid, _ in failures):
            print("  - ⚠️ Basic structural info is missing - this should be fixed!")
        if any(qid in [6, 7, 8, 9, 10] for qid, _ in failures):
            print("  - ⚠️ Signature info is missing - this should be fixed!")
    else:
        print(
            "  - PyShorthand is performing excellently! No improvements needed for structural/architectural questions."
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <results_file.json>")
        sys.exit(1)

    results_file = sys.argv[1]
    results = load_results(results_file)

    print("\n" + "=" * 80)
    print("PYSHORTHAND A/B TEST ANALYSIS")
    print("=" * 80)
    print(f"\nAnalyzing: {results_file}")
    print(f"Total tests: {len(results)}")

    analyze_results(results)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
