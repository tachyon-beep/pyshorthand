"""Performance benchmarks for PyShorthand parser.

Validates the "<1s for 10K lines" performance claim.
"""

import time
from pathlib import Path

from pyshort.core.parser import parse_file
from pyshort.core.tokenizer import Tokenizer
from pyshort.core.parser import Parser


def generate_synthetic_file(num_entities: int) -> str:
    """Generate synthetic PyShorthand file with N entities.

    Each entity has:
    - 10 state variables
    - 5 methods
    - Total: ~15 lines per entity
    """
    lines = [
        "# [M:SyntheticBenchmark] [Role:Core] [Risk:Low]",
        "# [N:batch] [D:dim] [H:hidden]",
        ""
    ]

    for i in range(num_entities):
        entity_num = i + 1
        lines.extend([
            f"[C:Entity{entity_num}]",
            f"  weights_{entity_num} ∈ f32[N, D]@GPU",
            f"  bias_{entity_num} ∈ f32[D]@GPU",
            f"  hidden_{entity_num} ∈ f32[N, H]@GPU",
            f"  mask_{entity_num} ∈ bool[N]@CPU",
            f"  learning_rate_{entity_num} ∈ f32@CPU",
            f"  momentum_{entity_num} ∈ f32@CPU",
            f"  epsilon_{entity_num} ∈ f32@CPU",
            f"  iterations_{entity_num} ∈ i32@CPU",
            f"  active_{entity_num} ∈ bool@CPU",
            f"  name_{entity_num} ∈ str@CPU",
            "",
        ])

    return "\n".join(lines)


def measure_parse_time(source: str, description: str) -> dict:
    """Measure parsing time for given source."""
    # Measure tokenization
    start = time.perf_counter()
    tokenizer = Tokenizer(source)
    tokens = tokenizer.tokenize()
    tokenize_time = time.perf_counter() - start

    # Measure parsing
    start = time.perf_counter()
    parser = Parser(tokens)
    ast = parser.parse()
    parse_time = time.perf_counter() - start

    total_time = tokenize_time + parse_time

    return {
        "description": description,
        "lines": source.count('\n') + 1,
        "tokens": len(tokens),
        "entities": len(ast.entities),
        "tokenize_time": tokenize_time,
        "parse_time": parse_time,
        "total_time": total_time,
        "lines_per_sec": (source.count('\n') + 1) / total_time if total_time > 0 else 0,
    }


def run_benchmarks():
    """Run performance benchmarks."""
    print("=" * 80)
    print("PYSHORTHAND PARSER PERFORMANCE BENCHMARKS")
    print("=" * 80)
    print()

    benchmarks = [
        (10, "Small file (10 entities, ~150 lines)"),
        (100, "Medium file (100 entities, ~1,500 lines)"),
        (500, "Large file (500 entities, ~7,500 lines)"),
        (667, "Extra large file (667 entities, ~10,000 lines)"),
        (1000, "Huge file (1000 entities, ~15,000 lines)"),
    ]

    results = []

    for num_entities, description in benchmarks:
        source = generate_synthetic_file(num_entities)
        result = measure_parse_time(source, description)
        results.append(result)

        print(f"\n{description}")
        print(f"  Lines: {result['lines']:,}")
        print(f"  Tokens: {result['tokens']:,}")
        print(f"  Entities: {result['entities']:,}")
        print(f"  Tokenize time: {result['tokenize_time']:.3f}s")
        print(f"  Parse time: {result['parse_time']:.3f}s")
        print(f"  Total time: {result['total_time']:.3f}s")
        print(f"  Speed: {result['lines_per_sec']:,.0f} lines/sec")

        # Check if meets performance target
        if result['lines'] >= 10000:
            target_met = result['total_time'] < 1.0
            status = "✓ PASS" if target_met else "✗ FAIL"
            print(f"  Performance target (<1s for 10K lines): {status}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Find 10K line result
    for result in results:
        if result['lines'] >= 10000:
            meets_spec = result['total_time'] < 1.0
            print(f"\n10K+ line performance:")
            print(f"  Lines: {result['lines']:,}")
            print(f"  Time: {result['total_time']:.3f}s")
            print(f"  Target: <1.0s")
            print(f"  Status: {'✓ MEETS SPEC' if meets_spec else '✗ FAILS SPEC'}")
            print(f"  Margin: {((1.0 - result['total_time']) / 1.0 * 100):.1f}%")
            break

    # Average speed
    avg_speed = sum(r['lines_per_sec'] for r in results) / len(results)
    print(f"\nAverage speed: {avg_speed:,.0f} lines/sec")
    print()


if __name__ == "__main__":
    run_benchmarks()
