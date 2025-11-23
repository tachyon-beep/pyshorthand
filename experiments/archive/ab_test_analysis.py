"""A/B Testing Analysis for PyShorthand v1.4 Compression.

Compares Python source files with PyShorthand decompiled output
to measure compression ratios and validate semantic preservation.
"""

import os
from typing import Dict, Tuple


def count_tokens(text: str) -> int:
    """Estimate token count (simple whitespace-based)."""
    return len(text.split())


def analyze_file(filepath: str) -> Dict[str, int]:
    """Analyze a file and return metrics."""
    with open(filepath, 'r') as f:
        content = f.read()

    lines = content.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    non_comment_lines = [line for line in non_empty_lines if not line.strip().startswith('#')]

    return {
        'chars': len(content),
        'lines': len(lines),
        'non_empty_lines': len(non_empty_lines),
        'code_lines': len(non_comment_lines),
        'tokens': count_tokens(content),
    }


def compare_files(py_file: str, pys_file: str) -> Tuple[Dict, Dict, Dict]:
    """Compare Python and PyShorthand files."""
    py_metrics = analyze_file(py_file)
    pys_metrics = analyze_file(pys_file)

    ratios = {
        'chars': pys_metrics['chars'] / py_metrics['chars'],
        'lines': pys_metrics['lines'] / py_metrics['lines'],
        'non_empty_lines': pys_metrics['non_empty_lines'] / py_metrics['non_empty_lines'],
        'code_lines': pys_metrics['code_lines'] / py_metrics['code_lines'],
        'tokens': pys_metrics['tokens'] / py_metrics['tokens'],
    }

    return py_metrics, pys_metrics, ratios


def print_comparison(name: str, py_metrics: Dict, pys_metrics: Dict, ratios: Dict):
    """Print formatted comparison."""
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")
    print(f"\n{'Metric':<20} {'Python':<15} {'PyShorthand':<15} {'Ratio':<10} {'Reduction'}")
    print(f"{'-'*70}")

    for metric in ['chars', 'lines', 'non_empty_lines', 'code_lines', 'tokens']:
        py_val = py_metrics[metric]
        pys_val = pys_metrics[metric]
        ratio = ratios[metric]
        reduction = (1 - ratio) * 100

        print(f"{metric:<20} {py_val:<15,} {pys_val:<15,} {ratio:>6.1%}    {reduction:>5.1f}%")


def print_summary(all_results: Dict[str, Tuple]):
    """Print overall summary."""
    print(f"\n\n{'='*70}")
    print(f"  OVERALL SUMMARY")
    print(f"{'='*70}\n")

    total_py = {'chars': 0, 'lines': 0, 'tokens': 0}
    total_pys = {'chars': 0, 'lines': 0, 'tokens': 0}

    for name, (py_metrics, pys_metrics, _) in all_results.items():
        for key in total_py:
            total_py[key] += py_metrics[key]
            total_pys[key] += pys_metrics[key]

    print(f"Total Python:")
    print(f"  Characters: {total_py['chars']:,}")
    print(f"  Lines:      {total_py['lines']:,}")
    print(f"  Tokens:     {total_py['tokens']:,}")

    print(f"\nTotal PyShorthand v1.4:")
    print(f"  Characters: {total_pys['chars']:,}")
    print(f"  Lines:      {total_pys['lines']:,}")
    print(f"  Tokens:     {total_pys['tokens']:,}")

    char_ratio = total_pys['chars'] / total_py['chars']
    line_ratio = total_pys['lines'] / total_py['lines']
    token_ratio = total_pys['tokens'] / total_py['tokens']

    print(f"\nCompression Ratios:")
    print(f"  Character reduction: {(1-char_ratio)*100:.1f}% (ratio: {char_ratio:.1%})")
    print(f"  Line reduction:      {(1-line_ratio)*100:.1f}% (ratio: {line_ratio:.1%})")
    print(f"  Token reduction:     {(1-token_ratio)*100:.1f}% (ratio: {token_ratio:.1%})")

    print(f"\n{'='*70}")


if __name__ == '__main__':
    test_files = [
        ('FastAPI Application', 'ab_test_fastapi.py', 'ab_test_fastapi.pys'),
        ('Neural Network', 'ab_test_neural_net.py', 'ab_test_neural_net.pys'),
        ('Algorithms', 'ab_test_algorithms.py', 'ab_test_algorithms.pys'),
    ]

    all_results = {}

    for name, py_file, pys_file in test_files:
        if os.path.exists(py_file) and os.path.exists(pys_file):
            py_metrics, pys_metrics, ratios = compare_files(py_file, pys_file)
            all_results[name] = (py_metrics, pys_metrics, ratios)
            print_comparison(name, py_metrics, pys_metrics, ratios)

    if all_results:
        print_summary(all_results)
