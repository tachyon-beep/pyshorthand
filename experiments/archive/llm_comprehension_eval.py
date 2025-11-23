#!/usr/bin/env python3
"""
LLM Comprehension Evaluation for PyShorthand.

Tests whether LLMs can better understand code architecture
from PyShorthand vs full Python source.
"""

import json
import os
import time
from pathlib import Path

from openai import OpenAI

from pyshort.decompiler.py2short import decompile_file

# Load API key
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    # Try loading from .env file
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if line.startswith("OPENROUTER_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    break

if not api_key:
    raise ValueError("OPENROUTER_API_KEY not found in environment or .env file")


# Initialize client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)


# Evaluation questions (architecture/design focused)
EVALUATION_QUESTIONS = [
    {
        "id": "q1_architecture",
        "question": "What are the main classes/components in this code and how do they relate to each other?",
        "focus": "architecture",
    },
    {
        "id": "q2_dependencies",
        "question": "What external frameworks or libraries does this code depend on?",
        "focus": "dependencies",
    },
    {
        "id": "q3_complexity",
        "question": "Which methods or functions are likely to be performance bottlenecks and why?",
        "focus": "complexity",
    },
    {
        "id": "q4_data_flow",
        "question": "Describe the data flow: what data structures are used and how is data transformed?",
        "focus": "data_flow",
    },
    {
        "id": "q5_purpose",
        "question": "In 2-3 sentences, what is the primary purpose of this code?",
        "focus": "purpose",
    },
]


def query_llm(prompt: str, code: str, model: str = "x-ai/grok-2-1212") -> tuple[str, float, int]:
    """Query the LLM with code and a question.

    Args:
        prompt: Question to ask about the code
        code: Code to analyze (Python or PyShorthand)
        model: Model to use

    Returns:
        Tuple of (response, time_taken, tokens_used)
    """
    messages = [{"role": "user", "content": f"{prompt}\n\nCode:\n```\n{code}\n```"}]

    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=500,  # Limit response length
    )
    elapsed = time.time() - start

    answer = response.choices[0].message.content
    tokens = response.usage.total_tokens if response.usage else 0

    return answer, elapsed, tokens


def evaluate_file(python_file: Path, questions: list[dict]) -> dict:
    """Evaluate comprehension for both Python and PyShorthand versions.

    Args:
        python_file: Path to Python source file
        questions: List of evaluation questions

    Returns:
        Evaluation results dictionary
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {python_file.name}")
    print(f"{'='*60}")

    # Read Python source
    with open(python_file) as f:
        python_code = f.read()

    # Generate PyShorthand
    pys_file = python_file.with_suffix(".pys")
    pyshorthand_code = decompile_file(str(python_file), str(pys_file))

    # Calculate token counts (rough estimate: 1 token ≈ 4 chars)
    python_tokens = len(python_code) // 4
    pys_tokens = len(pyshorthand_code) // 4
    compression = 1 - (pys_tokens / python_tokens)

    print(f"Python: {len(python_code)} chars, ~{python_tokens} tokens")
    print(f"PyShorthand: {len(pyshorthand_code)} chars, ~{pys_tokens} tokens")
    print(f"Compression: {compression:.1%}\n")

    results = {
        "file": str(python_file),
        "python_size": len(python_code),
        "pys_size": len(pyshorthand_code),
        "compression": f"{compression:.1%}",
        "questions": [],
    }

    # Ask each question for both versions
    for q in questions:
        print(f"Question ({q['id']}): {q['question']}")

        # Query with Python
        print("  Querying with Python code...")
        py_answer, py_time, py_tokens = query_llm(q["question"], python_code)

        # Query with PyShorthand
        print("  Querying with PyShorthand...")
        pys_answer, pys_time, pys_tokens = query_llm(q["question"], pyshorthand_code)

        # Calculate speedup
        speedup = py_time / pys_time if pys_time > 0 else 0

        q_result = {
            "question_id": q["id"],
            "question": q["question"],
            "focus": q["focus"],
            "python": {
                "answer": py_answer,
                "time": py_time,
                "tokens": py_tokens,
            },
            "pyshorthand": {
                "answer": pys_answer,
                "time": pys_time,
                "tokens": pys_tokens,
            },
            "speedup": f"{speedup:.2f}x",
            "token_reduction": f"{(1 - pys_tokens/py_tokens)*100:.1f}%",
        }

        results["questions"].append(q_result)

        print(f"  ✓ Python: {py_time:.2f}s, {py_tokens} tokens")
        print(f"  ✓ PyShorthand: {pys_time:.2f}s, {pys_tokens} tokens")
        print(
            f"  → Speedup: {speedup:.2f}x, Token reduction: {(1 - pys_tokens/py_tokens)*100:.1f}%"
        )
        print()

    return results


def compare_answers(results: dict) -> dict:
    """Analyze answer quality comparison.

    Args:
        results: Evaluation results

    Returns:
        Analysis dictionary
    """
    analysis = {
        "file": results["file"],
        "compression": results["compression"],
        "question_analysis": [],
    }

    for q in results["questions"]:
        py_answer = q["python"]["answer"]
        pys_answer = q["pyshorthand"]["answer"]

        # Basic similarity metrics
        py_len = len(py_answer)
        pys_len = len(pys_answer)

        # Check for key similarities
        similarity = {
            "question": q["question"],
            "python_length": py_len,
            "pys_length": pys_len,
            "speedup": q["speedup"],
            "token_reduction": q["token_reduction"],
        }

        analysis["question_analysis"].append(similarity)

    return analysis


def main():
    """Main evaluation entry point."""
    print("🔍 PyShorthand LLM Comprehension Evaluation")
    print("=" * 60)

    # Test files (use our validation repos)
    test_files = [
        Path("test_repos/nanoGPT/model.py"),
        Path("test_repos/minGPT/mingpt/model.py"),
        Path("test_repos/fastapi/fastapi/applications.py"),
    ]

    all_results = []

    for test_file in test_files:
        if not test_file.exists():
            print(f"⚠️  Skipping {test_file} (not found)")
            continue

        try:
            results = evaluate_file(test_file, EVALUATION_QUESTIONS)
            all_results.append(results)
        except Exception as e:
            print(f"❌ Error evaluating {test_file}: {e}")
            import traceback

            traceback.print_exc()

    # Save results
    output_file = Path("llm_comprehension_results.json")
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")

    # Generate summary
    print("\n📊 Summary Analysis\n")

    for result in all_results:
        print(f"File: {Path(result['file']).name}")
        print(f"  Compression: {result['compression']}")

        total_py_time = sum(q["python"]["time"] for q in result["questions"])
        total_pys_time = sum(q["pyshorthand"]["time"] for q in result["questions"])
        avg_speedup = total_py_time / total_pys_time if total_pys_time > 0 else 0

        total_py_tokens = sum(q["python"]["tokens"] for q in result["questions"])
        total_pys_tokens = sum(q["pyshorthand"]["tokens"] for q in result["questions"])
        token_reduction = (
            (1 - total_pys_tokens / total_py_tokens) * 100 if total_py_tokens > 0 else 0
        )

        print(f"  Avg speedup: {avg_speedup:.2f}x")
        print(f"  Token reduction: {token_reduction:.1f}%")
        print()


if __name__ == "__main__":
    main()
