#!/usr/bin/env python3
"""Quick LLM comprehension demo for PyShorthand."""

import json
import os
import time
from pathlib import Path

from openai import OpenAI

from pyshort.decompiler.py2short import decompile_file

# Load API key from .env
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            if line.startswith("OPENROUTER_API_KEY="):
                api_key = line.split("=", 1)[1].strip()
                os.environ["OPENROUTER_API_KEY"] = api_key
                break

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


def query_llm(prompt: str, code: str, max_retries: int = 3):
    """Query LLM with code."""
    messages = [{"role": "user", "content": f"{prompt}\n\nCode:\n```\n{code}\n```"}]

    for attempt in range(max_retries):
        try:
            start = time.time()
            response = client.chat.completions.create(
                model="x-ai/grok-4.1-fast",
                messages=messages,
                max_tokens=300,
            )
            elapsed = time.time() - start

            return response.choices[0].message.content, elapsed, response.usage.total_tokens

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2**attempt  # Exponential backoff
                print(f"  ⚠️  Error: {type(e).__name__}, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  ❌ Failed after {max_retries} attempts: {e}")
                raise


def main():
    print("🔍 PyShorthand vs Python - LLM Comprehension Demo\n")

    # Use nanoGPT model.py as test case
    test_file = Path("test_repos/nanoGPT/model.py")

    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        print("   Run validate_repos.py first to clone test repositories")
        return

    # Read Python
    with open(test_file) as f:
        python_code = f.read()

    # Generate PyShorthand
    pys_file = test_file.with_suffix(".pys")
    if not pys_file.exists():
        print("Generating PyShorthand...")
        pyshorthand_code = decompile_file(str(test_file), str(pys_file))
    else:
        with open(pys_file) as f:
            pyshorthand_code = f.read()

    # Show sizes
    py_tokens = len(python_code) // 4
    pys_tokens = len(pyshorthand_code) // 4
    compression = 1 - (pys_tokens / py_tokens)

    print(f"Test file: {test_file.name}")
    print(f"Python: {len(python_code)} chars, ~{py_tokens} tokens")
    print(f"PyShorthand: {len(pyshorthand_code)} chars, ~{pys_tokens} tokens")
    print(f"Compression: {compression:.1%}\n")

    # Two focused questions
    questions = [
        "What are the main classes in this code and what do they represent?",
        "What deep learning framework is this code using and what model architecture does it implement?",
    ]

    results = []

    for i, question in enumerate(questions, 1):
        print(f"Question {i}: {question}\n")

        # Python version
        print("  [Python] Querying LLM...")
        py_answer, py_time, py_tokens = query_llm(question, python_code)
        print(f"  ✓ Response in {py_time:.2f}s using {py_tokens} tokens")
        print(f"    Answer: {py_answer[:150]}{'...' if len(py_answer) > 150 else ''}\n")

        # PyShorthand version
        print("  [PyShorthand] Querying LLM...")
        pys_answer, pys_time, pys_tokens = query_llm(question, pyshorthand_code)
        print(f"  ✓ Response in {pys_time:.2f}s using {pys_tokens} tokens")
        print(f"    Answer: {pys_answer[:150]}{'...' if len(pys_answer) > 150 else ''}\n")

        # Compare
        speedup = py_time / pys_time if pys_time > 0 else 0
        token_reduction = (1 - pys_tokens / py_tokens) * 100

        print("  📊 Comparison:")
        print(f"     Speed: {speedup:.2f}x faster with PyShorthand")
        print(f"     Tokens: {token_reduction:.1f}% reduction")
        print(
            f"     Quality: {'Similar' if abs(len(py_answer) - len(pys_answer)) < 100 else 'Different'}\n"
        )

        results.append(
            {
                "question": question,
                "python": {"answer": py_answer, "time": py_time, "tokens": py_tokens},
                "pyshorthand": {"answer": pys_answer, "time": pys_time, "tokens": pys_tokens},
                "speedup": f"{speedup:.2f}x",
                "token_reduction": f"{token_reduction:.1f}%",
            }
        )

    # Save results
    with open("llm_demo_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 60)
    print("✅ Demo complete! Results saved to llm_demo_results.json")


if __name__ == "__main__":
    main()
