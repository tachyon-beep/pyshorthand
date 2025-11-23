#!/usr/bin/env python3
"""Quick test of ecosystem with just 3 questions (one per difficulty tier)."""

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ab_test_framework import load_test_suite

from pyshort.ecosystem.tools import CodebaseExplorer

# Test with just 3 questions
TEST_QUESTIONS = [1, 8, 20]  # Easy, Medium, Hard


def quick_test():
    """Run quick ecosystem test with 3 questions."""
    import requests

    questions = load_test_suite()
    test_questions = [q for q in questions if q.id in TEST_QUESTIONS]

    print(f"Quick Ecosystem Test - {len(test_questions)} questions")
    print("=" * 80)
    print()

    # Load PyShorthand overview
    with open(Path(__file__).parent.parent / "realworld_nanogpt.pys") as f:
        pyshorthand = f.read()

    pyshorthand_tokens = len(pyshorthand.split())

    # Create explorer
    explorer = CodebaseExplorer(Path(__file__).parent / "nanogpt_sample.py")

    api_key = os.environ.get("OPENROUTER_API_KEY")

    results = []

    for q in test_questions:
        print(f"Q{q.id} ({q.difficulty} - {q.category}):")
        print(f"  {q.question}")
        print()

        # Build context
        context_parts = [pyshorthand]
        tools_called = []
        tool_tokens = 0

        # Decide which tools to call based on category
        if q.category in ("signature", "structure"):
            # Try get_class_details
            for class_name in [
                "GPT",
                "Block",
                "CausalSelfAttention",
                "MLP",
                "LayerNorm",
                "GPTConfig",
            ]:
                if class_name in q.question:
                    details = explorer.get_class_details(class_name, expand_nested=True)
                    if details:
                        context_parts.append(f"\n\n# Details for {class_name}:\n{details}")
                        tools_called.append(f"get_class_details({class_name})")
                        tool_tokens += len(details.split())
                        print(f"  🔍 Called: get_class_details({class_name})")
                    break

        elif q.category in ("implementation", "architecture"):
            # Try get_implementation
            for method in [
                "forward",
                "generate",
                "configure_optimizers",
                "_init_weights",
                "from_pretrained",
            ]:
                if method in q.question:
                    for class_name in ["GPT", "Block", "CausalSelfAttention", "MLP"]:
                        impl = explorer.get_implementation(
                            f"{class_name}.{method}", include_context=False
                        )
                        if impl:
                            context_parts.append(
                                f"\n\n# Implementation of {class_name}.{method}:\n{impl}"
                            )
                            tools_called.append(f"get_implementation({class_name}.{method})")
                            tool_tokens += len(impl.split())
                            print(f"  🔍 Called: get_implementation({class_name}.{method})")
                            break
                    break

        # Build prompt
        context = "\n".join(context_parts)
        prompt = f"""You are analyzing a Python codebase. Answer the following question based on the provided code context.

Code context (PyShorthand overview + on-demand details):
{context}

Question: {q.question}

Provide a clear, concise answer."""

        total_tokens = len(prompt.split())

        print(
            f"  📊 Tokens: {pyshorthand_tokens} (base) + {tool_tokens} (tools) = {total_tokens} total"
        )
        print("  📡 Calling API...")

        # Call API
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "anthropic/claude-sonnet-4.5",
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=30,
            )

            response.raise_for_status()
            result = response.json()

            answer = result["choices"][0]["message"]["content"]

            # Simple correctness check
            answer_lower = answer.lower()
            correct_lower = q.correct_answer.lower()
            is_correct = correct_lower in answer_lower

            status = "✅" if is_correct else "❌"
            print(f"  {status} Answer: {answer[:100]}...")
            print()

            results.append(
                {
                    "question_id": q.id,
                    "difficulty": q.difficulty,
                    "category": q.category,
                    "tools_called": tools_called,
                    "pyshorthand_tokens": pyshorthand_tokens,
                    "tool_tokens": tool_tokens,
                    "total_tokens": total_tokens,
                    "is_correct": is_correct,
                    "answer": answer,
                }
            )

        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append(
                {
                    "question_id": q.id,
                    "error": str(e),
                }
            )

        time.sleep(1)  # Rate limiting

    # Summary
    print("=" * 80)
    print("QUICK TEST SUMMARY")
    print("=" * 80)
    print()

    correct = sum(1 for r in results if r.get("is_correct"))
    print(f"Accuracy: {correct}/{len(results)} ({correct/len(results)*100:.0f}%)")

    if results:
        avg_tokens = sum(r.get("total_tokens", 0) for r in results) / len(results)
        print(f"Avg tokens: {avg_tokens:.0f}")
        print()

        for r in results:
            if "error" not in r:
                status = "✅" if r["is_correct"] else "❌"
                print(
                    f"  {status} Q{r['question_id']} ({r['difficulty']}): {r['total_tokens']} tokens, {len(r['tools_called'])} tools"
                )

    print()
    print("If this looks good, run the full test with:")
    print("  python experiments/ab_test_ecosystem.py")


if __name__ == "__main__":
    quick_test()
