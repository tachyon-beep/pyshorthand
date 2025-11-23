#!/usr/bin/env python3
"""
Multi-Model A/B Testing Framework

Tests PyShorthand with multiple models to see if different models
handle compressed code differently.

Hypothesis: Newer models (Sonnet 4.5) might be better at "filling in gaps"
from incomplete information, or conversely, older models might benefit more
from the conciseness.
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")

# Import from the main framework
sys.path.insert(0, str(Path(__file__).parent))
from ab_test_framework import TestResult, load_test_suite, save_results


class MultiModelClient:
    """Client that can test with multiple models"""

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = model

    def ask_question(self, system_prompt: str, question: str, code_context: str) -> dict[str, Any]:
        """Ask a question given code context"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        full_prompt = f"""You are a code comprehension expert. Answer the following question about the provided codebase.

CODE CONTEXT:
```
{code_context}
```

QUESTION:
{question}

Provide a clear, concise answer. If you cannot determine the answer from the provided code, state "Cannot determine from provided code."
"""

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt},
            ],
        }

        start_time = time.time()
        response = requests.post(self.base_url, headers=headers, json=payload)
        response_time_ms = int((time.time() - start_time) * 1000)

        if response.status_code != 200:
            raise Exception(f"API error: {response.status_code} - {response.text}")

        result = response.json()

        return {
            "answer": result["choices"][0]["message"]["content"],
            "response_time_ms": response_time_ms,
            "prompt_tokens": result["usage"]["prompt_tokens"],
            "completion_tokens": result["usage"]["completion_tokens"],
            "total_tokens": result["usage"]["total_tokens"],
        }


def run_multimodel_experiment(
    original_code_path: str, pyshorthand_code_path: str, output_dir: str, models: list[str]
):
    """Run experiment with multiple models"""

    # Load API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment")

    # Load test suite
    questions = load_test_suite()
    print(f"\n📋 Loaded {len(questions)} test questions")
    print(f"🤖 Testing with {len(models)} models: {', '.join(models)}")

    # Load code contexts
    with open(original_code_path) as f:
        original_code = f.read()

    with open(pyshorthand_code_path) as f:
        pyshorthand_code = f.read()

    print(f"\n📄 Original code: {len(original_code)} chars")
    print(f"📄 PyShorthand code: {len(pyshorthand_code)} chars")
    print(f"📉 Size reduction: {100 * (1 - len(pyshorthand_code) / len(original_code)):.1f}%")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Run experiments for each model
    for model in models:
        print(f"\n{'='*80}")
        print(f"TESTING WITH MODEL: {model}")
        print(f"{'='*80}")

        client = MultiModelClient(api_key, model)
        all_results = []
        system_prompt = "You are an expert at reading and understanding code. Answer questions accurately and concisely."

        for question in questions:
            print(f"\n📝 Question {question.id} ({question.difficulty} - {question.category})")
            print(f"   {question.question}")

            # Test with original code
            try:
                result_orig = client.ask_question(system_prompt, question.question, original_code)
                is_correct_orig = question.correct_answer.lower() in result_orig["answer"].lower()

                test_result_orig = TestResult(
                    question_id=question.id,
                    format="original",
                    answer=result_orig["answer"],
                    response_time_ms=result_orig["response_time_ms"],
                    prompt_tokens=result_orig["prompt_tokens"],
                    completion_tokens=result_orig["completion_tokens"],
                    total_tokens=result_orig["total_tokens"],
                    is_correct=is_correct_orig,
                    completeness_score=1.0 if is_correct_orig else 0.5,
                    notes=f"model={model}",
                )
                all_results.append(test_result_orig)
                print(
                    f"   ✓ Original: {test_result_orig.response_time_ms}ms, "
                    f"{test_result_orig.total_tokens} tokens, "
                    f"{'✅' if is_correct_orig else '❌'}"
                )
            except Exception as e:
                print(f"   ❌ Original failed: {e}")
                all_results.append(
                    TestResult(
                        question_id=question.id,
                        format="original",
                        answer=f"ERROR: {e}",
                        response_time_ms=0,
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        is_correct=False,
                        completeness_score=0.0,
                        notes=f"model={model}",
                    )
                )

            time.sleep(1)  # Rate limiting

            # Test with PyShorthand
            try:
                result_pysh = client.ask_question(
                    system_prompt, question.question, pyshorthand_code
                )
                is_correct_pysh = question.correct_answer.lower() in result_pysh["answer"].lower()

                test_result_pysh = TestResult(
                    question_id=question.id,
                    format="pyshorthand",
                    answer=result_pysh["answer"],
                    response_time_ms=result_pysh["response_time_ms"],
                    prompt_tokens=result_pysh["prompt_tokens"],
                    completion_tokens=result_pysh["completion_tokens"],
                    total_tokens=result_pysh["total_tokens"],
                    is_correct=is_correct_pysh,
                    completeness_score=1.0 if is_correct_pysh else 0.5,
                    notes=f"model={model}",
                )
                all_results.append(test_result_pysh)
                print(
                    f"   ✓ PyShort:  {test_result_pysh.response_time_ms}ms, "
                    f"{test_result_pysh.total_tokens} tokens, "
                    f"{'✅' if is_correct_pysh else '❌'}"
                )
            except Exception as e:
                print(f"   ❌ PyShort failed: {e}")
                all_results.append(
                    TestResult(
                        question_id=question.id,
                        format="pyshorthand",
                        answer=f"ERROR: {e}",
                        response_time_ms=0,
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        is_correct=False,
                        completeness_score=0.0,
                        notes=f"model={model}",
                    )
                )

            time.sleep(1)  # Rate limiting

        # Save results for this model
        model_safe_name = model.replace("/", "_").replace(":", "_")
        results_file = f"{output_dir}/multimodel_{model_safe_name}_{timestamp}.json"
        save_results(all_results, results_file)

    print(f"\n{'='*80}")
    print("ALL EXPERIMENTS COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    models_to_test = [
        "anthropic/claude-3.5-sonnet",
        "anthropic/claude-sonnet-4.5",
    ]

    original_path = "test_repos/nanoGPT/model.py"
    pyshorthand_path = "realworld_nanogpt.pys"
    output_dir = "experiments/results"

    print("\n" + "=" * 80)
    print("PYSHORTHAND MULTI-MODEL A/B TESTING")
    print("=" * 80)

    run_multimodel_experiment(original_path, pyshorthand_path, output_dir, models_to_test)
