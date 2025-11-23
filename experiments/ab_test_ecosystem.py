#!/usr/bin/env python3
"""
A/B Test: PyShorthand Ecosystem vs Full Code

Tests the progressive disclosure approach against sending full code.
Measures accuracy, token usage, and cost.
"""

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ab_test_framework import Question, load_test_suite

from pyshort.ecosystem.tools import CodebaseExplorer


@dataclass
class EcosystemResult:
    """Result from ecosystem-based answer."""

    question_id: int
    answer: str
    is_correct: bool
    tools_called: list[str]  # Which tools were used
    pyshorthand_tokens: int  # Base PyShorthand overview
    tool_tokens: int  # Additional tokens from tool calls
    total_tokens: int  # Total input tokens
    response_time_ms: int
    completion_tokens: int
    notes: str = ""


class EcosystemAgent:
    """
    Simulated agent that uses PyShorthand ecosystem intelligently.

    Strategy:
    1. Start with PyShorthand overview
    2. Analyze question type
    3. Selectively call tools based on what's needed
    """

    def __init__(self, pyshorthand_path: Path, codebase_path: Path):
        """Initialize agent with PyShorthand and code explorer.

        Args:
            pyshorthand_path: Path to .pys file
            codebase_path: Path to original Python code
        """
        # Load PyShorthand overview
        with open(pyshorthand_path) as f:
            self.pyshorthand = f.read()

        self.pyshorthand_tokens = len(self.pyshorthand.split())  # Rough estimate

        # Create explorer for on-demand lookups
        self.explorer = CodebaseExplorer(codebase_path)

        # OpenRouter API setup
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment")

    def answer_question(
        self, question: Question, model: str = "anthropic/claude-sonnet-4.5"
    ) -> EcosystemResult:
        """Answer a question using the ecosystem approach.

        Args:
            question: Question to answer
            model: Model to use for answering

        Returns:
            EcosystemResult with answer and metadata
        """
        import requests

        # Build context based on question category
        context_parts = [self.pyshorthand]
        tools_called = []
        tool_tokens = 0

        # Decide which tools to call based on question category
        if question.category in ("signature", "structure"):
            # Need class details
            # Extract class name from question (simplified - would use smarter extraction)
            for class_name in ["GPT", "Block", "CausalSelfAttention", "MLP", "LayerNorm"]:
                if class_name in question.question:
                    details = self.explorer.get_class_details(class_name, expand_nested=True)
                    if details:
                        context_parts.append(f"\n\n# Details for {class_name}:\n{details}")
                        tools_called.append(f"get_class_details({class_name})")
                        tool_tokens += len(details.split())
                    break

        elif question.category in ("implementation", "architecture"):
            # Need implementations
            # Extract method name from question (simplified)
            for method in [
                "forward",
                "generate",
                "configure_optimizers",
                "_init_weights",
                "from_pretrained",
            ]:
                if method in question.question:
                    # Try to find which class
                    for class_name in ["GPT", "Block", "CausalSelfAttention", "MLP"]:
                        impl = self.explorer.get_implementation(
                            f"{class_name}.{method}", include_context=False
                        )
                        if impl:
                            context_parts.append(
                                f"\n\n# Implementation of {class_name}.{method}:\n{impl}"
                            )
                            tools_called.append(f"get_implementation({class_name}.{method})")
                            tool_tokens += len(impl.split())
                            break
                    break

        # Build final context
        context = "\n".join(context_parts)

        # Create prompt
        prompt = f"""You are analyzing a Python codebase. Answer the following question based on the provided code context.

Code context (PyShorthand overview + on-demand details):
{context}

Question: {question.question}

Provide a clear, concise answer."""

        # Call API
        start_time = time.time()

        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/yourusername/pyshorthand",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=30,
            )

            response.raise_for_status()
            result = response.json()

            answer = result["choices"][0]["message"]["content"]
            response_time_ms = int((time.time() - start_time) * 1000)

            # Evaluate correctness (simple string matching)
            answer_lower = answer.lower()
            correct_lower = question.correct_answer.lower()
            is_correct = correct_lower in answer_lower

            # Calculate tokens
            # Note: Using word count as approximation (actual would use tiktoken)
            prompt_tokens = len(prompt.split())
            completion_tokens = len(answer.split())

            return EcosystemResult(
                question_id=question.id,
                answer=answer,
                is_correct=is_correct,
                tools_called=tools_called,
                pyshorthand_tokens=self.pyshorthand_tokens,
                tool_tokens=tool_tokens,
                total_tokens=prompt_tokens,
                response_time_ms=response_time_ms,
                completion_tokens=completion_tokens,
                notes=f"model={model}, tools={len(tools_called)}",
            )

        except requests.exceptions.RequestException as e:
            return EcosystemResult(
                question_id=question.id,
                answer=f"Error: {e}",
                is_correct=False,
                tools_called=tools_called,
                pyshorthand_tokens=self.pyshorthand_tokens,
                tool_tokens=tool_tokens,
                total_tokens=0,
                response_time_ms=0,
                completion_tokens=0,
                notes=f"API error: {e}",
            )


def run_ecosystem_test(
    model: str = "anthropic/claude-sonnet-4.5",
) -> list[EcosystemResult]:
    """Run the ecosystem A/B test.

    Args:
        model: Model to test with

    Returns:
        List of results for all questions
    """
    # Load test suite
    questions = load_test_suite()

    # Create agent
    pyshorthand_path = Path(__file__).parent.parent / "realworld_nanogpt.pys"
    codebase_path = Path(__file__).parent / "nanogpt_sample.py"

    agent = EcosystemAgent(pyshorthand_path, codebase_path)

    # Run tests
    results = []
    print(f"\nRunning ecosystem test with {model}...")
    print(f"Questions: {len(questions)}")
    print()

    for i, question in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] Q{question.id}: {question.question[:60]}...")

        result = agent.answer_question(question, model=model)
        results.append(result)

        status = "✓" if result.is_correct else "✗"
        print(
            f"  {status} Tools: {', '.join(result.tools_called) if result.tools_called else 'none'}"
        )
        print(
            f"  Tokens: {result.pyshorthand_tokens} (base) + {result.tool_tokens} (tools) = {result.total_tokens}"
        )
        print()

        # Rate limiting
        time.sleep(1)

    return results


def save_results(results: list[EcosystemResult], model: str):
    """Save results to JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ecosystem_{model.replace('/', '_')}_{timestamp}.json"
    filepath = Path(__file__).parent / "results" / filename

    filepath.parent.mkdir(exist_ok=True)

    with open(filepath, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    print(f"Results saved to: {filepath}")
    return filepath


def analyze_results(results: list[EcosystemResult]):
    """Analyze and print results summary."""
    print("\n" + "=" * 80)
    print("ECOSYSTEM TEST RESULTS")
    print("=" * 80)
    print()

    # Overall accuracy
    correct = sum(1 for r in results if r.is_correct)
    accuracy = (correct / len(results)) * 100

    print(f"Accuracy: {correct}/{len(results)} ({accuracy:.1f}%)")
    print()

    # Token usage
    avg_pyshorthand = sum(r.pyshorthand_tokens for r in results) / len(results)
    avg_tool = sum(r.tool_tokens for r in results) / len(results)
    avg_total = sum(r.total_tokens for r in results) / len(results)

    print("Average tokens per question:")
    print(f"  PyShorthand base: {avg_pyshorthand:.0f}")
    print(f"  Tool calls: {avg_tool:.0f}")
    print(f"  Total: {avg_total:.0f}")
    print()

    # Tool usage breakdown
    tool_counts = {}
    for r in results:
        for tool in r.tools_called:
            tool_counts[tool] = tool_counts.get(tool, 0) + 1

    print("Tool usage:")
    if tool_counts:
        for tool, count in sorted(tool_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {tool}: {count} times")
    else:
        print("  (no tools called)")
    print()

    # Comparison to baseline (from previous test)
    print("Comparison to baselines:")
    print("  Full code: 35% accuracy, 5,348 tokens")
    print("  PyShorthand v1.5: 35% accuracy, 894 tokens")
    print(f"  Ecosystem: {accuracy:.1f}% accuracy, {avg_total:.0f} tokens")
    print()

    savings_vs_full = ((5348 - avg_total) / 5348) * 100
    print(f"Savings vs full code: {savings_vs_full:.1f}%")
    print()


def main():
    """Run ecosystem A/B test."""
    import argparse

    parser = argparse.ArgumentParser(description="Run PyShorthand ecosystem A/B test")
    parser.add_argument(
        "--model",
        default="anthropic/claude-sonnet-4.5",
        help="Model to test with",
    )
    args = parser.parse_args()

    print()
    print("╔═══════════════════════════════════════════════════════════════════════════╗")
    print("║         PyShorthand Ecosystem A/B Test                                   ║")
    print("╚═══════════════════════════════════════════════════════════════════════════╝")
    print()

    # Run test
    results = run_ecosystem_test(model=args.model)

    # Analyze results
    analyze_results(results)

    # Save results
    save_results(results, args.model)


if __name__ == "__main__":
    main()
