#!/usr/bin/env python3
"""
Test PyShorthand Ecosystem with GPT-5.1 + Reasoning Mode

GPT-5.1's reasoning should be much better at deciding which tools to call!
"""

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ab_test_framework import Question, load_test_suite

from pyshort.ecosystem.tools import CodebaseExplorer


@dataclass
class ReasoningResult:
    """Result from GPT-5.1 reasoning-based test."""

    question_id: int
    question: str
    difficulty: str
    category: str
    answer: str
    is_correct: bool
    tools_called: list[str]
    pyshorthand_tokens: int
    tool_tokens: int
    total_tokens: int
    reasoning_summary: str
    response_time_ms: int


class GPT5ReasoningAgent:
    """
    Agent using GPT-5.1's reasoning mode to decide which tools to call.

    The agent sees PyShorthand + available tools and reasons about:
    1. Can I answer from PyShorthand alone?
    2. If not, which tool(s) should I call?
    3. Do I need multiple tools?
    """

    def __init__(self, pyshorthand_path: Path, codebase_path: Path):
        """Initialize agent."""
        # Load PyShorthand overview
        with open(pyshorthand_path) as f:
            self.pyshorthand = f.read()

        self.pyshorthand_tokens = len(self.pyshorthand.split())

        # Create explorer for tool calls
        self.explorer = CodebaseExplorer(codebase_path)

        # OpenRouter client with GPT-5.1
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found")

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model = "openai/gpt-5.1"

    def answer_question(self, question: Question) -> ReasoningResult:
        """Answer question using GPT-5.1 reasoning to select tools."""

        print(f"\nQ{question.id} ({question.difficulty} - {question.category}):")
        print(f"  {question.question}")
        print()

        # Build initial prompt with PyShorthand + tool descriptions
        system_prompt = """You are analyzing a Python codebase using PyShorthand (a compressed code representation).

You have access to these tools for on-demand details:

1. **get_class_details(class_name)** - Get detailed class information
   - Shows method signatures with parameter/return types
   - Shows state variable types
   - Expands nested structures (ModuleDict, etc.)
   - Cost: ~200-400 tokens
   - Use when: Need exact types, signatures, or structure details

2. **get_implementation(ClassName.method_name)** - Get full Python code
   - Shows actual implementation with logic
   - Cost: ~300-500 tokens
   - Use when: Need to understand what code DOES (algorithms, control flow, specific operations)

3. **search_usage(symbol)** - Find where a class/method is used
   - Cost: ~50-100 tokens
   - Use when: Need to understand dependencies/usage

STRATEGY:
1. First try to answer from PyShorthand alone (it's free!)
2. If you need more detail, reason about which tool(s) to call
3. Be selective - each tool costs tokens
4. For implementation questions, you'll likely need get_implementation
5. For signature/type questions, get_class_details is usually enough

Format your response as:

REASONING: [Your reasoning about whether you need tools and which ones]
TOOL_CALLS: [List of tool calls, one per line, e.g. "get_class_details(GPT)" or "none"]
ANSWER: [Your final answer to the question]
"""

        user_prompt = f"""PyShorthand Overview:
```
{self.pyshorthand}
```

Question: {question.question}

Think carefully about what information you need to answer this question."""

        # First API call with reasoning
        start_time = time.time()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                extra_body={"reasoning": {"enabled": True}},
            )

            response_time_ms = int((time.time() - start_time) * 1000)

            # Extract response
            content = response.choices[0].message.content
            reasoning_details = getattr(response.choices[0].message, "reasoning_details", None)

            # Extract reasoning summary
            reasoning_summary = "No reasoning provided"
            if reasoning_details:
                reasoning_summary = str(reasoning_details)[:200] + "..."

            print(f"  🧠 Reasoning: {reasoning_summary[:100]}...")

            # Parse response for tool calls
            tools_called = []
            tool_tokens = 0
            tool_outputs = []

            if "TOOL_CALLS:" in content:
                tool_section = content.split("TOOL_CALLS:")[1].split("ANSWER:")[0].strip()

                if "none" not in tool_section.lower():
                    # Parse tool calls
                    for line in tool_section.split("\n"):
                        line = line.strip()
                        if not line or line == "none":
                            continue

                        # Try to parse tool call
                        if "get_class_details(" in line:
                            class_name = line.split("get_class_details(")[1].split(")")[0].strip()
                            details = self.explorer.get_class_details(
                                class_name, expand_nested=True
                            )
                            if details:
                                tool_outputs.append(
                                    f"# get_class_details({class_name}):\n{details}"
                                )
                                tools_called.append(f"get_class_details({class_name})")
                                tool_tokens += len(details.split())
                                print(f"  🔍 Called: get_class_details({class_name})")

                        elif "get_implementation(" in line:
                            target = line.split("get_implementation(")[1].split(")")[0].strip()
                            impl = self.explorer.get_implementation(target, include_context=False)
                            if impl:
                                tool_outputs.append(f"# get_implementation({target}):\n{impl}")
                                tools_called.append(f"get_implementation({target})")
                                tool_tokens += len(impl.split())
                                print(f"  🔍 Called: get_implementation({target})")

                        elif "search_usage(" in line:
                            symbol = line.split("search_usage(")[1].split(")")[0].strip()
                            usages = self.explorer.search_usage(symbol)
                            if usages:
                                usage_text = "\n".join(usages)
                                tool_outputs.append(f"# search_usage({symbol}):\n{usage_text}")
                                tools_called.append(f"search_usage({symbol})")
                                tool_tokens += len(usage_text.split())
                                print(f"  🔍 Called: search_usage({symbol})")

            # If tools were called, make second API call with results
            final_answer = content
            if tool_outputs:
                print("  💭 Reasoning with tool results...")

                tool_results = "\n\n".join(tool_outputs)

                # Prepare messages with preserved reasoning
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {
                        "role": "assistant",
                        "content": content,
                        "reasoning_details": reasoning_details if reasoning_details else None,
                    },
                    {
                        "role": "user",
                        "content": f"Here are the tool results:\n\n{tool_results}\n\nNow provide your final answer.",
                    },
                ]

                # Remove None reasoning_details if present
                if not reasoning_details:
                    del messages[2]["reasoning_details"]

                response2 = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    extra_body={"reasoning": {"enabled": True}},
                )

                final_answer = response2.choices[0].message.content

            # Extract final answer
            if "ANSWER:" in final_answer:
                answer = final_answer.split("ANSWER:")[1].strip()
            else:
                answer = final_answer

            # Evaluate correctness
            answer_lower = answer.lower()
            correct_lower = question.correct_answer.lower()
            is_correct = correct_lower in answer_lower

            # Calculate total tokens
            total_tokens = self.pyshorthand_tokens + tool_tokens

            status = "✅" if is_correct else "❌"
            print(
                f"  {status} Tokens: {self.pyshorthand_tokens} (base) + {tool_tokens} (tools) = {total_tokens}"
            )
            print(f"  {status} Answer: {answer[:100]}...")

            return ReasoningResult(
                question_id=question.id,
                question=question.question,
                difficulty=question.difficulty,
                category=question.category,
                answer=answer,
                is_correct=is_correct,
                tools_called=tools_called,
                pyshorthand_tokens=self.pyshorthand_tokens,
                tool_tokens=tool_tokens,
                total_tokens=total_tokens,
                reasoning_summary=reasoning_summary,
                response_time_ms=response_time_ms,
            )

        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback

            traceback.print_exc()

            return ReasoningResult(
                question_id=question.id,
                question=question.question,
                difficulty=question.difficulty,
                category=question.category,
                answer=f"Error: {e}",
                is_correct=False,
                tools_called=[],
                pyshorthand_tokens=self.pyshorthand_tokens,
                tool_tokens=0,
                total_tokens=self.pyshorthand_tokens,
                reasoning_summary=str(e),
                response_time_ms=0,
            )


def main():
    """Run GPT-5.1 reasoning test."""
    print()
    print("╔═══════════════════════════════════════════════════════════════════════════╗")
    print("║         PyShorthand Ecosystem + GPT-5.1 Reasoning                        ║")
    print("╚═══════════════════════════════════════════════════════════════════════════╝")
    print()

    # Load questions
    questions = load_test_suite()

    # Create agent
    pyshorthand_path = Path(__file__).parent.parent / "realworld_nanogpt.pys"
    codebase_path = Path(__file__).parent / "nanogpt_sample.py"

    agent = GPT5ReasoningAgent(pyshorthand_path, codebase_path)

    # Run tests
    results = []
    print(f"Testing {len(questions)} questions with GPT-5.1 reasoning mode...")
    print()

    for i, question in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}]", end=" ")
        result = agent.answer_question(question)
        results.append(result)
        time.sleep(1)  # Rate limiting

    # Summary
    print()
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print()

    correct = sum(1 for r in results if r.is_correct)
    accuracy = (correct / len(results)) * 100

    print(f"Accuracy: {correct}/{len(results)} ({accuracy:.1f}%)")
    print()

    # Token usage
    avg_base = sum(r.pyshorthand_tokens for r in results) / len(results)
    avg_tools = sum(r.tool_tokens for r in results) / len(results)
    avg_total = sum(r.total_tokens for r in results) / len(results)

    print("Average tokens:")
    print(f"  Base (PyShorthand): {avg_base:.0f}")
    print(f"  Tools: {avg_tools:.0f}")
    print(f"  Total: {avg_total:.0f}")
    print()

    # Tool usage stats
    all_tools = []
    for r in results:
        all_tools.extend(r.tools_called)

    print(f"Tool calls: {len(all_tools)} total")
    if all_tools:
        from collections import Counter

        tool_counts = Counter(all_tools)
        for tool, count in tool_counts.most_common():
            print(f"  {tool}: {count}")
    print()

    # Comparison
    print("Comparison to baselines:")
    print("  Full code: 35% accuracy, 5,348 tokens")
    print("  PyShorthand v1.5: 35% accuracy, 894 tokens")
    print("  Ecosystem (keyword): 40% accuracy, 328 tokens")
    print(f"  Ecosystem (GPT-5.1): {accuracy:.0f}% accuracy, {avg_total:.0f} tokens")
    print()

    savings = ((5348 - avg_total) / 5348) * 100
    print(f"Savings vs full code: {savings:.1f}%")
    print()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = Path(__file__).parent / "results" / f"ecosystem_gpt5_reasoning_{timestamp}.json"
    filepath.parent.mkdir(exist_ok=True)

    with open(filepath, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    print(f"Results saved to: {filepath}")


if __name__ == "__main__":
    main()
