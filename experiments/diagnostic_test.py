#!/usr/bin/env python3
"""
Diagnostic Test with Aggressive Tool-Calling Prompt

Watch GPT-5.1 reason about what tools it needs for complex multi-file questions.
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

from diagnostic_questions import load_diagnostic_suite

from pyshort.ecosystem.tools import CodebaseExplorer


@dataclass
class DiagnosticResult:
    """Result from diagnostic test."""

    question_id: int
    question: str
    answer: str
    is_correct: bool
    tools_called: list[str]
    expected_tools: list[str]
    tool_selection_quality: str  # "perfect", "good", "ok", "poor"
    reasoning_trace: str
    total_tokens: int
    crosses_files: int
    needs_implementation: bool


class AggressiveDiagnosticAgent:
    """
    Agent with AGGRESSIVE tool-calling prompt.

    Goals:
    1. Watch it reason about what information it needs
    2. See it decide between PyShorthand vs implementation
    3. Track multi-file reasoning
    """

    def __init__(self, pyshorthand_path: Path, codebase_path: Path):
        """Initialize agent."""
        # Load PyShorthand
        with open(pyshorthand_path) as f:
            self.pyshorthand = f.read()

        self.pyshorthand_tokens = len(self.pyshorthand.split())
        self.explorer = CodebaseExplorer(codebase_path)

        # OpenRouter client
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found")

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model = "openai/gpt-5.1"

    def answer_question(self, question):
        """Answer diagnostic question with aggressive tool calling."""

        print(f"\n{'='*80}")
        print(f"Q{question.id}: {question.question}")
        print(f"{'='*80}")
        print(
            f"Difficulty: Crosses {question.crosses_files} files, Implementation={question.needs_implementation}"
        )
        print(f"Expected tools: {question.expected_tools}")
        print()

        # AGGRESSIVE system prompt
        system_prompt = """You are an expert code analyst with access to both compressed (PyShorthand) and full code.

AVAILABLE TOOLS:

1. **get_class_details(class_name, expand_nested=True)**
   - Shows class structure, method signatures, state variable types
   - Shows nested structures (ModuleDict contents, etc.)
   - Cost: ~200-400 tokens
   - Use when: Need exact types, signatures, or structural details

2. **get_implementation(ClassName.method_name)**
   - Shows actual Python implementation with logic
   - Cost: ~300-500 tokens
   - Use when: Need to understand WHAT CODE DOES (algorithms, control flow, transformations)

3. **search_usage(symbol)**
   - Find where a class/method is used
   - Cost: ~50-100 tokens
   - Use when: Need to understand dependencies or call chains

STRATEGY (BE AGGRESSIVE):

1. **Start with PyShorthand** - it's free and shows architecture
2. **For complex/cross-file questions: CALL TOOLS LIBERALLY**
   - Architecture questions? → get_class_details for ALL relevant classes
   - Implementation questions? → get_implementation without hesitation
   - Dependency questions? → search_usage + get_class_details
3. **It's worth 500 tokens to get the RIGHT answer!**
4. **When uncertain, call MORE tools rather than guessing**
5. **For multi-file questions, fetch details for ALL involved files**

Think step-by-step:
- What information do I need to answer this?
- Is it in PyShorthand or do I need more detail?
- Which specific tools will give me that detail?
- Should I call multiple tools to build complete understanding?

FORMAT YOUR RESPONSE:

**REASONING:** [Your detailed reasoning about what you need and why]

**TOOL_CALLS:** [List each tool call on its own line, e.g.:]
get_class_details(GPT, expand_nested=True)
get_implementation(Block.forward)
search_usage(CausalSelfAttention)

**ANSWER:** [Your final answer]
"""

        user_prompt = f"""PyShorthand Overview:
```
{self.pyshorthand}
```

Question: {question.question}

This is a {'cross-file ' if question.crosses_files > 1 else ''}question that {'requires understanding implementation details' if question.needs_implementation else 'focuses on architecture/structure'}.

Think carefully about what information you need. Don't hesitate to call multiple tools!"""

        # First API call
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

            content = response.choices[0].message.content

            # Extract reasoning
            reasoning_trace = "No reasoning provided"
            if "REASONING:" in content:
                reasoning_section = content.split("REASONING:")[1].split("TOOL_CALLS:")[0].strip()
                reasoning_trace = reasoning_section
                print("🧠 REASONING:")
                print(f"   {reasoning_section[:300]}...")
                print()

            # Parse and execute tool calls
            tools_called = []
            tool_tokens = 0
            tool_outputs = []

            if "TOOL_CALLS:" in content:
                tool_section = content.split("TOOL_CALLS:")[1].split("ANSWER:")[0].strip()

                print("🔧 TOOL CALLS REQUESTED:")
                print(f"   {tool_section}")
                print()

                if "none" not in tool_section.lower():
                    for line in tool_section.split("\n"):
                        line = line.strip()
                        if not line or line == "none":
                            continue

                        # Parse and execute
                        if "get_class_details(" in line:
                            # Extract class name and args
                            parts = line.split("get_class_details(")[1].split(")")[0]
                            class_name = parts.split(",")[0].strip()
                            expand = "expand_nested=True" in parts or "True" in parts

                            details = self.explorer.get_class_details(
                                class_name, expand_nested=expand
                            )
                            if details:
                                tool_outputs.append(
                                    f"# get_class_details({class_name}):\n{details}"
                                )
                                tools_called.append(f"get_class_details({class_name})")
                                tool_tokens += len(details.split())
                                print(
                                    f"   ✓ Fetched get_class_details({class_name}) - {len(details)} chars"
                                )

                        elif "get_implementation(" in line:
                            target = line.split("get_implementation(")[1].split(")")[0].strip()
                            impl = self.explorer.get_implementation(target, include_context=False)
                            if impl:
                                tool_outputs.append(f"# get_implementation({target}):\n{impl}")
                                tools_called.append(f"get_implementation({target})")
                                tool_tokens += len(impl.split())
                                print(
                                    f"   ✓ Fetched get_implementation({target}) - {len(impl)} chars"
                                )

                        elif "search_usage(" in line:
                            symbol = line.split("search_usage(")[1].split(")")[0].strip()
                            usages = self.explorer.search_usage(symbol)
                            if usages:
                                usage_text = "\n".join(usages)
                                tool_outputs.append(f"# search_usage({symbol}):\n{usage_text}")
                                tools_called.append(f"search_usage({symbol})")
                                tool_tokens += len(usage_text.split())
                                print(
                                    f"   ✓ Fetched search_usage({symbol}) - {len(usages)} results"
                                )

            # If tools were called, make follow-up API call
            final_answer = content
            if tool_outputs:
                print()
                print("💭 SYNTHESIZING ANSWER WITH TOOL RESULTS...")
                print()

                tool_results = "\n\n".join(tool_outputs)

                follow_up = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": content},
                        {
                            "role": "user",
                            "content": f"Here are the tool results:\n\n{tool_results}\n\nNow provide your final answer.",
                        },
                    ],
                    extra_body={"reasoning": {"enabled": True}},
                )

                final_answer = follow_up.choices[0].message.content

            # Extract answer
            if "ANSWER:" in final_answer:
                answer = final_answer.split("ANSWER:")[1].strip()
            else:
                answer = final_answer

            # Evaluate correctness (fuzzy matching)
            answer_lower = answer.lower()
            correct_lower = question.correct_answer.lower()
            is_correct = correct_lower in answer_lower or any(
                word in answer_lower for word in correct_lower.split() if len(word) > 5
            )

            # Evaluate tool selection quality
            expected_set = set(question.expected_tools)
            called_set = set(tools_called)

            if called_set == expected_set:
                quality = "perfect"
            elif called_set.issuperset(expected_set):
                quality = "good (extra calls)"
            elif called_set.issubset(expected_set):
                quality = "ok (missed some)"
            else:
                quality = "poor (wrong tools)"

            total_tokens = self.pyshorthand_tokens + tool_tokens

            # Print result
            status = "✅" if is_correct else "❌"
            print(f"{status} ANSWER: {answer[:200]}...")
            print()
            print("📊 METRICS:")
            print(f"   Tools called: {len(tools_called)}")
            print(f"   Tool selection: {quality}")
            print(
                f"   Total tokens: {total_tokens} ({self.pyshorthand_tokens} base + {tool_tokens} tools)"
            )
            print(f"   Correct: {is_correct}")

            return DiagnosticResult(
                question_id=question.id,
                question=question.question,
                answer=answer,
                is_correct=is_correct,
                tools_called=tools_called,
                expected_tools=question.expected_tools,
                tool_selection_quality=quality,
                reasoning_trace=reasoning_trace,
                total_tokens=total_tokens,
                crosses_files=question.crosses_files,
                needs_implementation=question.needs_implementation,
            )

        except Exception as e:
            print(f"❌ ERROR: {e}")
            import traceback

            traceback.print_exc()

            return DiagnosticResult(
                question_id=question.id,
                question=question.question,
                answer=f"Error: {e}",
                is_correct=False,
                tools_called=[],
                expected_tools=question.expected_tools,
                tool_selection_quality="error",
                reasoning_trace=str(e),
                total_tokens=0,
                crosses_files=question.crosses_files,
                needs_implementation=question.needs_implementation,
            )


def main():
    """Run diagnostic test with aggressive prompting."""
    print()
    print("╔═══════════════════════════════════════════════════════════════════════════╗")
    print("║         DIAGNOSTIC TEST: Watch GPT-5.1 Reason About Tool Selection       ║")
    print("╚═══════════════════════════════════════════════════════════════════════════╝")
    print()
    print("Testing complex multi-file questions with AGGRESSIVE tool-calling prompt.")
    print("Watch GPT-5.1 decide what it needs: PyShorthand vs implementation vs both!")
    print()

    # Load questions
    questions = load_diagnostic_suite()

    # Create agent
    pyshorthand_path = Path(__file__).parent.parent / "realworld_nanogpt.pys"
    codebase_path = Path(__file__).parent / "nanogpt_sample.py"

    agent = AggressiveDiagnosticAgent(pyshorthand_path, codebase_path)

    # Run tests
    results = []
    print(f"Running {len(questions)} diagnostic questions...")
    print()

    for question in questions:
        result = agent.answer_question(question)
        results.append(result)
        time.sleep(2)  # Rate limiting

    # Summary
    print()
    print("=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)
    print()

    correct = sum(1 for r in results if r.is_correct)
    print(f"Accuracy: {correct}/{len(results)} ({correct/len(results)*100:.0f}%)")
    print()

    # Tool selection analysis
    perfect = sum(1 for r in results if r.tool_selection_quality == "perfect")
    good = sum(1 for r in results if "good" in r.tool_selection_quality)
    ok = sum(1 for r in results if "ok" in r.tool_selection_quality)
    poor = sum(1 for r in results if "poor" in r.tool_selection_quality)

    print("Tool Selection Quality:")
    print(f"  Perfect: {perfect}/{len(results)}")
    print(f"  Good: {good}/{len(results)}")
    print(f"  Ok: {ok}/{len(results)}")
    print(f"  Poor: {poor}/{len(results)}")
    print()

    # Token usage
    avg_tokens = sum(r.total_tokens for r in results) / len(results)
    print(f"Average tokens: {avg_tokens:.0f}")
    print()

    # By question type
    impl_questions = [r for r in results if r.needs_implementation]
    struct_questions = [r for r in results if not r.needs_implementation]

    if impl_questions:
        impl_correct = sum(1 for r in impl_questions if r.is_correct)
        print(
            f"Implementation questions: {impl_correct}/{len(impl_questions)} ({impl_correct/len(impl_questions)*100:.0f}%)"
        )

    if struct_questions:
        struct_correct = sum(1 for r in struct_questions if r.is_correct)
        print(
            f"Structural questions: {struct_correct}/{len(struct_questions)} ({struct_correct/len(struct_questions)*100:.0f}%)"
        )
    print()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = Path(__file__).parent / "results" / f"diagnostic_aggressive_{timestamp}.json"
    filepath.parent.mkdir(exist_ok=True)

    with open(filepath, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    print(f"Results saved to: {filepath}")


if __name__ == "__main__":
    main()
