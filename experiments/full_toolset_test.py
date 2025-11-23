#!/usr/bin/env python3
"""Test GPT-5.1 with full 8-tool ecosystem.

Tests whether GPT-5.1 can intelligently choose between:
- get_module_pyshorthand() - cheapest overview (800 tokens)
- get_class_pyshorthand(class) - single class (150 tokens)
- get_class_details(class) - detailed structure (250 tokens)
- get_implementation(method) - full code (400 tokens)
- get_context_pack(target) - dependency graph (200 tokens)
- trace_execution(entry) - execution flow (250 tokens)
- get_neighbors(symbol) - direct dependencies (100 tokens)
- search_usage(symbol) - find usages (75 tokens)
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Import ecosystem tools
from src.pyshort.ecosystem.tools import CodebaseExplorer

# Test questions designed to exercise different tool combinations
TEST_QUESTIONS = [
    {
        "id": 1,
        "question": "How many classes are in the codebase and what do they inherit from?",
        "expected_tools": ["get_module_pyshorthand"],
        "complexity": "simple_structural",
        "best_strategy": "Start with get_module_pyshorthand() - can answer entirely from that",
    },
    {
        "id": 2,
        "question": "What are all the methods in the GPT class and what are their signatures?",
        "expected_tools": ["get_class_pyshorthand", "get_class_details"],
        "complexity": "single_class_structural",
        "best_strategy": "Use get_class_pyshorthand(GPT) or get_class_details(GPT)",
    },
    {
        "id": 3,
        "question": "If I modify CausalSelfAttention, what other classes might be affected?",
        "expected_tools": ["get_context_pack", "search_usage"],
        "complexity": "dependency_analysis",
        "best_strategy": "Use get_context_pack(CausalSelfAttention) to see F1/F2 dependencies",
    },
    {
        "id": 4,
        "question": "Trace the execution flow when GPT.forward() is called - what gets executed in what order?",
        "expected_tools": ["trace_execution"],
        "complexity": "execution_flow",
        "best_strategy": "Use trace_execution(GPT.forward) to see call chain",
    },
    {
        "id": 5,
        "question": "How does the Block class implement residual connections? Show the exact code.",
        "expected_tools": ["get_implementation"],
        "complexity": "implementation_detail",
        "best_strategy": "Need actual code - use get_implementation(Block.forward)",
    },
    {
        "id": 6,
        "question": "What classes directly depend on LayerNorm and what depends on those?",
        "expected_tools": ["get_context_pack", "search_usage"],
        "complexity": "multi_hop_dependency",
        "best_strategy": "Use get_context_pack(LayerNorm, max_depth=2) for F1/F2 layers",
    },
    {
        "id": 7,
        "question": "What is the complete structure of the transformer ModuleDict in GPT?",
        "expected_tools": ["get_class_details", "get_class_pyshorthand"],
        "complexity": "nested_structure",
        "best_strategy": "Use get_class_details(GPT, expand_nested=True)",
    },
    {
        "id": 8,
        "question": "How does n_head flow from GPTConfig through the architecture? Which classes use it and how?",
        "expected_tools": ["get_context_pack", "get_class_details", "get_implementation"],
        "complexity": "cross_file_parameter_trace",
        "best_strategy": "Start with get_context_pack(GPTConfig), then get_implementation for classes that use n_head",
    },
]


def create_system_prompt() -> str:
    """Create system prompt with all 8 tools available."""
    return """You are an expert code analyst with access to 8 powerful tools for exploring a Python codebase.

AVAILABLE TOOLS (in order of cost, cheapest first):

1. get_module_pyshorthand()
   - Returns entire codebase in compressed PyShorthand format
   - Cost: ~800 tokens (but covers EVERYTHING)
   - Best for: Initial exploration, structural questions
   - Can answer: "How many classes?", "What inherits from X?", "Method signatures?"

2. search_usage(symbol)
   - Find where a class/method is used
   - Cost: ~75 tokens
   - Returns: List of usage locations

3. get_neighbors(symbol)
   - Get direct dependencies (callers + callees)
   - Cost: ~100 tokens
   - Returns: What this calls and what calls this

4. get_class_pyshorthand(class_name)
   - Single class in PyShorthand format
   - Cost: ~150 tokens
   - Best for: Focused class exploration

5. get_context_pack(target, max_depth=2)
   - Dependency graph with F0/F1/F2 layers
   - Cost: ~200 tokens
   - Returns: Target, immediate deps (F1), 2-hop deps (F2), peers, state
   - Best for: "What depends on what?", "Impact analysis"

6. trace_execution(entry_point, max_depth=10)
   - Trace runtime execution flow through calls
   - Cost: ~250 tokens
   - Returns: Execution path, call depth, variables in scope
   - Best for: "What happens when X is called?"

7. get_class_details(class_name, expand_nested=True)
   - Detailed class structure with types and signatures
   - Cost: ~250 tokens
   - Best for: Type information, nested structures

8. get_implementation(Class.method)
   - Full Python source code
   - Cost: ~400 tokens
   - Best for: "HOW does this work?", implementation details
   - Most expensive - only use when you need actual code!

STRATEGY (BE SMART BUT AGGRESSIVE):

1. START CHEAP:
   - For structural questions: call get_module_pyshorthand() first (it's comprehensive!)
   - For single class: use get_class_pyshorthand(class) or get_class_details(class)
   - For dependencies: use get_context_pack(target) or get_neighbors(symbol)

2. THEN DRILL DOWN:
   - Use get_context_pack() to understand relationships
   - Use trace_execution() for runtime behavior
   - Use get_implementation() ONLY when you need actual code

3. BE AGGRESSIVE:
   - Don't guess! If uncertain, call more tools
   - For complex questions, expect to call 3-5 tools
   - It's better to spend 500 tokens and get the RIGHT answer!

4. TOOL CALLING FORMAT:
   When you need a tool, output EXACTLY this format:

   TOOL_CALL: tool_name(arg1, arg2, ...)

   Examples:
   TOOL_CALL: get_module_pyshorthand()
   TOOL_CALL: get_class_details(GPT)
   TOOL_CALL: get_implementation(Block.forward)
   TOOL_CALL: get_context_pack(CausalSelfAttention, max_depth=2)
   TOOL_CALL: trace_execution(GPT.forward, max_depth=5)

IMPORTANT:
- Start with the cheapest tool that could answer the question
- If that's not enough, call additional tools
- Always explain your reasoning for which tools you chose
- Output "ANSWER:" followed by your final answer when done
"""


def parse_tool_calls(response_text: str) -> list:
    """Parse tool calls from GPT-5.1 response."""

    tool_calls = []
    lines = response_text.split("\n")

    for line in lines:
        if "TOOL_CALL:" in line:
            # Extract tool call after TOOL_CALL:
            call_str = line.split("TOOL_CALL:")[1].strip()
            tool_calls.append(call_str)

    return tool_calls


def execute_tool(tool_call: str, explorer: CodebaseExplorer) -> dict:
    """Execute a tool call and return result."""
    import re

    # Parse tool name and arguments
    match = re.match(r"(\w+)\((.*?)\)", tool_call)
    if not match:
        return {"error": f"Invalid tool call format: {tool_call}"}

    tool_name = match.group(1)
    args_str = match.group(2)

    # Parse arguments
    args = []
    kwargs = {}

    if args_str.strip():
        # Split by commas, but respect nested structures
        import ast

        try:
            # Try to parse as Python literal
            args_str_wrapped = f"[{args_str}]"
            parsed = ast.literal_eval(args_str_wrapped)
            args = parsed
        except:
            # Simple string parsing
            parts = [p.strip() for p in args_str.split(",")]
            for part in parts:
                if "=" in part:
                    k, v = part.split("=", 1)
                    try:
                        kwargs[k.strip()] = ast.literal_eval(v.strip())
                    except:
                        kwargs[k.strip()] = v.strip()
                else:
                    try:
                        args.append(ast.literal_eval(part))
                    except:
                        args.append(part)

    # Execute tool
    try:
        if tool_name == "get_module_pyshorthand":
            result = explorer.get_module_pyshorthand()
            return {"result": result, "tokens": len(result.split()) * 1.3 if result else 0}

        elif tool_name == "get_class_pyshorthand":
            result = explorer.get_class_pyshorthand(args[0] if args else "")
            return {"result": result, "tokens": len(result.split()) * 1.3 if result else 0}

        elif tool_name == "get_class_details":
            expand = kwargs.get("expand_nested", True)
            result = explorer.get_class_details(args[0] if args else "", expand_nested=expand)
            return {"result": result, "tokens": len(result.split()) * 1.3 if result else 0}

        elif tool_name == "get_implementation":
            result = explorer.get_implementation(args[0] if args else "")
            return {"result": result, "tokens": len(result.split()) * 1.3 if result else 0}

        elif tool_name == "get_context_pack":
            max_depth = kwargs.get("max_depth", 2) if kwargs else (args[1] if len(args) > 1 else 2)
            result = explorer.get_context_pack(args[0] if args else "", max_depth=max_depth)
            result_str = json.dumps(result, indent=2) if result else "None"
            return {"result": result_str, "tokens": len(result_str.split()) * 1.3}

        elif tool_name == "trace_execution":
            max_depth = (
                kwargs.get("max_depth", 10) if kwargs else (args[1] if len(args) > 1 else 10)
            )
            result = explorer.trace_execution(args[0] if args else "", max_depth=max_depth)
            result_str = json.dumps(result, indent=2) if result else "None"
            return {"result": result_str, "tokens": len(result_str.split()) * 1.3}

        elif tool_name == "get_neighbors":
            result = explorer.get_neighbors(args[0] if args else "")
            result_str = json.dumps(result, indent=2) if result else "None"
            return {"result": result_str, "tokens": len(result_str.split()) * 1.3}

        elif tool_name == "search_usage":
            result = explorer.search_usage(args[0] if args else "")
            result_str = "\n".join(result) if result else "No usages found"
            return {"result": result_str, "tokens": len(result_str.split()) * 1.3}

        else:
            return {"error": f"Unknown tool: {tool_name}"}

    except Exception as e:
        return {"error": f"Tool execution error: {str(e)}"}


def test_question(client, question_data: dict, explorer: CodebaseExplorer) -> dict:
    """Test a single question with GPT-5.1."""
    question = question_data["question"]

    print(f"\n{'='*80}")
    print(f"Q{question_data['id']}: {question}")
    print(f"Complexity: {question_data['complexity']}")
    print(f"Expected tools: {', '.join(question_data['expected_tools'])}")
    print(f"{'='*80}\n")

    messages = [
        {"role": "system", "content": create_system_prompt()},
        {"role": "user", "content": question},
    ]

    total_tokens = 0
    tools_used = []
    iterations = 0
    max_iterations = 10

    while iterations < max_iterations:
        iterations += 1

        print(f"\n--- Iteration {iterations} ---")

        # Call GPT-5.1
        response = client.chat.completions.create(
            model="openai/o1-2024-12-17",
            messages=messages,
        )

        assistant_message = response.choices[0].message.content
        print(f"\nGPT-5.1 Response:\n{assistant_message}\n")

        # Track tokens
        if hasattr(response, "usage"):
            total_tokens += response.usage.total_tokens

        # Check if there are tool calls
        tool_calls = parse_tool_calls(assistant_message)

        if not tool_calls:
            # No more tool calls, we're done
            if "ANSWER:" in assistant_message:
                # Extract answer
                answer = assistant_message.split("ANSWER:")[1].strip()
                return {
                    "question_id": question_data["id"],
                    "question": question,
                    "answer": answer,
                    "tools_used": tools_used,
                    "total_tokens": total_tokens,
                    "iterations": iterations,
                    "expected_tools": question_data["expected_tools"],
                    "complexity": question_data["complexity"],
                }
            else:
                # Response but no answer marker
                return {
                    "question_id": question_data["id"],
                    "question": question,
                    "answer": assistant_message,
                    "tools_used": tools_used,
                    "total_tokens": total_tokens,
                    "iterations": iterations,
                    "expected_tools": question_data["expected_tools"],
                    "complexity": question_data["complexity"],
                }

        # Execute tool calls
        tool_results = []
        for tool_call in tool_calls:
            print(f"Executing: {tool_call}")
            result = execute_tool(tool_call, explorer)
            tools_used.append(tool_call)

            if "error" in result:
                tool_results.append(f"Tool: {tool_call}\nError: {result['error']}")
            else:
                tool_results.append(f"Tool: {tool_call}\nResult:\n{result['result']}")
                total_tokens += int(result.get("tokens", 0))

        # Add assistant message and tool results to conversation
        messages.append({"role": "assistant", "content": assistant_message})
        messages.append({"role": "user", "content": "\n\n".join(tool_results)})

    return {
        "question_id": question_data["id"],
        "question": question,
        "answer": "Max iterations reached",
        "tools_used": tools_used,
        "total_tokens": total_tokens,
        "iterations": iterations,
        "expected_tools": question_data["expected_tools"],
        "complexity": question_data["complexity"],
    }


def main():
    """Run full toolset test."""
    print("=" * 80)
    print("GPT-5.1 Full Toolset Test")
    print("Testing all 8 ecosystem tools")
    print("=" * 80)

    # Initialize OpenRouter client
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not found in environment")
        return

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    # Initialize explorer (point to nanoGPT codebase)
    codebase_path = Path(__file__).parent.parent / "test_repos" / "nanoGPT" / "model.py"
    explorer = CodebaseExplorer(codebase_path)

    # Run tests
    results = []

    for question_data in TEST_QUESTIONS:
        try:
            result = test_question(client, question_data, explorer)
            results.append(result)

            # Print summary
            print(f"\n✓ Q{result['question_id']} completed:")
            print(f"  Tools used: {len(result['tools_used'])}")
            print(f"  Iterations: {result['iterations']}")
            print(f"  Total tokens: {result['total_tokens']}")

        except Exception as e:
            print(f"\n✗ Q{question_data['id']} failed: {e}")
            import traceback

            traceback.print_exc()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(__file__).parent / "results" / f"full_toolset_{timestamp}.json"
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")

    # Print summary statistics
    print("\n📊 SUMMARY STATISTICS\n")

    for result in results:
        print(f"Q{result['question_id']}: {result['complexity']}")
        print(f"  Expected: {', '.join(result['expected_tools'])}")
        print(
            f"  Actually used: {', '.join(result['tools_used']) if result['tools_used'] else 'None'}"
        )
        print(f"  Tokens: {result['total_tokens']}")
        print()


if __name__ == "__main__":
    main()
