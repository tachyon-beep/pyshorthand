#!/usr/bin/env python3
"""Test execution flow tracing for PyShorthand."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pyshort.core.tokenizer import Tokenizer
from pyshort.core.parser import Parser
from pyshort.analyzer.execution_flow import ExecutionFlowTracer, trace_execution


def test_simple_class_trace():
    """Test tracing a simple class with no references."""
    source = """
[C:Simple]
  value ∈ i32
"""
    tokenizer = Tokenizer(source)
    tokens = tokenizer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()

    flow = trace_execution(ast, "Simple", follow_calls=False)

    assert flow is not None, "Flow should be traced"
    assert flow.entry_point == "Simple", f"Entry point should be 'Simple', got {flow.entry_point}"
    assert len(flow.steps) == 1, f"Should have 1 step, got {len(flow.steps)}"
    assert flow.steps[0].entity_name == "Simple", "First step should be the class"
    assert flow.steps[0].depth == 0, "Entry point should have depth 0"

    print(f"✓ Simple class trace passed (steps: {len(flow.steps)})")
    return True


def test_reference_chain():
    """Test tracing through a chain of class references."""
    source = """
[C:Main]
  data ∈ i32
  processor ∈ [Ref:Processor]

[C:Processor]
  input ∈ i32
  transformer ∈ [Ref:Transformer]

[C:Transformer]
  value ∈ i32

[C:Finalizer]
  result ∈ i32
"""
    tokenizer = Tokenizer(source)
    tokens = tokenizer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()

    flow = trace_execution(ast, "Main", max_depth=5, follow_calls=True)

    assert flow is not None, "Flow should be traced"
    assert flow.entry_point == "Main", "Entry point should be 'Main'"

    # Should trace: Main -> Processor -> Transformer
    assert len(flow.steps) >= 2, f"Should have multiple steps, got {len(flow.steps)}"

    # Check that Main is the first step
    assert flow.steps[0].entity_name == "Main", "First step should be Main"
    assert flow.steps[0].depth == 0, "Main should be at depth 0"

    # Check max depth
    assert flow.max_depth > 0, f"Max depth should be > 0, got {flow.max_depth}"

    print(f"✓ Reference chain trace passed (steps: {len(flow.steps)}, max_depth: {flow.max_depth})")
    return True


def test_class_reference_trace():
    """Test tracing class references."""
    source = """
[C:Node]
  value ∈ i32
  next ∈ [Ref:Node]

[C:LinkedList]
  head ∈ [Ref:Node]
  tail ∈ [Ref:Node]
"""
    tokenizer = Tokenizer(source)
    tokens = tokenizer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()

    flow = trace_execution(ast, "LinkedList", max_depth=3, follow_calls=True)

    assert flow is not None, "Flow should be traced"
    assert flow.entry_point == "LinkedList", "Entry point should be 'LinkedList'"

    # Should trace LinkedList and its references to Node
    assert len(flow.steps) >= 1, f"Should have at least 1 step, got {len(flow.steps)}"

    # Check state access
    assert len(flow.state_accessed) > 0, f"Should access state, got {flow.state_accessed}"

    # LinkedList should reference Node
    assert "LinkedList" in [step.entity_name for step in flow.steps], "Should include LinkedList"

    print(f"✓ Class reference trace passed (steps: {len(flow.steps)}, state: {len(flow.state_accessed)})")
    return True


def test_max_depth_limiting():
    """Test that max_depth prevents infinite recursion."""
    source = """
[C:RecursiveNode]
  value ∈ i32
  left ∈ [Ref:RecursiveNode]
  right ∈ [Ref:RecursiveNode]
"""
    tokenizer = Tokenizer(source)
    tokens = tokenizer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()

    flow = trace_execution(ast, "RecursiveNode", max_depth=2, follow_calls=True)

    assert flow is not None, "Flow should be traced"
    assert flow.max_depth <= 2, f"Max depth should be <= 2, got {flow.max_depth}"

    print(f"✓ Max depth limiting passed (max_depth: {flow.max_depth})")
    return True


def test_state_tracking():
    """Test tracking state variables."""
    source = """
[C:Calculator]
  x ∈ i32
  y ∈ i32
  result ∈ i32
  temp ∈ i32
"""
    tokenizer = Tokenizer(source)
    tokens = tokenizer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()

    flow = trace_execution(ast, "Calculator", follow_calls=False)

    assert flow is not None, "Flow should be traced"
    assert len(flow.state_accessed) > 0, f"Should track state, got {flow.state_accessed}"

    # Should include state variables
    assert any("Calculator.x" in s for s in flow.state_accessed), \
        f"Should track 'Calculator.x', got {flow.state_accessed}"
    assert any("Calculator.y" in s for s in flow.state_accessed), \
        f"Should track 'Calculator.y', got {flow.state_accessed}"

    print(f"✓ State tracking passed (state: {flow.state_accessed})")
    return True


def test_no_follow_references():
    """Test tracing without following references."""
    source = """
[C:Main]
  value ∈ i32
  helper ∈ [Ref:Helper]

[C:Helper]
  data ∈ i32
"""
    tokenizer = Tokenizer(source)
    tokens = tokenizer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()

    flow = trace_execution(ast, "Main", follow_calls=False)

    assert flow is not None, "Flow should be traced"
    assert len(flow.steps) == 1, f"Should only trace Main (no following), got {len(flow.steps)}"
    assert flow.steps[0].entity_name == "Main", "Should only include Main"
    assert flow.max_depth == 0, f"Max depth should be 0, got {flow.max_depth}"

    print(f"✓ No follow references passed (steps: {len(flow.steps)})")
    return True


def test_execution_flow_to_dict():
    """Test serialization to dictionary."""
    source = """
[C:TestClass]
  x ∈ i32
  y ∈ i32
"""
    tokenizer = Tokenizer(source)
    tokens = tokenizer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()

    flow = trace_execution(ast, "TestClass")

    assert flow is not None, "Flow should be traced"

    flow_dict = flow.to_dict()

    assert "entry_point" in flow_dict, "Dict should have entry_point"
    assert flow_dict["entry_point"] == "TestClass", f"Entry point should be 'TestClass', got {flow_dict['entry_point']}"
    assert "total_steps" in flow_dict, "Dict should have total_steps"
    assert "max_depth" in flow_dict, "Dict should have max_depth"
    assert "execution_path" in flow_dict, "Dict should have execution_path"

    print(f"✓ Serialization passed (keys: {list(flow_dict.keys())})")
    return True


def test_execution_flow_summarize():
    """Test summary generation."""
    source = """
[C:Example]
  value ∈ i32
  next ∈ [Ref:Node]

[C:Node]
  data ∈ i32
"""
    tokenizer = Tokenizer(source)
    tokens = tokenizer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()

    flow = trace_execution(ast, "Example")

    assert flow is not None, "Flow should be traced"

    summary = flow.summarize()

    assert "Execution Flow:" in summary, "Summary should have title"
    assert "Example" in summary, "Summary should mention entry point"
    assert "Total steps:" in summary, "Summary should have step count"
    assert "Max call depth:" in summary, "Summary should have max depth"

    print("✓ Summary generation passed")
    print(f"\nExample summary:\n{summary}\n")
    return True


def test_missing_entry_point():
    """Test handling of missing entry point."""
    source = """
[C:Exists]
  value ∈ i32
"""
    tokenizer = Tokenizer(source)
    tokens = tokenizer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()

    flow = trace_execution(ast, "DoesNotExist")

    assert flow is None, "Should return None for missing entry point"

    print("✓ Missing entry point passed")
    return True


def test_tracer_class_directly():
    """Test using ExecutionFlowTracer class directly."""
    source = """
[C:Direct]
  value ∈ i32
  output ∈ i32
"""
    tokenizer = Tokenizer(source)
    tokens = tokenizer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()

    tracer = ExecutionFlowTracer()
    flow = tracer.trace_execution(ast, "Direct", max_depth=1)

    assert flow is not None, "Flow should be traced"
    assert flow.entry_point == "Direct", "Entry point should be 'Direct'"

    print("✓ Direct tracer usage passed")
    return True


def run_all_tests():
    """Run all execution flow tests."""
    tests = [
        ("Simple Class Trace", test_simple_class_trace),
        ("Reference Chain", test_reference_chain),
        ("Class Reference Trace", test_class_reference_trace),
        ("Max Depth Limiting", test_max_depth_limiting),
        ("State Tracking", test_state_tracking),
        ("No Follow References", test_no_follow_references),
        ("Serialization to Dict", test_execution_flow_to_dict),
        ("Summary Generation", test_execution_flow_summarize),
        ("Missing Entry Point", test_missing_entry_point),
        ("Direct Tracer Usage", test_tracer_class_directly),
    ]

    print("Testing Execution Flow Tracing:")
    print("=" * 70)

    passed = 0
    failed = 0

    for name, test_func in tests:
        print(f"\nTest: {name}")
        try:
            if test_func():
                passed += 1
            else:
                print(f"  ✗ FAIL")
                failed += 1
        except AssertionError as e:
            print(f"  ✗ FAIL - {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR - {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
