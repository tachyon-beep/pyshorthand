#!/usr/bin/env python3
"""Test visualization export for context packs and execution flows."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pyshort.core.tokenizer import Tokenizer
from pyshort.core.parser import Parser
from pyshort.analyzer import generate_context_pack, trace_execution


def test_context_pack_mermaid_export():
    """Test Mermaid export for context pack."""
    source = """
[C:NodeA]
  value ∈ i32
  next ∈ [Ref:NodeB]

[C:NodeB]
  value ∈ i32
  next ∈ [Ref:NodeC]

[C:NodeC]
  value ∈ i32
"""
    tokenizer = Tokenizer(source)
    tokens = tokenizer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()

    pack = generate_context_pack(ast, "NodeB", max_depth=2)

    assert pack is not None, "Context pack should be generated"

    mermaid = pack.to_mermaid()

    # Verify Mermaid structure
    assert "graph TB" in mermaid, "Should start with graph declaration"
    assert "NodeB" in mermaid, "Should include target node"
    assert "classDef f0" in mermaid, "Should define F0 styling"
    assert "classDef f1" in mermaid, "Should define F1 styling"
    assert "classDef f2" in mermaid, "Should define F2 styling"
    assert ":::" in mermaid, "Should use class styling syntax"

    print("✓ Context pack Mermaid export passed")
    print(f"\nGenerated Mermaid diagram:\n{mermaid}\n")
    return True


def test_context_pack_mermaid_direction():
    """Test Mermaid export with different directions."""
    source = """
[C:A]
  ref ∈ [Ref:B]

[C:B]
  value ∈ i32
"""
    tokenizer = Tokenizer(source)
    tokens = tokenizer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()

    pack = generate_context_pack(ast, "A")

    # Test different directions
    for direction in ["TB", "LR", "BT", "RL"]:
        mermaid = pack.to_mermaid(direction=direction)
        assert f"graph {direction}" in mermaid, f"Should use {direction} direction"

    print("✓ Context pack Mermaid direction test passed")
    return True


def test_context_pack_graphviz_export():
    """Test GraphViz export for context pack."""
    source = """
[C:Main]
  processor ∈ [Ref:Processor]

[C:Processor]
  transformer ∈ [Ref:Transformer]

[C:Transformer]
  value ∈ i32
"""
    tokenizer = Tokenizer(source)
    tokens = tokenizer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()

    pack = generate_context_pack(ast, "Main", max_depth=2)

    assert pack is not None, "Context pack should be generated"

    dot = pack.to_graphviz()

    # Verify GraphViz structure
    assert "digraph ContextPack" in dot, "Should be a digraph"
    assert "rankdir=TB" in dot, "Should have rankdir"
    assert "node [shape=box]" in dot, "Should define node shape"
    assert "Main" in dot, "Should include target node"
    assert "fillcolor" in dot, "Should include colors"
    assert "->" in dot, "Should have edges"

    print("✓ Context pack GraphViz export passed")
    print(f"\nGenerated DOT diagram:\n{dot}\n")
    return True


def test_execution_flow_mermaid_export():
    """Test Mermaid export for execution flow."""
    source = """
[C:Main]
  processor ∈ [Ref:Processor]

[C:Processor]
  transformer ∈ [Ref:Transformer]

[C:Transformer]
  value ∈ i32
"""
    tokenizer = Tokenizer(source)
    tokens = tokenizer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()

    flow = trace_execution(ast, "Main", max_depth=5, follow_calls=True)

    assert flow is not None, "Flow should be traced"

    mermaid = flow.to_mermaid()

    # Verify Mermaid structure
    assert "graph TB" in mermaid, "Should start with graph declaration"
    assert "Step1" in mermaid, "Should include step nodes"
    assert "Main" in mermaid, "Should include entry point"
    assert "classDef depth0" in mermaid, "Should define depth0 styling"
    assert "-->" in mermaid, "Should have arrows"

    print("✓ Execution flow Mermaid export passed")
    print(f"\nGenerated Mermaid diagram:\n{mermaid}\n")
    return True


def test_execution_flow_mermaid_direction():
    """Test Mermaid export with different directions."""
    source = """
[C:Simple]
  value ∈ i32
"""
    tokenizer = Tokenizer(source)
    tokens = tokenizer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()

    flow = trace_execution(ast, "Simple")

    # Test different directions
    for direction in ["TB", "LR"]:
        mermaid = flow.to_mermaid(direction=direction)
        assert f"graph {direction}" in mermaid, f"Should use {direction} direction"

    print("✓ Execution flow Mermaid direction test passed")
    return True


def test_execution_flow_graphviz_export():
    """Test GraphViz export for execution flow."""
    source = """
[C:EntryPoint]
  next ∈ [Ref:Handler]

[C:Handler]
  processor ∈ [Ref:Processor]

[C:Processor]
  value ∈ i32
"""
    tokenizer = Tokenizer(source)
    tokens = tokenizer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()

    flow = trace_execution(ast, "EntryPoint", max_depth=5, follow_calls=True)

    assert flow is not None, "Flow should be traced"

    dot = flow.to_graphviz()

    # Verify GraphViz structure
    assert "digraph ExecutionFlow" in dot, "Should be a digraph"
    assert "rankdir=TB" in dot, "Should have rankdir"
    assert "node [shape=box]" in dot, "Should define node shape"
    assert "Step1" in dot, "Should include step nodes"
    assert "EntryPoint" in dot, "Should include entry point"
    assert "fillcolor" in dot, "Should include colors"
    assert "->" in dot, "Should have edges"

    print("✓ Execution flow GraphViz export passed")
    print(f"\nGenerated DOT diagram:\n{dot}\n")
    return True


def test_mermaid_special_characters():
    """Test Mermaid export handles entity names correctly."""
    source = """
[C:MyClass_v2]
  data ∈ i32
  next ∈ [Ref:Helper_123]

[C:Helper_123]
  value ∈ i32
"""
    tokenizer = Tokenizer(source)
    tokens = tokenizer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()

    pack = generate_context_pack(ast, "MyClass_v2")

    mermaid = pack.to_mermaid()

    assert "MyClass_v2" in mermaid, "Should handle underscores in names"
    assert "Helper_123" in mermaid, "Should handle numbers in names"

    print("✓ Special characters in Mermaid export passed")
    return True


def test_graphviz_multiple_depths():
    """Test GraphViz export with multiple call depths."""
    source = """
[C:Level0]
  ref ∈ [Ref:Level1]

[C:Level1]
  ref ∈ [Ref:Level2]

[C:Level2]
  ref ∈ [Ref:Level3]

[C:Level3]
  value ∈ i32
"""
    tokenizer = Tokenizer(source)
    tokens = tokenizer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()

    flow = trace_execution(ast, "Level0", max_depth=5, follow_calls=True)

    assert flow is not None, "Flow should be traced"

    dot = flow.to_graphviz()

    # Should have multiple steps with different colors
    assert "Step1" in dot, "Should have step 1"
    assert "Step2" in dot, "Should have step 2"
    assert "Step3" in dot, "Should have step 3"
    assert "Step4" in dot, "Should have step 4"

    print("✓ GraphViz multiple depths export passed")
    return True


def test_empty_context_pack_visualization():
    """Test visualization of context pack with minimal entities."""
    source = """
[C:Standalone]
  value ∈ i32
"""
    tokenizer = Tokenizer(source)
    tokens = tokenizer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()

    pack = generate_context_pack(ast, "Standalone")

    assert pack is not None, "Context pack should be generated"

    mermaid = pack.to_mermaid()
    dot = pack.to_graphviz()

    # Should handle single-node graph
    assert "Standalone" in mermaid, "Mermaid should include standalone node"
    assert "Standalone" in dot, "GraphViz should include standalone node"

    print("✓ Empty context pack visualization passed")
    return True


def test_visualization_consistency():
    """Test that both formats export the same entities."""
    source = """
[C:A]
  ref ∈ [Ref:B]

[C:B]
  ref ∈ [Ref:C]

[C:C]
  value ∈ i32
"""
    tokenizer = Tokenizer(source)
    tokens = tokenizer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()

    pack = generate_context_pack(ast, "A", max_depth=2)

    assert pack is not None, "Context pack should be generated"

    mermaid = pack.to_mermaid()
    dot = pack.to_graphviz()

    # Check that both formats include all entities
    for entity in ["A", "B", "C"]:
        assert entity in mermaid, f"Mermaid should include {entity}"
        assert entity in dot, f"GraphViz should include {entity}"

    print("✓ Visualization consistency test passed")
    return True


def run_all_tests():
    """Run all visualization export tests."""
    tests = [
        ("Context Pack Mermaid Export", test_context_pack_mermaid_export),
        ("Context Pack Mermaid Direction", test_context_pack_mermaid_direction),
        ("Context Pack GraphViz Export", test_context_pack_graphviz_export),
        ("Execution Flow Mermaid Export", test_execution_flow_mermaid_export),
        ("Execution Flow Mermaid Direction", test_execution_flow_mermaid_direction),
        ("Execution Flow GraphViz Export", test_execution_flow_graphviz_export),
        ("Mermaid Special Characters", test_mermaid_special_characters),
        ("GraphViz Multiple Depths", test_graphviz_multiple_depths),
        ("Empty Context Pack Visualization", test_empty_context_pack_visualization),
        ("Visualization Consistency", test_visualization_consistency),
    ]

    print("Testing Visualization Export:")
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
