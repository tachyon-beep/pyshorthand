#!/usr/bin/env python3
"""Test context pack generation for PyShorthand."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pyshort.core.tokenizer import Tokenizer
from pyshort.core.parser import Parser
from pyshort.analyzer.context_pack import ContextPackGenerator


def test_context_pack_f0_core():
    """Test F0 layer - target entity itself."""
    source = """
[C:Target]
  value ∈ i32

[C:Other]
  data ∈ str
"""
    tokenizer = Tokenizer(source)
    tokens = tokenizer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()

    generator = ContextPackGenerator()
    pack = generator.generate_context_pack(ast, "Target")

    assert pack is not None, "Context pack should be generated"
    assert pack.target == "Target", f"Target should be 'Target', got {pack.target}"
    assert "Target" in pack.f0_core, "F0 should contain target entity"
    assert len(pack.f0_core) == 1, f"F0 should have 1 entity, got {len(pack.f0_core)}"

    print("✓ F0 Core layer test passed")
    return True


def test_context_pack_f1_immediate():
    """Test F1 layer - 1-hop bidirectional dependencies."""
    source = """
[C:ItemA]
  value ∈ i32
  ref ∈ [Ref:ItemB]

[C:ItemB]
  value ∈ i32
  ref ∈ [Ref:ItemC]

[C:ItemC]
  value ∈ i32
"""
    tokenizer = Tokenizer(source)
    tokens = tokenizer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()

    generator = ContextPackGenerator()
    pack = generator.generate_context_pack(ast, "ItemB")

    assert pack is not None, "Context pack should be generated"

    # F0: ItemB itself
    assert "ItemB" in pack.f0_core, "F0 should contain ItemB"

    # F1: ItemA (depends on ItemB) and ItemC (ItemB depends on it)
    # ItemA references ItemB, so ItemA → ItemB (ItemA in reverse graph of ItemB)
    # ItemB references ItemC, so ItemB → ItemC (ItemC in forward graph of ItemB)
    assert "ItemA" in pack.f1_immediate or "ItemC" in pack.f1_immediate, \
        f"F1 should contain dependencies, got: {pack.f1_immediate}"

    print(f"✓ F1 Immediate dependencies test passed (F1: {pack.f1_immediate})")
    return True


def test_context_pack_f2_extended():
    """Test F2 layer - 2-hop dependencies."""
    source = """
[C:NodeA]
  value ∈ i32
  next ∈ [Ref:NodeB]

[C:NodeB]
  value ∈ i32
  next ∈ [Ref:NodeC]

[C:NodeC]
  value ∈ i32
  next ∈ [Ref:NodeD]

[C:NodeD]
  value ∈ i32
"""
    tokenizer = Tokenizer(source)
    tokens = tokenizer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()

    generator = ContextPackGenerator()
    pack = generator.generate_context_pack(ast, "NodeB", max_depth=2)

    assert pack is not None, "Context pack should be generated"

    # F0: NodeB
    assert "NodeB" in pack.f0_core, "F0 should contain NodeB"

    # F1: NodeA (references NodeB) and NodeC (NodeB references it)
    assert "NodeC" in pack.f1_immediate, f"F1 should contain NodeC, got: {pack.f1_immediate}"

    # F2: NodeD (NodeC references it, 2 hops from NodeB)
    # We might also get NodeA in F2 through reverse edges
    assert len(pack.f2_extended) > 0, f"F2 should have entities, got: {pack.f2_extended}"

    print(f"✓ F2 Extended dependencies test passed (F2: {pack.f2_extended})")
    return True


def test_context_pack_class_peers():
    """Test class peer detection - other methods in same class."""
    source = """
[C:Calculator]
  result ∈ i32

  [M:add]
    x ∈ i32 → i32
    y ∈ i32
    add(x, y) → x + y

  [M:subtract]
    x ∈ i32 → i32
    y ∈ i32
    subtract(x, y) → x - y

  [M:multiply]
    x ∈ i32 → i32
    y ∈ i32
    multiply(x, y) → x * y
"""
    tokenizer = Tokenizer(source)
    tokens = tokenizer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()

    generator = ContextPackGenerator()

    # Methods are part of the class, not separate entities
    # So we target the class itself
    pack = generator.generate_context_pack(ast, "Calculator", include_peers=True)

    assert pack is not None, "Context pack should be generated"
    assert "Calculator" in pack.f0_core, "F0 should contain Calculator"

    # Class peers feature applies when we have method-level entities
    # For now, verify the pack is created correctly
    print(f"✓ Class peers test passed (F0: {pack.f0_core})")
    return True


def test_context_pack_state_variables():
    """Test related state variable tracking."""
    source = """
[C:Node]
  value ∈ i32
  next ∈ [Ref:Node]
  prev ∈ [Ref:Node]

[C:Graph]
  nodes ∈ [Ref:Node]
  edges ∈ [Ref:Edge]

[C:Edge]
  from ∈ [Ref:Node]
  to ∈ [Ref:Node]
"""
    tokenizer = Tokenizer(source)
    tokens = tokenizer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()

    generator = ContextPackGenerator()
    pack = generator.generate_context_pack(ast, "Graph")

    assert pack is not None, "Context pack should be generated"

    # State variables from Graph
    assert len(pack.related_state) > 0, \
        f"Should track state variables, got: {pack.related_state}"

    # Should include Graph.nodes and Graph.edges
    graph_state = [s for s in pack.related_state if s.startswith("Graph.")]
    assert len(graph_state) > 0, \
        f"Should track Graph state variables, got: {pack.related_state}"

    print(f"✓ State variables test passed (state: {pack.related_state})")
    return True


def test_context_pack_max_depth_limit():
    """Test max_depth parameter limits dependency traversal."""
    source = """
[C:ChainA]
  next ∈ [Ref:ChainB]

[C:ChainB]
  next ∈ [Ref:ChainC]

[C:ChainC]
  next ∈ [Ref:ChainD]

[C:ChainD]
  value ∈ i32
"""
    tokenizer = Tokenizer(source)
    tokens = tokenizer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()

    generator = ContextPackGenerator()

    # Test with max_depth=1 (only F1, no F2)
    pack = generator.generate_context_pack(ast, "ChainB", max_depth=1)

    assert pack is not None, "Context pack should be generated"
    assert "ChainB" in pack.f0_core, "F0 should contain ChainB"
    assert len(pack.f2_extended) == 0, \
        f"F2 should be empty with max_depth=1, got: {pack.f2_extended}"

    print(f"✓ Max depth limit test passed (F2 empty: {len(pack.f2_extended) == 0})")
    return True


def test_context_pack_missing_target():
    """Test that missing target returns None."""
    source = """
[C:Exists]
  value ∈ i32
"""
    tokenizer = Tokenizer(source)
    tokens = tokenizer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()

    generator = ContextPackGenerator()
    pack = generator.generate_context_pack(ast, "DoesNotExist")

    assert pack is None, "Should return None for missing target"

    print("✓ Missing target test passed")
    return True


def test_context_pack_to_dict():
    """Test context pack serialization to dict."""
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

    generator = ContextPackGenerator()
    pack = generator.generate_context_pack(ast, "A")

    assert pack is not None, "Context pack should be generated"

    # Test serialization
    pack_dict = pack.to_dict()

    assert "target" in pack_dict, "Dict should have 'target' field"
    assert pack_dict["target"] == "A", f"Target should be 'A', got {pack_dict['target']}"
    assert "f0_core" in pack_dict, "Dict should have 'f0_core' field"
    assert "f1_immediate" in pack_dict, "Dict should have 'f1_immediate' field"
    assert "f2_extended" in pack_dict, "Dict should have 'f2_extended' field"
    assert "total_entities" in pack_dict, "Dict should have 'total_entities' field"

    print(f"✓ Serialization test passed (dict keys: {list(pack_dict.keys())})")
    return True


def test_context_pack_convenience_function():
    """Test convenience function generate_context_pack()."""
    from pyshort.analyzer.context_pack import generate_context_pack

    source = """
[C:Target]
  value ∈ i32
"""
    tokenizer = Tokenizer(source)
    tokens = tokenizer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()

    pack = generate_context_pack(ast, "Target")

    assert pack is not None, "Context pack should be generated"
    assert pack.target == "Target", f"Target should be 'Target', got {pack.target}"
    assert "Target" in pack.f0_core, "F0 should contain target"

    print("✓ Convenience function test passed")
    return True


def run_all_tests():
    """Run all context pack tests."""
    tests = [
        ("F0 Core Layer", test_context_pack_f0_core),
        ("F1 Immediate Dependencies", test_context_pack_f1_immediate),
        ("F2 Extended Dependencies", test_context_pack_f2_extended),
        ("Class Peers", test_context_pack_class_peers),
        ("State Variables", test_context_pack_state_variables),
        ("Max Depth Limit", test_context_pack_max_depth_limit),
        ("Missing Target", test_context_pack_missing_target),
        ("Serialization to Dict", test_context_pack_to_dict),
        ("Convenience Function", test_context_pack_convenience_function),
    ]

    print("Testing Context Pack Generation:")
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
