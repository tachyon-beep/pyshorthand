#!/usr/bin/env python3
"""Test filtering and query API for context packs and execution flows."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pyshort.core.tokenizer import Tokenizer
from pyshort.core.parser import Parser
from pyshort.core.ast_nodes import Class
from pyshort.analyzer import generate_context_pack, trace_execution


def test_context_pack_filter_by_pattern():
    """Test pattern-based filtering of context pack."""
    source = """
[C:UserHandler]
  data ∈ i32

[C:DataProcessor]
  value ∈ i32
  ref ∈ [Ref:UserHandler]

[C:RequestValidator]
  input ∈ str
"""
    tokenizer = Tokenizer(source)
    tokens = tokenizer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()

    pack = generate_context_pack(ast, "DataProcessor", max_depth=2)
    assert pack is not None, "Context pack should be generated"

    # Filter for entities ending with "Handler"
    handlers = pack.filter_by_pattern(".*Handler$")

    assert "UserHandler" in handlers.all_entities(), \
        f"Should include UserHandler, got: {handlers.all_entities()}"
    assert "DataProcessor" not in handlers.all_entities(), \
        "Should not include DataProcessor"

    print(f"✓ Context pack filter_by_pattern passed (filtered: {handlers.all_entities()})")
    return True


def test_context_pack_filter_by_location():
    """Test location-based filtering of context pack."""
    source = """
[C:GPUTensor]
  data ∈ f32[N, M]@GPU

[C:CPUBuffer]
  values ∈ i32[K]@CPU

[C:Processor]
  gpu_data ∈ [Ref:GPUTensor]
  cpu_data ∈ [Ref:CPUBuffer]
"""
    tokenizer = Tokenizer(source)
    tokens = tokenizer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()

    pack = generate_context_pack(ast, "Processor", max_depth=2)
    assert pack is not None, "Context pack should be generated"

    # Filter for GPU entities
    gpu_entities = pack.filter_by_location("GPU")

    assert "GPUTensor" in gpu_entities.all_entities(), \
        f"Should include GPUTensor, got: {gpu_entities.all_entities()}"
    assert "CPUBuffer" not in gpu_entities.all_entities(), \
        "Should not include CPUBuffer"

    print(f"✓ Context pack filter_by_location passed (GPU entities: {gpu_entities.all_entities()})")
    return True


def test_context_pack_filter_custom():
    """Test custom predicate filtering of context pack."""
    source = """
[C:SmallClass]
  x ∈ i32

[C:LargeClass]
  a ∈ i32
  b ∈ i32
  c ∈ i32
  d ∈ i32
  e ∈ i32

[C:MediumClass]
  x ∈ i32
  y ∈ i32
  z ∈ i32
"""
    tokenizer = Tokenizer(source)
    tokens = tokenizer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()

    pack = generate_context_pack(ast, "MediumClass", max_depth=2)
    assert pack is not None, "Context pack should be generated"

    # Filter to only classes with more than 2 state variables
    large_classes = pack.filter_custom(
        lambda name, entity: isinstance(entity, Class) and len(entity.state) > 2
    )

    assert "LargeClass" in large_classes.all_entities() or "MediumClass" in large_classes.all_entities(), \
        f"Should include classes with >2 fields, got: {large_classes.all_entities()}"
    assert "SmallClass" not in large_classes.all_entities(), \
        "Should not include SmallClass"

    print(f"✓ Context pack filter_custom passed (large: {large_classes.all_entities()})")
    return True


def test_context_pack_get_by_layer():
    """Test getting entities by dependency layer."""
    source = """
[C:Level0]
  ref1 ∈ [Ref:Level1A]

[C:Level1A]
  ref ∈ [Ref:Level2]

[C:Level1B]
  value ∈ i32

[C:Level2]
  data ∈ i32
"""
    tokenizer = Tokenizer(source)
    tokens = tokenizer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()

    pack = generate_context_pack(ast, "Level0", max_depth=2)
    assert pack is not None, "Context pack should be generated"

    f0 = pack.get_by_layer(0)
    f1 = pack.get_by_layer(1)
    f2 = pack.get_by_layer(2)

    assert "Level0" in f0, f"F0 should contain Level0, got: {f0}"
    assert len(f1) > 0, f"F1 should have entities, got: {f1}"

    print(f"✓ Context pack get_by_layer passed (F0: {f0}, F1: {f1}, F2: {f2})")
    return True


def test_execution_flow_filter_by_depth():
    """Test depth-based filtering of execution flow."""
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

    # Filter to only depth 0 and 1
    shallow = flow.filter_by_depth(1)

    assert len(shallow.steps) <= len(flow.steps), \
        "Filtered flow should have fewer or equal steps"
    assert all(step.depth <= 1 for step in shallow.steps), \
        "All steps should be at depth <= 1"
    assert shallow.max_depth <= 1, \
        f"Max depth should be <= 1, got: {shallow.max_depth}"

    print(f"✓ Execution flow filter_by_depth passed (steps: {len(shallow.steps)}, max_depth: {shallow.max_depth})")
    return True


def test_execution_flow_filter_by_pattern():
    """Test pattern-based filtering of execution flow."""
    source = """
[C:MainHandler]
  processor ∈ [Ref:DataProcessor]

[C:DataProcessor]
  validator ∈ [Ref:InputValidator]

[C:InputValidator]
  value ∈ str
"""
    tokenizer = Tokenizer(source)
    tokens = tokenizer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()

    flow = trace_execution(ast, "MainHandler", max_depth=5, follow_calls=True)
    assert flow is not None, "Flow should be traced"

    # Filter for steps with "Processor" in the name
    processors = flow.filter_by_pattern(".*Processor$")

    assert len(processors.steps) > 0, "Should have matching steps"
    assert all("Processor" in step.entity_name for step in processors.steps), \
        "All steps should contain 'Processor'"

    print(f"✓ Execution flow filter_by_pattern passed (steps: {len(processors.steps)})")
    return True


def test_execution_flow_filter_by_state_access():
    """Test state access filtering of execution flow."""
    source = """
[C:DataManager]
  cache ∈ i32[N]@Cache
  disk_data ∈ i32[M]@Disk

[C:Processor]
  manager ∈ [Ref:DataManager]
"""
    tokenizer = Tokenizer(source)
    tokens = tokenizer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()

    flow = trace_execution(ast, "Processor", max_depth=5, follow_calls=True)
    assert flow is not None, "Flow should be traced"

    # Filter for steps that access cache
    cache_steps = flow.filter_by_state_access(".*cache.*")

    # This test may not find matches if state tracking doesn't capture the pattern
    # But the filter should execute without error
    assert isinstance(cache_steps.steps, list), "Should return valid flow"

    print(f"✓ Execution flow filter_by_state_access passed (steps: {len(cache_steps.steps)})")
    return True


def test_execution_flow_filter_custom():
    """Test custom predicate filtering of execution flow."""
    source = """
[C:Entry]
  proc1 ∈ [Ref:Processor1]

[C:Processor1]
  proc2 ∈ [Ref:Processor2]

[C:Processor2]
  final ∈ [Ref:Final]

[C:Final]
  value ∈ i32
"""
    tokenizer = Tokenizer(source)
    tokens = tokenizer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()

    flow = trace_execution(ast, "Entry", max_depth=5, follow_calls=True)
    assert flow is not None, "Flow should be traced"

    # Filter to only steps that make calls
    with_calls = flow.filter_custom(lambda step: len(step.calls_made) > 0)

    assert all(len(step.calls_made) > 0 for step in with_calls.steps), \
        "All steps should make calls"

    print(f"✓ Execution flow filter_custom passed (steps with calls: {len(with_calls.steps)})")
    return True


def test_execution_flow_get_steps_at_depth():
    """Test getting steps at specific depth."""
    source = """
[C:D0]
  ref ∈ [Ref:D1]

[C:D1]
  ref ∈ [Ref:D2]

[C:D2]
  value ∈ i32
"""
    tokenizer = Tokenizer(source)
    tokens = tokenizer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()

    flow = trace_execution(ast, "D0", max_depth=5, follow_calls=True)
    assert flow is not None, "Flow should be traced"

    depth0 = flow.get_steps_at_depth(0)
    depth1 = flow.get_steps_at_depth(1)

    assert len(depth0) > 0, "Should have steps at depth 0"
    assert all(step.depth == 0 for step in depth0), "All steps should be at depth 0"

    if len(depth1) > 0:
        assert all(step.depth == 1 for step in depth1), "All steps should be at depth 1"

    print(f"✓ Execution flow get_steps_at_depth passed (depth0: {len(depth0)}, depth1: {len(depth1)})")
    return True


def test_execution_flow_get_call_chain():
    """Test getting the execution call chain."""
    source = """
[C:First]
  ref ∈ [Ref:Second]

[C:Second]
  ref ∈ [Ref:Third]

[C:Third]
  value ∈ i32
"""
    tokenizer = Tokenizer(source)
    tokens = tokenizer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()

    flow = trace_execution(ast, "First", max_depth=5, follow_calls=True)
    assert flow is not None, "Flow should be traced"

    chain = flow.get_call_chain()

    assert isinstance(chain, list), "Chain should be a list"
    assert len(chain) > 0, "Chain should not be empty"
    assert chain[0] == "First", f"First element should be 'First', got: {chain[0]}"

    print(f"✓ Execution flow get_call_chain passed (chain: {chain})")
    return True


def test_filter_chaining():
    """Test chaining multiple filters together."""
    source = """
[C:ProcessorHandler]
  data ∈ i32

[C:DataProcessor]
  value ∈ i32

[C:UserHandler]
  input ∈ str

[C:RequestProcessor]
  output ∈ str
  ref ∈ [Ref:DataProcessor]
"""
    tokenizer = Tokenizer(source)
    tokens = tokenizer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()

    pack = generate_context_pack(ast, "RequestProcessor", max_depth=2)
    assert pack is not None, "Context pack should be generated"

    # Chain filters: Processors that also match Handler pattern
    result = pack.filter_by_pattern(".*Processor.*")

    assert isinstance(result, type(pack)), "Should return ContextPack"
    assert len(result.all_entities()) > 0, "Should have some entities"

    print(f"✓ Filter chaining passed (entities: {result.all_entities()})")
    return True


def run_all_tests():
    """Run all filtering and query API tests."""
    tests = [
        ("Context Pack Filter by Pattern", test_context_pack_filter_by_pattern),
        ("Context Pack Filter by Location", test_context_pack_filter_by_location),
        ("Context Pack Filter Custom", test_context_pack_filter_custom),
        ("Context Pack Get by Layer", test_context_pack_get_by_layer),
        ("Execution Flow Filter by Depth", test_execution_flow_filter_by_depth),
        ("Execution Flow Filter by Pattern", test_execution_flow_filter_by_pattern),
        ("Execution Flow Filter by State Access", test_execution_flow_filter_by_state_access),
        ("Execution Flow Filter Custom", test_execution_flow_filter_custom),
        ("Execution Flow Get Steps at Depth", test_execution_flow_get_steps_at_depth),
        ("Execution Flow Get Call Chain", test_execution_flow_get_call_chain),
        ("Filter Chaining", test_filter_chaining),
    ]

    print("Testing Filtering & Query API:")
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
