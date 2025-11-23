"""Integration tests for PyShorthand v1.4 features.

Tests the complete workflow: parse → validate → format for v1.4 tag types.
"""

import pytest

from pyshort.core.ast_nodes import DiagnosticSeverity
from pyshort.core.parser import parse_string
from pyshort.core.validator import Linter
from pyshort.formatter.formatter import Formatter


class TestV14EndToEnd:
    """End-to-end integration tests for v1.4."""

    def test_decorator_tag_workflow(self):
        """Test complete workflow with decorator tags."""
        source = """# [M:TestModule]
F:device() → str [Prop]
"""
        # Parse
        ast = parse_string(source)
        assert len(ast.functions) == 1
        assert ast.functions[0].tags[0].tag_type == "decorator"

        # Validate
        linter = Linter()
        diagnostics = linter.check(ast)
        errors = [d for d in diagnostics if d.severity == DiagnosticSeverity.ERROR]
        assert len(errors) == 0

        # Format
        formatter = Formatter()
        result = formatter.format_ast(ast)
        assert "[Prop]" in result

    def test_http_route_tag_workflow(self):
        """Test complete workflow with HTTP route tags."""
        source = """# [M:API]
F:get_user(id: i32) → User [GET/users/{id}]
"""
        # Parse
        ast = parse_string(source)
        assert len(ast.functions) == 1
        tag = ast.functions[0].tags[0]
        assert tag.tag_type == "http_route"
        assert tag.http_method == "GET"
        assert tag.http_path == "/users/{id}"

        # Validate
        linter = Linter()
        diagnostics = linter.check(ast)
        errors = [d for d in diagnostics if d.severity == DiagnosticSeverity.ERROR]
        assert len(errors) == 0

        # Format
        formatter = Formatter()
        result = formatter.format_ast(ast)
        assert "[GET /users/{id}]" in result

    def test_complexity_tag_workflow(self):
        """Test complete workflow with complexity tags."""
        source = """# [M:Algorithms]
F:process(data: List) → Result [O(N*M)]
"""
        # Parse
        ast = parse_string(source)
        assert len(ast.functions) == 1
        tag = ast.functions[0].tags[0]
        assert tag.tag_type == "complexity"
        assert tag.complexity == "O(N*M)"

        # Validate
        linter = Linter()
        diagnostics = linter.check(ast)
        errors = [d for d in diagnostics if d.severity == DiagnosticSeverity.ERROR]
        assert len(errors) == 0

        # Format
        formatter = Formatter()
        result = formatter.format_ast(ast)
        assert "[O(N*M)]" in result

    def test_combined_tags_workflow(self):
        """Test complete workflow with all tag types combined."""
        source = """# [M:NeuralNet]
F:forward(x: Tensor) → Tensor [Prop] [GET/api/forward] [NN:∇:Lin:MatMul] [O(B*N*D)]
"""
        # Parse
        ast = parse_string(source)
        assert len(ast.functions) == 1
        tags = ast.functions[0].tags
        assert len(tags) == 4

        tag_types = {tag.tag_type for tag in tags}
        assert "decorator" in tag_types
        assert "http_route" in tag_types
        assert "operation" in tag_types
        assert "complexity" in tag_types

        # Validate
        linter = Linter()
        diagnostics = linter.check(ast)
        errors = [d for d in diagnostics if d.severity == DiagnosticSeverity.ERROR]
        assert len(errors) == 0

        # Format (should maintain tag order)
        formatter = Formatter()
        result = formatter.format_ast(ast)
        assert "[Prop]" in result
        assert "[GET /api/forward]" in result
        assert "[NN:∇:Lin:MatMul]" in result
        assert "[O(B*N*D)]" in result

        # Verify order
        prop_idx = result.index("[Prop]")
        route_idx = result.index("[GET /api/forward]")
        op_idx = result.index("[NN:∇:Lin:MatMul]")
        comp_idx = result.index("[O(B*N*D)]")
        assert prop_idx < route_idx < op_idx < comp_idx

    def test_validator_catches_conflicts(self):
        """Test that validator catches v1.4 tag conflicts."""
        source = """# [M:Test]
F:bad_function() → None [Prop] [Static]
"""
        # Parse
        ast = parse_string(source)

        # Validate - should catch Prop+Static conflict
        linter = Linter()
        diagnostics = linter.check(ast)
        errors = [d for d in diagnostics if d.severity == DiagnosticSeverity.ERROR]
        assert len(errors) > 0
        assert any("conflicting" in d.message.lower() for d in errors)

    def test_validator_catches_invalid_rate_limit(self):
        """Test that validator catches invalid rate limit."""
        source = """# [M:Test]
F:limited() → None [RateLimit:-1]
"""
        # Parse
        ast = parse_string(source)

        # Validate
        linter = Linter()
        diagnostics = linter.check(ast)
        errors = [d for d in diagnostics if d.severity == DiagnosticSeverity.ERROR]
        assert len(errors) > 0
        assert any("must be positive" in d.message for d in errors)

    def test_roundtrip_preserves_tags(self):
        """Test that parse → format → parse preserves all tag information."""
        original = """# [M:TestModule]
F:api_call(x: i32) → str [Auth] [POST/api/call] [IO:Net] [O(1)]
"""
        # First parse
        ast1 = parse_string(original)
        func1 = ast1.functions[0]

        # Format
        formatter = Formatter()
        formatted = formatter.format_ast(ast1)

        # Second parse
        ast2 = parse_string(formatted)
        func2 = ast2.functions[0]

        # Compare tags
        assert len(func1.tags) == len(func2.tags)
        for tag1, tag2 in zip(func1.tags, func2.tags):
            assert tag1.base == tag2.base
            assert tag1.tag_type == tag2.tag_type
            assert tag1.qualifiers == tag2.qualifiers
            if tag1.tag_type == "http_route":
                assert tag1.http_method == tag2.http_method
                assert tag1.http_path == tag2.http_path


class TestV14RealWorldExamples:
    """Test v1.4 with real-world-like examples."""

    def test_fastapi_endpoint(self):
        """Test FastAPI-style endpoint documentation."""
        source = """# [M:UserAPI] [Role:Core]
F:get_user(user_id: i32) → User [GET/api/users/{user_id}] [Auth] [IO:Net:Async] [O(1)]
F:create_user(user: User) → User [POST/api/users] [Auth] [RateLimit:100] [IO:Net:Async] [O(1)]
F:update_user(user_id: i32, user: User) → User [PUT/api/users/{user_id}] [Auth] [IO:Net:Async] [O(1)]
F:delete_user(user_id: i32) → None [DELETE/api/users/{user_id}] [Auth] [IO:Net:Async] [O(1)]
"""
        ast = parse_string(source)
        assert len(ast.functions) == 4

        # Validate all endpoints
        linter = Linter()
        diagnostics = linter.check(ast)
        errors = [d for d in diagnostics if d.severity == DiagnosticSeverity.ERROR]
        assert len(errors) == 0

        # Check HTTP methods
        methods = [f.tags[0].http_method for f in ast.functions if f.tags and f.tags[0].tag_type == "http_route"]
        assert "GET" in methods
        assert "POST" in methods
        assert "PUT" in methods
        assert "DELETE" in methods

    def test_neural_network_model(self):
        """Test neural network model documentation."""
        source = """# [M:TransformerBlock]
F:__init__(config: Config) → None
F:forward(x: Tensor) → Tensor [NN:∇:Lin:MatMul:Thresh:Softmax] [O(B*N²*D)]
F:attention(q: Tensor, k: Tensor, v: Tensor) → Tensor [NN:∇:Lin:MatMul] [O(B*N²*D)]
F:feed_forward(x: Tensor) → Tensor [NN:∇:Lin:MatMul] [O(B*N*D)]
"""
        ast = parse_string(source)
        assert len(ast.functions) == 4

        # Validate
        linter = Linter()
        diagnostics = linter.check(ast)
        errors = [d for d in diagnostics if d.severity == DiagnosticSeverity.ERROR]
        assert len(errors) == 0

        # Check complexity tags
        complexities = [
            tag.base
            for f in ast.functions
            for tag in f.tags
            if tag.tag_type == "complexity"
        ]
        assert "O(B*N²*D)" in complexities
        assert "O(B*N*D)" in complexities

    def test_cached_property_pattern(self):
        """Test cached property pattern."""
        source = """# [M:Model]
[C:LargeModel]
  _cache ∈ Dict

  # Methods:
  # F:expensive_property() → Result [Prop] [Cached] [O(N²)]
  # F:invalidate_cache() → None
"""
        ast = parse_string(source)
        # Note: Comment-based methods might not parse, this tests the syntax

        linter = Linter()
        diagnostics = linter.check(ast)
        # Should not crash

    def test_algorithm_complexity_documentation(self):
        """Test algorithm with various complexity levels."""
        source = """# [M:Algorithms]
F:linear_search(arr: List, target: i32) → i32 [Iter:Sequential:O(N)]
F:binary_search(arr: List, target: i32) → i32 [Iter:O(log N)]
F:bubble_sort(arr: List) → List [Iter:Nested:O(N²)]
F:quick_sort(arr: List) → List [Iter:Nested:O(N log N)]
F:matrix_multiply(a: Matrix, b: Matrix) → Matrix [Lin:MatMul:O(N³)]
"""
        ast = parse_string(source)
        assert len(ast.functions) == 5

        # Validate
        linter = Linter()
        diagnostics = linter.check(ast)
        errors = [d for d in diagnostics if d.severity == DiagnosticSeverity.ERROR]
        assert len(errors) == 0

        # All should have complexity tags
        complexities = [
            tag.complexity
            for f in ast.functions
            for tag in f.tags
            if tag.complexity
        ]
        assert len(complexities) == 5


class TestV14BackwardCompatibility:
    """Test that v1.4 maintains backward compatibility with v1.3."""

    def test_v13_operation_tags_still_work(self):
        """Test that v1.3 operation tags still parse and validate."""
        source = """# [M:Test]
F:old_style(x: i32) → i32 [Lin:MatMul]
F:io_operation() → None [IO:Disk:Block]
F:iteration(arr: List) → None [Iter:Hot]
"""
        # Parse
        ast = parse_string(source)
        assert len(ast.functions) == 3

        # All should be operation tags
        for func in ast.functions:
            assert func.tags[0].tag_type == "operation"

        # Validate
        linter = Linter()
        diagnostics = linter.check(ast)
        errors = [d for d in diagnostics if d.severity == DiagnosticSeverity.ERROR]
        assert len(errors) == 0

        # Format
        formatter = Formatter()
        result = formatter.format_ast(ast)
        assert "[Lin:MatMul]" in result
        assert "[IO:Disk:Block]" in result
        assert "[Iter:Hot]" in result

    def test_mixed_v13_and_v14_tags(self):
        """Test mixing v1.3 and v1.4 tags in same file."""
        source = """# [M:MixedVersion]
F:old_function(x: i32) → i32 [Lin:MatMul]
F:new_function(x: i32) → i32 [Prop] [GET/api/data] [O(N)]
"""
        # Parse
        ast = parse_string(source)
        assert len(ast.functions) == 2

        # Validate
        linter = Linter()
        diagnostics = linter.check(ast)
        errors = [d for d in diagnostics if d.severity == DiagnosticSeverity.ERROR]
        assert len(errors) == 0

        # Format
        formatter = Formatter()
        result = formatter.format_ast(ast)
        assert "[Lin:MatMul]" in result  # v1.3
        assert "[Prop]" in result  # v1.4
        assert "[GET /api/data]" in result  # v1.4
        assert "[O(N)]" in result  # v1.4


class TestV14ErrorHandling:
    """Test error handling for v1.4 features."""

    def test_invalid_complexity_notation(self):
        """Test handling of invalid complexity notation."""
        # This should be caught during Tag creation, not parsing
        from pyshort.core.ast_nodes import Tag
        with pytest.raises(ValueError, match="Invalid complexity notation"):
            Tag(base="O(", tag_type="complexity")

    def test_missing_http_path(self):
        """Test handling of HTTP route without path."""
        from pyshort.core.ast_nodes import Tag
        with pytest.raises(ValueError, match="must have both http_method and http_path"):
            Tag(base="GET", tag_type="http_route", http_method="GET", http_path=None)

    def test_invalid_http_method(self):
        """Test handling of invalid HTTP method."""
        from pyshort.core.ast_nodes import Tag
        with pytest.raises(ValueError, match="Invalid HTTP method"):
            Tag(base="INVALID", tag_type="http_route", http_method="INVALID", http_path="/path")

    def test_parse_handles_malformed_gracefully(self):
        """Test that parser handles malformed input gracefully."""
        # Parser currently skips malformed functions instead of raising errors
        source = """# [M:Test]
F:bad_func() → None [
"""
        ast = parse_string(source)
        # Malformed function should be skipped
        assert len(ast.functions) == 0
        # Metadata should still be parsed
        assert ast.metadata.module_name == "Test"
