"""Unit tests for v1.4 formatter enhancements (tag grouping)."""

import pytest

from pyshort.core.ast_nodes import Function, Metadata, PyShortAST, Tag
from pyshort.formatter.formatter import FormatConfig, Formatter


class TestFormatterV14TagGrouping:
    """Test formatter v1.4 tag grouping and ordering."""

    def test_format_single_operation_tag(self):
        """Test formatting single operation tag."""
        formatter = Formatter()
        tags = [Tag(base="Lin", qualifiers=["MatMul"], tag_type="operation")]
        result = formatter._format_tags(tags)
        assert result == "[Lin:MatMul]"

    def test_format_single_complexity_tag(self):
        """Test formatting single complexity tag."""
        formatter = Formatter()
        tags = [Tag(base="O(N)", tag_type="complexity")]
        result = formatter._format_tags(tags)
        assert result == "[O(N)]"

    def test_format_single_decorator_tag(self):
        """Test formatting single decorator tag."""
        formatter = Formatter()
        tags = [Tag(base="Prop", tag_type="decorator")]
        result = formatter._format_tags(tags)
        assert result == "[Prop]"

    def test_format_single_http_route_tag(self):
        """Test formatting single HTTP route tag."""
        formatter = Formatter()
        tags = [Tag(base="GET /users", tag_type="http_route", http_method="GET", http_path="/users")]
        result = formatter._format_tags(tags)
        assert result == "[GET /users]"

    def test_format_tags_in_correct_order(self):
        """Test that tags are formatted in correct order: Decorator, Route, Operation, Complexity."""
        formatter = Formatter()
        tags = [
            Tag(base="O(N)", tag_type="complexity"),  # Should be last
            Tag(base="Lin", qualifiers=["MatMul"], tag_type="operation"),  # Should be third
            Tag(base="GET /api", tag_type="http_route", http_method="GET", http_path="/api"),  # Should be second
            Tag(base="Prop", tag_type="decorator"),  # Should be first
        ]
        result = formatter._format_tags(tags)
        # Expected order: [Prop] [GET /api] [Lin:MatMul] [O(N)]
        assert result == "[Prop] [GET /api] [Lin:MatMul] [O(N)]"

    def test_format_multiple_decorator_tags(self):
        """Test formatting multiple decorator tags."""
        formatter = Formatter()
        tags = [
            Tag(base="Prop", tag_type="decorator"),
            Tag(base="Cached", qualifiers=["TTL", "60"], tag_type="decorator"),
        ]
        result = formatter._format_tags(tags)
        assert result == "[Prop] [Cached:TTL:60]"

    def test_format_multiple_operation_tags(self):
        """Test formatting multiple operation tags."""
        formatter = Formatter()
        tags = [
            Tag(base="NN", qualifiers=["∇"], tag_type="operation"),
            Tag(base="Lin", qualifiers=["MatMul"], tag_type="operation"),
        ]
        result = formatter._format_tags(tags)
        assert result == "[NN:∇] [Lin:MatMul]"

    def test_format_combined_neural_net_tag(self):
        """Test formatting combined neural net tag."""
        formatter = Formatter()
        tags = [Tag(base="NN", qualifiers=["∇", "Lin", "MatMul"], tag_type="operation")]
        result = formatter._format_tags(tags)
        assert result == "[NN:∇:Lin:MatMul]"

    def test_format_empty_tag_list(self):
        """Test formatting empty tag list."""
        formatter = Formatter()
        result = formatter._format_tags([])
        assert result == ""

    def test_format_custom_tag(self):
        """Test formatting custom tag type."""
        formatter = Formatter()
        tags = [Tag(base="CustomTag", qualifiers=["Arg1"], tag_type="custom")]
        result = formatter._format_tags(tags)
        assert result == "[CustomTag:Arg1]"


class TestFormatterV14FunctionSignatures:
    """Test formatting function signatures with v1.4 tags."""

    def test_format_function_with_decorator_tag(self):
        """Test formatting function with decorator tag."""
        func = Function(
            name="device",
            tags=[Tag(base="Prop", tag_type="decorator")],
        )
        ast = PyShortAST(metadata=Metadata(module_name="Test"), functions=[func])
        formatter = Formatter()
        result = formatter.format_ast(ast)
        assert "[Prop]" in result
        assert "F:device" in result

    def test_format_function_with_http_route_tag(self):
        """Test formatting function with HTTP route tag."""
        func = Function(
            name="get_user",
            tags=[Tag(base="GET /users/{id}", tag_type="http_route", http_method="GET", http_path="/users/{id}")],
        )
        ast = PyShortAST(metadata=Metadata(module_name="Test"), functions=[func])
        formatter = Formatter()
        result = formatter.format_ast(ast)
        assert "[GET /users/{id}]" in result

    def test_format_function_with_complexity_tag(self):
        """Test formatting function with complexity tag."""
        func = Function(
            name="process",
            tags=[Tag(base="O(N)", tag_type="complexity")],
        )
        ast = PyShortAST(metadata=Metadata(module_name="Test"), functions=[func])
        formatter = Formatter()
        result = formatter.format_ast(ast)
        assert "[O(N)]" in result

    def test_format_function_with_all_tag_types(self):
        """Test formatting function with all tag types in correct order."""
        func = Function(
            name="forward",
            tags=[
                Tag(base="O(B*N*D)", tag_type="complexity"),
                Tag(base="NN", qualifiers=["∇", "Lin"], tag_type="operation"),
                Tag(base="GET /api", tag_type="http_route", http_method="GET", http_path="/api"),
                Tag(base="Prop", tag_type="decorator"),
            ],
        )
        ast = PyShortAST(metadata=Metadata(module_name="Test"), functions=[func])
        formatter = Formatter()
        result = formatter.format_ast(ast)

        # Tags should appear in order: decorator, route, operation, complexity
        assert "F:forward" in result
        prop_idx = result.index("[Prop]")
        route_idx = result.index("[GET /api]")
        op_idx = result.index("[NN:∇:Lin]")
        comp_idx = result.index("[O(B*N*D)]")

        # Verify order
        assert prop_idx < route_idx < op_idx < comp_idx

    def test_format_function_with_modifiers_and_tags(self):
        """Test formatting function with both modifiers and tags."""
        func = Function(
            name="process",
            modifiers=["Async"],
            tags=[Tag(base="IO", qualifiers=["Net"], tag_type="operation")],
        )
        ast = PyShortAST(metadata=Metadata(module_name="Test"), functions=[func])
        formatter = Formatter()
        result = formatter.format_ast(ast)

        assert "[Async]" in result
        assert "[IO:Net]" in result

    def test_format_function_with_decorator_and_qualifiers(self):
        """Test formatting function with decorator tag with qualifiers."""
        func = Function(
            name="cached_func",
            tags=[Tag(base="Cached", qualifiers=["TTL", "60"], tag_type="decorator")],
        )
        ast = PyShortAST(metadata=Metadata(module_name="Test"), functions=[func])
        formatter = Formatter()
        result = formatter.format_ast(ast)

        assert "[Cached:TTL:60]" in result

    def test_format_function_backward_compatible_v13(self):
        """Test that v1.3 style tags still format correctly."""
        func = Function(
            name="process",
            tags=[Tag(base="Lin", qualifiers=["MatMul"], tag_type="operation")],
        )
        ast = PyShortAST(metadata=Metadata(module_name="Test"), functions=[func])
        formatter = Formatter()
        result = formatter.format_ast(ast)

        assert "[Lin:MatMul]" in result


class TestFormatterV14UnicodeASCII:
    """Test formatter Unicode/ASCII preference with v1.4 tags."""

    def test_format_unicode_preference(self):
        """Test formatting with Unicode preference."""
        from pyshort.core.ast_nodes import TypeSpec
        func = Function(
            name="forward",
            return_type=TypeSpec(base_type="Tensor"),
            tags=[Tag(base="NN", qualifiers=["∇"], tag_type="operation")],
        )
        ast = PyShortAST(metadata=Metadata(module_name="Test"), functions=[func])
        formatter = Formatter(FormatConfig(prefer_unicode=True))
        result = formatter.format_ast(ast)

        assert "→" in result  # Unicode arrow
        assert "[NN:∇]" in result  # Unicode gradient

    def test_format_ascii_preference(self):
        """Test formatting with ASCII preference."""
        from pyshort.core.ast_nodes import TypeSpec
        func = Function(
            name="forward",
            return_type=TypeSpec(base_type="Tensor"),
            tags=[Tag(base="NN", qualifiers=["∇"], tag_type="operation")],
        )
        ast = PyShortAST(metadata=Metadata(module_name="Test"), functions=[func])
        formatter = Formatter(FormatConfig(prefer_unicode=False))
        result = formatter.format_ast(ast)

        assert "->" in result  # ASCII arrow
        # Note: Tag content might still have Unicode, depends on implementation


class TestFormatterV14EdgeCases:
    """Test formatter edge cases with v1.4."""

    def test_format_function_no_tags(self):
        """Test formatting function without tags."""
        func = Function(name="simple")
        ast = PyShortAST(metadata=Metadata(module_name="Test"), functions=[func])
        formatter = Formatter()
        result = formatter.format_ast(ast)

        assert "F:simple" in result
        # Should not have any tag brackets for this function
        lines = result.split("\n")
        func_line = [l for l in lines if "F:simple" in l][0]
        # Count brackets in function line
        assert func_line.count("[") == 0

    def test_format_function_only_decorator(self):
        """Test formatting function with only decorator tag."""
        func = Function(
            name="prop_func",
            tags=[Tag(base="Prop", tag_type="decorator")],
        )
        ast = PyShortAST(metadata=Metadata(module_name="Test"), functions=[func])
        formatter = Formatter()
        result = formatter.format_ast(ast)

        assert "F:prop_func" in result
        assert "[Prop]" in result

    def test_format_function_only_complexity(self):
        """Test formatting function with only complexity tag."""
        func = Function(
            name="complex_func",
            tags=[Tag(base="O(N²)", tag_type="complexity")],
        )
        ast = PyShortAST(metadata=Metadata(module_name="Test"), functions=[func])
        formatter = Formatter()
        result = formatter.format_ast(ast)

        assert "[O(N²)]" in result

    def test_format_preserves_tag_order_in_group(self):
        """Test that multiple tags of same type preserve order."""
        func = Function(
            name="multi",
            tags=[
                Tag(base="Auth", tag_type="decorator"),
                Tag(base="RateLimit", qualifiers=["100"], tag_type="decorator"),
            ],
        )
        ast = PyShortAST(metadata=Metadata(module_name="Test"), functions=[func])
        formatter = Formatter()
        result = formatter.format_ast(ast)

        # Both decorators should appear
        assert "[Auth]" in result
        assert "[RateLimit:100]" in result

        # Auth should come before RateLimit (order preserved)
        auth_idx = result.index("[Auth]")
        rate_idx = result.index("[RateLimit:100]")
        assert auth_idx < rate_idx


class TestFormatterV14Integration:
    """Integration tests for v1.4 formatter."""

    def test_format_complete_example(self):
        """Test formatting a complete example with v1.4 features."""
        func = Function(
            name="api_endpoint",
            tags=[
                Tag(base="POST /api/process", tag_type="http_route", http_method="POST", http_path="/api/process"),
                Tag(base="Auth", tag_type="decorator"),
                Tag(base="NN", qualifiers=["∇", "Lin"], tag_type="operation"),
                Tag(base="O(B*N*D)", tag_type="complexity"),
            ],
        )
        ast = PyShortAST(metadata=Metadata(module_name="TestAPI"), functions=[func])
        formatter = Formatter()
        result = formatter.format_ast(ast)

        # Check all elements present
        assert "# [M:TestAPI]" in result
        assert "F:api_endpoint" in result
        assert "[Auth]" in result
        assert "[POST /api/process]" in result
        assert "[NN:∇:Lin]" in result
        assert "[O(B*N*D)]" in result

    def test_roundtrip_formatting(self):
        """Test that formatting is idempotent."""
        func = Function(
            name="test",
            tags=[
                Tag(base="Prop", tag_type="decorator"),
                Tag(base="Lin", qualifiers=["MatMul"], tag_type="operation"),
                Tag(base="O(N)", tag_type="complexity"),
            ],
        )
        ast = PyShortAST(metadata=Metadata(module_name="Test"), functions=[func])
        formatter = Formatter()

        # Format once
        result1 = formatter.format_ast(ast)

        # Parse and format again
        from pyshort.core.parser import parse_string
        ast2 = parse_string(result1)
        result2 = formatter.format_ast(ast2)

        # Should be identical
        assert result1 == result2
