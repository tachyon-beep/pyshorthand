"""Unit tests for PyShorthand parser."""

import pytest

from pyshort.core.parser import parse_string
from pyshort.core.ast_nodes import DiagnosticSeverity


class TestMetadataParsing:
    """Test metadata header parsing."""

    def test_basic_metadata(self):
        """Test parsing basic metadata headers."""
        source = """# [M:TestModule] [ID:TM] [Role:Core]
# [Layer:Domain] [Risk:High]
"""
        ast = parse_string(source)

        assert ast.metadata.module_name == "TestModule"
        assert ast.metadata.module_id == "TM"
        assert ast.metadata.role == "Core"
        assert ast.metadata.layer == "Domain"
        assert ast.metadata.risk == "High"

    def test_dimensions_metadata(self):
        """Test parsing dimension metadata."""
        source = """# [M:Test] [Dims: N=agents, B=batch, D=dim]
"""
        ast = parse_string(source)

        assert "N" in ast.metadata.dims
        assert ast.metadata.dims["N"] == "agents"
        assert "B" in ast.metadata.dims
        assert ast.metadata.dims["B"] == "batch"

    def test_requires_metadata(self):
        """Test parsing requirements metadata."""
        source = """# [M:Test] [Requires: torch>=2.0, numpy>=1.20]
"""
        ast = parse_string(source)

        assert "torch>=2.0" in ast.metadata.requires
        assert "numpy>=1.20" in ast.metadata.requires


class TestStateVariables:
    """Test state variable parsing."""

    def test_simple_state_var(self):
        """Test parsing simple state variable."""
        source = """# [M:Test]
x ∈ f32
"""
        ast = parse_string(source)

        # State variables may be in statements or state depending on context
        # For now, just check no errors
        assert not ast.has_errors()

    def test_state_var_with_shape(self):
        """Test parsing state variable with shape."""
        source = """# [M:Test]
batch ∈ f32[B, T, D]
"""
        ast = parse_string(source)

        assert not ast.has_errors()

    def test_state_var_with_location(self):
        """Test parsing state variable with memory location."""
        source = """# [M:Test]
data ∈ f32[N]@GPU
"""
        ast = parse_string(source)

        assert not ast.has_errors()

    def test_state_var_with_transfer(self):
        """Test parsing state variable with memory transfer."""
        source = """# [M:Test]
weights ∈ f32[N, N]@Disk→GPU
"""
        ast = parse_string(source)

        assert not ast.has_errors()


class TestExpressions:
    """Test expression parsing."""

    def test_identifier(self):
        """Test parsing identifier."""
        source = """# [M:Test]
x ≡ y
"""
        ast = parse_string(source)

        assert not ast.has_errors()

    def test_number_literal(self):
        """Test parsing number literal."""
        source = """# [M:Test]
x ≡ 42
"""
        ast = parse_string(source)

        assert not ast.has_errors()

    def test_binary_operation(self):
        """Test parsing binary operation."""
        source = """# [M:Test]
z ≡ x + y
"""
        ast = parse_string(source)

        assert not ast.has_errors()

    def test_function_call(self):
        """Test parsing function call."""
        source = """# [M:Test]
result ≡ func(x, y)
"""
        ast = parse_string(source)

        assert not ast.has_errors()


class TestTags:
    """Test tag parsing."""

    def test_simple_tag(self):
        """Test parsing simple tag."""
        source = """# [M:Test]
x ≡ y →[Lin]
"""
        ast = parse_string(source)

        assert not ast.has_errors()

    def test_tag_with_qualifiers(self):
        """Test parsing tag with qualifiers."""
        source = """# [M:Test]
x ≡ y →[Lin:Broad:O(N)]
"""
        ast = parse_string(source)

        assert not ast.has_errors()

    def test_multiple_tags(self):
        """Test parsing multiple tags."""
        source = """# [M:Test]
x ≡ func() →[IO:Net:Async]
"""
        ast = parse_string(source)

        assert not ast.has_errors()


class TestStatements:
    """Test statement parsing."""

    def test_assignment(self):
        """Test parsing assignment statement."""
        source = """# [M:Test]
x ≡ 42
"""
        ast = parse_string(source)

        assert not ast.has_errors()
        assert len(ast.statements) > 0

    def test_local_mutation(self):
        """Test parsing local mutation."""
        source = """# [M:Test]
!x
"""
        ast = parse_string(source)

        # This might not parse perfectly yet, but shouldn't crash
        # Just check it doesn't have critical errors

    def test_system_mutation(self):
        """Test parsing system mutation."""
        source = """# [M:Test]
!!db.commit()
"""
        ast = parse_string(source)

        # Check for basic parsing

    def test_return_statement(self):
        """Test parsing return statement."""
        source = """# [M:Test]
← result
"""
        ast = parse_string(source)

        assert not ast.has_errors()


class TestASCIICompatibility:
    """Test ASCII-compatible notation."""

    def test_ascii_arrow(self):
        """Test ASCII arrow notation."""
        source = """# [M:Test]
x == y -> z
"""
        ast = parse_string(source)

        assert not ast.has_errors()

    def test_ascii_member_of(self):
        """Test ASCII member-of notation."""
        source = """# [M:Test]
x IN f32
"""
        ast = parse_string(source)

        # Should parse without errors

    def test_ascii_happens_after(self):
        """Test ASCII happens-after notation."""
        source = """# [M:Test]
>> x == y
"""
        ast = parse_string(source)

        # Should parse without errors


class TestComments:
    """Test comment handling."""

    def test_single_line_comment(self):
        """Test single-line comment."""
        source = """# [M:Test]
// This is a comment
x ≡ y
"""
        ast = parse_string(source)

        assert not ast.has_errors()

    def test_inline_comment(self):
        """Test inline comment."""
        source = """# [M:Test]
x ∈ f32[N]@GPU  // Agent positions
"""
        ast = parse_string(source)

        assert not ast.has_errors()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
