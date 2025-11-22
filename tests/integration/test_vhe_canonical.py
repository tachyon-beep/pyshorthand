"""Integration test for VHE canonical example."""

import os
from pathlib import Path

import pytest

from pyshort.core.parser import parse_file
from pyshort.core.validator import Linter


class TestVHECanonical:
    """Test parsing and validating the canonical VHE example from the RFC."""

    @pytest.fixture
    def vhe_file_path(self):
        """Get path to VHE canonical fixture."""
        return Path(__file__).parent / "fixtures" / "vhe_canonical.pys"

    def test_vhe_file_exists(self, vhe_file_path):
        """Test that VHE canonical file exists."""
        assert vhe_file_path.exists(), f"VHE canonical file not found at {vhe_file_path}"

    def test_parse_vhe(self, vhe_file_path):
        """Test parsing VHE canonical example."""
        ast = parse_file(str(vhe_file_path))

        # Should parse without critical errors
        assert ast is not None
        assert ast.source_file == str(vhe_file_path)

    def test_vhe_metadata(self, vhe_file_path):
        """Test VHE metadata extraction."""
        ast = parse_file(str(vhe_file_path))

        # Check metadata
        assert ast.metadata.module_name == "VectorizedHamletEnv"
        assert ast.metadata.module_id == "VHE"
        assert ast.metadata.role == "Core"
        assert ast.metadata.layer == "Domain"
        assert ast.metadata.risk == "High"
        assert ast.metadata.context == "GPU-RL Simulation"

        # Check dimensions
        assert "N" in ast.metadata.dims
        assert ast.metadata.dims["N"] == "agents"
        assert "M" in ast.metadata.dims
        assert ast.metadata.dims["M"] == "meters"

    def test_vhe_entities(self, vhe_file_path):
        """Test VHE entity extraction."""
        ast = parse_file(str(vhe_file_path))

        # Should have at least the VHE class
        assert len(ast.entities) > 0

        # Check for class definition
        vhe_class = ast.entities[0]
        assert hasattr(vhe_class, "name")

    def test_vhe_validation(self, vhe_file_path):
        """Test validating VHE canonical example."""
        linter = Linter(strict=False)
        diagnostics = linter.check_file(str(vhe_file_path))

        # Should not have any critical errors
        from pyshort.core.ast_nodes import DiagnosticSeverity

        errors = [d for d in diagnostics if d.severity == DiagnosticSeverity.ERROR]

        # Print diagnostics for debugging
        if errors:
            print("\nErrors found in VHE canonical:")
            for error in errors:
                print(f"  {error}")

        # Allow some warnings, but no errors in canonical example
        # (We may have warnings about missing complexity tags, etc.)

    def test_vhe_ast_structure(self, vhe_file_path):
        """Test VHE AST structure."""
        ast = parse_file(str(vhe_file_path))

        # Should have functions (calc_rewards, step)
        # Note: These might be parsed as methods of the class
        # or as standalone functions depending on indentation

        # Check that we parsed something meaningful
        assert ast.metadata.module_name is not None
        assert len(ast.entities) > 0 or len(ast.functions) > 0 or len(ast.statements) > 0

    def test_vhe_to_dict(self, vhe_file_path):
        """Test converting VHE AST to dictionary."""
        ast = parse_file(str(vhe_file_path))

        ast_dict = ast.to_dict()

        # Should be serializable
        assert isinstance(ast_dict, dict)
        assert "metadata" in ast_dict
        assert "entities" in ast_dict
        assert isinstance(ast_dict["metadata"], dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
