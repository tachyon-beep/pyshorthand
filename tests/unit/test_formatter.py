"""Unit tests for PyShorthand formatter."""

import unittest
from pyshort.core.tokenizer import Tokenizer
from pyshort.core.parser import Parser
from pyshort.formatter.formatter import Formatter, FormatConfig


class TestFormatter(unittest.TestCase):
    """Test auto-formatting functionality."""

    def test_basic_formatting(self):
        """Test basic formatting with alignment."""
        source = """# [M:Test]
[C:Example]
  x∈f32[N]@CPU
  meters ∈ f32[N, M]@GPU
"""
        tokenizer = Tokenizer(source)
        tokens = tokenizer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()

        formatter = Formatter()
        formatted = formatter.format_ast(ast)

        # Should have alignment
        self.assertIn("x      ∈", formatted)
        self.assertIn("meters ∈", formatted)

    def test_location_based_sorting(self):
        """Test sorting state vars by location."""
        source = """# [M:Test]
[C:Example]
  x∈f32[N]@CPU
  y∈f32[N]@GPU
  z∈f32[N]@Disk
"""
        tokenizer = Tokenizer(source)
        tokens = tokenizer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()

        config = FormatConfig(sort_state_by="location")
        formatter = Formatter(config)
        formatted = formatter.format_ast(ast)

        # GPU should come before CPU before Disk
        lines = formatted.split("\n")
        state_lines = [l for l in lines if "∈" in l]

        self.assertEqual(len(state_lines), 3)
        # Check ordering (GPU, CPU, Disk)
        self.assertIn("y", state_lines[0])  # GPU first
        self.assertIn("x", state_lines[1])  # CPU second
        self.assertIn("z", state_lines[2])  # Disk third

    def test_ascii_preference(self):
        """Test ASCII symbol preference."""
        source = """# [M:Test]
[C:Example]
  x ∈ f32[N]@CPU
"""
        tokenizer = Tokenizer(source)
        tokens = tokenizer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()

        config = FormatConfig(prefer_unicode=False)
        formatter = Formatter(config)
        formatted = formatter.format_ast(ast)

        # Should use ASCII IN instead of Unicode ∈
        self.assertIn("IN", formatted)
        self.assertNotIn("∈", formatted)

    def test_metadata_preservation(self):
        """Test that metadata is preserved."""
        source = """# [M:TestModule] [Role:Core] [Risk:High]
[C:Example]
  x ∈ f32[N]@CPU
"""
        tokenizer = Tokenizer(source)
        tokens = tokenizer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()

        formatter = Formatter()
        formatted = formatter.format_ast(ast)

        # Metadata should be at the top
        self.assertTrue(formatted.startswith("# [M:TestModule]"))
        self.assertIn("[Role:Core]", formatted)
        self.assertIn("[Risk:High]", formatted)


if __name__ == "__main__":
    unittest.main()
