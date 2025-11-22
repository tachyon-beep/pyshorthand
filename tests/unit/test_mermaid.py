"""Unit tests for Mermaid diagram generation."""

import unittest
from pyshort.core.tokenizer import Tokenizer
from pyshort.core.parser import Parser
from pyshort.visualization.mermaid import generate_mermaid, MermaidConfig


class TestMermaid(unittest.TestCase):
    """Test Mermaid diagram generation."""

    def test_flowchart_generation(self):
        """Test basic flowchart generation."""
        source = """# [M:Test]

[C:VHE]
  ◊ [Ref:Substrate]

  pos ∈ f32[N,D]@GPU
"""
        tokenizer = Tokenizer(source)
        tokens = tokenizer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()

        diagram = generate_mermaid(ast, diagram_type="flowchart")

        self.assertIn("flowchart TB", diagram)
        self.assertIn("VHE", diagram)
        self.assertIn("Substrate", diagram)
        # Simple references use dashed arrows
        self.assertIn("-..->", diagram)

    def test_class_diagram_generation(self):
        """Test class diagram generation."""
        source = """# [M:Test]

[C:VHE]
  pos ∈ f32[N,D]@GPU
  meters ∈ f32[N,M]@GPU
"""
        tokenizer = Tokenizer(source)
        tokens = tokenizer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()

        diagram = generate_mermaid(ast, diagram_type="classDiagram")

        self.assertIn("classDiagram", diagram)
        self.assertIn("class VHE", diagram)
        self.assertIn("pos:", diagram)
        self.assertIn("meters:", diagram)
        self.assertIn("f32[N, D]@GPU", diagram)

    def test_risk_coloring(self):
        """Test risk-based coloring."""
        source = """# [M:Test] [Risk:High]

[C:VHE]
"""
        tokenizer = Tokenizer(source)
        tokens = tokenizer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()

        diagram = generate_mermaid(
            ast,
            diagram_type="flowchart",
            color_by_risk=True
        )

        self.assertIn("style VHE fill:#ff6b6b", diagram)  # Red for High risk

    def test_direction_setting(self):
        """Test diagram direction setting."""
        source = """# [M:Test]

[C:VHE]
"""
        tokenizer = Tokenizer(source)
        tokens = tokenizer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()

        # Test left-to-right
        diagram_lr = generate_mermaid(ast, diagram_type="flowchart", direction="LR")
        self.assertIn("flowchart LR", diagram_lr)

        # Test top-to-bottom
        diagram_tb = generate_mermaid(ast, diagram_type="flowchart", direction="TB")
        self.assertIn("flowchart TB", diagram_tb)

    def test_dependencies_shown(self):
        """Test that dependencies are shown as edges."""
        source = """# [M:Test]

[C:VHE]
  ◊ [Ref:Substrate]
"""
        tokenizer = Tokenizer(source)
        tokens = tokenizer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()

        diagram = generate_mermaid(ast, diagram_type="flowchart")

        # Should have edges to dependencies
        self.assertIn("Substrate", diagram)
        # Check there's an edge
        self.assertIn("-..->", diagram)


if __name__ == "__main__":
    unittest.main()
