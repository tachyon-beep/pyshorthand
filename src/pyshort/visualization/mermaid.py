"""Mermaid diagram generator for PyShorthand ASTs.

Generates Mermaid syntax for documentation-friendly visualizations:
- Dataflow graphs: Entity/function dependencies with colored edges by risk
- Entity relationship diagrams: Class structures with state and methods
- Architecture diagrams: Module-level overview with layer organization
"""

from typing import List, Set, Dict, Optional
from dataclasses import dataclass
from ..core.ast_nodes import (
    PyShortAST,
    Entity,
    Class,
    Function,
    StateVar,
    Reference,
    DiagnosticSeverity,
)


@dataclass
class MermaidConfig:
    """Configuration for Mermaid diagram generation."""

    diagram_type: str = "flowchart"  # "flowchart", "classDiagram", "graph"
    direction: str = "TB"  # TB (top-bottom), LR (left-right), RL, BT
    show_state_vars: bool = True
    show_methods: bool = True
    show_dependencies: bool = True
    show_transfers: bool = True
    color_by_risk: bool = True
    include_metadata: bool = True
    max_label_length: int = 30


class MermaidGenerator:
    """Generates Mermaid diagrams from PyShorthand ASTs."""

    def __init__(self, config: Optional[MermaidConfig] = None):
        self.config = config or MermaidConfig()
        self.node_counter = 0
        self.node_ids: Dict[str, str] = {}

    def generate(self, ast: PyShortAST) -> str:
        """Generate Mermaid diagram from AST."""
        if self.config.diagram_type == "flowchart":
            return self.generate_flowchart(ast)
        elif self.config.diagram_type == "classDiagram":
            return self.generate_class_diagram(ast)
        else:
            return self.generate_graph(ast)

    def generate_flowchart(self, ast: PyShortAST) -> str:
        """Generate a flowchart showing dataflow and dependencies."""
        lines = [f"flowchart {self.config.direction}"]

        # Add metadata as subgraph if requested
        if self.config.include_metadata and ast.metadata.module_name:
            lines.append(f"  subgraph Module[\"{ast.metadata.module_name}\"]")
            if ast.metadata.role:
                lines.append(f"    direction {self.config.direction}")

            # Add entities
            for entity in ast.entities:
                node_id = self._get_node_id(entity.name)
                if isinstance(entity, Class):
                    label = self._format_class_label(entity)
                    shape = self._get_shape_for_entity(entity)
                    lines.append(f"    {node_id}{shape}")

            # Add functions
            for func in ast.functions:
                node_id = self._get_node_id(func.name)
                label = self._format_function_label(func)
                lines.append(f"    {node_id}[[\"{label}\"]]")

            lines.append("  end")
        else:
            # No subgraph, just add nodes
            for entity in ast.entities:
                node_id = self._get_node_id(entity.name)
                shape = self._get_shape_for_entity(entity)
                lines.append(f"  {node_id}{shape}")

            for func in ast.functions:
                node_id = self._get_node_id(func.name)
                label = self._format_function_label(func)
                lines.append(f"  {node_id}[[\"{label}\"]]")

        # Add dependencies as edges
        if self.config.show_dependencies:
            for entity in ast.entities:
                if isinstance(entity, Class):
                    self._add_dependency_edges(entity, lines)

        # Add styling based on risk
        if self.config.color_by_risk and ast.metadata.risk:
            risk_color = self._get_risk_color(ast.metadata.risk)
            for entity in ast.entities:
                node_id = self._get_node_id(entity.name)
                lines.append(f"  style {node_id} fill:{risk_color}")

        return "\n".join(lines)

    def generate_class_diagram(self, ast: PyShortAST) -> str:
        """Generate a UML-like class diagram."""
        lines = ["classDiagram"]

        # Add classes
        for entity in ast.entities:
            if isinstance(entity, Class):
                lines.append(f"  class {entity.name} {{")

                # Add state variables
                if self.config.show_state_vars:
                    for state_var in entity.state:
                        type_spec = self._format_type_spec(state_var)
                        lines.append(f"    {state_var.name}: {type_spec}")

                # Add methods
                if self.config.show_methods:
                    for method in entity.methods:
                        params = ", ".join(
                            f"{p.name}: {p.type_spec or 'Any'}"
                            for p in method.parameters
                        )
                        ret_type = method.return_type or "void"
                        lines.append(f"    {method.name}({params}): {ret_type}")

                lines.append("  }")

                # Add dependencies
                if self.config.show_dependencies:
                    for dep in entity.dependencies:
                        # dep is a Reference with ref_id field
                        target = dep.ref_id
                        lines.append(f"  {entity.name} --> {target}")

        return "\n".join(lines)

    def generate_graph(self, ast: PyShortAST) -> str:
        """Generate a simple graph showing relationships."""
        lines = [f"graph {self.config.direction}"]

        # Add nodes
        for entity in ast.entities:
            node_id = self._get_node_id(entity.name)
            lines.append(f"  {node_id}[\"{entity.name}\"]")

        # Add edges from dependencies
        for entity in ast.entities:
            if isinstance(entity, Class):
                for dep in entity.dependencies:
                    # dep is a Reference with ref_id field
                    target = dep.ref_id
                    target_id = self._get_node_id(target)
                    entity_id = self._get_node_id(entity.name)
                    lines.append(f"  {entity_id} --> {target_id}")

        return "\n".join(lines)

    def _get_node_id(self, name: str) -> str:
        """Get or create a sanitized node ID."""
        if name not in self.node_ids:
            # Sanitize name for Mermaid (alphanumeric + underscore)
            sanitized = "".join(c if c.isalnum() else "_" for c in name)
            self.node_ids[name] = sanitized
        return self.node_ids[name]

    def _format_class_label(self, cls: Class) -> str:
        """Format a class label for display."""
        label_parts = [cls.name]

        if self.config.show_state_vars and cls.state:
            state_count = len(cls.state)
            label_parts.append(f"({state_count} vars)")

        label = " ".join(label_parts)

        # Truncate if too long
        if len(label) > self.config.max_label_length:
            label = label[:self.config.max_label_length - 3] + "..."

        return label

    def _format_function_label(self, func: Function) -> str:
        """Format a function label for display."""
        param_count = len(func.parameters)
        label = f"{func.name}({param_count} params)"

        if len(label) > self.config.max_label_length:
            label = label[:self.config.max_label_length - 3] + "..."

        return label

    def _format_type_spec(self, state_var: StateVar) -> str:
        """Format type specification for display."""
        # StateVar has a type_spec field which is a TypeSpec object
        # TypeSpec has __str__ method that formats it nicely
        if state_var.type_spec:
            return str(state_var.type_spec)
        return "Any"

    def _get_shape_for_entity(self, entity: Entity) -> str:
        """Get Mermaid shape syntax for entity."""
        if isinstance(entity, Class):
            # Rectangle with rounded corners for classes
            label = self._format_class_label(entity)
            return f"[\"{label}\"]"
        else:
            # Default rectangle
            return f"[\"{entity.name}\"]"

    def _add_dependency_edges(self, cls: Class, lines: List[str]) -> None:
        """Add edges for class dependencies."""
        entity_id = self._get_node_id(cls.name)

        for dep in cls.dependencies:
            # dep is a Reference with ref_id field
            target = dep.ref_id
            target_id = self._get_node_id(target)

            # Choose edge style based on dependency type
            if ":" in dep.ref_id:
                # Typed reference (e.g., NN:Model)
                edge = "-->"  # Solid arrow
            else:
                # Simple reference
                edge = "-..->"  # Dashed arrow

            lines.append(f"  {entity_id} {edge} {target_id}")

    def _get_risk_color(self, risk: str) -> str:
        """Get color code for risk level."""
        risk_colors = {
            "High": "#ff6b6b",      # Red
            "Medium": "#ffd93d",    # Yellow
            "Low": "#6bcf7f",       # Green
            "Critical": "#c92a2a",  # Dark red
        }
        return risk_colors.get(risk, "#a8dadc")  # Default light blue


def generate_mermaid(
    ast: PyShortAST,
    diagram_type: str = "flowchart",
    direction: str = "TB",
    **kwargs
) -> str:
    """
    Convenience function to generate Mermaid diagram.

    Args:
        ast: PyShorthand AST to visualize
        diagram_type: "flowchart", "classDiagram", or "graph"
        direction: "TB", "LR", "RL", or "BT"
        **kwargs: Additional config options

    Returns:
        Mermaid diagram syntax as string

    Example:
        >>> from pyshort.core.parser import parse_file
        >>> from pyshort.visualization.mermaid import generate_mermaid
        >>>
        >>> ast = parse_file("example.pys")
        >>> diagram = generate_mermaid(ast, diagram_type="flowchart")
        >>> print(diagram)
    """
    config = MermaidConfig(
        diagram_type=diagram_type,
        direction=direction,
        **kwargs
    )
    generator = MermaidGenerator(config)
    return generator.generate(ast)


def save_mermaid(
    ast: PyShortAST,
    output_path: str,
    diagram_type: str = "flowchart",
    **kwargs
) -> None:
    """
    Generate and save Mermaid diagram to file.

    Args:
        ast: PyShorthand AST to visualize
        output_path: Path to save .mmd file
        diagram_type: "flowchart", "classDiagram", or "graph"
        **kwargs: Additional config options
    """
    diagram = generate_mermaid(ast, diagram_type=diagram_type, **kwargs)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(diagram)
