"""PyShorthand auto-formatter implementation.

Provides consistent, opinionated formatting for PyShorthand files.
"""

from dataclasses import dataclass

from pyshort.core.ast_nodes import Class, Function, PyShortAST, Statement, StateVar
from pyshort.core.parser import parse_file, parse_string
from pyshort.core.symbols import to_ascii, to_unicode


@dataclass
class FormatConfig:
    """Configuration for PyShorthand formatter."""

    indent: int = 2
    align_types: bool = True
    prefer_unicode: bool = True
    sort_state_by: str = "location"  # "location", "name", "none"
    max_line_length: int = 100
    blank_lines_around_functions: int = 1
    blank_lines_around_classes: int = 2


class Formatter:
    """Formats PyShorthand code for consistency."""

    def __init__(self, config: FormatConfig | None = None):
        """Initialize formatter.

        Args:
            config: Formatting configuration
        """
        self.config = config or FormatConfig()

    def format_ast(self, ast: PyShortAST) -> str:
        """Format a PyShorthand AST.

        Args:
            ast: AST to format

        Returns:
            Formatted PyShorthand string
        """
        lines = []

        # Format metadata header
        if ast.metadata.module_name or ast.metadata.role:
            lines.extend(self._format_metadata(ast))
            lines.append("")  # Blank line after metadata

        # Format entities
        for entity in ast.entities:
            if isinstance(entity, Class):
                lines.extend(self._format_class(entity))
                lines.extend([""] * self.config.blank_lines_around_classes)

        # Format standalone functions
        for func in ast.functions:
            lines.extend(self._format_function(func))
            lines.extend([""] * self.config.blank_lines_around_functions)

        # Format standalone statements
        if ast.statements:
            lines.extend(self._format_statements(ast.statements))

        # Remove trailing blank lines
        while lines and not lines[-1].strip():
            lines.pop()

        result = "\n".join(lines)

        # Apply unicode/ascii preference
        result = to_unicode(result) if self.config.prefer_unicode else to_ascii(result)

        return result

    def _format_metadata(self, ast: PyShortAST) -> list[str]:
        """Format metadata header."""
        lines = []
        meta = ast.metadata

        # First line: [M:Name] [ID:ID] [Role:X] [Layer:X] [Risk:X]
        parts = []
        if meta.module_name:
            parts.append(f"[M:{meta.module_name}]")
        if meta.module_id:
            parts.append(f"[ID:{meta.module_id}]")
        if meta.role:
            parts.append(f"[Role:{meta.role}]")
        if meta.layer:
            parts.append(f"[Layer:{meta.layer}]")
        if meta.risk:
            parts.append(f"[Risk:{meta.risk}]")

        if parts:
            lines.append("# " + " ".join(parts))

        # Second line: [Context:X] [Dims:...]
        parts = []
        if meta.context:
            parts.append(f"[Context: {meta.context}]")
        if meta.dims:
            dims_str = ", ".join(f"{k}={v}" for k, v in meta.dims.items())
            parts.append(f"[Dims: {dims_str}]")

        if parts:
            lines.append("# " + " ".join(parts))

        # Third line: [Requires:...] [Owner:...]
        parts = []
        if meta.requires:
            reqs_str = ", ".join(meta.requires)
            parts.append(f"[Requires: {reqs_str}]")
        if meta.owner:
            parts.append(f"[Owner: {meta.owner}]")

        if parts:
            lines.append("# " + " ".join(parts))

        return lines

    def _format_class(self, cls: Class) -> list[str]:
        """Format a class definition.

        v1.5 supports:
        - [P:Name] for protocols
        - [C:List<T>] for generics
        - [C:Foo] ◊ Base1, Base2 for inheritance
        - [Abstract] and [Protocol] tags
        """
        lines = []

        # v1.5: Use P: prefix for protocols
        prefix = "P" if cls.is_protocol else "C"

        # Class declaration with optional generic parameters
        if cls.generic_params:
            generic_str = ", ".join(cls.generic_params)
            lines.append(f"[{prefix}:{cls.name}<{generic_str}>]")
        else:
            lines.append(f"[{prefix}:{cls.name}]")

        # v1.5: Add [Abstract] or [Protocol] tags if present
        tags = []
        if cls.is_abstract:
            tags.append("[Abstract]")
        if cls.is_protocol and prefix != "P":
            # Only add [Protocol] tag if not already using P: prefix
            tags.append("[Protocol]")
        if tags:
            lines[-1] += " " + " ".join(tags)

        # v1.5: Inheritance (◊ Base1, Base2)
        if cls.base_classes:
            bases = ", ".join(cls.base_classes)
            lines.append(f"  ◊ {bases}")
            lines.append("")

        # v1.4: Dependencies (kept separate from inheritance)
        if cls.dependencies:
            deps = ", ".join(f"[Ref:{dep.ref_id}]" for dep in cls.dependencies)
            lines.append(f"  Dependencies: {deps}")
            lines.append("")

        # State variables (sorted and aligned)
        if cls.state:
            lines.extend(self._format_state_variables(cls.state))
            lines.append("")

        # Methods
        for i, method in enumerate(cls.methods):
            lines.extend(self._format_function(method, indent=1))
            if i < len(cls.methods) - 1:
                lines.append("")

        return lines

    def _format_state_variables(self, state_vars: list[StateVar]) -> list[str]:
        """Format state variables with alignment."""
        if not state_vars:
            return []

        # Sort if configured
        sorted_vars = list(state_vars)
        if self.config.sort_state_by == "location":
            # Sort by location: @GPU, @CPU, @Disk, @Net, then no location
            location_order = {"GPU": 0, "CPU": 1, "Disk": 2, "Net": 3}

            def sort_key(sv: StateVar) -> tuple:
                if sv.type_spec and sv.type_spec.location:
                    return (location_order.get(sv.type_spec.location, 99), sv.name)
                elif sv.type_spec and sv.type_spec.transfer:
                    return (location_order.get(sv.type_spec.transfer[1], 99), sv.name)
                return (99, sv.name)

            sorted_vars.sort(key=sort_key)
        elif self.config.sort_state_by == "name":
            sorted_vars.sort(key=lambda sv: sv.name)

        # Calculate alignment if needed
        max_name_len = max(len(sv.name) for sv in sorted_vars) if self.config.align_types else 0

        lines = []
        for sv in sorted_vars:
            line = self._format_state_var(sv, max_name_len)
            lines.append(line)

        return lines

    def _format_state_var(self, sv: StateVar, align_to: int = 0) -> str:
        """Format a single state variable."""
        indent = " " * self.config.indent
        name = sv.name

        if align_to > 0:
            name = name.ljust(align_to)

        line = f"{indent}{name} ∈ {sv.type_spec}" if sv.type_spec else f"{indent}{name}"

        if sv.comment:
            # Add inline comment
            line += f"  // {sv.comment}"

        return line

    def _format_tags(self, tags: list) -> str:
        """Format tags in v1.4 grouped order.

        Order: [Decorators] [HTTP Routes] [Operations] [Complexity]

        Args:
            tags: List of Tag objects

        Returns:
            Formatted tag string
        """
        if not tags:
            return ""

        # Group tags by type
        decorator_tags = [
            str(t) for t in tags if hasattr(t, "tag_type") and t.tag_type == "decorator"
        ]
        route_tags = [str(t) for t in tags if hasattr(t, "tag_type") and t.tag_type == "http_route"]
        operation_tags = [
            str(t) for t in tags if hasattr(t, "tag_type") and t.tag_type == "operation"
        ]
        complexity_tags = [
            str(t) for t in tags if hasattr(t, "tag_type") and t.tag_type == "complexity"
        ]
        custom_tags = [str(t) for t in tags if hasattr(t, "tag_type") and t.tag_type == "custom"]

        # Combine in order
        ordered = decorator_tags + route_tags + operation_tags + complexity_tags + custom_tags
        return " ".join(ordered) if ordered else ""

    def _format_function(self, func: Function, indent: int = 0) -> list[str]:
        """Format a function definition."""
        lines = []
        indent_str = " " * (self.config.indent * indent)
        body_indent = " " * (self.config.indent * (indent + 1))

        # Function signature
        params_str = ", ".join(
            f"{p.name}:{p.type_spec}" if p.type_spec else p.name for p in func.params
        )
        sig = f"{indent_str}F:{func.name}({params_str})"

        if func.return_type:
            sig += f" → {func.return_type}"

        # Add modifiers/tags (v1.4: grouped by type)
        tag_parts = []
        if func.modifiers:
            tag_parts.extend([f"[{m}]" for m in func.modifiers])
        if func.tags:
            formatted_tags = self._format_tags(func.tags)
            if formatted_tags:
                tag_parts.append(formatted_tags)

        if tag_parts:
            sig += " " + " ".join(tag_parts)

        lines.append(sig)

        # Contracts
        if func.preconditions:
            for pre in func.preconditions:
                lines.append(f"{body_indent}[Pre]  {pre}")
        if func.postconditions:
            for post in func.postconditions:
                lines.append(f"{body_indent}[Post] {post}")
        if func.errors:
            errors_str = ", ".join(func.errors)
            lines.append(f"{body_indent}[Err]  {errors_str}")

        if func.preconditions or func.postconditions or func.errors:
            lines.append("")

        # Body
        for stmt in func.body:
            formatted_stmt = self._format_statement(stmt, indent + 1)
            if formatted_stmt:
                lines.append(formatted_stmt)

        return lines

    def _format_statements(self, statements: list[Statement]) -> list[str]:
        """Format a list of statements."""
        return [
            self._format_statement(stmt, 0)
            for stmt in statements
            if self._format_statement(stmt, 0)
        ]

    def _format_statement(self, stmt: Statement, indent: int = 0) -> str:
        """Format a single statement."""
        indent_str = " " * (self.config.indent * indent)

        # Phase markers
        if stmt.statement_type == "phase":
            return ""  # Skip phase markers in formatted output

        # Comments
        if stmt.comment:
            return f"{indent_str}// {stmt.comment}"

        # Profiling
        prefix = ""
        if stmt.profiling:
            prefix = f"⏱{stmt.profiling} "

        # Build statement
        if stmt.statement_type == "return":
            if stmt.rhs:
                return f"{indent_str}{prefix}← {stmt.rhs}"
            return f"{indent_str}{prefix}←"

        if stmt.statement_type == "assertion":
            return f"{indent_str}{prefix}⊢ {stmt.rhs}"

        if stmt.statement_type == "causal":
            return f"{indent_str}{prefix}⊳ {stmt.rhs}"

        if stmt.statement_type == "conditional":
            tags_str = self._format_tags(stmt.tags)
            if stmt.condition:
                return (
                    f"{indent_str}{prefix}?{stmt.condition} → {tags_str}"
                    if tags_str
                    else f"{indent_str}{prefix}?{stmt.condition}"
                )
            return f"{indent_str}{prefix}?"

        if stmt.lhs and stmt.operator and stmt.rhs:
            # Assignment or mutation (v1.4: grouped tags)
            tags_str = self._format_tags(stmt.tags)
            tag_suffix = f" → {tags_str}" if tags_str else ""
            return f"{indent_str}{prefix}{stmt.lhs} {stmt.operator} {stmt.rhs}{tag_suffix}"

        if stmt.operator == "!!" and stmt.rhs:
            # System mutation (v1.4: grouped tags)
            tags_str = self._format_tags(stmt.tags)
            tag_suffix = f" → {tags_str}" if tags_str else ""
            return f"{indent_str}{prefix}!!{stmt.rhs}{tag_suffix}"

        return ""


def format_string(source: str, config: FormatConfig | None = None) -> str:
    """Format PyShorthand source code.

    Args:
        source: Source code string
        config: Formatting configuration

    Returns:
        Formatted source code
    """
    ast = parse_string(source)
    formatter = Formatter(config)
    return formatter.format_ast(ast)


def format_file(
    file_path: str, config: FormatConfig | None = None, in_place: bool = False
) -> str:
    """Format a PyShorthand file.

    Args:
        file_path: Path to .pys file
        config: Formatting configuration
        in_place: If True, overwrite the file

    Returns:
        Formatted source code
    """
    ast = parse_file(file_path)
    formatter = Formatter(config)
    formatted = formatter.format_ast(ast)

    if in_place:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(formatted)

    return formatted
