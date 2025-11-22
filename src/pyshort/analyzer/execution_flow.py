#!/usr/bin/env python3
"""Execution flow tracing for PyShorthand code.

This module provides execution flow analysis, tracing the call path
through functions and summarizing everything in scope.

Unlike context packs (static dependency layers), execution flow tracing
follows the actual runtime path through function calls.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Callable

from pyshort.core.ast_nodes import (
    Class,
    Entity,
    Function,
    Module,
    Statement,
)


@dataclass
class ExecutionStep:
    """A single step in the execution flow."""

    entity_name: str  # Function/method being executed
    depth: int  # Call depth (0 = entry point)
    statement: Optional[Statement] = None  # The statement being executed
    variables_in_scope: Set[str] = field(default_factory=set)
    calls_made: List[str] = field(default_factory=list)  # Functions called from here
    state_accessed: Set[str] = field(default_factory=set)  # State vars accessed


@dataclass
class ExecutionFlow:
    """Complete execution flow trace for a function."""

    entry_point: str  # Starting function/method
    steps: List[ExecutionStep] = field(default_factory=list)
    max_depth: int = 0
    total_functions_called: int = 0
    variables_accessed: Set[str] = field(default_factory=set)
    state_accessed: Set[str] = field(default_factory=set)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "entry_point": self.entry_point,
            "total_steps": len(self.steps),
            "max_depth": self.max_depth,
            "total_functions_called": self.total_functions_called,
            "variables_accessed": sorted(self.variables_accessed),
            "state_accessed": sorted(self.state_accessed),
            "execution_path": [
                {
                    "depth": step.depth,
                    "entity": step.entity_name,
                    "calls": step.calls_made,
                    "scope": sorted(step.variables_in_scope),
                }
                for step in self.steps
            ],
        }

    def summarize(self) -> str:
        """Generate a human-readable summary of the execution flow."""
        lines = [
            f"Execution Flow: {self.entry_point}",
            "=" * 70,
            f"Total steps: {len(self.steps)}",
            f"Max call depth: {self.max_depth}",
            f"Functions called: {self.total_functions_called}",
            f"Variables accessed: {len(self.variables_accessed)}",
            f"State accessed: {len(self.state_accessed)}",
            "",
            "Execution Path:",
        ]

        for i, step in enumerate(self.steps, 1):
            indent = "  " * step.depth
            lines.append(f"{i:3d}. {indent}{step.entity_name}")
            if step.calls_made:
                calls_str = ", ".join(step.calls_made)
                lines.append(f"     {indent}  → calls: {calls_str}")

        if self.variables_accessed:
            lines.append("")
            lines.append("Variables in scope:")
            for var in sorted(self.variables_accessed):
                lines.append(f"  - {var}")

        if self.state_accessed:
            lines.append("")
            lines.append("State accessed:")
            for state in sorted(self.state_accessed):
                lines.append(f"  - {state}")

        return "\n".join(lines)

    def to_mermaid(self, direction: str = "TB") -> str:
        """Export execution flow as Mermaid diagram.

        Args:
            direction: Graph direction (TB=top-bottom, LR=left-right)

        Returns:
            Mermaid flowchart string ready for rendering

        Example:
            ```mermaid
            graph TB
                Step1[Main]:::depth0
                Step2[Processor]:::depth1
                Step3[Transformer]:::depth2
                Step1 --> Step2
                Step2 --> Step3
                classDef depth0 fill:#ff6b6b,stroke:#c92a2a,color:#fff
                classDef depth1 fill:#51cf66,stroke:#2f9e44,color:#000
                classDef depth2 fill:#74c0fc,stroke:#1971c2,color:#000
            ```
        """
        lines = [f"graph {direction}"]

        # Track unique depth values for styling
        depths = set()

        # Add nodes with step numbers and depth-based styling
        for i, step in enumerate(self.steps, 1):
            depths.add(step.depth)
            node_id = f"Step{i}"
            label = step.entity_name
            if step.calls_made:
                call_list = ", ".join(step.calls_made[:2])  # Show first 2 calls
                if len(step.calls_made) > 2:
                    call_list += "..."
                label += f"<br/>→ {call_list}"

            lines.append(f"    {node_id}[\"{label}\"]:::depth{step.depth}")

        lines.append("")

        # Add edges showing execution flow
        for i in range(len(self.steps) - 1):
            src = f"Step{i+1}"
            dst = f"Step{i+2}"
            lines.append(f"    {src} --> {dst}")

        # Add class definitions for each depth level
        lines.append("")
        colors = [
            ("#ff6b6b", "#c92a2a", "#fff"),  # Red for depth 0
            ("#51cf66", "#2f9e44", "#000"),  # Green for depth 1
            ("#74c0fc", "#1971c2", "#000"),  # Blue for depth 2
            ("#ffd43b", "#f08c00", "#000"),  # Yellow for depth 3
            ("#da77f2", "#9c36b5", "#fff"),  # Purple for depth 4+
        ]

        for depth in sorted(depths):
            color_idx = min(depth, len(colors) - 1)
            fill, stroke, text = colors[color_idx]
            lines.append(f"    classDef depth{depth} fill:{fill},stroke:{stroke},color:{text},stroke-width:2px")

        return "\n".join(lines)

    def to_graphviz(self) -> str:
        """Export execution flow as GraphViz DOT format.

        Returns:
            DOT format string for GraphViz rendering

        Example:
            digraph ExecutionFlow {
                rankdir=TB;
                node [shape=box];

                Step1 [label="Main", style=filled, fillcolor="#ff6b6b"];
                Step2 [label="Processor", style=filled, fillcolor="#51cf66"];
                Step3 [label="Transformer", style=filled, fillcolor="#74c0fc"];

                Step1 -> Step2;
                Step2 -> Step3;
            }
        """
        lines = ["digraph ExecutionFlow {"]
        lines.append("    rankdir=TB;")
        lines.append("    node [shape=box];")
        lines.append("")

        # Color scheme based on depth
        colors = [
            ("#ff6b6b", "white"),  # Red for depth 0
            ("#51cf66", "black"),  # Green for depth 1
            ("#74c0fc", "black"),  # Blue for depth 2
            ("#ffd43b", "black"),  # Yellow for depth 3
            ("#da77f2", "white"),  # Purple for depth 4+
        ]

        # Add nodes with labels and depth-based colors
        for i, step in enumerate(self.steps, 1):
            node_id = f"Step{i}"
            label = step.entity_name

            # Add call info to label
            if step.calls_made:
                call_list = ", ".join(step.calls_made[:2])
                if len(step.calls_made) > 2:
                    call_list += "..."
                label += f"\\n→ {call_list}"

            color_idx = min(step.depth, len(colors) - 1)
            fill, font = colors[color_idx]

            lines.append(f'    {node_id} [label="{label}", style=filled, fillcolor="{fill}", fontcolor={font}, penwidth=2];')

        lines.append("")

        # Add edges showing execution flow
        for i in range(len(self.steps) - 1):
            src = f"Step{i+1}"
            dst = f"Step{i+2}"
            lines.append(f"    {src} -> {dst};")

        lines.append("}")

        return "\n".join(lines)

    def filter_by_depth(self, max_depth: int) -> "ExecutionFlow":
        """Filter execution flow to only include steps up to a certain depth.

        Args:
            max_depth: Maximum depth to include

        Returns:
            New ExecutionFlow with only steps at depth <= max_depth

        Example:
            shallow = flow.filter_by_depth(1)  # Only entry point and direct calls
        """
        filtered_steps = [step for step in self.steps if step.depth <= max_depth]

        # Recalculate aggregates
        new_flow = ExecutionFlow(
            entry_point=self.entry_point,
            steps=filtered_steps,
        )
        new_flow.max_depth = max(step.depth for step in filtered_steps) if filtered_steps else 0
        new_flow.total_functions_called = len(set(step.entity_name for step in filtered_steps))
        new_flow.variables_accessed = set()
        new_flow.state_accessed = set()

        for step in filtered_steps:
            new_flow.variables_accessed.update(step.variables_in_scope)
            new_flow.state_accessed.update(step.state_accessed)

        return new_flow

    def filter_by_pattern(self, pattern: str) -> "ExecutionFlow":
        """Filter execution flow to only include steps matching entity name pattern.

        Args:
            pattern: Regex pattern to match entity names

        Returns:
            New ExecutionFlow with only matching steps

        Example:
            handlers = flow.filter_by_pattern(".*Handler$")
            processors = flow.filter_by_pattern("^Process.*")
        """
        regex = re.compile(pattern)
        filtered_steps = [step for step in self.steps if regex.search(step.entity_name)]

        new_flow = ExecutionFlow(
            entry_point=self.entry_point,
            steps=filtered_steps,
        )
        new_flow.max_depth = max(step.depth for step in filtered_steps) if filtered_steps else 0
        new_flow.total_functions_called = len(set(step.entity_name for step in filtered_steps))
        new_flow.variables_accessed = set()
        new_flow.state_accessed = set()

        for step in filtered_steps:
            new_flow.variables_accessed.update(step.variables_in_scope)
            new_flow.state_accessed.update(step.state_accessed)

        return new_flow

    def filter_by_state_access(self, state_pattern: str) -> "ExecutionFlow":
        """Filter execution flow to only include steps that access matching state.

        Args:
            state_pattern: Pattern to match state variable names

        Returns:
            New ExecutionFlow with only steps accessing matching state

        Example:
            gpu_ops = flow.filter_by_state_access(".*@GPU")
            cache_hits = flow.filter_by_state_access(".*cache.*")
        """
        regex = re.compile(state_pattern)
        filtered_steps = []

        for step in self.steps:
            if any(regex.search(state) for state in step.state_accessed):
                filtered_steps.append(step)

        new_flow = ExecutionFlow(
            entry_point=self.entry_point,
            steps=filtered_steps,
        )
        new_flow.max_depth = max(step.depth for step in filtered_steps) if filtered_steps else 0
        new_flow.total_functions_called = len(set(step.entity_name for step in filtered_steps))
        new_flow.variables_accessed = set()
        new_flow.state_accessed = set()

        for step in filtered_steps:
            new_flow.variables_accessed.update(step.variables_in_scope)
            new_flow.state_accessed.update(step.state_accessed)

        return new_flow

    def filter_custom(self, predicate: Callable[[ExecutionStep], bool]) -> "ExecutionFlow":
        """Filter execution flow using a custom predicate function.

        Args:
            predicate: Function that takes ExecutionStep and returns bool

        Returns:
            New ExecutionFlow with only matching steps

        Example:
            # Filter to only steps that make calls
            with_calls = flow.filter_custom(lambda s: len(s.calls_made) > 0)

            # Filter by depth and name
            deep_processors = flow.filter_custom(
                lambda s: s.depth >= 2 and "Processor" in s.entity_name
            )
        """
        filtered_steps = [step for step in self.steps if predicate(step)]

        new_flow = ExecutionFlow(
            entry_point=self.entry_point,
            steps=filtered_steps,
        )
        new_flow.max_depth = max(step.depth for step in filtered_steps) if filtered_steps else 0
        new_flow.total_functions_called = len(set(step.entity_name for step in filtered_steps))
        new_flow.variables_accessed = set()
        new_flow.state_accessed = set()

        for step in filtered_steps:
            new_flow.variables_accessed.update(step.variables_in_scope)
            new_flow.state_accessed.update(step.state_accessed)

        return new_flow

    def get_steps_at_depth(self, depth: int) -> List[ExecutionStep]:
        """Get all execution steps at a specific depth.

        Args:
            depth: Call depth to retrieve

        Returns:
            List of steps at that depth

        Example:
            entry_points = flow.get_steps_at_depth(0)
            first_level = flow.get_steps_at_depth(1)
        """
        return [step for step in self.steps if step.depth == depth]

    def get_call_chain(self) -> List[str]:
        """Get the execution call chain as a list of entity names.

        Returns:
            List of entity names in execution order

        Example:
            chain = flow.get_call_chain()
            # ['Main', 'Processor', 'Transformer', 'Finalizer']
        """
        return [step.entity_name for step in self.steps]


class ExecutionFlowTracer:
    """Traces execution flow through PyShorthand code."""

    def __init__(self):
        self.entity_map: Dict[str, Entity] = {}
        self.function_map: Dict[str, Function] = {}
        self.class_map: Dict[str, Class] = {}
        self.call_graph: Dict[str, Set[str]] = {}  # entity -> functions it calls

    def trace_execution(
        self,
        module: Module,
        entry_point: str,
        max_depth: int = 10,
        follow_calls: bool = True,
    ) -> Optional[ExecutionFlow]:
        """Trace execution flow starting from an entry point.

        Args:
            module: The PyShorthand module to analyze
            entry_point: The function/method to start tracing from
            max_depth: Maximum call depth to trace (prevents infinite recursion)
            follow_calls: If True, recursively trace into function calls

        Returns:
            ExecutionFlow object with the trace, or None if entry point not found
        """
        self._build_maps(module)

        if entry_point not in self.entity_map and entry_point not in self.function_map:
            return None

        flow = ExecutionFlow(entry_point=entry_point)
        visited = set()  # Track visited functions to prevent infinite loops

        self._trace_entity(
            entity_name=entry_point,
            flow=flow,
            depth=0,
            max_depth=max_depth,
            follow_calls=follow_calls,
            visited=visited,
        )

        flow.max_depth = max(step.depth for step in flow.steps) if flow.steps else 0
        flow.total_functions_called = len(set(step.entity_name for step in flow.steps))

        return flow

    def _build_maps(self, module: Module) -> None:
        """Build entity and function maps from the module."""
        self.entity_map.clear()
        self.function_map.clear()
        self.class_map.clear()
        self.call_graph.clear()

        # Map all entities
        for entity in module.entities:
            self.entity_map[entity.name] = entity
            if isinstance(entity, Class):
                self.class_map[entity.name] = entity

        # Map all functions
        for func in module.functions:
            self.function_map[func.name] = func
            self.entity_map[func.name] = func

        # Build call graph
        self._build_call_graph(module)

    def _build_call_graph(self, module: Module) -> None:
        """Build a graph of function calls."""
        # Analyze functions
        for func in module.functions:
            calls = self._extract_calls_from_statements(func.body)
            self.call_graph[func.name] = calls

        # Analyze classes (state variable references)
        for entity in module.entities:
            if isinstance(entity, Class):
                calls = set()
                # Check state variable type references
                for state_var in entity.state:
                    if state_var.type_spec and state_var.type_spec.base_type.startswith("Ref:"):
                        ref_name = state_var.type_spec.base_type[4:]  # Extract name from Ref:X
                        calls.add(ref_name)
                self.call_graph[entity.name] = calls

    def _extract_calls_from_statements(self, statements: List[Statement]) -> Set[str]:
        """Extract function calls from a list of statements."""
        calls = set()

        for stmt in statements:
            # Look for function calls in the statement body
            body = stmt.body

            # Simple pattern matching for function calls: name(...)
            # This is a simplified version - real implementation would parse expressions
            if "(" in body and ")" in body:
                # Extract potential function names
                parts = body.split("(")
                for part in parts[:-1]:  # All but the last part
                    # Get the last word before (
                    words = part.strip().split()
                    if words:
                        potential_func = words[-1]
                        # Check if it's alphanumeric (likely a function name)
                        if potential_func.replace("_", "").isalnum():
                            if potential_func in self.function_map or potential_func in self.entity_map:
                                calls.add(potential_func)

        return calls

    def _trace_entity(
        self,
        entity_name: str,
        flow: ExecutionFlow,
        depth: int,
        max_depth: int,
        follow_calls: bool,
        visited: Set[str],
    ) -> None:
        """Recursively trace execution through an entity."""
        # Prevent infinite recursion
        if depth > max_depth:
            return

        # Check if we've already visited this function at this depth
        visit_key = f"{entity_name}:{depth}"
        if visit_key in visited:
            return
        visited.add(visit_key)

        # Get the entity
        entity = self.entity_map.get(entity_name) or self.function_map.get(entity_name)
        if not entity:
            return

        # Track variables in scope
        variables_in_scope = set()
        state_accessed = set()

        # If it's a function, get its parameters and local vars
        if isinstance(entity, Function):
            variables_in_scope.update(param.name for param in entity.inputs)
            variables_in_scope.update(param.name for param in entity.outputs)

            # Get calls made by this function
            calls_made = list(self.call_graph.get(entity_name, set()))

            # Create execution step
            step = ExecutionStep(
                entity_name=entity_name,
                depth=depth,
                variables_in_scope=variables_in_scope,
                calls_made=calls_made,
                state_accessed=state_accessed,
            )
            flow.steps.append(step)
            flow.variables_accessed.update(variables_in_scope)

            # Follow calls if enabled
            if follow_calls:
                for called_func in calls_made:
                    self._trace_entity(
                        entity_name=called_func,
                        flow=flow,
                        depth=depth + 1,
                        max_depth=max_depth,
                        follow_calls=follow_calls,
                        visited=visited,
                    )

        # If it's a class, track its state
        elif isinstance(entity, Class):
            state_accessed.update(f"{entity_name}.{var.name}" for var in entity.state)

            # Get references from state
            calls_made = list(self.call_graph.get(entity_name, set()))

            # Create execution step
            step = ExecutionStep(
                entity_name=entity_name,
                depth=depth,
                variables_in_scope=variables_in_scope,
                calls_made=calls_made,
                state_accessed=state_accessed,
            )
            flow.steps.append(step)
            flow.state_accessed.update(state_accessed)

            # Follow references if enabled
            if follow_calls:
                for ref_entity in calls_made:
                    self._trace_entity(
                        entity_name=ref_entity,
                        flow=flow,
                        depth=depth + 1,
                        max_depth=max_depth,
                        follow_calls=follow_calls,
                        visited=visited,
                    )


def trace_execution(
    module: Module,
    entry_point: str,
    max_depth: int = 10,
    follow_calls: bool = True,
) -> Optional[ExecutionFlow]:
    """Convenience function to trace execution flow.

    Args:
        module: The PyShorthand module to analyze
        entry_point: The function/method to start tracing from
        max_depth: Maximum call depth to trace
        follow_calls: If True, recursively trace into function calls

    Returns:
        ExecutionFlow object with the trace, or None if entry point not found
    """
    tracer = ExecutionFlowTracer()
    return tracer.trace_execution(module, entry_point, max_depth, follow_calls)
