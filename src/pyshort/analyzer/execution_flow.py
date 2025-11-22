#!/usr/bin/env python3
"""Execution flow tracing for PyShorthand code.

This module provides execution flow analysis, tracing the call path
through functions and summarizing everything in scope.

Unlike context packs (static dependency layers), execution flow tracing
follows the actual runtime path through function calls.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

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
