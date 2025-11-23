"""
Circular Reference Validator for PyShorthand

Detects circular dependencies in type references across classes and data structures.
Part of P20: Circular reference validation.
"""

from dataclasses import dataclass

from ..core.ast_nodes import Class, Data, Entity, Module, TypeSpec


@dataclass
class CircularReferenceError:
    """Represents a detected circular reference."""

    cycle: list[str]  # The cycle path, e.g., ["A", "B", "C", "A"]
    message: str
    is_self_reference: bool = False  # True if A → A (single node cycle)

    def __str__(self) -> str:
        return self.message


class CircularReferenceValidator:
    """Validates PyShorthand AST for circular type references."""

    def __init__(self):
        self.dependency_graph: dict[str, set[str]] = {}
        self.entity_map: dict[str, Entity] = {}

    def validate(
        self, module: Module, include_self_references: bool = False
    ) -> list[CircularReferenceError]:
        """
        Validate a module for circular references.

        Args:
            module: The parsed PyShorthand module
            include_self_references: If True, report self-references (A → A) as cycles.
                                    If False, only report multi-node cycles.
                                    Default is False since self-references are common
                                    in recursive data structures (linked lists, trees).

        Returns:
            List of CircularReferenceError objects (empty if no cycles)
        """
        # Build dependency graph
        self._build_dependency_graph(module)

        # Detect cycles
        errors = []
        visited = set()
        rec_stack = set()

        for entity_name in self.dependency_graph.keys():
            if entity_name not in visited:
                cycle = self._detect_cycle_dfs(entity_name, visited, rec_stack, [])
                if cycle:
                    # Check if it's a self-reference
                    is_self_ref = len(cycle) == 2 and cycle[0] == cycle[1]

                    # Skip self-references if requested
                    if is_self_ref and not include_self_references:
                        continue

                    errors.append(
                        CircularReferenceError(
                            cycle=cycle,
                            message=f"Circular reference detected: {' → '.join(cycle)}",
                            is_self_reference=is_self_ref,
                        )
                    )

        return errors

    def _build_dependency_graph(self, module: Module) -> None:
        """Build a dependency graph from entities in the module."""
        self.dependency_graph.clear()
        self.entity_map.clear()

        # First pass: Register all entities
        for entity in module.entities:
            if isinstance(entity, (Class, Data)):
                self.entity_map[entity.name] = entity
                self.dependency_graph[entity.name] = set()

        # Second pass: Extract dependencies
        for entity in module.entities:
            if isinstance(entity, Class):
                deps = self._extract_dependencies_from_class(entity)
                self.dependency_graph[entity.name] = deps
            elif isinstance(entity, Data):
                deps = self._extract_dependencies_from_data(entity)
                self.dependency_graph[entity.name] = deps

    def _extract_dependencies_from_class(self, cls: Class) -> set[str]:
        """Extract type dependencies from a class."""
        dependencies = set()

        # Check state variables
        for state_var in cls.state:
            deps = self._extract_type_references(state_var.type_spec)
            dependencies.update(deps)

        # Check method parameters and return types
        for method in cls.methods:
            # Check parameters
            for param in method.params:
                if param.type_spec:
                    deps = self._extract_type_references(param.type_spec)
                    dependencies.update(deps)

            # Check return type
            if method.return_type:
                deps = self._extract_type_references(method.return_type)
                dependencies.update(deps)

        return dependencies

    def _extract_dependencies_from_data(self, data: Data) -> set[str]:
        """Extract type dependencies from a data structure."""
        dependencies = set()

        for field in data.fields:
            deps = self._extract_type_references(field.type_spec)
            dependencies.update(deps)

        return dependencies

    def _extract_type_references(self, type_spec: TypeSpec | None) -> set[str]:
        """Extract entity names referenced in a type specification."""
        if not type_spec:
            return set()

        references = set()

        # Handle union types
        if type_spec.union_types:
            for union_type in type_spec.union_types:
                ref = self._extract_ref_name(union_type)
                if ref:
                    references.add(ref)
        else:
            # Handle single type
            ref = self._extract_ref_name(type_spec.base_type)
            if ref:
                references.add(ref)

        return references

    def _extract_ref_name(self, type_name: str) -> str | None:
        """
        Extract the referenced entity name from a type string.

        Examples:
            "[Ref:Node]" -> "Node"
            "Ref:Node" -> "Node"
            "i32" -> None (built-in type)
            "Node" -> "Node" (if it's a known entity)
        """
        # Handle reference syntax: [Ref:Name] or Ref:Name
        if "Ref:" in type_name:
            # Extract the name after "Ref:"
            parts = type_name.split("Ref:")
            if len(parts) > 1:
                # Remove brackets and get the name
                ref_name = parts[1].strip("[]").strip()
                return ref_name
            return None

        # Check if it's a known entity (not a built-in type)
        if type_name in self.entity_map:
            return type_name

        # Built-in types don't create dependencies
        builtin_types = {
            "i32",
            "i64",
            "f32",
            "f64",
            "bool",
            "str",
            "obj",
            "Map",
            "Str",
            "Any",
        }
        if type_name in builtin_types:
            return None

        # If we don't recognize it, it might be a forward reference
        # We'll conservatively treat it as a potential dependency
        return None

    def _detect_cycle_dfs(
        self,
        node: str,
        visited: set[str],
        rec_stack: set[str],
        path: list[str],
    ) -> list[str] | None:
        """
        Detect cycles using depth-first search with recursion stack.

        Args:
            node: Current node to visit
            visited: Set of all visited nodes
            rec_stack: Set of nodes in current recursion stack
            path: Current path being explored

        Returns:
            The cycle path if a cycle is detected, None otherwise
        """
        visited.add(node)
        rec_stack.add(node)
        path.append(node)

        # Visit all neighbors
        for neighbor in self.dependency_graph.get(node, set()):
            if neighbor not in self.entity_map:
                # Skip references to undefined types
                continue

            if neighbor not in visited:
                # Recurse
                cycle = self._detect_cycle_dfs(neighbor, visited, rec_stack, path.copy())
                if cycle:
                    return cycle
            elif neighbor in rec_stack:
                # Found a cycle!
                # Build the cycle path from where we found the back edge
                cycle_start_idx = path.index(neighbor)
                cycle_path = path[cycle_start_idx:] + [neighbor]
                return cycle_path

        # Backtrack
        rec_stack.remove(node)
        return None

    def get_dependency_graph(self) -> dict[str, set[str]]:
        """Get the built dependency graph (for debugging/inspection)."""
        return self.dependency_graph.copy()
