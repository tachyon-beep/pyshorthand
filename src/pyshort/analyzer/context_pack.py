"""
Context Pack Generator for PyShorthand

Generates dependency-aware context packs for functions and classes.
For a given entity F, generates:
- F0: The target entity itself
- F1: All entities that call or are called by F
- F2: All entities that call or are called by F1
- Related: Class peers, global variables, state variables

Perfect for LLM context, code review, documentation, and refactoring.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from ..core.ast_nodes import Class, Data, Entity, Function, Module


@dataclass
class ContextPack:
    """A context pack containing an entity and its dependency layers."""

    target: str  # The target entity name
    target_entity: Entity | None = None  # The actual entity object

    # Dependency layers
    f0_core: set[str] = field(default_factory=set)  # Target itself
    f1_immediate: set[str] = field(default_factory=set)  # Direct dependencies
    f2_extended: set[str] = field(default_factory=set)  # 2-hop dependencies

    # Related entities
    class_peers: set[str] = field(default_factory=set)  # Other methods in same class
    related_globals: set[str] = field(default_factory=set)  # Global variables referenced
    related_state: set[str] = field(default_factory=set)  # State variables in class

    # Entity objects for serialization
    entities: dict[str, Entity] = field(default_factory=dict)  # All entities in pack

    def all_entities(self) -> set[str]:
        """Get all entity names in the context pack."""
        return self.f0_core | self.f1_immediate | self.f2_extended | self.class_peers

    def layer_count(self, entity_name: str) -> int:
        """Get the layer depth of an entity (0 = core, 1 = immediate, 2 = extended)."""
        if entity_name in self.f0_core:
            return 0
        elif entity_name in self.f1_immediate:
            return 1
        elif entity_name in self.f2_extended:
            return 2
        return -1

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "target": self.target,
            "f0_core": sorted(list(self.f0_core)),
            "f1_immediate": sorted(list(self.f1_immediate)),
            "f2_extended": sorted(list(self.f2_extended)),
            "class_peers": sorted(list(self.class_peers)),
            "related_globals": sorted(list(self.related_globals)),
            "related_state": sorted(list(self.related_state)),
            "total_entities": len(self.all_entities()),
        }

    def to_mermaid(self, direction: str = "TB") -> str:
        """Export context pack as Mermaid diagram.

        Args:
            direction: Graph direction (TB=top-bottom, LR=left-right, BT=bottom-top, RL=right-left)

        Returns:
            Mermaid diagram string ready for rendering

        Example:
            ```mermaid
            graph TB
                Target[Target]:::f0
                A[EntityA]:::f1
                B[EntityB]:::f1
                C[EntityC]:::f2
                Target --> A
                Target --> B
                A --> C
                classDef f0 fill:#ff6b6b,stroke:#c92a2a,color:#fff
                classDef f1 fill:#51cf66,stroke:#2f9e44,color:#000
                classDef f2 fill:#74c0fc,stroke:#1971c2,color:#000
            ```
        """
        lines = [f"graph {direction}"]

        # Add nodes with styling
        all_nodes = self.f0_core | self.f1_immediate | self.f2_extended

        for node in sorted(all_nodes):
            if node in self.f0_core:
                lines.append(f"    {node}[{node}]:::f0")
            elif node in self.f1_immediate:
                lines.append(f"    {node}[{node}]:::f1")
            elif node in self.f2_extended:
                lines.append(f"    {node}[{node}]:::f2")

        # Add class peers with different style
        if self.class_peers:
            for peer in sorted(self.class_peers):
                if peer not in all_nodes:
                    lines.append(f"    {peer}[{peer}]:::peer")

        # Add edges (using stored entity relationships)
        edges = set()

        # Connect F0 to F1
        for f1_entity in sorted(self.f1_immediate):
            edges.add((self.target, f1_entity))

        # Connect F1 to F2
        for f2_entity in sorted(self.f2_extended):
            # Find which F1 entity connects to this F2
            for f1_entity in self.f1_immediate:
                edges.add((f1_entity, f2_entity))

        for src, dst in sorted(edges):
            lines.append(f"    {src} --> {dst}")

        # Add class definitions for styling
        lines.append("")
        lines.append("    classDef f0 fill:#ff6b6b,stroke:#c92a2a,color:#fff,stroke-width:3px")
        lines.append("    classDef f1 fill:#51cf66,stroke:#2f9e44,color:#000,stroke-width:2px")
        lines.append("    classDef f2 fill:#74c0fc,stroke:#1971c2,color:#000,stroke-width:1px")
        lines.append(
            "    classDef peer fill:#ffd43b,stroke:#f08c00,color:#000,stroke-width:1px,stroke-dasharray: 5 5"
        )

        return "\n".join(lines)

    def to_graphviz(self) -> str:
        """Export context pack as GraphViz DOT format.

        Returns:
            DOT format string for GraphViz rendering

        Example:
            digraph ContextPack {
                rankdir=TB;
                node [shape=box];

                Target [style=filled, fillcolor="#ff6b6b", fontcolor=white];
                EntityA [style=filled, fillcolor="#51cf66"];
                EntityB [style=filled, fillcolor="#51cf66"];
                EntityC [style=filled, fillcolor="#74c0fc"];

                Target -> EntityA;
                Target -> EntityB;
                EntityA -> EntityC;
            }
        """
        lines = ["digraph ContextPack {"]
        lines.append("    rankdir=TB;")
        lines.append("    node [shape=box];")
        lines.append("")

        # Define nodes with colors
        all_nodes = self.f0_core | self.f1_immediate | self.f2_extended

        for node in sorted(all_nodes):
            if node in self.f0_core:
                lines.append(
                    f'    {node} [style=filled, fillcolor="#ff6b6b", fontcolor=white, penwidth=3];'
                )
            elif node in self.f1_immediate:
                lines.append(f'    {node} [style=filled, fillcolor="#51cf66", penwidth=2];')
            elif node in self.f2_extended:
                lines.append(f'    {node} [style=filled, fillcolor="#74c0fc", penwidth=1];')

        # Add class peers
        if self.class_peers:
            for peer in sorted(self.class_peers):
                if peer not in all_nodes:
                    lines.append(f'    {peer} [style="filled,dashed", fillcolor="#ffd43b"];')

        lines.append("")

        # Add edges
        edges = set()

        for f1_entity in sorted(self.f1_immediate):
            edges.add((self.target, f1_entity))

        for f2_entity in sorted(self.f2_extended):
            for f1_entity in self.f1_immediate:
                edges.add((f1_entity, f2_entity))

        for src, dst in sorted(edges):
            lines.append(f"    {src} -> {dst};")

        lines.append("}")

        return "\n".join(lines)

    def filter_by_tag(self, tag_pattern: str) -> "ContextPack":
        """Filter context pack to only include entities with matching tags.

        Args:
            tag_pattern: Tag pattern to match (e.g., "Risk:High", "O(N^2)", "GPU")

        Returns:
            New ContextPack with only matching entities

        Example:
            high_risk = pack.filter_by_tag("Risk:High")
            gpu_entities = pack.filter_by_tag("GPU")
        """
        filtered_entities = {}
        matching_names = set()

        for name, entity in self.entities.items():
            if hasattr(entity, "tags") and entity.tags:
                for tag in entity.tags:
                    tag_str = str(tag)
                    if tag_pattern in tag_str or re.search(tag_pattern, tag_str):
                        filtered_entities[name] = entity
                        matching_names.add(name)
                        break

        # Create filtered pack
        return ContextPack(
            target=self.target,
            target_entity=self.target_entity,
            f0_core=self.f0_core & matching_names,
            f1_immediate=self.f1_immediate & matching_names,
            f2_extended=self.f2_extended & matching_names,
            class_peers=self.class_peers & matching_names,
            related_globals=self.related_globals,
            related_state=self.related_state,
            entities=filtered_entities,
        )

    def filter_by_complexity(self, complexity_pattern: str) -> "ContextPack":
        """Filter context pack to only include entities with matching complexity.

        Args:
            complexity_pattern: Complexity pattern (e.g., "O(N^2)", "O(N)", "O(1)")

        Returns:
            New ContextPack with only matching entities

        Example:
            quadratic = pack.filter_by_complexity("O(N^2)")
            linear_or_worse = pack.filter_by_complexity("O(N.*)")
        """
        filtered_entities = {}
        matching_names = set()

        for name, entity in self.entities.items():
            if hasattr(entity, "tags") and entity.tags:
                for tag in entity.tags:
                    tag_str = str(tag)
                    # Match complexity in tag (e.g., "O(N^2)", "Lin:O(N)")
                    if complexity_pattern in tag_str or re.search(complexity_pattern, tag_str):
                        filtered_entities[name] = entity
                        matching_names.add(name)
                        break

        return ContextPack(
            target=self.target,
            target_entity=self.target_entity,
            f0_core=self.f0_core & matching_names,
            f1_immediate=self.f1_immediate & matching_names,
            f2_extended=self.f2_extended & matching_names,
            class_peers=self.class_peers & matching_names,
            related_globals=self.related_globals,
            related_state=self.related_state,
            entities=filtered_entities,
        )

    def filter_by_location(self, location: str) -> "ContextPack":
        """Filter context pack to only include entities with specific memory location.

        Args:
            location: Memory location (e.g., "GPU", "CPU", "Disk", "Cache")

        Returns:
            New ContextPack with only matching entities

        Example:
            gpu_ops = pack.filter_by_location("GPU")
            disk_io = pack.filter_by_location("Disk")
        """
        filtered_entities = {}
        matching_names = set()

        for name, entity in self.entities.items():
            # Check if entity has state variables with location specs
            if isinstance(entity, Class) and hasattr(entity, "state"):
                for state_var in entity.state:
                    if state_var.type_spec and state_var.type_spec.location == location:
                        filtered_entities[name] = entity
                        matching_names.add(name)
                        break

        return ContextPack(
            target=self.target,
            target_entity=self.target_entity,
            f0_core=self.f0_core & matching_names,
            f1_immediate=self.f1_immediate & matching_names,
            f2_extended=self.f2_extended & matching_names,
            class_peers=self.class_peers & matching_names,
            related_globals=self.related_globals,
            related_state=self.related_state,
            entities=filtered_entities,
        )

    def filter_by_pattern(self, pattern: str) -> "ContextPack":
        """Filter context pack to only include entities matching name pattern.

        Args:
            pattern: Regex pattern to match entity names

        Returns:
            New ContextPack with only matching entities

        Example:
            handlers = pack.filter_by_pattern(".*Handler$")
            utils = pack.filter_by_pattern("^util_.*")
        """
        regex = re.compile(pattern)
        matching_names = {name for name in self.all_entities() if regex.search(name)}

        filtered_entities = {
            name: entity for name, entity in self.entities.items() if name in matching_names
        }

        return ContextPack(
            target=self.target,
            target_entity=self.target_entity,
            f0_core=self.f0_core & matching_names,
            f1_immediate=self.f1_immediate & matching_names,
            f2_extended=self.f2_extended & matching_names,
            class_peers=self.class_peers & matching_names,
            related_globals=self.related_globals,
            related_state=self.related_state,
            entities=filtered_entities,
        )

    def filter_custom(self, predicate: Callable[[str, Entity], bool]) -> "ContextPack":
        """Filter context pack using a custom predicate function.

        Args:
            predicate: Function that takes (entity_name, entity) and returns bool

        Returns:
            New ContextPack with only matching entities

        Example:
            # Filter to only classes
            classes = pack.filter_custom(lambda name, e: isinstance(e, Class))

            # Filter by state variable count
            large = pack.filter_custom(
                lambda n, e: isinstance(e, Class) and len(e.state) > 5
            )
        """
        matching_names = set()
        filtered_entities = {}

        for name, entity in self.entities.items():
            if predicate(name, entity):
                matching_names.add(name)
                filtered_entities[name] = entity

        return ContextPack(
            target=self.target,
            target_entity=self.target_entity,
            f0_core=self.f0_core & matching_names,
            f1_immediate=self.f1_immediate & matching_names,
            f2_extended=self.f2_extended & matching_names,
            class_peers=self.class_peers & matching_names,
            related_globals=self.related_globals,
            related_state=self.related_state,
            entities=filtered_entities,
        )

    def get_by_layer(self, layer: int) -> set[str]:
        """Get all entities at a specific dependency layer.

        Args:
            layer: Layer number (0=F0, 1=F1, 2=F2)

        Returns:
            Set of entity names at that layer

        Example:
            core = pack.get_by_layer(0)  # F0
            immediate = pack.get_by_layer(1)  # F1
        """
        if layer == 0:
            return self.f0_core.copy()
        elif layer == 1:
            return self.f1_immediate.copy()
        elif layer == 2:
            return self.f2_extended.copy()
        else:
            return set()


class ContextPackGenerator:
    """Generates context packs for PyShorthand entities."""

    def __init__(self):
        self.entity_map: dict[str, Entity] = {}
        self.dependency_graph: dict[str, set[str]] = {}  # Who depends on whom (forward)
        self.reverse_graph: dict[str, set[str]] = {}  # Who is depended on by whom (backward)
        self.class_map: dict[str, str] = {}  # Method name -> Class name mapping

    def generate_context_pack(
        self,
        module: Module,
        target_name: str,
        max_depth: int = 2,
        include_peers: bool = True,
    ) -> ContextPack | None:
        """
        Generate a context pack for a target entity.

        Args:
            module: The PyShorthand module
            target_name: Name of the target function/class
            max_depth: Maximum dependency depth (1 = F1 only, 2 = F1 + F2)
            include_peers: If True, include class peers for class methods

        Returns:
            ContextPack or None if target not found
        """
        # Build graphs
        self._build_graphs(module)

        # Check if target exists
        if target_name not in self.entity_map:
            return None

        pack = ContextPack(
            target=target_name,
            target_entity=self.entity_map[target_name],
        )

        # F0: Core - the target itself
        pack.f0_core.add(target_name)
        pack.entities[target_name] = self.entity_map[target_name]

        # F1: Immediate dependencies (1-hop)
        f1_entities = self._get_neighbors(target_name)
        pack.f1_immediate = f1_entities
        for entity_name in f1_entities:
            if entity_name in self.entity_map:
                pack.entities[entity_name] = self.entity_map[entity_name]

        # F2: Extended dependencies (2-hop) if requested
        if max_depth >= 2:
            f2_entities = set()
            for f1_entity in f1_entities:
                f2_neighbors = self._get_neighbors(f1_entity)
                f2_entities.update(f2_neighbors)

            # Remove entities already in F0 or F1
            f2_entities -= pack.f0_core
            f2_entities -= pack.f1_immediate
            pack.f2_extended = f2_entities

            for entity_name in f2_entities:
                if entity_name in self.entity_map:
                    pack.entities[entity_name] = self.entity_map[entity_name]

        # Add class peers if requested
        if include_peers:
            self._add_class_peers(pack)

        # Add related state variables
        self._add_related_state(pack)

        return pack

    def _build_graphs(self, module: Module) -> None:
        """Build forward and reverse dependency graphs."""
        self.entity_map.clear()
        self.dependency_graph.clear()
        self.reverse_graph.clear()
        self.class_map.clear()

        # Register all entities
        for entity in module.entities:
            if isinstance(entity, (Class, Data, Function)):
                self.entity_map[entity.name] = entity
                self.dependency_graph[entity.name] = set()
                self.reverse_graph[entity.name] = set()

                # Track class methods
                if isinstance(entity, Class):
                    for method in entity.methods:
                        self.class_map[method.name] = entity.name

        # Build dependency edges
        for entity in module.entities:
            if isinstance(entity, Class):
                deps = self._extract_class_dependencies(entity)
                self.dependency_graph[entity.name] = deps

                # Build reverse edges
                for dep in deps:
                    if dep in self.reverse_graph:
                        self.reverse_graph[dep].add(entity.name)

            elif isinstance(entity, Data):
                deps = self._extract_data_dependencies(entity)
                self.dependency_graph[entity.name] = deps

                for dep in deps:
                    if dep in self.reverse_graph:
                        self.reverse_graph[dep].add(entity.name)

            elif isinstance(entity, Function):
                # Functions can reference classes/data
                # For now, just track type dependencies
                deps = set()
                for param in entity.params:
                    if param.type_spec:
                        type_ref = self._extract_type_reference(param.type_spec.base_type)
                        if type_ref:
                            deps.add(type_ref)

                self.dependency_graph[entity.name] = deps

                for dep in deps:
                    if dep in self.reverse_graph:
                        self.reverse_graph[dep].add(entity.name)

    def _get_neighbors(self, entity_name: str) -> set[str]:
        """Get all neighbors (callees + callers) of an entity."""
        neighbors = set()

        # Add dependencies (entities this one depends on)
        if entity_name in self.dependency_graph:
            neighbors.update(self.dependency_graph[entity_name])

        # Add reverse dependencies (entities that depend on this one)
        if entity_name in self.reverse_graph:
            neighbors.update(self.reverse_graph[entity_name])

        return neighbors

    def _extract_class_dependencies(self, cls: Class) -> set[str]:
        """Extract all dependencies from a class."""
        deps = set()

        # State variable dependencies
        for state_var in cls.state:
            if state_var.type_spec:
                type_ref = self._extract_type_reference(state_var.type_spec.base_type)
                if type_ref:
                    deps.add(type_ref)

                # Handle union types
                if state_var.type_spec.union_types:
                    for union_type in state_var.type_spec.union_types:
                        type_ref = self._extract_type_reference(union_type)
                        if type_ref:
                            deps.add(type_ref)

        # Method parameter dependencies
        for method in cls.methods:
            for param in method.params:
                if param.type_spec:
                    type_ref = self._extract_type_reference(param.type_spec.base_type)
                    if type_ref:
                        deps.add(type_ref)

        return deps

    def _extract_data_dependencies(self, data: Data) -> set[str]:
        """Extract all dependencies from a data structure."""
        deps = set()

        for data_field in data.fields:
            if data_field.type_spec:
                type_ref = self._extract_type_reference(data_field.type_spec.base_type)
                if type_ref:
                    deps.add(type_ref)

                # Handle union types
                if data_field.type_spec.union_types:
                    for union_type in data_field.type_spec.union_types:
                        type_ref = self._extract_type_reference(union_type)
                        if type_ref:
                            deps.add(type_ref)

        return deps

    def _extract_type_reference(self, type_str: str) -> str | None:
        """Extract entity name from type string."""
        if "Ref:" in type_str:
            parts = type_str.split("Ref:")
            if len(parts) > 1:
                return parts[1].strip("[]").strip()

        # Check if it's a known entity
        if type_str in self.entity_map:
            return type_str

        return None

    def _add_class_peers(self, pack: ContextPack) -> None:
        """Add class peer methods to the context pack."""
        # Find if target or any F1 entities are methods
        all_current = pack.f0_core | pack.f1_immediate

        for entity_name in all_current:
            if entity_name in self.class_map:
                # This is a method, find its class
                class_name = self.class_map[entity_name]

                if class_name in self.entity_map:
                    cls = self.entity_map[class_name]
                    if isinstance(cls, Class):
                        # Add all methods from this class as peers
                        for method in cls.methods:
                            if method.name not in all_current:
                                pack.class_peers.add(method.name)
                                # Note: Methods are part of class, not separate entities

    def _add_related_state(self, pack: ContextPack) -> None:
        """Add related state variables to the context pack."""
        # For each class in the pack, track its state variables
        all_entities = pack.f0_core | pack.f1_immediate | pack.f2_extended

        for entity_name in all_entities:
            if entity_name in self.entity_map:
                entity = self.entity_map[entity_name]

                if isinstance(entity, Class):
                    for state_var in entity.state:
                        pack.related_state.add(f"{entity_name}.{state_var.name}")


def generate_context_pack(
    module: Module,
    target_name: str,
    max_depth: int = 2,
    include_peers: bool = True,
) -> ContextPack | None:
    """
    Convenience function to generate a context pack.

    Args:
        module: The PyShorthand module
        target_name: Name of target entity
        max_depth: Maximum dependency depth (default 2 for F1 + F2)
        include_peers: Include class peer methods (default True)

    Returns:
        ContextPack or None if target not found
    """
    generator = ContextPackGenerator()
    return generator.generate_context_pack(module, target_name, max_depth, include_peers)
