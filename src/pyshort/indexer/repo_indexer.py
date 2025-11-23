"""Repository indexer for PyShorthand.

Scans entire repositories and generates PyShorthand specs with dependency analysis.
"""

import ast
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from pyshort.decompiler import decompile_file


@dataclass
class EntityInfo:
    """Information about a single entity (class/function)."""

    name: str
    type: str  # 'class' or 'function'
    file_path: str
    module_path: str
    line_number: int
    state_vars: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    dependencies: set[str] = field(default_factory=set)


@dataclass
class ModuleInfo:
    """Information about a Python module."""

    module_path: str
    file_path: str
    entities: list[EntityInfo] = field(default_factory=list)
    imports: set[str] = field(default_factory=set)
    line_count: int = 0
    pyshorthand: str = ""


@dataclass
class RepositoryIndex:
    """Complete repository index."""

    root_path: str
    modules: dict[str, ModuleInfo] = field(default_factory=dict)
    entity_map: dict[str, EntityInfo] = field(default_factory=dict)  # name -> entity
    dependency_graph: dict[str, set[str]] = field(default_factory=dict)  # module -> dependencies
    statistics: dict[str, int] = field(default_factory=dict)


class RepositoryIndexer:
    """Index Python repositories and generate PyShorthand specs."""

    def __init__(self, root_path: str, exclude_patterns: list[str] | None = None):
        """Initialize indexer.

        Args:
            root_path: Root directory of repository
            exclude_patterns: Patterns to exclude (e.g., 'venv', '__pycache__')
        """
        self.root_path = Path(root_path).resolve()
        self.exclude_patterns = exclude_patterns or [
            "venv",
            "env",
            ".venv",
            "__pycache__",
            ".git",
            ".pytest_cache",
            "node_modules",
            "dist",
            "build",
            ".eggs",
            "*.egg-info",
        ]
        self.index = RepositoryIndex(root_path=str(self.root_path))

    def should_exclude(self, path: Path) -> bool:
        """Check if path should be excluded."""
        # Check dot directories first (before loop for efficiency)
        if path.name.startswith(".") and path.is_dir():
            return True

        # Check exclusion patterns against path components (not full string)
        path_parts = path.parts
        for pattern in self.exclude_patterns:
            # Handle glob patterns
            if "*" in pattern:
                if path.match(pattern):
                    return True
            else:
                # Check if pattern matches any path component
                # e.g., "test" should match "/foo/test/bar" but not "/home/latest/foo"
                if pattern in path_parts:
                    return True

        return False

    def find_python_files(self) -> list[Path]:
        """Find all Python files in repository."""
        python_files = []

        for path in self.root_path.rglob("*.py"):
            if not self.should_exclude(path):
                python_files.append(path)

        return sorted(python_files)

    def get_module_path(self, file_path: Path) -> str:
        """Convert file path to Python module path.

        Examples:
            src/foo/bar.py -> foo.bar
            foo/bar/baz.py -> foo.bar.baz
        """
        # Get relative path from root
        try:
            rel_path = file_path.relative_to(self.root_path)
        except ValueError:
            # File is outside root
            return file_path.stem

        # Remove .py extension
        parts = list(rel_path.parts[:-1]) + [rel_path.stem]

        # Remove 'src' if it's the first part
        if parts and parts[0] == "src":
            parts = parts[1:]

        # Remove __init__ from module path
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]

        return ".".join(parts) if parts else file_path.stem

    def extract_imports(self, source: str) -> set[str]:
        """Extract import statements from Python source."""
        imports = set()

        try:
            tree = ast.parse(source)

            # Only iterate module-level nodes, not nested scopes
            for node in tree.body:
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split(".")[0])

                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split(".")[0])

        except SyntaxError:
            pass  # Skip files with syntax errors

        return imports

    def extract_entities(self, source: str, file_path: str, module_path: str) -> list[EntityInfo]:
        """Extract entity information from Python source."""
        entities = []

        try:
            tree = ast.parse(source)

            # Only iterate module-level nodes, not nested scopes
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    entity = EntityInfo(
                        name=node.name,
                        type="class",
                        file_path=file_path,
                        module_path=module_path,
                        line_number=node.lineno,
                    )

                    # Extract method names
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            entity.methods.append(item.name)

                    # Extract dependencies from base classes
                    for base in node.bases:
                        if isinstance(base, ast.Name):
                            entity.dependencies.add(base.id)
                        elif isinstance(base, ast.Attribute):
                            # Handle torch.nn.Module, etc.
                            parts = []
                            current = base
                            while isinstance(current, ast.Attribute):
                                parts.insert(0, current.attr)
                                current = current.value
                            if isinstance(current, ast.Name):
                                parts.insert(0, current.id)
                            entity.dependencies.add(".".join(parts))

                    entities.append(entity)

                elif isinstance(node, ast.FunctionDef):
                    # Top-level functions
                    entity = EntityInfo(
                        name=node.name,
                        type="function",
                        file_path=file_path,
                        module_path=module_path,
                        line_number=node.lineno,
                    )
                    entities.append(entity)

        except SyntaxError:
            pass

        return entities

    def index_file(self, file_path: Path) -> ModuleInfo | None:
        """Index a single Python file."""
        try:
            with open(file_path, encoding="utf-8") as f:
                source = f.read()

            module_path = self.get_module_path(file_path)

            # Extract imports
            imports = self.extract_imports(source)

            # Extract entities
            entities = self.extract_entities(source, str(file_path), module_path)

            # Generate PyShorthand
            pyshorthand = ""
            try:
                pyshorthand = decompile_file(str(file_path))
            except Exception:
                pass  # Decompilation failed, skip

            # Count lines
            line_count = source.count("\n") + 1

            module_info = ModuleInfo(
                module_path=module_path,
                file_path=str(file_path),
                entities=entities,
                imports=imports,
                line_count=line_count,
                pyshorthand=pyshorthand,
            )

            return module_info

        except Exception:
            # Skip files that can't be read/parsed
            return None

    def build_dependency_graph(self):
        """Build module-level dependency graph."""
        # Pre-build set of all module paths for O(1) lookups
        all_modules = set(self.index.modules.keys())

        for module_path, module_info in self.index.modules.items():
            dependencies = set()

            # Add imports that reference other modules in the repo
            for imp in module_info.imports:
                # Fast exact match check using set lookup O(1)
                if imp in all_modules:
                    dependencies.add(imp)

                # Check for sub-modules (e.g., imp="foo" matches "foo.bar")
                # Still need to iterate, but much less common
                for other_module in all_modules:
                    if other_module.startswith(imp + "."):
                        dependencies.add(other_module)

            self.index.dependency_graph[module_path] = dependencies

    def compute_statistics(self):
        """Compute repository statistics."""
        total_files = len(self.index.modules)
        total_lines = sum(m.line_count for m in self.index.modules.values())
        total_entities = len(self.index.entity_map)
        total_classes = sum(1 for e in self.index.entity_map.values() if e.type == "class")
        total_functions = sum(1 for e in self.index.entity_map.values() if e.type == "function")

        self.index.statistics = {
            "total_files": total_files,
            "total_lines": total_lines,
            "total_entities": total_entities,
            "total_classes": total_classes,
            "total_functions": total_functions,
            "avg_lines_per_file": total_lines // total_files if total_files > 0 else 0,
        }

    def index_repository(self, verbose: bool = False) -> RepositoryIndex:
        """Index entire repository.

        Args:
            verbose: Print progress messages

        Returns:
            Complete repository index
        """
        python_files = self.find_python_files()

        if verbose:
            print(f"Found {len(python_files)} Python files")

        # Index each file
        for i, file_path in enumerate(python_files):
            if verbose and (i % 10 == 0 or i == len(python_files) - 1):
                print(f"  Indexing: {i + 1}/{len(python_files)} files...", end="\r")

            module_info = self.index_file(file_path)

            if module_info:
                self.index.modules[module_info.module_path] = module_info

                # Add entities to global map
                for entity in module_info.entities:
                    # Use fully qualified name: module.EntityName
                    # Handle empty module_path (e.g., from __init__.py at root)
                    if module_info.module_path:
                        fqn = f"{module_info.module_path}.{entity.name}"
                    else:
                        fqn = entity.name
                    self.index.entity_map[fqn] = entity

        if verbose:
            print()  # New line after progress

        # Build dependency graph
        self.build_dependency_graph()

        # Compute statistics
        self.compute_statistics()

        return self.index

    def save_index(self, output_path: str):
        """Save index to JSON file."""

        # Helper function to convert EntityInfo to dict with sets as lists
        def entity_to_dict(entity: EntityInfo) -> dict:
            entity_dict = asdict(entity)
            # Convert set to list for JSON serialization
            entity_dict["dependencies"] = list(entity_dict["dependencies"])
            return entity_dict

        # Convert to serializable format
        data = {
            "root_path": self.index.root_path,
            "modules": {
                path: {
                    "module_path": info.module_path,
                    "file_path": info.file_path,
                    "entities": [entity_to_dict(e) for e in info.entities],
                    "imports": list(info.imports),
                    "line_count": info.line_count,
                }
                for path, info in self.index.modules.items()
            },
            "dependency_graph": {k: list(v) for k, v in self.index.dependency_graph.items()},
            "statistics": self.index.statistics,
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    def generate_report(self) -> str:
        """Generate human-readable index report."""
        lines = []
        lines.append("=" * 80)
        lines.append("REPOSITORY INDEX REPORT")
        lines.append("=" * 80)
        lines.append(f"Repository: {self.index.root_path}")
        lines.append("")

        # Statistics
        stats = self.index.statistics
        lines.append("Statistics:")
        lines.append(f"  Total Python files: {stats['total_files']}")
        lines.append(f"  Total lines of code: {stats['total_lines']:,}")
        lines.append(f"  Average lines per file: {stats['avg_lines_per_file']}")
        lines.append(f"  Total entities: {stats['total_entities']}")
        lines.append(f"    Classes: {stats['total_classes']}")
        lines.append(f"    Functions: {stats['total_functions']}")
        lines.append("")

        # Top modules by entity count
        modules_by_entity_count = sorted(
            self.index.modules.items(), key=lambda x: len(x[1].entities), reverse=True
        )[:10]

        if modules_by_entity_count:
            lines.append("Top modules by entity count:")
            for module_path, module_info in modules_by_entity_count:
                lines.append(f"  {module_path}: {len(module_info.entities)} entities")
            lines.append("")

        # Dependency insights
        if self.index.dependency_graph:
            most_dependencies = max(
                self.index.dependency_graph.items(), key=lambda x: len(x[1]), default=(None, set())
            )

            if most_dependencies[0]:
                lines.append("Module with most dependencies:")
                lines.append(f"  {most_dependencies[0]}: {len(most_dependencies[1])} dependencies")
                lines.append("")

        lines.append("=" * 80)
        return "\n".join(lines)

    def generate_dependency_graph_mermaid(self, max_nodes: int = 20) -> str:
        """Generate Mermaid dependency graph.

        Args:
            max_nodes: Maximum number of nodes to include

        Returns:
            Mermaid graph definition
        """
        lines = []
        lines.append("```mermaid")
        lines.append("graph TD")
        lines.append("")

        # Limit to most connected modules
        modules_by_connections = sorted(
            self.index.dependency_graph.items(), key=lambda x: len(x[1]), reverse=True
        )[:max_nodes]

        # Create nodes and edges
        included_modules = {mod for mod, _ in modules_by_connections}

        # Add nodes
        for module_path, _ in modules_by_connections:
            # Simplify module name for display
            display_name = module_path.split(".")[-1] if "." in module_path else module_path
            node_id = module_path.replace(".", "_")
            lines.append(f'    {node_id}["{display_name}"]')

        lines.append("")

        # Add edges (dependencies)
        for module_path, dependencies in modules_by_connections:
            node_id = module_path.replace(".", "_")

            for dep in dependencies:
                if dep in included_modules:
                    dep_id = dep.replace(".", "_")
                    lines.append(f"    {node_id} --> {dep_id}")

        lines.append("```")
        return "\n".join(lines)

    def generate_entity_map_report(self, limit: int = 50) -> str:
        """Generate report of all entities in the repository.

        Args:
            limit: Maximum entities to show

        Returns:
            Entity map report
        """
        lines = []
        lines.append("=" * 80)
        lines.append("ENTITY MAP")
        lines.append("=" * 80)
        lines.append(f"Total entities: {len(self.index.entity_map)}")
        lines.append("")

        # Group entities by module
        entities_by_module = defaultdict(list)
        for fqn, entity in self.index.entity_map.items():
            entities_by_module[entity.module_path].append(entity)

        # Sort modules by number of entities
        sorted_modules = sorted(entities_by_module.items(), key=lambda x: len(x[1]), reverse=True)[
            :limit
        ]

        for module_path, entities in sorted_modules:
            lines.append(f"\n{module_path}:")
            for entity in sorted(entities, key=lambda e: e.name):
                type_indicator = "C" if entity.type == "class" else "F"
                lines.append(f"  [{type_indicator}] {entity.name}")
                if entity.methods and len(entity.methods) <= 5:
                    for method in entity.methods[:5]:
                        lines.append(f"      • {method}()")

        lines.append("")
        lines.append("=" * 80)
        return "\n".join(lines)


def index_repository(
    root_path: str, output_path: str | None = None, verbose: bool = False
) -> RepositoryIndex:
    """Index a repository and optionally save results.

    Args:
        root_path: Path to repository root
        output_path: Optional path to save index JSON
        verbose: Print progress messages

    Returns:
        Repository index
    """
    indexer = RepositoryIndexer(root_path)
    index = indexer.index_repository(verbose=verbose)

    if output_path:
        indexer.save_index(output_path)

    return index
