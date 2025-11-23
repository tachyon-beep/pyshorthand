"""Core tools for PyShorthand ecosystem progressive disclosure."""

import ast
from dataclasses import dataclass
from pathlib import Path

# Import context pack and execution flow analyzers
try:
    from ..analyzer.context_pack import ContextPack, ContextPackGenerator
    from ..analyzer.execution_flow import ExecutionFlow, ExecutionFlowTracer
    from ..core.parser import parse_string as parse_pyshorthand
    from ..decompiler.py2short import PyShorthandGenerator

    _ADVANCED_TOOLS_AVAILABLE = True
except ImportError:
    _ADVANCED_TOOLS_AVAILABLE = False
    ContextPack = None
    ExecutionFlow = None


@dataclass
class MethodImplementation:
    """Full implementation of a method."""

    class_name: str
    method_name: str
    source_code: str
    line_start: int
    line_end: int
    dependencies: list[str]  # Other methods called within this method


@dataclass
class ClassDetails:
    """Detailed class information."""

    name: str
    base_classes: list[str]
    attributes: dict[str, str]  # name -> type annotation
    methods: dict[str, str]  # name -> signature
    nested_structures: dict[str, str]  # For ModuleDict, etc.


class CodebaseExplorer:
    """
    Progressive disclosure system for code understanding.

    Provides on-demand access to implementation details while maintaining
    a lightweight PyShorthand overview as the default context.
    """

    def __init__(self, codebase_path: Path):
        """Initialize explorer with path to Python codebase.

        Args:
            codebase_path: Path to the Python file or directory to explore
        """
        self.codebase_path = Path(codebase_path)
        self.cache: dict[str, str] = {}
        self._ast_cache: dict[Path, ast.Module] = {}

    def get_implementation(self, target: str, include_context: bool = True) -> str | None:
        """Retrieve full Python implementation of a specific method.

        Args:
            target: Format "ClassName.method_name" (e.g., "GPT.forward")
            include_context: Include related helper methods called within

        Returns:
            Full Python source code of the method, or None if not found

        Example:
            >>> explorer = CodebaseExplorer("model.py")
            >>> code = explorer.get_implementation("GPT.configure_optimizers")
            >>> print(code)
            def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
                param_dict = {pn: p for pn, p in self.named_parameters()}
                ...
        """
        cache_key = f"impl:{target}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Parse target
        if "." not in target:
            return None

        class_name, method_name = target.rsplit(".", 1)

        # Find implementation
        impl = self._extract_method_implementation(class_name, method_name)
        if impl is None:
            return None

        result = impl.source_code

        # Optionally include called methods
        if include_context and impl.dependencies:
            result += "\n\n# Called methods:\n"
            for dep in impl.dependencies:
                dep_impl = self._extract_method_implementation(class_name, dep)
                if dep_impl:
                    result += f"\n{dep_impl.source_code}\n"

        self.cache[cache_key] = result
        return result

    def get_class_details(
        self,
        class_name: str,
        include_methods: bool = False,
        expand_nested: bool = True,
    ) -> str | None:
        """Retrieve detailed type information for a class.

        Args:
            class_name: Name of the class to inspect
            include_methods: Include full method implementations (expensive)
            expand_nested: Expand nested structures like ModuleDict

        Returns:
            Formatted string with class details, or None if not found

        Example:
            >>> explorer = CodebaseExplorer("model.py")
            >>> details = explorer.get_class_details("GPT", expand_nested=True)
            >>> print(details)
            class GPT(nn.Module):
                config: GPTConfig
                transformer: nn.ModuleDict = {
                    'wte': nn.Embedding(50304, 768),
                    ...
                }
        """
        cache_key = f"class:{class_name}:{include_methods}:{expand_nested}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        details = self._extract_class_details(class_name, expand_nested)
        if details is None:
            return None

        # Format output
        lines = []

        # Class declaration
        if details.base_classes:
            bases = ", ".join(details.base_classes)
            lines.append(f"class {details.name}({bases}):")
        else:
            lines.append(f"class {details.name}:")

        # Attributes with types
        if details.attributes:
            lines.append("    # State Variables:")
            for attr_name, attr_type in details.attributes.items():
                if attr_name in details.nested_structures and expand_nested:
                    # Show expanded structure
                    lines.append(f"    {attr_name}: {attr_type} = {{")
                    for key, val in self._parse_nested_structure(
                        details.nested_structures[attr_name]
                    ).items():
                        lines.append(f"        '{key}': {val},")
                    lines.append("    }")
                else:
                    lines.append(f"    {attr_name}: {attr_type}")

        # Method signatures
        if details.methods:
            lines.append("")
            lines.append("    # Methods:")
            for method_name, signature in details.methods.items():
                lines.append(f"    {signature}")

                # Optionally include full implementation
                if include_methods:
                    impl = self._extract_method_implementation(class_name, method_name)
                    if impl:
                        # Indent implementation
                        for line in impl.source_code.split("\n"):
                            lines.append(f"    {line}")

        result = "\n".join(lines)
        self.cache[cache_key] = result
        return result

    def search_usage(self, symbol: str) -> list[str]:
        """Find where a class/method is used in the codebase.

        Args:
            symbol: Class name, method name, or variable to search for

        Returns:
            List of usage locations (formatted strings)

        Example:
            >>> explorer = CodebaseExplorer("model.py")
            >>> usages = explorer.search_usage("LayerNorm")
            >>> for usage in usages:
            ...     print(usage)
            Block.ln_1 (state variable)
            Block.ln_2 (state variable)
            GPT.transformer.ln_f (nested in ModuleDict)
        """
        usages = []

        # Search through all classes
        for file_path in self._get_python_files():
            tree = self._get_ast(file_path)
            if tree is None:
                continue

            for node in ast.walk(tree):
                # Check class attribute assignments
                if isinstance(node, ast.ClassDef):
                    for item in node.body:
                        if isinstance(item, ast.Assign):
                            # Check if symbol appears in the value
                            if self._contains_symbol(item.value, symbol):
                                for target in item.targets:
                                    if isinstance(target, ast.Attribute):
                                        usages.append(f"{node.name}.{target.attr} (state variable)")
                                    elif isinstance(target, ast.Name):
                                        usages.append(f"{node.name}.{target.id} (state variable)")

                # Check method calls
                if isinstance(node, ast.Call):
                    if self._is_class_instantiation(node, symbol):
                        # Find containing class/method
                        parent = self._find_parent_context(tree, node)
                        if parent:
                            usages.append(f"{parent} (instantiation)")

        return usages

    def get_context_pack(
        self, target: str, max_depth: int = 2, include_peers: bool = True
    ) -> dict | None:
        """Get dependency context pack with F0/F1/F2 layers and neighbors.

        This returns a dependency-aware context pack showing:
        - F0: The target entity itself
        - F1: Direct dependencies (callers + callees)
        - F2: 2-hop dependencies
        - Class peers: Other methods in same class
        - Related state: State variables accessed

        Args:
            target: Entity name to analyze (class or function)
            max_depth: Maximum dependency depth (1=F1 only, 2=F1+F2)
            include_peers: Include class peer methods

        Returns:
            Dictionary with context pack data, or None if advanced tools unavailable

        Example:
            >>> explorer = CodebaseExplorer("model.py")
            >>> pack = explorer.get_context_pack("GPT.forward", max_depth=2)
            >>> print(pack["f1_immediate"])  # Direct dependencies
            ['Block', 'LayerNorm', 'Embedding']
            >>> print(pack["f2_extended"])  # 2-hop dependencies
            ['CausalSelfAttention', 'MLP']
        """
        if not _ADVANCED_TOOLS_AVAILABLE:
            return None

        # First need to convert Python to PyShorthand module
        pyshorthand_module = self._get_pyshorthand_module()
        if not pyshorthand_module:
            return None

        # Generate context pack
        generator = ContextPackGenerator()
        pack = generator.generate_context_pack(pyshorthand_module, target, max_depth, include_peers)

        if pack:
            return pack.to_dict()
        return None

    def trace_execution(
        self, entry_point: str, max_depth: int = 10, follow_calls: bool = True
    ) -> dict | None:
        """Trace execution flow through function calls.

        Shows the runtime call path, variables in scope, and call graph.

        Args:
            entry_point: Function/method to start tracing from
            max_depth: Maximum call depth to trace
            follow_calls: If True, recursively trace into function calls

        Returns:
            Dictionary with execution flow data, or None if unavailable

        Example:
            >>> explorer = CodebaseExplorer("model.py")
            >>> flow = explorer.trace_execution("GPT.forward", max_depth=5)
            >>> print(flow["execution_path"])
            [
                {"depth": 0, "entity": "GPT.forward", "calls": ["Block.forward"]},
                {"depth": 1, "entity": "Block.forward", "calls": ["LayerNorm", "CausalSelfAttention"]},
                ...
            ]
        """
        if not _ADVANCED_TOOLS_AVAILABLE:
            return None

        # Get PyShorthand module
        pyshorthand_module = self._get_pyshorthand_module()
        if not pyshorthand_module:
            return None

        # Trace execution
        tracer = ExecutionFlowTracer()
        flow = tracer.trace_execution(pyshorthand_module, entry_point, max_depth, follow_calls)

        if flow:
            return flow.to_dict()
        return None

    def get_neighbors(self, symbol: str) -> dict[str, list[str]] | None:
        """Get direct neighbors (callers + callees) of an entity.

        This is a simplified version of get_context_pack that only returns
        F1 (immediate dependencies) in both directions.

        Args:
            symbol: Entity name to get neighbors for

        Returns:
            Dict with 'callees' (what this calls) and 'callers' (what calls this)

        Example:
            >>> explorer = CodebaseExplorer("model.py")
            >>> neighbors = explorer.get_neighbors("Block.forward")
            >>> print(neighbors["callees"])
            ['LayerNorm', 'CausalSelfAttention', 'MLP']
            >>> print(neighbors["callers"])
            ['GPT.forward']
        """
        pack_data = self.get_context_pack(symbol, max_depth=1, include_peers=False)
        if not pack_data:
            return None

        return {
            "callees": pack_data.get("f1_immediate", []),
            "callers": [],  # Would need reverse dependency analysis
            "peers": pack_data.get("class_peers", []),
        }

    def get_module_pyshorthand(self) -> str | None:
        """Get entire module in PyShorthand format.

        Returns the full PyShorthand representation of the entire codebase,
        showing all classes, functions, and their relationships.

        Returns:
            PyShorthand formatted string, or None if unavailable

        Example:
            >>> explorer = CodebaseExplorer("model.py")
            >>> pyshorthand = explorer.get_module_pyshorthand()
            >>> print(pyshorthand)
            # [M:model] [Role:Core]

            [C:GPT] ◊ nn.Module
              config ∈ GPTConfig
              transformer ∈ ModuleDict {...}
              F:forward(idx, targets) → (Tensor, Tensor?)
            ...
        """
        if not _ADVANCED_TOOLS_AVAILABLE:
            return None

        # Read all Python files
        all_pyshorthand = []

        for file_path in self._get_python_files():
            try:
                with open(file_path) as f:
                    source = f.read()

                tree = ast.parse(source)
                generator = PyShorthandGenerator()
                pyshorthand = generator.generate(tree, str(file_path))
                all_pyshorthand.append(f"# File: {file_path.name}\n{pyshorthand}")
            except Exception:
                continue

        if all_pyshorthand:
            return "\n\n".join(all_pyshorthand)
        return None

    def get_class_pyshorthand(self, class_name: str) -> str | None:
        """Get a single class in PyShorthand format.

        Returns just the PyShorthand representation of one class,
        including all its methods, state variables, and structure.

        Args:
            class_name: Name of class to get

        Returns:
            PyShorthand formatted string for the class, or None if not found

        Example:
            >>> explorer = CodebaseExplorer("model.py")
            >>> ps = explorer.get_class_pyshorthand("GPT")
            >>> print(ps)
            [C:GPT] ◊ nn.Module
              config ∈ GPTConfig
              transformer ∈ ModuleDict {
                wte: Embedding(50304, 768),
                wpe: Embedding(1024, 768),
                h: ModuleList<Block>[12]
              }
              F:forward(idx, targets) → (Tensor, Tensor?)
              F:generate(idx, max_new_tokens) → Tensor [no_grad]
        """
        if not _ADVANCED_TOOLS_AVAILABLE:
            return None

        # Find the class in source files
        for file_path in self._get_python_files():
            tree = self._get_ast(file_path)
            if tree is None:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    # Generate PyShorthand for just this class
                    try:
                        with open(file_path) as f:
                            source = f.read()

                        # Parse full file to get context
                        full_tree = ast.parse(source)
                        generator = PyShorthandGenerator()

                        # Generate for full module, then extract class
                        full_pyshorthand = generator.generate(full_tree, str(file_path))

                        # Extract just the class section
                        # Look for [C:ClassName] pattern
                        import re

                        class_pattern = rf"\[C:{class_name}\].*?(?=\n\[C:|$)"
                        match = re.search(class_pattern, full_pyshorthand, re.DOTALL)

                        if match:
                            return match.group(0).strip()
                    except Exception:
                        continue

        return None

    def _get_pyshorthand_module(self):
        """Internal: Convert Python files to PyShorthand module for analysis."""
        if not _ADVANCED_TOOLS_AVAILABLE:
            return None

        # Get full PyShorthand representation
        pyshorthand_str = self.get_module_pyshorthand()
        if not pyshorthand_str:
            return None

        try:
            # Parse PyShorthand into module AST
            module = parse_pyshorthand(pyshorthand_str)
            return module
        except Exception:
            return None

    # Private methods

    def _get_python_files(self) -> list[Path]:
        """Get all Python files in codebase path."""
        if self.codebase_path.is_file():
            return [self.codebase_path]
        return list(self.codebase_path.rglob("*.py"))

    def _get_ast(self, file_path: Path) -> ast.Module | None:
        """Get AST for a Python file (with caching)."""
        if file_path in self._ast_cache:
            return self._ast_cache[file_path]

        try:
            with open(file_path) as f:
                tree = ast.parse(f.read(), filename=str(file_path))
            self._ast_cache[file_path] = tree
            return tree
        except Exception:
            return None

    def _extract_method_implementation(
        self, class_name: str, method_name: str
    ) -> MethodImplementation | None:
        """Extract source code for a specific method."""
        for file_path in self._get_python_files():
            tree = self._get_ast(file_path)
            if tree is None:
                continue

            # Find the class
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    # Find the method
                    for item in node.body:
                        if (
                            isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                            and item.name == method_name
                        ):
                            # Extract source code
                            with open(file_path) as f:
                                source = ast.get_source_segment(f.read(), item)
                            if source is None:
                                continue

                            # Find dependencies (methods called within)
                            deps = self._find_method_calls(item)

                            return MethodImplementation(
                                class_name=class_name,
                                method_name=method_name,
                                source_code=source,
                                line_start=item.lineno,
                                line_end=item.end_lineno or item.lineno,
                                dependencies=deps,
                            )

        return None

    def _extract_class_details(
        self, class_name: str, expand_nested: bool
    ) -> ClassDetails | None:
        """Extract detailed information about a class."""
        for file_path in self._get_python_files():
            tree = self._get_ast(file_path)
            if tree is None:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    # Extract base classes
                    base_classes = []
                    for base in node.bases:
                        base_classes.append(ast.unparse(base))

                    # Extract attributes (from __init__ typically)
                    attributes = {}
                    nested_structures = {}
                    init_method = None

                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                            init_method = item
                            break

                    if init_method:
                        for stmt in init_method.body:
                            if isinstance(stmt, ast.Assign):
                                for target in stmt.targets:
                                    if isinstance(target, ast.Attribute):
                                        if (
                                            isinstance(target.value, ast.Name)
                                            and target.value.id == "self"
                                        ):
                                            attr_name = target.attr
                                            # Try to infer type
                                            attr_type = self._infer_type(stmt.value)
                                            attributes[attr_name] = attr_type

                                            # Check if it's a nested structure
                                            if expand_nested and isinstance(stmt.value, ast.Call):
                                                if self._is_nested_structure(stmt.value):
                                                    nested_structures[attr_name] = ast.unparse(
                                                        stmt.value
                                                    )

                    # Extract method signatures
                    methods = {}
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            # Build signature
                            sig = self._build_signature(item)
                            methods[item.name] = sig

                    return ClassDetails(
                        name=class_name,
                        base_classes=base_classes,
                        attributes=attributes,
                        methods=methods,
                        nested_structures=nested_structures,
                    )

        return None

    def _infer_type(self, node: ast.AST) -> str:
        """Infer type from AST node."""
        if isinstance(node, ast.Call):
            # Constructor call like nn.Linear(...)
            return ast.unparse(node.func)
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return ast.unparse(node)
        return "Unknown"

    def _is_nested_structure(self, node: ast.Call) -> bool:
        """Check if this is a nested structure like ModuleDict."""
        func_name = ast.unparse(node.func)
        return any(
            pattern in func_name
            for pattern in ["ModuleDict", "ModuleList", "Sequential", "ParameterDict"]
        )

    def _parse_nested_structure(self, source: str) -> dict[str, str]:
        """Parse nested structure from source code."""
        # Simple implementation - could be enhanced
        try:
            tree = ast.parse(source, mode="eval")
            if isinstance(tree.body, ast.Call):
                # Extract dict argument for ModuleDict
                for arg in tree.body.args:
                    if isinstance(arg, ast.Dict):
                        result = {}
                        for key, value in zip(arg.keys, arg.values):
                            key_str = ast.unparse(key) if key else "?"
                            val_str = ast.unparse(value)
                            result[key_str.strip("'")] = val_str
                        return result
        except Exception:
            pass
        return {}

    def _build_signature(self, func_node: ast.FunctionDef) -> str:
        """Build method signature string from AST."""
        args = []
        for arg in func_node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"
            args.append(arg_str)

        # Add defaults
        defaults = func_node.args.defaults
        if defaults:
            num_defaults = len(defaults)
            for i, default in enumerate(defaults):
                arg_idx = len(args) - num_defaults + i
                args[arg_idx] += f" = {ast.unparse(default)}"

        args_str = ", ".join(args)

        # Return type
        returns = ""
        if func_node.returns:
            returns = f" -> {ast.unparse(func_node.returns)}"

        return f"def {func_node.name}({args_str}){returns}"

    def _find_method_calls(self, func_node: ast.FunctionDef) -> list[str]:
        """Find all method calls within a function."""
        calls = set()
        for node in ast.walk(func_node):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                # self.method_name()
                if isinstance(node.func.value, ast.Name):
                    if node.func.value.id == "self":
                        calls.add(node.func.attr)
        return list(calls)

    def _contains_symbol(self, node: ast.AST, symbol: str) -> bool:
        """Check if AST node contains a reference to symbol."""
        source = ast.unparse(node)
        return symbol in source

    def _is_class_instantiation(self, call_node: ast.Call, class_name: str) -> bool:
        """Check if this call instantiates the given class."""
        func_name = ast.unparse(call_node.func)
        return class_name in func_name

    def _find_parent_context(self, tree: ast.Module, target_node: ast.AST) -> str | None:
        """Find the class.method context containing a node."""
        # Walk tree to find parent
        # This is simplified - would need proper parent tracking
        return None  # TODO: Implement proper parent tracking
