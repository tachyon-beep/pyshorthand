"""Python AST to PyShorthand decompiler implementation."""

import ast
import re
from dataclasses import dataclass
from pathlib import Path


class PyShorthandGenerator:
    """Generate PyShorthand from Python AST."""

    def __init__(self, aggressive: bool = False, with_confidence: bool = False):
        """Initialize generator.

        Args:
            aggressive: If True, use aggressive type inference
            with_confidence: If True, include confidence scores in output
        """
        self.aggressive = aggressive
        self.with_confidence = with_confidence
        self.imports: set[str] = set()
        self.local_classes: set[str] = set()  # Classes defined in this module
        self.dependencies: list[str] = []
        self.import_map: dict[str, str] = {}  # alias -> full module path

    def generate(self, tree: ast.Module, source_file: str | None = None) -> str:
        """Generate PyShorthand from Python AST.

        Args:
            tree: Python AST module
            source_file: Original source file path (for metadata)

        Returns:
            PyShorthand source code
        """
        lines = []

        # Extract imports first (needed for dependency analysis)
        self._extract_imports(tree)

        # Collect all class names defined in this module
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                self.local_classes.add(node.name)

        # Extract metadata from module docstring
        module_metadata = self._extract_module_metadata(tree, source_file)

        # Generate metadata header
        module_name = module_metadata.get("name", "UnnamedModule")
        role = module_metadata.get("role", "Core")
        lines.append(f"# [M:{module_name}] [Role:{role}]")

        # Add risk if found
        if "risk" in module_metadata:
            lines[-1] = lines[-1].rstrip("]") + f"] [Risk:{module_metadata['risk']}]"

        lines.append("")

        # Extract classes
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]

        for cls in classes:
            entity_lines = self._generate_entity(cls, tree)
            lines.extend(entity_lines)
            lines.append("")

        # Extract module-level functions (including async functions)
        functions = [
            node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]

        if functions:
            lines.append("# Module-level functions")
            for func in functions:
                func_line = self._generate_function_signature(func)
                lines.append(f"{func_line}")
            lines.append("")

        return "\n".join(lines)

    def _extract_module_name(self, tree: ast.Module, source_file: str | None) -> str:
        """Extract module name from AST or file path."""
        # Try to find module name from docstring or file path
        if tree.body and isinstance(tree.body[0], ast.Expr):
            if isinstance(tree.body[0].value, ast.Constant):
                docstring = tree.body[0].value.value
                if isinstance(docstring, str):
                    # Extract first line of docstring as module name
                    first_line = docstring.strip().split("\n")[0]
                    # Clean up common patterns
                    if "." in first_line:
                        return first_line.split(".")[0]
                    return first_line[:50]  # Limit length

        # Fallback to file name
        if source_file:
            return Path(source_file).stem

        return "UnnamedModule"

    def _extract_imports(self, tree: ast.Module):
        """Extract import statements and build import map."""
        # Only iterate module-level nodes, not nested scopes
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name
                    as_name = alias.asname if alias.asname else alias.name
                    self.imports.add(module_name)
                    self.import_map[as_name] = module_name

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    self.imports.add(node.module)
                    for alias in node.names:
                        imported_name = alias.name
                        as_name = alias.asname if alias.asname else imported_name
                        # Store mapping from alias to module.name
                        full_name = f"{node.module}.{imported_name}"
                        self.import_map[as_name] = full_name

    def _extract_module_metadata(
        self, tree: ast.Module, source_file: str | None
    ) -> dict[str, str]:
        """Extract metadata from module docstring and file."""
        metadata = {}

        # Get module name
        metadata["name"] = self._extract_module_name(tree, source_file)

        # Try to extract metadata from docstring
        if tree.body and isinstance(tree.body[0], ast.Expr):
            if isinstance(tree.body[0].value, ast.Constant):
                docstring = tree.body[0].value.value
                if isinstance(docstring, str):
                    # Look for role/risk/layer tags in docstring
                    tags = self._extract_docstring_tags(docstring)
                    metadata.update(tags)

        return metadata

    def _extract_docstring_tags(self, docstring: str) -> dict[str, str]:
        """Extract PyShorthand tags from docstring.

        Recognizes patterns like:
        - Role: Core|Service|Util
        - Risk: High|Med|Low
        - Layer: API|Logic|Data
        - :O(N) for complexity
        """
        tags = {}

        # Role pattern
        role_match = re.search(
            r"Role:\s*(Core|Service|Util|API|Logic|Data)", docstring, re.IGNORECASE
        )
        if role_match:
            tags["role"] = role_match.group(1).capitalize()

        # Risk pattern
        risk_match = re.search(r"Risk:\s*(High|Med|Low)", docstring, re.IGNORECASE)
        if risk_match:
            tags["risk"] = risk_match.group(1).capitalize()

        # Layer pattern
        layer_match = re.search(r"Layer:\s*(API|Logic|Data)", docstring, re.IGNORECASE)
        if layer_match:
            tags["layer"] = layer_match.group(1)

        return tags

    def _generate_entity(self, cls: ast.ClassDef, tree: ast.Module) -> list[str]:
        """Generate PyShorthand entity from Python class.

        0.9.0-RC1 enhancements:
        - Extract base classes as inheritance (◊ notation)
        - Detect abstract classes ([Abstract] tag)
        - Detect Protocol classes ([P:Name] prefix)
        - Extract generic parameters from typing (e.g., Generic[T])

        Args:
            cls: Python ClassDef node
            tree: Module AST (for context)

        Returns:
            Lines of PyShorthand code
        """
        lines = []

        # Check for special patterns
        is_dataclass = self._is_dataclass(cls)
        is_pydantic = self._is_pydantic_model(cls)
        web_framework = self._detect_web_framework(cls)

        # 0.9.0-RC1: Detect abstract and protocol classes
        is_abstract = self._is_abstract_class(cls)
        is_protocol = self._is_protocol_class(cls)
        generic_params = self._extract_generic_params(cls)

        # 0.9.0-RC1: Use P: prefix for protocols
        prefix = "P" if is_protocol else "C"

        # Entity header with optional generic parameters
        if generic_params:
            header = f"[{prefix}:{cls.name}<{', '.join(generic_params)}>]"
        else:
            header = f"[{prefix}:{cls.name}]"

        # 0.9.0-RC1: Add [Abstract] tag if applicable
        if is_abstract and not is_protocol:
            header += " [Abstract]"

        # Add pattern annotations as comments
        if is_dataclass:
            header += " # @dataclass"
        elif is_pydantic:
            header += " # Pydantic"
        elif web_framework:
            header += f" # {web_framework}"
        lines.append(header)

        # 0.9.0-RC1: Extract inheritance (◊ Base1, Base2)
        base_classes = self._extract_base_classes(cls)

        if base_classes:
            lines.append(f"  ◊ {', '.join(base_classes)}")

        # Extract class attributes with type hints
        state_vars = self._extract_state_variables(cls, is_dataclass or is_pydantic)

        if state_vars:
            for var in state_vars:
                lines.append(f"  {var}")
        else:
            lines.append("  # No typed attributes found")

        # Extract methods as comments (parser doesn't support F:name syntax in entities yet)
        methods = [
            node for node in cls.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]

        if methods:
            lines.append("")
            lines.append("  # Methods:")
            for method in methods:
                if method.name.startswith("_") and method.name != "__init__":
                    continue  # Skip private methods except __init__

                # Generate signature with legacy tags
                sig = self._generate_function_signature(method, indent="  # ")
                lines.append(sig)

        return lines

    def _extract_class_dependencies(self, cls: ast.ClassDef) -> list[str]:
        """Extract dependencies from base classes."""
        dependencies = []

        for base in cls.bases:
            dep_name = None

            if isinstance(base, ast.Name):
                dep_name = base.id
            elif isinstance(base, ast.Attribute):
                # Handle torch.nn.Module, etc.
                parts = []
                current = base
                while isinstance(current, ast.Attribute):
                    parts.insert(0, current.attr)
                    current = current.value
                if isinstance(current, ast.Name):
                    parts.insert(0, current.id)
                dep_name = ".".join(parts)

            if dep_name:
                # If it's a local class, use [Ref:Name]
                if dep_name in self.local_classes:
                    dependencies.append(f"[Ref:{dep_name}]")
                # Otherwise, if it's imported, use module reference
                elif dep_name in self.import_map:
                    # For common frameworks, simplify
                    full_name = self.import_map[dep_name]
                    if "fastapi" in full_name.lower():
                        dependencies.append("[Ref:FastAPI]")
                    elif "pydantic" in full_name.lower():
                        dependencies.append("[Ref:Pydantic]")
                    elif "torch.nn" in full_name.lower() or "nn.Module" in dep_name:
                        dependencies.append("[Ref:PyTorch]")
                    elif "flask" in full_name.lower():
                        dependencies.append("[Ref:Flask]")
                    else:
                        # Generic external dependency
                        dependencies.append(f"[Ref:{dep_name}]")

        return dependencies

    def _extract_base_classes(self, cls: ast.ClassDef) -> list[str]:
        """Extract base classes for 0.9.0-RC1 inheritance notation.

        Args:
            cls: Python ClassDef node

        Returns:
            List of base class names (e.g., ['nn.Module', 'ABC'])
        """
        bases = []

        for base in cls.bases:
            base_name = None

            if isinstance(base, ast.Name):
                base_name = base.id
            elif isinstance(base, ast.Attribute):
                # Handle nn.Module, abc.ABC, etc.
                parts = []
                current = base
                while isinstance(current, ast.Attribute):
                    parts.insert(0, current.attr)
                    current = current.value
                if isinstance(current, ast.Name):
                    parts.insert(0, current.id)
                base_name = ".".join(parts)
            elif isinstance(base, ast.Subscript):
                # Handle Generic[T], Protocol[T], etc. - skip these
                # Generic/Protocol are handled separately
                continue

            if base_name and base_name not in ("Generic", "Protocol"):
                # Filter out ABC if already marked as abstract
                if base_name not in ("ABC", "abc.ABC") or not self._is_abstract_class(cls):
                    bases.append(base_name)

        return bases

    def _is_abstract_class(self, cls: ast.ClassDef) -> bool:
        """Check if class is abstract (0.9.0-RC1).

        Args:
            cls: Python ClassDef node

        Returns:
            True if class inherits from ABC or has abstractmethod decorators
        """
        # Check if inherits from ABC
        for base in cls.bases:
            if isinstance(base, ast.Name) and base.id in ("ABC",):
                return True
            elif isinstance(base, ast.Attribute):
                # Check for abc.ABC
                if base.attr == "ABC":
                    return True

        # Check for @abstractmethod decorators
        for node in cls.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for dec in node.decorator_list:
                    if isinstance(dec, ast.Name) and "abstract" in dec.id.lower() or isinstance(dec, ast.Attribute) and "abstract" in dec.attr.lower():
                        return True

        return False

    def _is_protocol_class(self, cls: ast.ClassDef) -> bool:
        """Check if class is a Protocol (0.9.0-RC1).

        Args:
            cls: Python ClassDef node

        Returns:
            True if class inherits from Protocol
        """
        for base in cls.bases:
            if isinstance(base, ast.Name) and base.id == "Protocol" or isinstance(base, ast.Attribute) and base.attr == "Protocol":
                return True
            elif isinstance(base, ast.Subscript):
                # Handle Protocol[T]
                if isinstance(base.value, ast.Name) and base.value.id == "Protocol":
                    return True

        return False

    def _extract_generic_params(self, cls: ast.ClassDef) -> list[str]:
        """Extract generic type parameters from class (0.9.0-RC1).

        Args:
            cls: Python ClassDef node

        Returns:
            List of generic parameter names (e.g., ['T', 'U'])
        """
        for base in cls.bases:
            if isinstance(base, ast.Subscript):
                # Check if this is Generic[T] or Generic[T, U]
                if isinstance(base.value, ast.Name) and base.value.id == "Generic":
                    params = []
                    if isinstance(base.slice, ast.Tuple):
                        # Generic[T, U, V]
                        for elt in base.slice.elts:
                            if isinstance(elt, ast.Name):
                                params.append(elt.id)
                    elif isinstance(base.slice, ast.Name):
                        # Generic[T]
                        params.append(base.slice.id)
                    return params

        return []

    def _is_dataclass(self, cls: ast.ClassDef) -> bool:
        """Check if class is a dataclass."""
        for decorator in cls.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id == "dataclass":
                return True
            elif isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Name) and decorator.func.id == "dataclass":
                    return True
        return False

    def _is_pydantic_model(self, cls: ast.ClassDef) -> bool:
        """Check if class is a Pydantic model."""
        for base in cls.bases:
            if isinstance(base, ast.Name):
                if "BaseModel" in base.id or "Pydantic" in base.id:
                    return True
            elif isinstance(base, ast.Attribute):
                if base.attr == "BaseModel":
                    return True
        return False

    def _detect_web_framework(self, cls: ast.ClassDef) -> str | None:
        """Detect which web framework this class uses."""
        # Check base classes
        for base in cls.bases:
            base_name = ""
            if isinstance(base, ast.Name):
                base_name = base.id
            elif isinstance(base, ast.Attribute):
                base_name = base.attr

            if "APIRouter" in base_name or "FastAPI" in base_name:
                return "FastAPI"
            elif "Flask" in base_name or "Blueprint" in base_name:
                return "Flask"
            elif "APIView" in base_name or "ViewSet" in base_name:
                return "Django REST"

        # Check decorators on methods
        for node in cls.body:
            if isinstance(node, ast.FunctionDef):
                for dec in node.decorator_list:
                    dec_name = ""
                    if isinstance(dec, ast.Name):
                        dec_name = dec.id
                    elif isinstance(dec, ast.Attribute):
                        dec_name = dec.attr

                    if dec_name in ("get", "post", "put", "delete", "patch"):
                        return "FastAPI/Flask"

        return None

    def _extract_state_variables(self, cls: ast.ClassDef, is_special: bool = False) -> list[str]:
        """Extract state variables from class.

        Looks for:
        1. Class-level annotated assignments
        2. Instance attributes in __init__

        Args:
            cls: Class definition
            is_special: True if dataclass or Pydantic model (prioritize class-level annotations)
        """
        state_vars = []

        # 1. Class-level annotations (important for dataclass/Pydantic)
        for node in cls.body:
            if isinstance(node, ast.AnnAssign):
                var_name = self._get_name(node.target)
                type_spec = self._convert_type_annotation(node.annotation)

                # For dataclass/Pydantic, check for default values
                if is_special and isinstance(node.value, ast.Constant):
                    # Include default value as comment
                    default = node.value.value
                    state_vars.append(f"{var_name} ∈ {type_spec}  # default: {default}")
                else:
                    state_vars.append(f"{var_name} ∈ {type_spec}")

        # 2. Instance attributes in __init__
        init_method = None
        for node in cls.body:
            if isinstance(node, ast.FunctionDef) and node.name == "__init__":
                init_method = node
                break

        if init_method:
            for node in ast.walk(init_method):
                # Look for self.attr = ...
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Attribute):
                            if isinstance(target.value, ast.Name) and target.value.id == "self":
                                attr_name = target.attr

                                # Try to infer type from assignment
                                type_spec = self._infer_type(node.value)

                                # Check if we already have this from annotations
                                if not any(attr_name in sv for sv in state_vars):
                                    state_vars.append(f"{attr_name} ∈ {type_spec}")

                # Also handle annotated assignments in __init__
                elif isinstance(node, ast.AnnAssign):
                    if isinstance(node.target, ast.Attribute):
                        if (
                            isinstance(node.target.value, ast.Name)
                            and node.target.value.id == "self"
                        ):
                            attr_name = node.target.attr
                            type_spec = self._convert_type_annotation(node.annotation)

                            if not any(attr_name in sv for sv in state_vars):
                                state_vars.append(f"{attr_name} ∈ {type_spec}")

        return state_vars

    def _generate_function_signature(self, func: ast.FunctionDef, indent: str = "") -> str:
        """Generate function signature in PyShorthand format with legacy tags.

        Args:
            func: Function definition node
            indent: Indentation prefix

        Returns:
            Function signature string with tags
        """
        # Extract parameters
        params = []
        for arg in func.args.args:
            if arg.arg == "self":
                continue

            param_str = arg.arg
            if arg.annotation:
                type_str = self._convert_type_annotation(arg.annotation)
                param_str = f"{arg.arg}: {type_str}"

            params.append(param_str)

        params_str = ", ".join(params)

        # Extract return type
        return_type = "Unknown"
        if func.returns:
            return_type = self._convert_type_annotation(func.returns)

        # Build base signature
        sig = f"{indent}F:{func.name}({params_str}) → {return_type}"

        # Extract legacy tags
        tags = self._extract_function_tags(func)
        if tags:
            sig += " " + " ".join(tags)

        return sig

    def _extract_function_tags(self, func: ast.FunctionDef) -> list[str]:
        """Extract all legacy tags for a function.

        Tags are ordered: Decorator → Route → Operation → Complexity

        Args:
            func: Function definition node

        Returns:
            List of tag strings like ["[Prop]", "[GET /api/data]", "[O(N)]"]
        """
        tags = []

        # 1. Decorator tags (Prop, Static, Class, Cached, Auth, etc.)
        decorator_tags = self._extract_decorator_tags(func)
        tags.extend(decorator_tags)

        # 2. HTTP route tags
        route_tag = self._extract_http_route_tag(func)
        if route_tag:
            tags.append(route_tag)

        # 3. Operation tags (analyze function body)
        operation_tags = self._extract_operation_tags(func)
        tags.extend(operation_tags)

        # 4. Complexity tag (from docstring or analysis)
        complexity_tag = self._extract_complexity_tag(func)
        if complexity_tag:
            tags.append(complexity_tag)

        return tags

    def _extract_decorator_tags(self, func: ast.FunctionDef) -> list[str]:
        """Extract decorator tags from function decorators.

        Recognizes:
        - @property → [Prop]
        - @staticmethod → [Static]
        - @classmethod → [Class]
        - @cached_property → [Cached]
        - @lru_cache → [Cached]
        - @login_required, @require_auth, etc. → [Auth]
        - Custom decorators → [DecoratorName]

        Args:
            func: Function definition node

        Returns:
            List of decorator tag strings
        """
        tags = []

        for decorator in func.decorator_list:
            dec_name = None

            # Simple decorator: @property
            if isinstance(decorator, ast.Name):
                dec_name = decorator.id

            # Decorator call: @lru_cache(maxsize=128)
            elif isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Name):
                    dec_name = decorator.func.id
                elif isinstance(decorator.func, ast.Attribute):
                    dec_name = decorator.func.attr

            # Attribute decorator: @functools.lru_cache
            elif isinstance(decorator, ast.Attribute):
                dec_name = decorator.attr

            if dec_name:
                # Map Python decorators to PyShorthand decorator tags
                if dec_name == "property":
                    tags.append("[Prop]")
                elif dec_name == "staticmethod":
                    tags.append("[Static]")
                elif dec_name == "classmethod":
                    tags.append("[Class]")
                elif dec_name in ("cached_property", "lru_cache", "cache"):
                    # Check for TTL or maxsize arguments
                    if isinstance(decorator, ast.Call):
                        # Look for maxsize or ttl keyword arg
                        for kw in decorator.keywords:
                            if kw.arg in ("maxsize", "ttl"):
                                if isinstance(kw.value, ast.Constant):
                                    tags.append(f"[Cached:TTL:{kw.value.value}]")
                                    break
                        else:
                            tags.append("[Cached]")
                    else:
                        tags.append("[Cached]")
                elif dec_name in (
                    "login_required",
                    "require_auth",
                    "authenticated",
                    "auth_required",
                ):
                    tags.append("[Auth]")
                # Skip HTTP route decorators (handled separately)
                elif dec_name not in ("get", "post", "put", "delete", "patch", "route"):
                    # Custom decorator - add as-is if not too generic
                    if dec_name not in ("wraps", "contextmanager"):
                        tags.append(f"[{dec_name}]")

        return tags

    def _extract_http_route_tag(self, func: ast.FunctionDef) -> str | None:
        """Extract HTTP route tag from web framework decorators.

        Recognizes:
        - @app.get("/path") → [GET /path]
        - @router.post("/users/{id}") → [POST /users/{id}]
        - @route("/path", methods=["GET"]) → [GET /path]

        Args:
            func: Function definition node

        Returns:
            HTTP route tag string or None
        """
        for decorator in func.decorator_list:
            # FastAPI/Flask style: @app.get("/path")
            if isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Attribute):
                    http_method = decorator.func.attr.upper()

                    # Check if it's an HTTP method
                    if http_method in ("GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"):
                        # Extract path from first argument
                        if decorator.args and isinstance(decorator.args[0], ast.Constant):
                            path = decorator.args[0].value
                            return f"[{http_method} {path}]"
                        return f"[{http_method}]"

                    # @route decorator with methods argument
                    elif http_method == "ROUTE":
                        path = None
                        method = "GET"  # Default

                        # Extract path from first argument
                        if decorator.args and isinstance(decorator.args[0], ast.Constant):
                            path = decorator.args[0].value

                        # Look for methods keyword argument
                        for kw in decorator.keywords:
                            if kw.arg == "methods":
                                if isinstance(kw.value, ast.List):
                                    # Get first method
                                    if kw.value.elts and isinstance(kw.value.elts[0], ast.Constant):
                                        method = kw.value.elts[0].value

                        if path:
                            return f"[{method} {path}]"

        return None

    def _extract_operation_tags(self, func: ast.FunctionDef) -> list[str]:
        """Extract operation tags by analyzing function body.

        Detects:
        - Neural network operations: [NN:∇] (if torch operations)
        - Linear algebra: [Lin:MatMul] (matrix operations)
        - I/O operations: [IO:Disk], [IO:Net]
        - Iteration: [Iter:Hot] (loops)
        - Synchronization: [Sync:Lock]

        Args:
            func: Function definition node

        Returns:
            List of operation tag strings
        """
        tags = []
        has_loops = False
        has_io = False
        has_torch = False
        has_matmul = False
        has_async = False

        # Analyze function body
        for node in ast.walk(func):
            # Detect loops
            if isinstance(node, (ast.For, ast.While)):
                has_loops = True

            # Detect I/O operations
            elif isinstance(node, ast.Call):
                func_name = self._get_name(node.func)

                # File I/O
                if any(io_op in func_name.lower() for io_op in ["open", "read", "write", "file"]):
                    has_io = True

                # Network I/O (only if calling actual network functions, not just containing words)
                if any(
                    func_name.lower().startswith(net_op) or f".{net_op}" in func_name.lower()
                    for net_op in ["request", "fetch", "socket", "urlopen", "get(", "post("]
                ):
                    if "Net" not in str(tags):
                        tags.append("[IO:Net]")

                # Torch operations
                if "torch" in func_name.lower() or func_name in ("matmul", "mm", "bmm"):
                    has_torch = True

                # Matrix multiplication
                if func_name in ("matmul", "mm", "bmm", "dot"):
                    has_matmul = True

            # Detect async operations
            elif isinstance(node, (ast.Await, ast.AsyncFor, ast.AsyncWith)):
                has_async = True

        # Generate operation tags based on analysis
        if has_torch:
            # Check for gradient operations
            has_gradient = any(
                isinstance(node, ast.Attribute) and node.attr in ("backward", "grad")
                for node in ast.walk(func)
            )
            if has_gradient and has_matmul:
                tags.append("[NN:∇:Lin:MatMul]")
            elif has_gradient:
                tags.append("[NN:∇]")
            elif has_matmul:
                tags.append("[NN:Lin:MatMul]")

        elif has_matmul:
            tags.append("[Lin:MatMul]")

        if has_io and not any("IO:Net" in t for t in tags):
            tags.append("[IO:Disk]")

        if has_loops:
            # Detect nested loops
            loop_depth = self._calculate_loop_depth(func)
            if loop_depth > 1:
                tags.append("[Iter:Nested]")
            else:
                tags.append("[Iter]")

        if has_async:
            tags.append("[IO:Async]")

        return tags

    def _extract_complexity_tag(self, func: ast.FunctionDef) -> str | None:
        """Extract complexity tag from docstring or analyze function body.

        Looks for:
        - Explicit O(...) notation in docstring
        - Pattern-based complexity detection (nested loops, etc.)

        Args:
            func: Function definition node

        Returns:
            Complexity tag string or None
        """
        # First, try to extract from docstring
        docstring = ast.get_docstring(func)
        if docstring:
            # Look for O(...) pattern
            complexity_match = re.search(r"O\([^)]+\)", docstring)
            if complexity_match:
                complexity = complexity_match.group(0)
                return f"[{complexity}]"

            # Look for complexity annotations like "Complexity: O(N)"
            complexity_match = re.search(r"Complexity:\s*(O\([^)]+\))", docstring, re.IGNORECASE)
            if complexity_match:
                complexity = complexity_match.group(1)
                return f"[{complexity}]"

            # Look for "Time: O(N)" or "Runtime: O(N)"
            time_match = re.search(r"(?:Time|Runtime):\s*(O\([^)]+\))", docstring, re.IGNORECASE)
            if time_match:
                complexity = time_match.group(1)
                return f"[{complexity}]"

        # Pattern-based complexity detection
        loop_depth = self._calculate_loop_depth(func)
        if loop_depth == 1:
            return "[O(N)]"
        elif loop_depth == 2:
            return "[O(N²)]"
        elif loop_depth == 3:
            return "[O(N³)]"

        # Default for simple functions
        if loop_depth == 0:
            # Check if it's just a simple return
            if len(func.body) == 1:
                return "[O(1)]"

        return None

    def _calculate_loop_depth(self, func: ast.FunctionDef) -> int:
        """Calculate maximum loop nesting depth in function.

        Args:
            func: Function definition node

        Returns:
            Maximum loop depth (0 = no loops)
        """
        max_depth = 0
        current_depth = 0

        class LoopDepthVisitor(ast.NodeVisitor):
            def __init__(self):
                self.max_depth = 0
                self.current_depth = 0

            def visit_For(self, node):
                self.current_depth += 1
                self.max_depth = max(self.max_depth, self.current_depth)
                self.generic_visit(node)
                self.current_depth -= 1

            def visit_While(self, node):
                self.current_depth += 1
                self.max_depth = max(self.max_depth, self.current_depth)
                self.generic_visit(node)
                self.current_depth -= 1

        visitor = LoopDepthVisitor()
        visitor.visit(func)
        return visitor.max_depth

    def _convert_type_annotation(self, annotation: ast.expr) -> str:
        """Convert Python type annotation to PyShorthand type spec.

        Args:
            annotation: Python AST annotation node

        Returns:
            PyShorthand type specification
        """
        # Handle simple names: int, float, str, etc.
        if isinstance(annotation, ast.Name):
            python_type = annotation.id

            # Check if it's a local class
            if python_type in self.local_classes:
                return f"[Ref:{python_type}]"

            return self._map_python_type(python_type)

        # Handle subscripted types: List[int], Dict[str, int], etc.
        if isinstance(annotation, ast.Subscript):
            base = self._get_name(annotation.value)

            # Handle Optional[T] - extract the inner type
            # Supports: Optional[T], typing.Optional[T]
            if base == "Optional" or base.endswith(".Optional"):
                if isinstance(annotation.slice, ast.Name):
                    inner_type = annotation.slice.id
                    # Check if it's a local class
                    if inner_type in self.local_classes:
                        return f"[Ref:{inner_type}]?"
                    return f"{self._map_python_type(inner_type)}?"
                return "Unknown?"

            # Handle Union[X, None] - equivalent to Optional[X]
            if base == "Union" or base.endswith(".Union"):
                # Check if it's Union[X, None] pattern
                if isinstance(annotation.slice, ast.Tuple):
                    types_in_union = annotation.slice.elts
                    # Look for None type
                    has_none = any(
                        isinstance(t, ast.Constant) and t.value is None for t in types_in_union
                    )
                    if has_none and len(types_in_union) == 2:
                        # This is Union[X, None], equivalent to Optional[X]
                        non_none_type = next(
                            t
                            for t in types_in_union
                            if not (isinstance(t, ast.Constant) and t.value is None)
                        )
                        if isinstance(non_none_type, ast.Name):
                            inner_type = non_none_type.id
                            if inner_type in self.local_classes:
                                return f"[Ref:{inner_type}]?"
                            return f"{self._map_python_type(inner_type)}?"
                # General Union (not Optional pattern) - just use first type for now
                # TODO: Add proper Union type support
                if isinstance(annotation.slice, ast.Tuple) and annotation.slice.elts:
                    first_type = annotation.slice.elts[0]
                    if isinstance(first_type, ast.Name):
                        return self._map_python_type(first_type.id)
                return "Unknown"

            # Handle List, Tuple, etc.
            if base in ("List", "list"):
                return "list"

            # Handle Tensor, torch.Tensor, etc.
            if "Tensor" in base or "tensor" in base.lower():
                return "f32[N]@GPU"  # Default to GPU tensor with unknown shape N

        # Handle attribute access: torch.Tensor, np.ndarray
        if isinstance(annotation, ast.Attribute):
            full_name = self._get_attribute_name(annotation)
            if "Tensor" in full_name:
                return "f32[N]@GPU"
            if "ndarray" in full_name:
                return "f32[N]@CPU"

        # Fallback
        return "Unknown"  # Use 'Unknown' as valid identifier instead of '?'

    def _map_python_type(self, python_type: str) -> str:
        """Map Python type to PyShorthand type.

        Args:
            python_type: Python type name

        Returns:
            PyShorthand type name
        """
        type_map = {
            "int": "i32",
            "float": "f32",
            "str": "str",
            "bool": "bool",
            "list": "list",
            "dict": "dict",
            "tuple": "tuple",
        }
        return type_map.get(python_type, python_type)

    def _infer_type(self, node: ast.expr) -> str:
        """Infer type from assignment value.

        Args:
            node: Python AST expression node

        Returns:
            Inferred PyShorthand type
        """
        # Number literals
        if isinstance(node, ast.Constant):
            # Check bool BEFORE int since bool is subclass of int in Python
            if isinstance(node.value, bool):
                return "bool"
            elif isinstance(node.value, int):
                return "i32"
            elif isinstance(node.value, float):
                return "f32"
            elif isinstance(node.value, str):
                return "str"

        # List literal
        if isinstance(node, ast.List):
            return "list"

        # Dict literal
        if isinstance(node, ast.Dict):
            return "dict"

        # Function calls
        if isinstance(node, ast.Call):
            func_name = self._get_name(node.func)

            # Check for local class instantiation
            if func_name in self.local_classes:
                return f"[Ref:{func_name}]"

            # torch.zeros, torch.ones, etc.
            if "zeros" in func_name or "ones" in func_name or "randn" in func_name:
                return "f32[N]@GPU"

            # numpy arrays
            if "array" in func_name:
                return "f32[N]@CPU"

            # PyTorch nn.Module components
            if "Linear" in func_name:
                return "Linear"  # nn.Linear
            if "Conv" in func_name:
                return "Conv"  # nn.Conv2d, etc.
            if "LayerNorm" in func_name or "BatchNorm" in func_name:
                return "Norm"  # Normalization layers
            if "ModuleList" in func_name:
                return "ModuleList"
            if "Embedding" in func_name:
                return "Embedding"
            if "Dropout" in func_name:
                return "Dropout"
            if "Attention" in func_name:
                return "Attention"

        return "Unknown"  # Use 'Unknown' as valid identifier instead of '?'

    def _get_name(self, node: ast.expr) -> str:
        """Extract name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return self._get_attribute_name(node)
        return "Unknown"  # Use 'Unknown' as valid identifier instead of '?'

    def _get_attribute_name(self, node: ast.Attribute) -> str:
        """Get full attribute name like 'torch.Tensor'."""
        parts = [node.attr]
        current = node.value
        while isinstance(current, ast.Attribute):
            parts.insert(0, current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.insert(0, current.id)
        return ".".join(parts)


def decompile(source: str, aggressive: bool = False) -> str:
    """Decompile Python source code to PyShorthand.

    Args:
        source: Python source code string
        aggressive: If True, use aggressive type inference

    Returns:
        PyShorthand code
    """
    tree = ast.parse(source)
    generator = PyShorthandGenerator(aggressive=aggressive)
    return generator.generate(tree)


def decompile_file(
    input_path: str, output_path: str | None = None, aggressive: bool = False
) -> str:
    """Decompile Python file to PyShorthand.

    Args:
        input_path: Path to Python source file
        output_path: Path to output .pys file (optional)
        aggressive: If True, use aggressive type inference

    Returns:
        PyShorthand code

    Raises:
        IOError: If input file cannot be read
        SyntaxError: If Python source has syntax errors
        RuntimeError: If output file cannot be written
    """
    try:
        with open(input_path, encoding="utf-8") as f:
            source = f.read()
    except OSError as e:
        raise OSError(f"Cannot read input file '{input_path}': {e}")
    except UnicodeDecodeError as e:
        raise OSError(f"Cannot decode input file '{input_path}' as UTF-8: {e}")

    try:
        tree = ast.parse(source, filename=input_path)
    except SyntaxError as e:
        raise SyntaxError(f"Syntax error in '{input_path}' at line {e.lineno}: {e.msg}")

    generator = PyShorthandGenerator(aggressive=aggressive)
    result = generator.generate(tree, source_file=input_path)

    if output_path:
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result)
        except OSError as e:
            raise RuntimeError(f"Cannot write output file '{output_path}': {e}")

    return result


# Backwards compatibility alias
PyShortDecompiler = PyShorthandGenerator
