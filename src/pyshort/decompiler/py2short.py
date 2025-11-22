"""Python AST to PyShorthand decompiler implementation."""

import ast
import re
from typing import List, Dict, Optional, Set, Tuple
from pathlib import Path
from dataclasses import dataclass


@dataclass
class InferredType:
    """Type inference result with confidence."""
    type_spec: str
    confidence: float  # 0.0 to 1.0
    reason: str


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
        self.imports: Set[str] = set()
        self.local_classes: Set[str] = set()  # Classes defined in this module
        self.dependencies: List[str] = []
        self.import_map: Dict[str, str] = {}  # alias -> full module path

    def generate(self, tree: ast.Module, source_file: Optional[str] = None) -> str:
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
        module_name = module_metadata.get('name', 'UnnamedModule')
        role = module_metadata.get('role', 'Core')
        lines.append(f"# [M:{module_name}] [Role:{role}]")

        # Add risk if found
        if 'risk' in module_metadata:
            lines[-1] = lines[-1].rstrip(']') + f"] [Risk:{module_metadata['risk']}]"

        lines.append("")

        # Extract classes
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]

        for cls in classes:
            entity_lines = self._generate_entity(cls, tree)
            lines.extend(entity_lines)
            lines.append("")

        # Extract module-level functions
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]

        if functions:
            lines.append("# Module-level functions")
            for func in functions:
                func_line = self._generate_function_signature(func)
                lines.append(f"{func_line}")
            lines.append("")

        return "\n".join(lines)

    def _extract_module_name(self, tree: ast.Module, source_file: Optional[str]) -> str:
        """Extract module name from AST or file path."""
        # Try to find module name from docstring or file path
        if tree.body and isinstance(tree.body[0], ast.Expr):
            if isinstance(tree.body[0].value, ast.Constant):
                docstring = tree.body[0].value.value
                if isinstance(docstring, str):
                    # Extract first line of docstring as module name
                    first_line = docstring.strip().split('\n')[0]
                    # Clean up common patterns
                    if '.' in first_line:
                        return first_line.split('.')[0]
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

    def _extract_module_metadata(self, tree: ast.Module, source_file: Optional[str]) -> Dict[str, str]:
        """Extract metadata from module docstring and file."""
        metadata = {}

        # Get module name
        metadata['name'] = self._extract_module_name(tree, source_file)

        # Try to extract metadata from docstring
        if tree.body and isinstance(tree.body[0], ast.Expr):
            if isinstance(tree.body[0].value, ast.Constant):
                docstring = tree.body[0].value.value
                if isinstance(docstring, str):
                    # Look for role/risk/layer tags in docstring
                    tags = self._extract_docstring_tags(docstring)
                    metadata.update(tags)

        return metadata

    def _extract_docstring_tags(self, docstring: str) -> Dict[str, str]:
        """Extract PyShorthand tags from docstring.

        Recognizes patterns like:
        - Role: Core|Service|Util
        - Risk: High|Med|Low
        - Layer: API|Logic|Data
        - :O(N) for complexity
        """
        tags = {}

        # Role pattern
        role_match = re.search(r'Role:\s*(Core|Service|Util|API|Logic|Data)', docstring, re.IGNORECASE)
        if role_match:
            tags['role'] = role_match.group(1).capitalize()

        # Risk pattern
        risk_match = re.search(r'Risk:\s*(High|Med|Low)', docstring, re.IGNORECASE)
        if risk_match:
            tags['risk'] = risk_match.group(1).capitalize()

        # Layer pattern
        layer_match = re.search(r'Layer:\s*(API|Logic|Data)', docstring, re.IGNORECASE)
        if layer_match:
            tags['layer'] = layer_match.group(1)

        return tags

    def _generate_entity(self, cls: ast.ClassDef, tree: ast.Module) -> List[str]:
        """Generate PyShorthand entity from Python class.

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
        is_fastapi_route = self._is_fastapi_route_class(cls)
        web_framework = self._detect_web_framework(cls)

        # Entity header with pattern annotations
        header = f"[C:{cls.name}]"
        if is_dataclass:
            header += " # @dataclass"
        elif is_pydantic:
            header += " # Pydantic"
        elif web_framework:
            header += f" # {web_framework}"
        lines.append(header)

        # Extract dependencies from base classes
        dependencies = self._extract_class_dependencies(cls)

        if dependencies:
            lines.append(f"  ◊ {', '.join(dependencies)}")

        # Extract class attributes with type hints
        state_vars = self._extract_state_variables(cls, is_dataclass or is_pydantic)

        if state_vars:
            for var in state_vars:
                lines.append(f"  {var}")
        else:
            lines.append("  # No typed attributes found")

        # Extract methods as comments (parser doesn't support F:name syntax in entities yet)
        methods = [node for node in cls.body if isinstance(node, ast.FunctionDef)]

        if methods:
            lines.append("")
            lines.append("  # Methods:")
            for method in methods:
                if method.name.startswith('_') and method.name != '__init__':
                    continue  # Skip private methods except __init__

                # Check for route decorators (FastAPI/Flask)
                route_info = self._extract_route_info(method)
                sig = self._generate_function_signature(method, indent="  # ")

                if route_info:
                    sig += f" [{route_info}]"

                lines.append(sig)

        return lines

    def _extract_class_dependencies(self, cls: ast.ClassDef) -> List[str]:
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
                dep_name = '.'.join(parts)

            if dep_name:
                # If it's a local class, use [Ref:Name]
                if dep_name in self.local_classes:
                    dependencies.append(f"[Ref:{dep_name}]")
                # Otherwise, if it's imported, use module reference
                elif dep_name in self.import_map:
                    # For common frameworks, simplify
                    full_name = self.import_map[dep_name]
                    if 'fastapi' in full_name.lower():
                        dependencies.append("[Ref:FastAPI]")
                    elif 'pydantic' in full_name.lower():
                        dependencies.append("[Ref:Pydantic]")
                    elif 'torch.nn' in full_name.lower() or 'nn.Module' in dep_name:
                        dependencies.append("[Ref:PyTorch]")
                    elif 'flask' in full_name.lower():
                        dependencies.append("[Ref:Flask]")
                    else:
                        # Generic external dependency
                        dependencies.append(f"[Ref:{dep_name}]")

        return dependencies

    def _is_dataclass(self, cls: ast.ClassDef) -> bool:
        """Check if class is a dataclass."""
        for decorator in cls.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id == 'dataclass':
                return True
            elif isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Name) and decorator.func.id == 'dataclass':
                    return True
        return False

    def _is_pydantic_model(self, cls: ast.ClassDef) -> bool:
        """Check if class is a Pydantic model."""
        for base in cls.bases:
            if isinstance(base, ast.Name):
                if 'BaseModel' in base.id or 'Pydantic' in base.id:
                    return True
            elif isinstance(base, ast.Attribute):
                if base.attr == 'BaseModel':
                    return True
        return False

    def _is_fastapi_route_class(self, cls: ast.ClassDef) -> bool:
        """Check if class contains FastAPI routes."""
        for node in cls.body:
            if isinstance(node, ast.FunctionDef):
                if self._extract_route_info(node):
                    return True
        return False

    def _detect_web_framework(self, cls: ast.ClassDef) -> Optional[str]:
        """Detect which web framework this class uses."""
        # Check base classes
        for base in cls.bases:
            base_name = ""
            if isinstance(base, ast.Name):
                base_name = base.id
            elif isinstance(base, ast.Attribute):
                base_name = base.attr

            if 'APIRouter' in base_name or 'FastAPI' in base_name:
                return "FastAPI"
            elif 'Flask' in base_name or 'Blueprint' in base_name:
                return "Flask"
            elif 'APIView' in base_name or 'ViewSet' in base_name:
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

                    if dec_name in ('get', 'post', 'put', 'delete', 'patch'):
                        return "FastAPI/Flask"

        return None

    def _extract_route_info(self, method: ast.FunctionDef) -> Optional[str]:
        """Extract route information from method decorators."""
        for decorator in method.decorator_list:
            # Check for @app.get("/path"), @router.post("/path"), etc.
            if isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Attribute):
                    http_method = decorator.func.attr
                    if http_method in ('get', 'post', 'put', 'delete', 'patch', 'route'):
                        # Try to extract path
                        if decorator.args and isinstance(decorator.args[0], ast.Constant):
                            path = decorator.args[0].value
                            return f"{http_method.upper()} {path}"
                        return http_method.upper()

            # Check for @route decorator
            elif isinstance(decorator, ast.Name):
                if decorator.id == 'route':
                    return "ROUTE"

        return None

    def _extract_state_variables(self, cls: ast.ClassDef, is_special: bool = False) -> List[str]:
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
            if isinstance(node, ast.FunctionDef) and node.name == '__init__':
                init_method = node
                break

        if init_method:
            for node in ast.walk(init_method):
                # Look for self.attr = ...
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Attribute):
                            if isinstance(target.value, ast.Name) and target.value.id == 'self':
                                attr_name = target.attr

                                # Try to infer type from assignment
                                type_spec = self._infer_type(node.value)

                                # Check if we already have this from annotations
                                if not any(attr_name in sv for sv in state_vars):
                                    state_vars.append(f"{attr_name} ∈ {type_spec}")

                # Also handle annotated assignments in __init__
                elif isinstance(node, ast.AnnAssign):
                    if isinstance(node.target, ast.Attribute):
                        if isinstance(node.target.value, ast.Name) and node.target.value.id == 'self':
                            attr_name = node.target.attr
                            type_spec = self._convert_type_annotation(node.annotation)

                            if not any(attr_name in sv for sv in state_vars):
                                state_vars.append(f"{attr_name} ∈ {type_spec}")

        return state_vars

    def _generate_function_signature(self, func: ast.FunctionDef, indent: str = "") -> str:
        """Generate function signature in PyShorthand format.

        Args:
            func: Function definition node
            indent: Indentation prefix

        Returns:
            Function signature string
        """
        # Extract parameters
        params = []
        for arg in func.args.args:
            if arg.arg == 'self':
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

        return f"{indent}F:{func.name}({params_str}) → {return_type}"

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
            if base == 'Optional' or base.endswith('.Optional'):
                if isinstance(annotation.slice, ast.Name):
                    inner_type = annotation.slice.id
                    # Check if it's a local class
                    if inner_type in self.local_classes:
                        return f"[Ref:{inner_type}]?"
                    return f"{self._map_python_type(inner_type)}?"
                return "Unknown?"

            # Handle Union[X, None] - equivalent to Optional[X]
            if base == 'Union' or base.endswith('.Union'):
                # Check if it's Union[X, None] pattern
                if isinstance(annotation.slice, ast.Tuple):
                    types_in_union = annotation.slice.elts
                    # Look for None type
                    has_none = any(
                        isinstance(t, ast.Constant) and t.value is None
                        for t in types_in_union
                    )
                    if has_none and len(types_in_union) == 2:
                        # This is Union[X, None], equivalent to Optional[X]
                        non_none_type = next(
                            t for t in types_in_union
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
            if base in ('List', 'list'):
                # Try to get element type
                if isinstance(annotation.slice, ast.Name):
                    elem_type = self._map_python_type(annotation.slice.id)
                    return f"list"  # Just use list type without shape
                return "list"

            # Handle Tensor, torch.Tensor, etc.
            if 'Tensor' in base or 'tensor' in base.lower():
                return "f32[N]@GPU"  # Default to GPU tensor with unknown shape N

        # Handle attribute access: torch.Tensor, np.ndarray
        if isinstance(annotation, ast.Attribute):
            full_name = self._get_attribute_name(annotation)
            if 'Tensor' in full_name:
                return "f32[N]@GPU"
            if 'ndarray' in full_name:
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
            'int': 'i32',
            'float': 'f32',
            'str': 'str',
            'bool': 'bool',
            'list': 'list',
            'dict': 'dict',
            'tuple': 'tuple',
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
            if 'zeros' in func_name or 'ones' in func_name or 'randn' in func_name:
                return "f32[N]@GPU"

            # numpy arrays
            if 'array' in func_name:
                return "f32[N]@CPU"

            # PyTorch nn.Module components
            if 'Linear' in func_name:
                return "Linear"  # nn.Linear
            if 'Conv' in func_name:
                return "Conv"  # nn.Conv2d, etc.
            if 'LayerNorm' in func_name or 'BatchNorm' in func_name:
                return "Norm"  # Normalization layers
            if 'ModuleList' in func_name:
                return "ModuleList"
            if 'Embedding' in func_name:
                return "Embedding"
            if 'Dropout' in func_name:
                return "Dropout"
            if 'Attention' in func_name:
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


def decompile_file(input_path: str, output_path: Optional[str] = None, aggressive: bool = False) -> str:
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
        with open(input_path, 'r', encoding='utf-8') as f:
            source = f.read()
    except IOError as e:
        raise IOError(f"Cannot read input file '{input_path}': {e}")
    except UnicodeDecodeError as e:
        raise IOError(f"Cannot decode input file '{input_path}' as UTF-8: {e}")

    try:
        tree = ast.parse(source, filename=input_path)
    except SyntaxError as e:
        raise SyntaxError(f"Syntax error in '{input_path}' at line {e.lineno}: {e.msg}")

    generator = PyShorthandGenerator(aggressive=aggressive)
    result = generator.generate(tree, source_file=input_path)

    if output_path:
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result)
        except IOError as e:
            raise RuntimeError(f"Cannot write output file '{output_path}': {e}")

    return result
