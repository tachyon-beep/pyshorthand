# PyShorthand Decompiler - Priority Improvements Plan

**Date**: November 22, 2025
**Estimated Total Effort**: 7-10 hours
**Expected Impact**: 60% → 85% completeness

---

## Priority 1: Enhanced Type Inference (2-3 hours)

### Current Problem

**30-50% of attributes are "Unknown"** because the decompiler only infers types from:
1. Literal values (`self.x = 10` → `i32`)
2. Explicit type hints (`x: torch.Tensor` → `f32[N]@GPU`)
3. Known function calls (`nn.Linear()` → `Linear`)

**Missing cases:**
```python
def __init__(self, config: GPTConfig):
    self.config = config           # → config ∈ Unknown (should be [Ref:GPTConfig])
    self.n_head = config.n_head    # → n_head ∈ Unknown (should be i32)
    self.flash = hasattr(...)      # → flash ∈ Unknown (should be bool)
    self.model = MyModel()         # → model ∈ [Ref:MyModel] (already works!)
```

---

### Solution Design

#### 1.1 Parameter Type Tracking

**What**: Track parameter types and propagate to assignments

**Implementation**:
```python
class PyShorthandGenerator:
    def __init__(self):
        # ... existing ...
        self.param_types: Dict[str, str] = {}  # param_name -> type_spec

    def _extract_state_variables(self, cls: ast.ClassDef, is_special: bool = False):
        # Find __init__ method
        init_method = self._find_init_method(cls)
        if not init_method:
            return []

        # Step 1: Build parameter type map
        self._build_param_type_map(init_method)

        # Step 2: Analyze assignments with parameter context
        for node in ast.walk(init_method):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute):
                        if isinstance(target.value, ast.Name) and target.value.id == 'self':
                            attr_name = target.attr

                            # Try parameter tracking first
                            type_spec = self._infer_from_parameter(node.value)

                            # Fallback to existing inference
                            if not type_spec:
                                type_spec = self._infer_type(node.value)

                            state_vars.append(f"{attr_name} ∈ {type_spec}")

    def _build_param_type_map(self, func: ast.FunctionDef):
        """Build map of parameter names to their type specs."""
        self.param_types.clear()

        for arg in func.args.args:
            if arg.arg == 'self':
                continue

            if arg.annotation:
                type_spec = self._convert_type_annotation(arg.annotation)
                self.param_types[arg.arg] = type_spec

    def _infer_from_parameter(self, node: ast.expr) -> Optional[str]:
        """Infer type from parameter assignment.

        Example: self.config = config → look up 'config' in param_types
        """
        if isinstance(node, ast.Name):
            return self.param_types.get(node.id)

        return None
```

**Example transformation**:
```python
# Python input
class Model:
    def __init__(self, config: GPTConfig, dim: int):
        self.config = config      # Currently: Unknown
        self.dim = dim            # Currently: Unknown
        self.layers = []          # Currently: list

# Current output
[C:Model]
  config ∈ Unknown
  dim ∈ Unknown
  layers ∈ list

# After improvement
[C:Model]
  config ∈ [Ref:GPTConfig]  # ✅ Inferred from parameter
  dim ∈ i32                  # ✅ Inferred from parameter
  layers ∈ list
```

**Test cases to add**:
```python
def test_parameter_type_propagation():
    source = """
class Model:
    def __init__(self, config: GPTConfig, dim: int):
        self.config = config
        self.dim = dim
"""
    result = decompile(source)
    assert "config ∈ [Ref:GPTConfig]" in result
    assert "dim ∈ i32" in result
```

---

#### 1.2 Attribute Access Inference

**What**: Infer types from attribute access chains

**Implementation**:
```python
def _infer_from_attribute_access(self, node: ast.Attribute) -> Optional[str]:
    """Infer type from attribute access.

    Examples:
    - config.n_head (where config: GPTConfig) → i32
    - model.encoder (where model: Transformer) → [Ref:Encoder]
    """
    if isinstance(node.value, ast.Name):
        base_name = node.value.id
        attr_name = node.attr

        # Look up base type
        base_type = self.param_types.get(base_name)
        if not base_type:
            return None

        # If base is a local class reference, could look up its attributes
        # For now, make educated guesses based on naming
        if attr_name.startswith('n_'):
            return 'i32'  # Common pattern: n_head, n_layer, n_embd
        elif attr_name.endswith('_rate') or attr_name == 'dropout':
            return 'f32'
        elif attr_name.endswith('_enabled') or attr_name.startswith('is_'):
            return 'bool'

    return None

def _infer_type(self, node: ast.expr) -> str:
    """Enhanced type inference with attribute access."""
    # ... existing literal checks ...

    # NEW: Attribute access inference
    if isinstance(node, ast.Attribute):
        inferred = self._infer_from_attribute_access(node)
        if inferred:
            return inferred

    # ... rest of existing inference ...
```

**Example transformation**:
```python
# Python input
class Model:
    def __init__(self, config: GPTConfig):
        self.n_head = config.n_head      # Currently: Unknown
        self.dropout = config.dropout    # Currently: Unknown

# Current output
[C:Model]
  n_head ∈ Unknown
  dropout ∈ Unknown

# After improvement
[C:Model]
  n_head ∈ i32     # ✅ Pattern: n_* → i32
  dropout ∈ f32    # ✅ Pattern: dropout → f32
```

---

#### 1.3 Built-in Function Inference

**What**: Infer types from built-in function returns

**Implementation**:
```python
def _infer_from_builtin_call(self, node: ast.Call) -> Optional[str]:
    """Infer types from built-in function calls.

    Examples:
    - hasattr(...) → bool
    - isinstance(...) → bool
    - len(...) → i32
    - range(...) → list
    - open(...) → file handle
    """
    func_name = self._get_name(node.func)

    builtin_returns = {
        'hasattr': 'bool',
        'isinstance': 'bool',
        'issubclass': 'bool',
        'callable': 'bool',
        'len': 'i32',
        'sum': 'f32',  # Could be i32, but f32 is safer
        'max': 'f32',
        'min': 'f32',
        'range': 'list',
        'enumerate': 'list',
        'zip': 'list',
        'map': 'list',
        'filter': 'list',
        'sorted': 'list',
        'reversed': 'list',
        'dict': 'dict',
        'list': 'list',
        'tuple': 'tuple',
        'set': 'set',
    }

    return builtin_returns.get(func_name)

def _infer_type(self, node: ast.expr) -> str:
    """Enhanced type inference with builtins."""
    # ... existing checks ...

    # Function calls
    if isinstance(node, ast.Call):
        # NEW: Check builtins first
        builtin_type = self._infer_from_builtin_call(node)
        if builtin_type:
            return builtin_type

        # ... existing inference ...
```

**Example transformation**:
```python
# Python input
class Model:
    def __init__(self):
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        self.num_params = len(self.parameters())

# Current output
[C:Model]
  flash ∈ Unknown
  num_params ∈ Unknown

# After improvement
[C:Model]
  flash ∈ bool       # ✅ hasattr() → bool
  num_params ∈ i32   # ✅ len() → i32
```

---

#### 1.4 Binary Operation Inference

**What**: Infer types from operations

**Implementation**:
```python
def _infer_from_binop(self, node: ast.BinOp) -> Optional[str]:
    """Infer type from binary operation.

    Examples:
    - x + y (where x: i32) → i32
    - x / y → f32 (division always returns float)
    - x * y (where x: str) → str
    """
    # Infer operand types
    left_type = self._infer_type(node.left)
    right_type = self._infer_type(node.right)

    # Division always produces float
    if isinstance(node.op, ast.Div):
        return 'f32'

    # Floor division can be int or float
    if isinstance(node.op, ast.FloorDiv):
        if left_type in ('i32', 'i64'):
            return left_type
        return 'f32'

    # For other ops, use left operand type
    if left_type != 'Unknown':
        return left_type

    return None
```

**Example transformation**:
```python
# Python input
class Model:
    def __init__(self, dim: int):
        self.half_dim = dim // 2
        self.scale = 1.0 / dim

# Current output
[C:Model]
  half_dim ∈ Unknown
  scale ∈ Unknown

# After improvement
[C:Model]
  half_dim ∈ i32    # ✅ i32 // i32 → i32
  scale ∈ f32       # ✅ / → f32
```

---

### Implementation Checklist

- [ ] Add `param_types` tracking to `__init__`
- [ ] Implement `_build_param_type_map()`
- [ ] Implement `_infer_from_parameter()`
- [ ] Implement `_infer_from_attribute_access()`
- [ ] Implement `_infer_from_builtin_call()`
- [ ] Implement `_infer_from_binop()`
- [ ] Update `_infer_type()` to call new methods
- [ ] Add 10+ test cases for new inference patterns
- [ ] Run validation suite to ensure no regressions

**Estimated Effort**: 2-3 hours
**Expected Impact**: Reduce "Unknown" from 40% to 15%

---

## Priority 2: Complexity Tag Generation (3-4 hours)

### Current Problem

**No complexity annotations** in generated PyShorthand, so LLMs can't answer:
- "Which methods are slow?"
- "What are the performance bottlenecks?"
- "What's the computational complexity?"

---

### Solution Design

#### 2.1 Docstring Complexity Extraction

**What**: Parse `:O(N)` annotations from docstrings

**Implementation**:
```python
def _extract_complexity_from_docstring(self, func: ast.FunctionDef) -> Optional[str]:
    """Extract complexity annotation from docstring.

    Recognizes patterns:
    - :O(N) - Big-O notation
    - :O(N*M) - Multi-variable
    - :O(N^2) - Exponential
    - Complexity: O(N) - Natural language
    """
    docstring = ast.get_docstring(func)
    if not docstring:
        return None

    # Pattern 1: :O(...)
    match = re.search(r':O\(([^)]+)\)', docstring)
    if match:
        return f"O({match.group(1)})"

    # Pattern 2: Complexity: O(...)
    match = re.search(r'Complexity:\s*O\(([^)]+)\)', docstring, re.IGNORECASE)
    if match:
        return f"O({match.group(1)})"

    return None

def _generate_function_signature(self, func: ast.FunctionDef, indent: str = "") -> str:
    """Enhanced with complexity tags."""
    # ... existing signature generation ...

    # Extract complexity
    complexity = self._extract_complexity_from_docstring(func)

    # Build tags
    tags = []
    if complexity:
        tags.append(complexity)

    # Add to signature
    if tags:
        signature += f" [{':'.join(tags)}]"

    return signature
```

**Example transformation**:
```python
# Python input
def process_batch(data):
    """Process a batch of data.

    Complexity: O(N*M) where N=batch_size, M=features
    """
    for item in data:
        for feature in item:
            ...

# Current output
# F:process_batch(data) → Unknown

# After improvement
# F:process_batch(data) → Unknown [O(N*M)]
```

---

#### 2.2 Pattern-Based Complexity Detection

**What**: Detect complexity from code patterns

**Implementation**:
```python
class ComplexityAnalyzer:
    """Analyze function complexity from AST patterns."""

    def analyze(self, func: ast.FunctionDef) -> Optional[str]:
        """Analyze function and return complexity estimate.

        Detection patterns:
        - Single loop → O(N)
        - Nested loops → O(N^2)
        - Recursive calls → O(N) or O(log N)
        - Matrix operations → O(N*M*D)
        - Sorting → O(N log N)
        """
        # Count loop nesting
        max_depth = self._max_loop_depth(func)

        if max_depth == 0:
            # Check for specific operations
            if self._has_matmul(func):
                return "O(N*M)"
            if self._has_sort(func):
                return "O(N log N)"
            return "O(1)"

        elif max_depth == 1:
            return "O(N)"

        elif max_depth == 2:
            return "O(N²)"

        else:
            return f"O(N^{max_depth})"

    def _max_loop_depth(self, node: ast.AST, current_depth: int = 0) -> int:
        """Calculate maximum loop nesting depth."""
        max_depth = current_depth

        for child in ast.walk(node):
            if isinstance(child, (ast.For, ast.While)):
                child_depth = self._max_loop_depth(child, current_depth + 1)
                max_depth = max(max_depth, child_depth)

        return max_depth

    def _has_matmul(self, node: ast.AST) -> bool:
        """Check for matrix multiplication operations."""
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                func_name = self._get_call_name(child)
                if func_name in ('matmul', 'dot', '@'):
                    return True
        return False

    def _has_sort(self, node: ast.AST) -> bool:
        """Check for sorting operations."""
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                func_name = self._get_call_name(child)
                if func_name in ('sort', 'sorted'):
                    return True
        return False

    def _get_call_name(self, call: ast.Call) -> str:
        """Extract function name from call."""
        if isinstance(call.func, ast.Name):
            return call.func.id
        elif isinstance(call.func, ast.Attribute):
            return call.func.attr
        return ""


# In PyShorthandGenerator
def __init__(self):
    # ... existing ...
    self.complexity_analyzer = ComplexityAnalyzer()

def _generate_function_signature(self, func: ast.FunctionDef, indent: str = "") -> str:
    """Enhanced with pattern-based complexity."""
    # ... existing ...

    # Try docstring first
    complexity = self._extract_complexity_from_docstring(func)

    # Fallback to pattern analysis
    if not complexity and self.aggressive:
        complexity = self.complexity_analyzer.analyze(func)

    # ... rest ...
```

**Example transformation**:
```python
# Python input
def process_matrix(A, B):
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                C[i][j] += A[i][k] * B[k][j]
    return C

# Current output
# F:process_matrix(A, B) → Unknown

# After improvement (with --aggressive)
# F:process_matrix(A, B) → Unknown [O(N³)]
```

---

#### 2.3 Operation-Specific Tags

**What**: Tag specific operation types

**Implementation**:
```python
class OperationTagger:
    """Detect specific operation types."""

    def tag_operations(self, func: ast.FunctionDef) -> List[str]:
        """Return list of operation tags.

        Tags:
        - [Lin] - Linear algebra
        - [Lin:MatMul] - Matrix multiplication
        - [Iter] - Iteration
        - [Iter:Hot] - Inner loop (hot path)
        - [IO:Net] - Network I/O
        - [IO:Disk] - Disk I/O
        - [NN:∇] - Neural network with gradients
        - [Thresh] - Thresholding (softmax, relu, etc.)
        """
        tags = []

        # Check for matrix operations
        if self._has_operation(func, ['matmul', 'dot', '@', 'mm']):
            tags.append('Lin:MatMul')
        elif self._has_operation(func, ['transpose', 'T', 'reshape']):
            tags.append('Lin')

        # Check for NN operations
        if self._has_operation(func, ['softmax', 'relu', 'gelu', 'sigmoid']):
            tags.append('Thresh')

        if self._has_operation(func, ['backward', 'grad', 'autograd']):
            tags.append('NN:∇')

        # Check for I/O
        if self._has_operation(func, ['open', 'read', 'write', 'save', 'load']):
            tags.append('IO:Disk')

        if self._has_operation(func, ['requests', 'http', 'socket', 'get', 'post']):
            tags.append('IO:Net')

        # Check for iteration
        loop_depth = self._max_loop_depth(func)
        if loop_depth > 0:
            if loop_depth >= 2:
                tags.append('Iter:Hot')
            else:
                tags.append('Iter')

        return tags

    def _has_operation(self, node: ast.AST, operation_names: List[str]) -> bool:
        """Check if function contains any of the specified operations."""
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                func_name = self._get_call_name(child)
                if func_name in operation_names:
                    return True
            elif isinstance(child, ast.Attribute):
                if child.attr in operation_names:
                    return True
        return False


# In _generate_function_signature
def _generate_function_signature(self, func: ast.FunctionDef, indent: str = "") -> str:
    """Enhanced with operation tags."""
    # ... existing ...

    # Get operation tags
    op_tags = self.operation_tagger.tag_operations(func)

    # Combine with complexity
    all_tags = []
    all_tags.extend(op_tags)
    if complexity:
        all_tags.append(complexity)

    # Format
    if all_tags:
        signature += f" [{':'.join(all_tags)}]"

    return signature
```

**Example transformation**:
```python
# Python input
def attention(q, k, v):
    scores = torch.matmul(q, k.transpose(-1, -2))
    scores = torch.softmax(scores, dim=-1)
    return torch.matmul(scores, v)

# Current output
# F:attention(q, k, v) → f32[N]@GPU

# After improvement
# F:attention(q, k, v) → f32[N]@GPU [Lin:MatMul:Thresh:O(N*M)]
```

---

### Implementation Checklist

- [ ] Implement `_extract_complexity_from_docstring()`
- [ ] Implement `ComplexityAnalyzer` class
- [ ] Implement `OperationTagger` class
- [ ] Add `--aggressive` flag support for pattern detection
- [ ] Update `_generate_function_signature()` to include tags
- [ ] Add 15+ test cases for different complexity patterns
- [ ] Test on real PyTorch/NumPy code

**Estimated Effort**: 3-4 hours
**Expected Impact**: 30-40% of methods tagged with complexity/operation info

---

## Priority 3: Decorator Extraction (2-3 hours)

### Current Problem

**Route and decorator information is lost**, so LLMs can't answer:
- "What are the API endpoints?"
- "Which methods are properties vs regular methods?"
- "What decorators are applied?"

---

### Solution Design

#### 3.1 Web Framework Route Extraction

**What**: Extract HTTP routes from FastAPI/Flask decorators

**Enhancement to existing `_extract_route_info()`**:
```python
def _extract_route_info(self, method: ast.FunctionDef) -> Optional[Dict[str, any]]:
    """Extract comprehensive route information.

    Returns dict with:
    - method: HTTP method (GET, POST, etc.)
    - path: Route path
    - params: Path parameters
    - decorators: Other decorators
    """
    route_info = {
        'method': None,
        'path': None,
        'params': [],
        'decorators': []
    }

    for decorator in method.decorator_list:
        # FastAPI/Flask style: @app.get("/path")
        if isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Attribute):
                http_method = decorator.func.attr.upper()
                if http_method in ('GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS', 'HEAD'):
                    route_info['method'] = http_method

                    # Extract path
                    if decorator.args and isinstance(decorator.args[0], ast.Constant):
                        path = decorator.args[0].value
                        route_info['path'] = path

                        # Extract path parameters
                        import re
                        params = re.findall(r'\{(\w+)\}', path)
                        route_info['params'] = params

                    # Extract query parameters, dependencies, etc. from kwargs
                    for keyword in decorator.keywords:
                        if keyword.arg in ('response_model', 'status_code', 'tags'):
                            route_info['decorators'].append(f"{keyword.arg}=...")

        # Django REST: @api_view(['GET', 'POST'])
        elif isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Name):
                if decorator.func.id == 'api_view':
                    if decorator.args and isinstance(decorator.args[0], ast.List):
                        methods = []
                        for elt in decorator.args[0].elts:
                            if isinstance(elt, ast.Constant):
                                methods.append(elt.value)
                        route_info['method'] = '/'.join(methods)

        # Flask style: @app.route("/path", methods=['GET', 'POST'])
        elif isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Attribute):
                if decorator.func.attr == 'route':
                    if decorator.args and isinstance(decorator.args[0], ast.Constant):
                        route_info['path'] = decorator.args[0].value

                    for keyword in decorator.keywords:
                        if keyword.arg == 'methods':
                            if isinstance(keyword.value, ast.List):
                                methods = []
                                for elt in keyword.value.elts:
                                    if isinstance(elt, ast.Constant):
                                        methods.append(elt.value)
                                route_info['method'] = '/'.join(methods)

    # Format for output
    if route_info['method'] and route_info['path']:
        parts = [route_info['method'], route_info['path']]
        if route_info['decorators']:
            parts.append('(' + ', '.join(route_info['decorators']) + ')')
        return ' '.join(parts)

    return None
```

**Example transformation**:
```python
# Python input
from fastapi import FastAPI, Depends

app = FastAPI()

class UserAPI:
    @app.get("/users/{user_id}", response_model=User, tags=["users"])
    def get_user(self, user_id: int):
        ...

    @app.post("/users", status_code=201)
    def create_user(self, user: UserCreate):
        ...

# Current output
[C:UserAPI]
  # Methods:
  # F:get_user(user_id: i32) → Unknown
  # F:create_user(user: [Ref:UserCreate]) → Unknown

# After improvement
[C:UserAPI]
  # Methods:
  # F:get_user(user_id: i32) → Unknown [GET /users/{user_id}]
  # F:create_user(user: [Ref:UserCreate]) → Unknown [POST /users]
```

---

#### 3.2 Property and Method Decorators

**What**: Distinguish properties, static methods, class methods

**Implementation**:
```python
def _extract_method_modifiers(self, method: ast.FunctionDef) -> List[str]:
    """Extract method modifiers from decorators.

    Recognizes:
    - @property → [Prop]
    - @staticmethod → [Static]
    - @classmethod → [Class]
    - @cached_property → [Cached]
    - @abstractmethod → [Abstract]
    """
    modifiers = []

    for decorator in method.decorator_list:
        if isinstance(decorator, ast.Name):
            name = decorator.id

            if name == 'property':
                modifiers.append('Prop')
            elif name == 'staticmethod':
                modifiers.append('Static')
            elif name == 'classmethod':
                modifiers.append('Class')
            elif name == 'cached_property':
                modifiers.append('Cached')
            elif name == 'abstractmethod':
                modifiers.append('Abstract')

        elif isinstance(decorator, ast.Attribute):
            name = decorator.attr

            if name == 'setter':
                modifiers.append('Setter')
            elif name == 'deleter':
                modifiers.append('Deleter')

    return modifiers

def _generate_entity(self, cls: ast.ClassDef, tree: ast.Module) -> List[str]:
    """Enhanced entity generation with decorator support."""
    # ... existing code ...

    # Extract methods with modifiers
    if methods:
        lines.append("")
        lines.append("  # Methods:")
        for method in methods:
            if method.name.startswith('_') and method.name != '__init__':
                continue

            # Get modifiers
            modifiers = self._extract_method_modifiers(method)

            # Get route info
            route_info = self._extract_route_info(method)

            # Generate signature
            sig = self._generate_function_signature(method, indent="  # ")

            # Add modifiers
            if modifiers:
                sig += f" [{':'.join(modifiers)}]"

            # Add route info
            if route_info:
                sig += f" [{route_info}]"

            lines.append(sig)

    return lines
```

**Example transformation**:
```python
# Python input
from functools import cached_property

class Model:
    @property
    def device(self):
        return self._device

    @cached_property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def from_pretrained(path):
        ...

# Current output
[C:Model]
  # Methods:
  # F:device() → Unknown
  # F:num_parameters() → Unknown
  # F:from_pretrained(path) → Unknown

# After improvement
[C:Model]
  # Methods:
  # F:device() → Unknown [Prop]
  # F:num_parameters() → Unknown [Cached]
  # F:from_pretrained(path) → Unknown [Static]
```

---

#### 3.3 Custom Decorator Tracking

**What**: Track application-specific decorators

**Implementation**:
```python
def _extract_custom_decorators(self, method: ast.FunctionDef) -> List[str]:
    """Extract custom/domain-specific decorators.

    Examples:
    - @require_auth → [Auth]
    - @rate_limit(100) → [RateLimit:100]
    - @cache(ttl=60) → [Cache:60s]
    """
    custom = []

    for decorator in method.decorator_list:
        # Simple decorator: @decorator_name
        if isinstance(decorator, ast.Name):
            name = decorator.id
            if name not in ('property', 'staticmethod', 'classmethod', 'abstractmethod'):
                custom.append(name)

        # Decorator with args: @decorator(args)
        elif isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Name):
                name = decorator.func.id

                # Extract simple arguments
                args = []
                for arg in decorator.args:
                    if isinstance(arg, ast.Constant):
                        args.append(str(arg.value))

                if args:
                    custom.append(f"{name}:{','.join(args)}")
                else:
                    custom.append(name)

    return custom
```

**Example transformation**:
```python
# Python input
class API:
    @require_auth
    @rate_limit(requests=100, window=60)
    def get_data(self, user_id: int):
        ...

# Current output
[C:API]
  # Methods:
  # F:get_data(user_id: i32) → Unknown

# After improvement
[C:API]
  # Methods:
  # F:get_data(user_id: i32) → Unknown [require_auth] [rate_limit:100,60]
```

---

### Implementation Checklist

- [ ] Enhance `_extract_route_info()` with full route details
- [ ] Implement `_extract_method_modifiers()`
- [ ] Implement `_extract_custom_decorators()`
- [ ] Update `_generate_entity()` to include all decorator info
- [ ] Add configuration for which decorators to include
- [ ] Add 10+ test cases for different decorator patterns
- [ ] Test on FastAPI, Flask, Django REST codebases

**Estimated Effort**: 2-3 hours
**Expected Impact**: Complete API contract information, better method classification

---

## Testing Strategy

### Unit Tests

Each improvement needs comprehensive unit tests:

```python
# tests/unit/test_enhanced_inference.py
class TestEnhancedTypeInference:
    def test_parameter_propagation(self): ...
    def test_attribute_access_inference(self): ...
    def test_builtin_function_inference(self): ...
    def test_binary_operation_inference(self): ...

# tests/unit/test_complexity_tags.py
class TestComplexityGeneration:
    def test_docstring_extraction(self): ...
    def test_single_loop_detection(self): ...
    def test_nested_loop_detection(self): ...
    def test_matmul_detection(self): ...
    def test_operation_tagging(self): ...

# tests/unit/test_decorator_extraction.py
class TestDecoratorExtraction:
    def test_fastapi_routes(self): ...
    def test_flask_routes(self): ...
    def test_property_decorator(self): ...
    def test_staticmethod_decorator(self): ...
    def test_custom_decorators(self): ...
```

### Integration Tests

Validate on real-world code:

```python
# tests/integration/test_improvements.py
def test_pytorch_model_with_improvements():
    """Test all improvements on a real PyTorch model."""
    source = Path("test_repos/nanoGPT/model.py").read_text()
    result = decompile(source, aggressive=True)

    # Should have fewer Unknowns
    unknown_count = result.count("∈ Unknown")
    assert unknown_count < 10  # Down from ~30

    # Should have complexity tags
    assert "[O(" in result
    assert "[Lin:MatMul" in result

    # Should have decorator info (if any)
    ...
```

---

## Configuration Options

Add flags to control new features:

```python
class PyShorthandGenerator:
    def __init__(
        self,
        aggressive: bool = False,
        with_confidence: bool = False,
        infer_complexity: bool = True,      # NEW
        extract_decorators: bool = True,     # NEW
        complexity_mode: str = 'auto'        # NEW: 'auto', 'docstring', 'pattern', 'both'
    ):
        ...
```

**CLI flags**:
```bash
# Conservative (default)
py2short model.py

# Aggressive type inference
py2short model.py --aggressive

# Disable complexity inference
py2short model.py --no-complexity

# Only use docstring complexity (skip pattern matching)
py2short model.py --complexity-mode=docstring
```

---

## Expected Results

### Before Improvements (Current)
```python
# nanoGPT model.py
[C:CausalSelfAttention]
  c_attn ∈ Linear
  c_proj ∈ Linear
  attn_dropout ∈ Dropout
  resid_dropout ∈ Dropout
  n_head ∈ Unknown          # ❌
  n_embd ∈ Unknown          # ❌
  flash ∈ Unknown           # ❌

  # Methods:
  # F:__init__(config) → Unknown
  # F:forward(x) → Unknown   # ❌ No complexity
```

### After All Improvements
```python
# nanoGPT model.py
[C:CausalSelfAttention]
  c_attn ∈ Linear
  c_proj ∈ Linear
  attn_dropout ∈ Dropout
  resid_dropout ∈ Dropout
  n_head ∈ i32              # ✅ Inferred from config.n_head
  n_embd ∈ i32              # ✅ Inferred from config.n_embd
  flash ∈ bool              # ✅ Inferred from hasattr()

  # Methods:
  # F:__init__(config: [Ref:GPTConfig]) → Unknown
  # F:forward(x: f32[N]@GPU) → f32[N]@GPU [Lin:MatMul:Thresh:O(N*D)]  # ✅ Tagged
```

---

## Success Metrics

| Metric | Current | After Improvements | Target |
|--------|---------|-------------------|--------|
| **Unknown types** | 40% | 15% | <15% ✅ |
| **Methods with complexity** | 0% | 35% | >30% ✅ |
| **Route information** | 0% | 90% | >80% ✅ |
| **Overall completeness** | 60% | 85% | >80% ✅ |

---

## Implementation Order

**Week 1** (2-3 hours):
1. Enhanced type inference - Highest impact
2. Add tests for type inference

**Week 2** (3-4 hours):
3. Complexity tag generation
4. Add tests for complexity

**Week 3** (2-3 hours):
5. Decorator extraction
6. Add tests for decorators
7. Integration testing on all 14 validation files

**Total**: 7-10 hours spread over 2-3 weeks

---

## Risk Mitigation

**Risks**:
1. Type inference heuristics could be wrong
2. Complexity detection might be inaccurate
3. Breaking changes to existing users

**Mitigations**:
1. Use conservative defaults, require `--aggressive` for risky inference
2. Prefer docstring annotations (ground truth) over patterns
3. Add regression tests to prevent breaking existing output
4. Add `--legacy` mode to disable new features

---

**Ready to implement?** Each improvement is modular and can be tackled independently. Start with type inference for maximum impact!
