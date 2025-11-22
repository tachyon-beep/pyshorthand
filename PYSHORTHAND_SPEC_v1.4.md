# PyShorthand Protocol Specification v1.4

**Status**: Draft
**Date**: November 22, 2025
**Changes from v1.3.1**:
- Added complexity tags (Section 3.5.3)
- Added decorator tags (Section 3.5.4)
- Added HTTP route tags (Section 3.5.5)
- Enhanced function signature syntax (Section 3.4.2)

---

## 1. Overview

PyShorthand is a high-density intermediate representation for Python code optimized for LLM comprehension. This specification extends v1.3.1 with enhanced semantic annotations for complexity analysis, decorator information, and web framework support.

---

## 2. Core Syntax (Unchanged from v1.3.1)

### 2.1 Metadata Headers
```
# [M:ModuleName] [Role:Core|Glue|Script] [Layer:Domain|Infra|Adapter|Test]
# [Risk:High|Med|Low] [Context:description] [Dims:N=batch,D=dim]
```

### 2.2 Entity Declarations
```
[C:ClassName]  # Class
[D:DataName]   # Data structure
[I:Interface]  # Interface/Protocol
[F:FuncName]   # Function
```

### 2.3 State Variables
```
name ∈ Type[Shape]@Location
```

---

## 3. Enhanced Features (v1.4)

### 3.1 Function Signatures with Tags

**Syntax**:
```
F:function_name(params) → ReturnType [Tags]
```

**Tags** can include:
- Complexity tags: `[O(N)]`, `[O(N*M)]`
- Operation tags: `[Lin:MatMul]`, `[Iter:Hot]`, `[NN:∇]`
- Decorator tags: `[Prop]`, `[Static]`, `[Cached]`
- Route tags: `[GET /path]`, `[POST /api/users]`

**Examples**:
```python
# Simple function with complexity
F:process_batch(data: list) → list [O(N)]

# Matrix operation with complexity and operation tags
F:matmul(A: f32[N,M], B: f32[M,K]) → f32[N,K] [Lin:MatMul:O(N*M*K)]

# Web API endpoint
F:get_user(user_id: i32) → User [GET /users/{user_id}]

# Property decorator
F:device() → str [Prop]

# Static method
F:from_pretrained(path: str) → Model [Static]

# Cached property with complexity
F:num_parameters() → i32 [Cached:O(N)]

# Neural network forward pass
F:forward(x: f32[B,N,D]@GPU) → f32[B,N,D]@GPU [NN:∇:Lin:MatMul:Thresh:O(B*N*D)]
```

---

### 3.2 Complexity Tags

**Purpose**: Indicate computational complexity and performance characteristics

**Syntax**: `[O(complexity)]` or as part of operation tags

**Standard Forms**:
- `O(1)` - Constant time
- `O(log N)` - Logarithmic
- `O(N)` - Linear
- `O(N log N)` - Linearithmic (sorting)
- `O(N²)` or `O(N^2)` - Quadratic
- `O(N³)` or `O(N^3)` - Cubic
- `O(N*M)` - Multi-variable
- `O(N*M*D)` - Three variables
- `O(2^N)` - Exponential

**Variable Conventions**:
- `N` - Primary dimension (batch size, sequence length)
- `M` - Secondary dimension (features, vocab size)
- `D` - Embedding dimension
- `B` - Batch size (when distinct from N)
- `T` - Time steps
- `H` - Number of heads
- `V` - Vocabulary size

**Examples**:
```python
F:linear_search(arr: list, target: i32) → i32? [O(N)]
F:bubble_sort(arr: list) → list [O(N²)]
F:attention(Q: f32[B,N,D], K: f32[B,M,D]) → f32[B,N,M] [O(B*N*M*D)]
```

---

### 3.3 Operation Tags

**Purpose**: Classify computational operations for analysis

**Base Types** (existing):
- `Lin` - Linear/algebraic operations
- `Thresh` - Thresholding/branching
- `Iter` - Iteration
- `Map` - Mapping/lookup
- `Stoch` - Stochastic/random
- `IO` - Input/Output
- `Sync` - Concurrency synchronization
- `NN` - Neural network operations
- `Heur` - Heuristic/business logic

**Qualifiers**:

**Linear Operations** (`Lin:`):
- `MatMul` - Matrix multiplication
- `Broad` - Broadcasting
- `Reduce` - Reduction operations (sum, mean)
- `Transpose` - Transpose/reshape

**Iteration** (`Iter:`):
- `Hot` - Inner loop (performance critical)
- `Scan` - Sequential scan
- `Sequential` - Must be sequential
- `Random` - Random access pattern
- `Strided` - Strided access

**Thresholding** (`Thresh:`):
- `Mask` - Masking operation
- `Cond` - Conditional branching
- `Clamp` - Value clamping
- `Softmax` - Softmax activation
- `ReLU` - ReLU activation

**Neural Network** (`NN:`):
- `∇` or `Grad` - Gradient computation
- `Inf` - Inference only (no gradients)
- `Train` - Training mode

**I/O Operations** (`IO:`):
- `Net` - Network I/O
- `Disk` - Disk I/O
- `Async` - Asynchronous I/O
- `Block` - Blocking I/O
- `Stream` - Streaming I/O

**Examples**:
```python
F:matmul(A, B) → Tensor [Lin:MatMul:O(N*M*D)]
F:softmax(x) → Tensor [Thresh:Softmax:O(N)]
F:train_step(batch) → f32 [NN:∇:O(B*N*D)]
F:load_model(path) → Model [IO:Disk:Block]
```

**Combining Tags**:
Multiple tags separated by colons, most specific first:
```python
F:transformer_forward(x) → Tensor [NN:∇:Lin:MatMul:Thresh:Softmax:O(B*N²*D)]
```

---

### 3.4 Decorator Tags

**Purpose**: Preserve Python decorator semantics

#### 3.4.1 Standard Python Decorators

**Property Decorators**:
- `[Prop]` - `@property`
- `[Setter]` - `@name.setter`
- `[Deleter]` - `@name.deleter`
- `[Cached]` - `@cached_property` or `@functools.lru_cache`

**Method Decorators**:
- `[Static]` - `@staticmethod`
- `[Class]` - `@classmethod`
- `[Abstract]` - `@abstractmethod`

**Examples**:
```python
[C:Model]
  _device ∈ str

  # Methods:
  # F:__init__() → Unknown
  # F:device() → str [Prop]
  # F:to(device: str) → Model
  # F:from_pretrained(path: str) → Model [Static:Class]
  # F:forward(x: Tensor) → Tensor [Abstract]
  # F:num_parameters() → i32 [Cached:O(N)]
```

#### 3.4.2 Web Framework Decorators

**FastAPI/Flask Routes**:
```
[HTTP_METHOD /path]
[HTTP_METHOD /path/with/{params}]
```

**HTTP Methods**:
- `GET` - Read operation
- `POST` - Create operation
- `PUT` - Update/replace operation
- `PATCH` - Partial update
- `DELETE` - Delete operation
- `OPTIONS` - Options query
- `HEAD` - Headers only

**Route Parameters**:
- Path parameters: `{param_name}`
- Query parameters: Not shown in tag (in function params)

**Examples**:
```python
[C:UserAPI]
  # F:get_user(user_id: i32) → User [GET /users/{user_id}]
  # F:list_users(skip: i32, limit: i32) → list [GET /users]
  # F:create_user(user: UserCreate) → User [POST /users]
  # F:update_user(user_id: i32, user: UserUpdate) → User [PUT /users/{user_id}]
  # F:delete_user(user_id: i32) → Unknown [DELETE /users/{user_id}]
```

**Django REST Framework**:
```python
[C:UserViewSet]
  # F:list(request) → Response [GET /api/users]
  # F:retrieve(request, pk: i32) → Response [GET /api/users/{pk}]
  # F:create(request) → Response [POST /api/users]
```

#### 3.4.3 Custom Decorators

**Syntax**: `[decorator_name]` or `[decorator_name:args]`

**Common Patterns**:
- `[Auth]` or `[require_auth]` - Authentication required
- `[RateLimit:100]` - Rate limiting (100 requests)
- `[Retry:3]` - Retry decorator (3 attempts)
- `[Cache:60s]` - Caching (60 second TTL)
- `[Timeout:30s]` - Timeout decorator

**Examples**:
```python
[C:SecureAPI]
  # F:get_sensitive_data(user: User) → Data [Auth:GET /api/sensitive]
  # F:bulk_operation(items: list) → list [RateLimit:10:Timeout:60s]
```

---

### 3.5 Tag Composition Rules

**Order of Tags** (recommendation):
1. Decorator tags (`[Prop]`, `[Static]`, etc.)
2. HTTP routes (`[GET /path]`)
3. Operation tags (`[NN:∇]`, `[Lin:MatMul]`)
4. Complexity tags (`[O(N)]`)
5. Custom decorators

**Examples**:
```python
# Property with complexity
# F:num_params() → i32 [Prop:Cached:O(N)]

# API endpoint with operation and complexity
# F:forward(x: Tensor) → Tensor [POST /api/inference:NN:Inf:O(B*N*D)]

# Static constructor with I/O
# F:load(path: str) → Model [Static:IO:Disk:O(N)]
```

**Multiple Decorators**:
```python
# F:process_request(data) → Response [Auth:RateLimit:100:GET /api/process]
```

---

### 3.6 Backward Compatibility

**v1.3.1 Compliance**: All v1.3.1 syntax remains valid. New tags are optional.

**Migration Path**:
1. Existing `.pys` files work without changes
2. New tags can be added incrementally
3. Tools should treat missing tags gracefully

**Validator Behavior**:
- Default mode: Accept both v1.3.1 and v1.4 syntax
- Strict mode (`--strict=1.4`): Require v1.4 features
- Legacy mode (`--legacy=1.3.1`): Reject v1.4 features

---

## 4. Examples

### 4.1 PyTorch Transformer Model

```python
# [M:Transformer] [Role:Core] [Layer:Domain] [Risk:Med]
# [Context:Neural network architecture] [Dims:B=batch,N=seq,D=dim,H=heads]

[C:MultiHeadAttention]
  ◊ [Ref:PyTorch]

  num_heads ∈ i32
  d_model ∈ i32
  q_proj ∈ Linear
  k_proj ∈ Linear
  v_proj ∈ Linear
  out_proj ∈ Linear
  dropout ∈ Dropout

  # Methods:
  # F:__init__(d_model: i32, num_heads: i32) → Unknown
  # F:forward(query: f32[B,N,D]@GPU, key: f32[B,M,D]@GPU, value: f32[B,M,D]@GPU) → f32[B,N,D]@GPU [NN:∇:Lin:MatMul:Thresh:Softmax:O(B*N*M*D)]

[C:TransformerBlock]
  ◊ [Ref:PyTorch]

  attention ∈ [Ref:MultiHeadAttention]
  norm1 ∈ Norm
  norm2 ∈ Norm
  mlp ∈ [Ref:MLP]
  dropout ∈ Dropout

  # Methods:
  # F:__init__(d_model: i32) → Unknown
  # F:forward(x: f32[B,N,D]@GPU) → f32[B,N,D]@GPU [NN:∇:O(B*N²*D)]

[C:Transformer]
  ◊ [Ref:PyTorch]

  config ∈ [Ref:TransformerConfig]
  embed ∈ Embedding
  blocks ∈ list  # ModuleList of TransformerBlock
  norm ∈ Norm

  # Methods:
  # F:__init__(config: [Ref:TransformerConfig]) → Unknown
  # F:forward(x: i64[B,N]@GPU) → f32[B,N,V]@GPU [NN:∇:O(B*N²*D)]
  # F:from_pretrained(path: str) → [Ref:Transformer] [Static:IO:Disk:O(N)]
  # F:num_parameters() → i32 [Prop:Cached:O(N)]
```

### 4.2 FastAPI Web Service

```python
# [M:UserAPI] [Role:Service] [Layer:API] [Risk:Med]

[C:User]  # Pydantic
  ◊ [Ref:Pydantic]
  id ∈ i32
  email ∈ str
  name ∈ str
  created_at ∈ str  # datetime

[C:UserCreate]  # Pydantic
  ◊ [Ref:Pydantic]
  email ∈ str
  name ∈ str
  password ∈ str

[C:UserAPI]
  ◊ [Ref:FastAPI]

  db ∈ Unknown  # Database connection

  # Methods:
  # F:__init__(db) → Unknown
  # F:get_user(user_id: i32) → [Ref:User] [GET /users/{user_id}:O(1)]
  # F:list_users(skip: i32, limit: i32) → list [GET /users:O(N)]
  # F:create_user(user: [Ref:UserCreate]) → [Ref:User] [POST /users:IO:Disk:O(1)]
  # F:update_user(user_id: i32, user: [Ref:UserCreate]) → [Ref:User] [PUT /users/{user_id}:IO:Disk:O(1)]
  # F:delete_user(user_id: i32) → Unknown [Auth:DELETE /users/{user_id}:IO:Disk:O(1)]
```

### 4.3 Data Processing Pipeline

```python
# [M:DataPipeline] [Role:Core] [Layer:Domain] [Risk:High]
# [Dims:N=samples,M=features,D=dims]

[C:DataProcessor]
  batch_size ∈ i32
  num_workers ∈ i32
  cache ∈ dict

  # Methods:
  # F:__init__(batch_size: i32) → Unknown
  # F:load_data(path: str) → f32[N,M]@CPU [IO:Disk:Stream:O(N)]
  # F:preprocess(data: f32[N,M]@CPU) → f32[N,D]@GPU [Lin:MatMul:O(N*M*D)]
  # F:create_batches(data: f32[N,D]) → list [Iter:O(N)]
  # F:cached_transform(data) → f32[N,D] [Cached:Map:Hash:O(1)]
  # F:parallel_process(items: list) → list [Iter:||:O(N)]
```

---

## 5. Implementation Notes

### 5.1 Parser Changes

**New token types**:
- HTTP method keywords: `GET`, `POST`, `PUT`, `DELETE`, `PATCH`, `OPTIONS`, `HEAD`
- Decorator keywords: `Prop`, `Static`, `Class`, `Cached`, `Abstract`, `Setter`, `Deleter`
- Operation qualifiers: `MatMul`, `Broad`, `Softmax`, etc.

**Tag parsing**:
```python
# Function signature with tags
F:name(params) → type [tag1:tag2:tag3]

# Multiple independent tag groups
F:name(params) → type [Prop] [O(N)]
F:name(params) → type [GET /path] [O(1)]
```

### 5.2 Validator Changes

**New validation rules**:
1. Validate complexity notation (O(N), O(N*M), etc.)
2. Validate HTTP routes (valid paths, parameter syntax)
3. Validate decorator compatibility (can't have both Prop and Static)
4. Validate tag ordering recommendations

**Warning levels**:
- Error: Invalid syntax
- Warning: Non-standard tag ordering
- Info: Missing complexity annotations

### 5.3 Formatter Changes

**Tag formatting**:
- Align tags vertically in method lists
- Sort tags by category (decorators, routes, operations, complexity)
- Option to strip custom decorators for cleaner output

### 5.4 Decompiler Changes

**Generation logic**:
1. Extract complexity from docstrings first (`:O(N)`)
2. Infer complexity from patterns (loop nesting, matmul calls)
3. Extract decorators from AST decorator_list
4. Detect HTTP routes from FastAPI/Flask/Django decorators
5. Combine tags in recommended order

---

## 6. Validation Rules

### 6.1 Complexity Tags

**Valid**:
- `[O(1)]`, `[O(N)]`, `[O(N²)]`, `[O(N*M*D)]`
- `[O(log N)]`, `[O(N log N)]`
- `[O(2^N)]` (exponential)

**Invalid**:
- `[O(n)]` (lowercase - should be uppercase N)
- `[O()]` (empty)
- `[O(N*)]` (incomplete)

### 6.2 HTTP Routes

**Valid**:
- `[GET /users]`
- `[POST /api/users/{user_id}]`
- `[PUT /v1/items/{item_id}/details]`

**Invalid**:
- `[GET]` (missing path)
- `[/users]` (missing method)
- `[GET users]` (missing leading /)

### 6.3 Decorator Tags

**Valid**:
- `[Prop]`, `[Static]`, `[Cached]`
- `[Auth]`, `[RateLimit:100]`

**Invalid**:
- `[Prop:Static]` (conflicting: can't be both property and static)
- `[Prop:Class]` (conflicting: can't be both property and classmethod)

---

## 7. Tooling Support

### 7.1 CLI Flags

**Decompiler** (`py2short`):
```bash
# Generate with complexity tags (docstrings only)
py2short model.py --complexity=docstring

# Generate with inferred complexity (aggressive)
py2short model.py --complexity=inferred --aggressive

# Include all decorators
py2short api.py --decorators=all

# Include only standard decorators (skip custom)
py2short api.py --decorators=standard

# Skip complexity tags entirely
py2short model.py --no-complexity
```

**Parser** (`pyshort-parse`):
```bash
# Parse with v1.4 features
pyshort-parse model.pys --version=1.4

# Strict v1.4 validation
pyshort-parse model.pys --strict=1.4
```

**Linter** (`pyshort-lint`):
```bash
# Require complexity tags on all methods
pyshort-lint model.pys --require-complexity

# Require HTTP routes on all public methods in API classes
pyshort-lint api.pys --require-routes
```

### 7.2 Configuration

**`.pyshortrc`**:
```ini
[decompiler]
complexity_mode = inferred  # docstring|inferred|both|none
decorators = all            # all|standard|none
include_custom_decorators = true

[validator]
spec_version = 1.4
require_complexity = false
require_routes = false
allow_legacy = true

[formatter]
tag_alignment = vertical
tag_order = decorator,route,operation,complexity
```

---

## 8. Migration Guide

### 8.1 From v1.3.1 to v1.4

**Step 1**: No changes required - v1.4 is backward compatible

**Step 2** (optional): Add complexity tags
```bash
# Generate with complexity inference
py2short --complexity=inferred model.py
```

**Step 3** (optional): Add decorator tags
```bash
# Regenerate with decorator extraction
py2short --decorators=all api.py
```

**Step 4** (optional): Validate with v1.4
```bash
pyshort-lint --strict=1.4 model.pys
```

### 8.2 Updating Existing Specs

**Before** (v1.3.1):
```python
[C:Model]
  # F:forward(x: Tensor) → Tensor
  # F:from_pretrained(path: str) → Model
```

**After** (v1.4):
```python
[C:Model]
  # F:forward(x: f32[N,D]@GPU) → f32[N,D]@GPU [NN:∇:O(N*D)]
  # F:from_pretrained(path: str) → Model [Static:IO:Disk]
```

---

## 9. Appendix

### 9.1 Complete Tag Reference

**Complexity**:
- O(1), O(log N), O(N), O(N log N), O(N²), O(N³), O(N*M), O(N*M*D), O(2^N)

**Operations**:
- Lin: Linear, Broad, MatMul, Reduce, Transpose
- Thresh: Mask, Cond, Clamp, Softmax, ReLU
- Iter: Hot, Scan, Sequential, Random, Strided
- NN: ∇, Inf, Train
- IO: Net, Disk, Async, Block, Stream
- Sync: Lock, Atomic, Barrier, Await
- Map: Hash, Cache
- Stoch: Seed, Dist

**Decorators**:
- Standard: Prop, Setter, Deleter, Static, Class, Abstract, Cached
- HTTP: GET, POST, PUT, DELETE, PATCH, OPTIONS, HEAD
- Custom: Any user-defined decorator name

### 9.2 Reserved Keywords

**Added in v1.4**:
- HTTP methods: GET, POST, PUT, DELETE, PATCH, OPTIONS, HEAD
- Decorators: Prop, Static, Class, Cached, Abstract, Setter, Deleter
- Operations: MatMul, Softmax, ReLU, etc.

**Conflict Resolution**:
If a user-defined identifier conflicts with a reserved keyword, prefix with underscore:
```python
# Avoid
F:get() → Unknown  # Conflicts with HTTP GET?

# Better
F:get_data() → Unknown
F:_get() → Unknown  # If you must
```

---

**Specification Version**: 1.4.0
**Published**: November 22, 2025
**Authors**: PyShorthand Contributors
**License**: MIT
