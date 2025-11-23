# PyShorthand Protocol Specification v1.5

**Status**: Draft
**Date**: November 23, 2025
**Changes from v1.4**:
- **Added inheritance notation** (Section 2.2.1) - `◊` symbol
- **Added nested structure expansion** (Section 2.3.1) - `{}` syntax
- **Added generic type parameters** (Section 2.4) - `<T>` syntax
- **Added abstract markers** (Section 3.4.4) - `[Abstract]` tag
- **Enhanced class declarations** with base class support

---

## 1. Overview

PyShorthand is a high-density intermediate representation for Python code optimized for LLM comprehension. Version 1.5 extends v1.4 with inheritance notation, generic types, and nested structure expansion based on empirical validation testing.

### 1.1 Design Goals for v1.5

1. **Preserve Inheritance Information** - Empirical tests showed LLMs need explicit inheritance (Q4 failures in Sonnet 3.5/4.5 tests)
2. **Expand Nested Structures** - Make complex compositions clearer (ModuleDict, nested objects)
3. **Support Generic Types** - Better type safety and clarity
4. **Mark Abstract Patterns** - Distinguish interfaces from implementations

---

## 2. Core Syntax (Enhanced from v1.4)

### 2.1 Metadata Headers (Unchanged)
```
# [M:ModuleName] [Role:Core|Glue|Script] [Layer:Domain|Infra|Adapter|Test]
# [Risk:High|Med|Low] [Context:description] [Dims:N=batch,D=dim]
```

### 2.2 Entity Declarations

#### 2.2.1 Class Declarations with Inheritance ✨ NEW

**Syntax**:
```
[C:ClassName] ◊ BaseClass
[C:ClassName] ◊ BaseClass, MixinA, MixinB
```

The `◊` symbol indicates inheritance/extension. Multiple bases are comma-separated.

**Examples**:
```python
# Single inheritance
[C:LayerNorm] ◊ nn.Module
  weight ∈ Parameter
  bias ∈ Parameter?

# Multiple inheritance (mixins)
[C:MyModel] ◊ nn.Module, Serializable, Configurable

# No explicit inheritance (defaults to object)
[C:Config]
  batch_size ∈ i32
```

**Implicit Bases**:
- If no `◊` is shown, inheritance from `object` is implied
- Framework-specific bases (like `nn.Module`) should be explicit

#### 2.2.2 Other Entity Types (Unchanged)

```
[D:DataName]   # Data structure/dataclass
[I:Interface]  # Interface/Protocol
[F:FuncName]   # Function
[P:Protocol]   # Protocol (typing.Protocol)
[E:EnumName]   # Enum
```

---

### 2.3 State Variables

#### 2.3.1 Nested Structure Expansion ✨ NEW

**Syntax**: Use `{}` to expand composite structures inline

**Simple Form** (v1.4):
```
transformer ∈ ModuleDict
```

**Expanded Form** (v1.5):
```
transformer ∈ ModuleDict {
  wte: Embedding
  wpe: Embedding
  drop: Dropout
  h: ModuleList<Block>
  ln_f: LayerNorm
}
```

**Rules**:
- Use `{}` for dict-like or composite structures
- Each field on its own line, indented
- Field format: `key: Type` or `key: Type<GenericParam>`
- Use for `ModuleDict`, `nn.ModuleDict`, plain `dict`, or custom composites

**Examples**:
```python
# PyTorch Module with nested structure
[C:GPT] ◊ nn.Module
  config ∈ GPTConfig
  transformer ∈ ModuleDict {
    wte: Embedding(vocab_size, n_embd)
    wpe: Embedding(block_size, n_embd)
    drop: Dropout(p=dropout)
    h: ModuleList<Block>  # List of Block modules
    ln_f: LayerNorm
  }
  lm_head ∈ Linear

# Configuration with nested dict
[D:AppConfig]
  database ∈ dict {
    host: str
    port: i32
    credentials: dict {
      username: str
      password: str
    }
  }

# Simple nested structure
[C:Cache]
  storage ∈ dict {
    data: Any
    metadata: dict
  }
```

**When to Expand**:
- ✅ Expand when structure is important (ModuleDict, critical config)
- ✅ Expand for 2-5 key fields
- ❌ Don't expand for >10 fields (too verbose)
- ❌ Don't expand simple dicts with dynamic keys

---

### 2.4 Generic Type Parameters ✨ NEW

**Syntax**: `Type<Param1, Param2, ...>`

**Standard Generics**:
```python
List<T>              # List of T
Dict<K, V>           # Dictionary from K to V
Optional<T>          # T or None (also T?)
Callable<T→U>        # Function from T to U
Tuple<T1, T2, T3>    # Tuple of specific types
Union<T1, T2>        # Union of types
```

**Custom Generics**:
```python
[C:LinkedList<T>]
  head ∈ Node<T>?
  tail ∈ LinkedList<T>?

[C:Node<T>]
  value ∈ T
  next ∈ Node<T>?
```

**PyTorch Generics**:
```python
ModuleList<Block>           # List of Block modules
Sequential<Layer>           # Sequential of Layer modules
DataLoader<Dataset>         # DataLoader of Dataset type
```

**Function Generics**:
```python
# Generic function signature
F:map<T, U>(fn: Callable<T→U>, items: List<T>) → List<U>

# Constrained generics (informal)
F:sum<T: Numeric>(items: List<T>) → T
```

**Examples**:
```python
[C:Transformer<T>] ◊ nn.Module
  layers ∈ ModuleList<TransformerBlock<T>>

  # F:forward(x: T) → T

[C:Cache<K, V>]
  storage ∈ Dict<K, V>

  # F:get(key: K) → Optional<V>
  # F:put(key: K, value: V) → None
```

---

### 2.5 State Variable Syntax (Complete)

**Full Syntax**:
```
name ∈ Type<Generic>[Shape]@Location {expanded}
```

**Components**:
- `name` - Variable name
- `∈` - "is of type" (U+2208)
- `Type` - Base type
- `<Generic>` - Optional generic parameters ✨ NEW
- `[Shape]` - Optional shape annotation
- `@Location` - Optional location (GPU, CPU)
- `{expanded}` - Optional nested structure ✨ NEW

**Examples**:
```python
# Simple type
weight ∈ Tensor

# With shape
weight ∈ Tensor[N, D]

# With location
weight ∈ Tensor[N, D]@GPU

# With generics ✨
layers ∈ ModuleList<Block>

# With expansion ✨
config ∈ dict {
  lr: f32
  epochs: i32
}

# All together
cache ∈ LRUCache<str, Tensor[N]@GPU> {
  max_size: i32
  eviction_policy: str
}
```

---

## 3. Enhanced Features (v1.4 + v1.5)

### 3.1 Function Signatures with Tags

**Syntax** (unchanged from v1.4):
```
F:function_name(params) → ReturnType [Tags]
```

**New: Generic Function Signatures** ✨
```
F:map<T, U>(fn: Callable<T→U>, items: List<T>) → List<U>
F:filter<T>(pred: Callable<T→bool>, items: List<T>) → List<T>
```

**Examples**:
```python
# Generic function
F:identity<T>(x: T) → T

# With constraints (informal notation)
F:sum<T: Numeric>(items: List<T>) → T [O(N)]

# Complex generic with tags
F:batch_process<T, U>(
  items: List<T>,
  fn: Callable<T→U>
) → List<U> [Iter:O(N)]
```

---

### 3.2 Complexity Tags (Unchanged from v1.4)

**Standard Forms**:
- `O(1)` - Constant time
- `O(log N)` - Logarithmic
- `O(N)` - Linear
- `O(N²)` - Quadratic
- `O(N*M)` - Multi-variable
- `O(2^N)` - Exponential

See v1.4 spec for complete details.

---

### 3.3 Operation Tags (Unchanged from v1.4)

**Base Types**:
- `Lin` - Linear/algebraic operations
- `Iter` - Iteration
- `NN` - Neural network operations
- `IO` - Input/Output
- `Sync` - Concurrency

See v1.4 spec for complete details.

---

### 3.4 Decorator Tags

#### 3.4.1 Standard Python Decorators (from v1.4)

- `[Prop]` - `@property`
- `[Static]` - `@staticmethod`
- `[Class]` - `@classmethod`
- `[Cached]` - `@cached_property`

#### 3.4.2 Web Framework Decorators (from v1.4)

- `[GET /path]` - GET endpoint
- `[POST /path]` - POST endpoint
- `[PUT /path]`, `[DELETE /path]`, etc.

#### 3.4.3 Custom Decorators (from v1.4)

- `[Auth]` - Authentication required
- `[RateLimit:100]` - Rate limiting
- `[Retry:3]` - Retry decorator

#### 3.4.4 Abstract/Interface Markers ✨ NEW

**Purpose**: Mark abstract classes and methods

**Class-Level**:
```
[C:Animal] ◊ ABC [Abstract]
[I:Drawable] [Protocol]
```

**Method-Level**:
```
# F:make_sound() → str [Abstract]
# F:draw() → None [Abstract]
```

**Examples**:
```python
# Abstract base class
[C:Model] ◊ nn.Module, ABC [Abstract]
  config ∈ Config

  # Methods:
  # F:__init__(config: Config) → None
  # F:forward(x: Tensor) → Tensor [Abstract]  # Must be implemented
  # F:loss(pred: Tensor, target: Tensor) → Tensor [Abstract]
  # F:num_parameters() → i32 [Cached:O(N)]  # Concrete implementation

# Protocol (structural typing)
[P:Drawable] [Protocol]
  # F:draw(canvas: Canvas) → None [Abstract]
  # F:get_bounds() → Rect [Abstract]

# Concrete implementation
[C:Circle] ◊ Shape  # Implements Drawable protocol
  radius ∈ f32
  center ∈ Point

  # F:draw(canvas: Canvas) → None  # Implements abstract method
  # F:get_bounds() → Rect
  # F:area() → f32 [Prop:O(1)]
```

**Conventions**:
- Use `[Abstract]` for classes that cannot be instantiated
- Use `[Protocol]` for structural type protocols (PEP 544)
- Mark abstract methods with `[Abstract]` tag
- Concrete implementations don't need `[Abstract]` tag

---

### 3.5 Tag Composition Rules (Enhanced)

**Order of Tags** (recommendation):
1. Decorator tags (`[Prop]`, `[Static]`, `[Class]`, `[Abstract]` ✨)
2. HTTP routes (`[GET /path]`)
3. Operation tags (`[NN:∇]`, `[Lin:MatMul]`)
4. Complexity tags (`[O(N)]`)
5. Custom decorators (`[Auth]`, `[RateLimit]`)

**Examples**:
```python
# Abstract property with complexity
# F:num_params() → i32 [Abstract:Prop:Cached:O(N)]

# Static constructor with I/O and generics
# F:load<T>(path: str) → T [Static:IO:Disk:O(N)]

# API endpoint with auth and rate limiting
# F:get_data() → Data [GET /api/data:Auth:RateLimit:100]
```

---

## 4. Complete Examples

### 4.1 nanoGPT with v1.5 Features

```python
# [M:nanoGPT] [Role:Core] [Layer:Domain]

[C:LayerNorm] ◊ nn.Module
  weight ∈ Parameter[ndim]
  bias ∈ Parameter[ndim]?

  # F:__init__(ndim: i32, bias: bool) → None
  # F:forward(input: Tensor) → Tensor [O(N)]

[C:CausalSelfAttention] ◊ nn.Module
  c_attn ∈ Linear
  c_proj ∈ Linear
  attn_dropout ∈ Dropout
  resid_dropout ∈ Dropout
  n_head ∈ i32
  n_embd ∈ i32
  flash ∈ bool

  # F:__init__(config: GPTConfig) → None
  # F:forward(x: Tensor[B,T,C]) → Tensor[B,T,C] [NN:∇:O(B*T²*C)]

[C:MLP] ◊ nn.Module
  c_fc ∈ Linear
  gelu ∈ GELU
  c_proj ∈ Linear
  dropout ∈ Dropout

  # F:__init__(config: GPTConfig) → None
  # F:forward(x: Tensor) → Tensor [NN:∇:Lin:O(N*D)]

[C:Block] ◊ nn.Module
  ln_1 ∈ LayerNorm
  attn ∈ CausalSelfAttention
  ln_2 ∈ LayerNorm
  mlp ∈ MLP

  # F:__init__(config: GPTConfig) → None
  # F:forward(x: Tensor) → Tensor [NN:∇]

[D:GPTConfig] @dataclass
  block_size ∈ i32  # default: 1024
  vocab_size ∈ i32  # default: 50304
  n_layer ∈ i32  # default: 12
  n_head ∈ i32  # default: 12
  n_embd ∈ i32  # default: 768
  dropout ∈ f32  # default: 0.0
  bias ∈ bool  # default: True

[C:GPT] ◊ nn.Module
  config ∈ GPTConfig
  transformer ∈ ModuleDict {
    wte: Embedding(vocab_size, n_embd)
    wpe: Embedding(block_size, n_embd)
    drop: Dropout(p=dropout)
    h: ModuleList<Block>
    ln_f: LayerNorm
  }
  lm_head ∈ Linear(n_embd, vocab_size, bias=False)

  # Methods:
  # F:__init__(config: GPTConfig) → None [Iter:O(N)]
  # F:get_num_params(non_embedding: bool) → i32 [O(N)]
  # F:forward(idx: i64[B,T], targets: i64[B,T]?) → Tuple<Tensor, Tensor?> [NN:∇:Iter:O(B*T*N)]
  # F:crop_block_size(block_size: i32) → None [Iter:O(N)]
  # F:from_pretrained(cls, model_type: str, override_args: dict?) → GPT [Class:IO:Net:Iter:O(N)]
  # F:configure_optimizers(weight_decay: f32, learning_rate: f32, betas: Tuple<f32,f32>, device_type: str) → AdamW [O(N)]
  # F:estimate_mfu(fwdbwd_per_iter: i32, dt: f32) → f32 [O(1)]
  # F:generate(idx: i64[B,T], max_new_tokens: i32, temperature: f32, top_k: i32?) → i64[B,T+max_new_tokens] [@torch.no_grad:Iter:O(max_new_tokens*B*T)]
```

### 4.2 Abstract Base Class Example

```python
# [M:models.base] [Role:Core] [Layer:Domain]

[C:BaseModel] ◊ nn.Module, ABC [Abstract]
  config ∈ Config
  device ∈ str

  # F:__init__(config: Config) → None
  # F:forward(x: Tensor) → Tensor [Abstract:NN:∇]
  # F:loss(pred: Tensor, target: Tensor) → Tensor [Abstract]
  # F:device() → str [Prop]
  # F:to(device: str) → BaseModel
  # F:num_parameters() → i32 [Cached:O(N)]
  # F:from_pretrained(path: str) → BaseModel [Class:Static:IO:Disk:Abstract]

[C:Transformer] ◊ BaseModel
  layers ∈ ModuleList<TransformerBlock>
  embeddings ∈ Embedding

  # F:forward(x: i64[B,T]) → Tensor[B,T,D] [NN:∇:O(B*T²*D)]  # Implements abstract
  # F:loss(pred: Tensor, target: Tensor) → Tensor [O(B*T)]  # Implements abstract
```

### 4.3 Generic Container Example

```python
# [M:utils.containers] [Role:Glue]

[C:LRUCache<K, V>]
  capacity ∈ i32
  cache ∈ OrderedDict<K, V>

  # F:__init__(capacity: i32) → None
  # F:get(key: K) → Optional<V> [O(1)]
  # F:put(key: K, value: V) → None [O(1)]
  # F:clear() → None [O(1)]
  # F:size() → i32 [Prop:O(1)]

[C:DataLoader<T>]
  dataset ∈ Dataset<T>
  batch_size ∈ i32
  shuffle ∈ bool

  # F:__init__(dataset: Dataset<T>, batch_size: i32, shuffle: bool) → None
  # F:__iter__() → Iterator<List<T>>
  # F:__len__() → i32 [O(1)]
```

---

## 5. Migration Guide: v1.4 → v1.5

### 5.1 Add Inheritance Information

**v1.4**:
```python
[C:LayerNorm]
  weight ∈ Parameter
```

**v1.5**:
```python
[C:LayerNorm] ◊ nn.Module
  weight ∈ Parameter
```

### 5.2 Expand Critical Nested Structures

**v1.4**:
```python
[C:GPT]
  transformer ∈ ModuleDict
```

**v1.5**:
```python
[C:GPT] ◊ nn.Module
  transformer ∈ ModuleDict {
    wte: Embedding
    wpe: Embedding
    h: ModuleList<Block>
    ln_f: LayerNorm
  }
```

### 5.3 Add Generic Types

**v1.4**:
```python
layers ∈ ModuleList
```

**v1.5**:
```python
layers ∈ ModuleList<TransformerBlock>
```

### 5.4 Mark Abstract Classes/Methods

**v1.4**:
```python
[C:BaseModel]
  # F:forward(x) → Tensor
```

**v1.5**:
```python
[C:BaseModel] ◊ nn.Module, ABC [Abstract]
  # F:forward(x) → Tensor [Abstract]
```

---

## 6. Validation Rules (v1.5)

### 6.1 Inheritance Rules

1. Base classes should be valid Python classes or known framework types
2. Multiple inheritance: separate with commas, no spaces around `◊`
3. Abstract classes must have `[Abstract]` tag
4. Abstract methods must be marked with `[Abstract]` tag

### 6.2 Generic Rules

1. Generic parameters use `<>` angle brackets
2. Standard Python generics: `List`, `Dict`, `Optional`, `Tuple`, `Union`, `Callable`
3. Custom generics should be declared in class definition
4. Use arrow `→` for callable types: `Callable<T→U>`

### 6.3 Nested Structure Rules

1. Use `{}` for structures with 2-10 fields
2. Don't expand for >10 fields (too verbose)
3. Each field on its own line, indented 2 spaces
4. Field format: `key: Type` or `key: Type<Generic>`
5. Nested expansion allowed (up to 2-3 levels)

---

## 7. Tooling Support

### 7.1 Parser Support

Parsers must support:
- `◊` symbol for inheritance (U+25CA)
- `<>` for generic parameters
- `{}` for nested structure expansion
- `[Abstract]` and `[Protocol]` tags

### 7.2 Formatter Support

Formatters should:
- Align `◊` symbols for readability
- Indent nested structures by 2 spaces
- Preserve generic parameter order
- Sort tags according to composition rules

### 7.3 Validator Support

Validators should check:
- Valid base class names
- Matching generic parameters
- Abstract classes have abstract methods
- Nested structure depth < 4 levels

---

## 8. Backward Compatibility

### 8.1 v1.4 → v1.5

- ✅ All v1.4 files are valid v1.5 (additive changes only)
- ✅ New features are optional
- ✅ Parsers should accept both v1.4 and v1.5

### 8.2 Deprecations

None. All v1.4 features remain valid in v1.5.

---

## Appendix A: Symbol Reference

| Symbol | Unicode | Purpose | Example |
|--------|---------|---------|---------|
| `∈` | U+2208 | Type membership | `x ∈ Tensor` |
| `→` | U+2192 | Function return | `F:foo() → i32` |
| `◊` | U+25CA | Inheritance | `[C:Foo] ◊ Bar` |
| `<>` | U+003C, U+003E | Generics | `List<T>` |
| `{}` | U+007B, U+007D | Nesting | `dict { }` |
| `?` | U+003F | Optional | `Optional<T>` or `T?` |
| `@` | U+0040 | Location | `Tensor@GPU` |
| `[]` | U+005B, U+005D | Shape/Tags | `[N,D]` or `[O(N)]` |

---

**End of Specification v1.5**
