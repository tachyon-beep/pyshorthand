# PyShorthand v1.4 Specification - Implementation Checklist

**Status**: In Progress
**Started**: November 22, 2025

---

## ✅ Completed

### 1. Specification & Core Symbols
- [x] Created PYSHORTHAND_SPEC_v1.4.md (844 lines)
- [x] Updated symbols.py with new tag sets
  - [x] DECORATOR_TAGS
  - [x] HTTP_METHODS
  - [x] Expanded COMMON_QUALIFIERS
  - [x] Helper functions (is_decorator_tag, is_http_method, etc.)

---

## 🔄 In Progress

### 2. AST Nodes (src/pyshort/core/ast_nodes.py)
**Status**: Pending
**Estimated Time**: 30 minutes

**Changes Needed**:
- [ ] Update `Tag` class to support new tag types
  - [ ] Add `tag_type` field: "operation" | "complexity" | "decorator" | "http_route" | "custom"
  - [ ] Add parsing for HTTP routes (method + path)
  - [ ] Add parsing for complexity notation O(...)
- [ ] Update `Function` class to store tags list
- [ ] Add validation in `Tag.__post_init__` for new types

**Example**:
```python
@dataclass
class Tag:
    base: str
    qualifiers: List[str] = field(default_factory=list)
    tag_type: str = "operation"  # NEW
    http_method: Optional[str] = None  # NEW: for route tags
    http_path: Optional[str] = None    # NEW: for route tags
```

---

### 3. Parser (src/pyshort/core/parser.py)
**Status**: Pending
**Estimated Time**: 1-2 hours

**Changes Needed**:
- [ ] Update `parse_tag()` to recognize new tag formats
  - [ ] Parse `[Prop]`, `[Static]`, etc. as decorator tags
  - [ ] Parse `[GET /path]`, `[POST /api/users]` as HTTP routes
  - [ ] Parse `[O(N)]`, `[O(N*M)]` as complexity tags
  - [ ] Parse combined tags: `[Prop:Cached:O(N)]`
- [ ] Update `parse_function_signature()` to handle tags after return type
  - Current: `F:name(params) → type`
  - New: `F:name(params) → type [Tags]`
- [ ] Handle multiple tag groups: `[Prop] [O(N)]`

**Example**:
```python
def parse_tag(self, token_stream: TokenStream) -> Tag:
    # Match [...]
    self.consume(TokenType.LBRACKET)
    content = self.read_until(TokenType.RBRACKET)

    # Check for HTTP route
    route = parse_http_route(content)
    if route:
        return Tag(
            base=route[0],
            tag_type="http_route",
            http_method=route[0],
            http_path=route[1]
        )

    # Check for complexity
    if is_complexity_tag(content):
        return Tag(
            base=content,
            tag_type="complexity"
        )

    # Check for decorator
    if is_decorator_tag(content.split(':')[0]):
        parts = content.split(':')
        return Tag(
            base=parts[0],
            qualifiers=parts[1:],
            tag_type="decorator"
        )

    # Operation tag (existing logic)
    ...
```

---

### 4. Validator (src/pyshort/core/validator.py)
**Status**: Pending
**Estimated Time**: 1 hour

**Changes Needed**:
- [ ] Add `ComplexityTagValidator` rule
  - [ ] Validate O(...) notation syntax
  - [ ] Check for valid variable names (N, M, D, B, etc.)
  - [ ] Warn on unusual complexity (O(N^10), etc.)
- [ ] Add `DecoratorTagValidator` rule
  - [ ] Check for conflicting decorators (Prop + Static)
  - [ ] Validate decorator arguments (RateLimit:100)
- [ ] Add `HTTPRouteValidator` rule
  - [ ] Validate route path syntax (starts with /)
  - [ ] Validate parameter syntax ({param_name})
  - [ ] Check HTTP method is valid
- [ ] Update `TagValidator` for v1.4 tag types

**Example**:
```python
class ComplexityTagValidator(ValidationRule):
    def validate(self, ast: PyShortAST) -> List[Diagnostic]:
        diagnostics = []

        for entity in ast.entities:
            for func in entity.functions:
                for tag in func.tags:
                    if tag.tag_type == "complexity":
                        # Validate O(...) syntax
                        if not re.match(r'^O\([^)]+\)$', tag.base):
                            diagnostics.append(Diagnostic(
                                level="error",
                                message=f"Invalid complexity notation: {tag.base}",
                                code="E301"
                            ))

        return diagnostics
```

---

### 5. Formatter (src/pyshort/formatter/formatter.py)
**Status**: Pending
**Estimated Time**: 30 minutes

**Changes Needed**:
- [ ] Update tag formatting to group by type
  - [ ] Decorators first: `[Prop]`
  - [ ] HTTP routes next: `[GET /path]`
  - [ ] Operations: `[Lin:MatMul]`
  - [ ] Complexity last: `[O(N)]`
- [ ] Add `--tag-order` config option
- [ ] Add alignment for tags in method lists

**Example**:
```python
def format_tags(self, tags: List[Tag]) -> str:
    # Group by type
    decorator_tags = [t for t in tags if t.tag_type == "decorator"]
    route_tags = [t for t in tags if t.tag_type == "http_route"]
    operation_tags = [t for t in tags if t.tag_type == "operation"]
    complexity_tags = [t for t in tags if t.tag_type == "complexity"]

    # Format each group
    parts = []
    for tag_list in [decorator_tags, route_tags, operation_tags, complexity_tags]:
        if tag_list:
            parts.append(self.format_tag_group(tag_list))

    return ' '.join(parts)
```

---

### 6. Decompiler (src/pyshort/decompiler/py2short.py)
**Status**: Pending (but implementation plan exists)
**Estimated Time**: 7-10 hours (as per IMPROVEMENT_PLAN.md)

**Priority 1**: Enhanced Type Inference (2-3 hours)
- [ ] Parameter type tracking
- [ ] Attribute access inference
- [ ] Built-in function inference

**Priority 2**: Complexity Tag Generation (3-4 hours)
- [ ] Docstring complexity extraction
- [ ] Pattern-based detection (loop nesting)
- [ ] Operation tagging (MatMul, Softmax, etc.)

**Priority 3**: Decorator Extraction (2-3 hours)
- [ ] HTTP route extraction
- [ ] Property/static method detection
- [ ] Custom decorator tracking

*Note: Detailed implementation in IMPROVEMENT_PLAN.md*

---

### 7. Tests
**Status**: Pending
**Estimated Time**: 2-3 hours

**Test Files Needed**:
- [ ] `tests/unit/test_symbols_v14.py` - Test new symbol functions
- [ ] `tests/unit/test_parser_v14.py` - Test new tag parsing
- [ ] `tests/unit/test_validator_v14.py` - Test new validation rules
- [ ] `tests/integration/test_v14_features.py` - End-to-end tests

**Test Cases**:
```python
def test_complexity_tag_parsing():
    source = "# F:process(x) → y [O(N*M)]"
    ast = parse(source)
    assert ast.functions[0].tags[0].tag_type == "complexity"
    assert ast.functions[0].tags[0].base == "O(N*M)"

def test_http_route_parsing():
    source = "# F:get_user(id: i32) → User [GET /users/{id}]"
    ast = parse(source)
    tag = ast.functions[0].tags[0]
    assert tag.tag_type == "http_route"
    assert tag.http_method == "GET"
    assert tag.http_path == "/users/{id}"

def test_decorator_tag_parsing():
    source = "# F:device() → str [Prop]"
    ast = parse(source)
    assert ast.functions[0].tags[0].tag_type == "decorator"
    assert ast.functions[0].tags[0].base == "Prop"

def test_combined_tags():
    source = "# F:forward(x) → Tensor [NN:∇:Lin:MatMul:O(N*D)]"
    ast = parse(source)
    tags = ast.functions[0].tags
    # Should parse as single combined operation tag + complexity
    ...
```

---

### 8. Documentation Updates
**Status**: Pending
**Estimated Time**: 1 hour

**Files to Update**:
- [ ] README.md - Mention v1.4 features
- [ ] ARCHITECTURE.md - Update with new tag types
- [ ] STATUS.md - Mark v1.4 features as implemented
- [ ] Examples in docs/ - Add v1.4 examples

---

## 📊 Progress Summary

| Component | Status | Time Est. | Priority |
|-----------|--------|-----------|----------|
| Specification | ✅ Complete | - | - |
| Symbols | ✅ Complete | - | - |
| AST Nodes | ⏳ Pending | 30min | High |
| Parser | ⏳ Pending | 1-2hrs | High |
| Validator | ⏳ Pending | 1hr | Medium |
| Formatter | ⏳ Pending | 30min | Medium |
| Decompiler | ⏳ Pending | 7-10hrs | Low* |
| Tests | ⏳ Pending | 2-3hrs | High |
| Docs | ⏳ Pending | 1hr | Low |

*Low priority because implementation plan exists (IMPROVEMENT_PLAN.md)

**Total Estimated Time**: 13-19 hours
**Core Toolchain** (AST + Parser + Validator + Formatter + Tests): 6-9 hours

---

## 🎯 Recommended Approach

### Phase 1: Core Toolchain Support (6-9 hours)
1. **AST Nodes** (30 min) - Foundation for everything
2. **Parser** (1-2 hrs) - Can parse v1.4 syntax
3. **Tests** (2-3 hrs) - Ensure correctness
4. **Validator** (1 hr) - Validate new tags
5. **Formatter** (30 min) - Pretty-print new syntax

**Result**: Can parse, validate, and format v1.4 PyShorthand files

### Phase 2: Generation (7-10 hours)
6. **Decompiler** (7-10 hrs) - Generate v1.4 from Python

**Result**: Can auto-generate v1.4 PyShorthand from Python code

### Phase 3: Polish (1 hour)
7. **Documentation** (1 hr) - Update all docs

---

## 🚀 Quick Start

To continue implementation:

```bash
# Start with AST nodes
vim src/pyshort/core/ast_nodes.py

# Then parser
vim src/pyshort/core/parser.py

# Write tests as you go
vim tests/unit/test_parser_v14.py
```

**Test-driven approach**:
1. Write test for new feature
2. Implement in AST/Parser
3. Verify test passes
4. Repeat

---

## 📝 Notes

- All changes maintain backward compatibility with v1.3.1
- Decompiler improvements can be done separately (detailed in IMPROVEMENT_PLAN.md)
- Focus on toolchain first, generation second
- Tests are critical - v1.4 adds significant complexity

---

**Last Updated**: November 22, 2025
**Next Step**: Update AST nodes
