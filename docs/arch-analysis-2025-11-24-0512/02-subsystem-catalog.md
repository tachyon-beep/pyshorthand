# COMPREHENSIVE SUBSYSTEM CATALOG

## PyShorthand Toolchain Architecture Analysis

**Analysis Date:** 2025-11-24
**Repository:** /home/john/pyshorthand
**Analysis Depth:** Very Thorough (Code-level inspection)

---

## BATCH 1 - CORE PIPELINE (Parsing & Validation)

### 1. TOKENIZER

#### Subsystem Overview

- **Name**: Lexical Analyzer / Tokenizer
- **Location**: `src/pyshort/core/tokenizer.py` (548 lines)
- **Primary Responsibility**: Breaks PyShorthand source code into lexical tokens, supporting both Unicode (∈, →) and ASCII (IN, ->) notation
- **Confidence Level**: High

#### Key Components

**Classes:**

- `TokenType(Enum)` (lines 12-77): Defines ~60 token types including literals, operators, delimiters, and special symbols
- `Token(dataclass)` (lines 79-90): Immutable token representation with type, value, line, and column
- `Tokenizer` (lines 92-547): Main lexical analyzer with stateful position tracking

**Key Methods:**

- `tokenize() -> list[Token]` (lines 339-547): Main entry point, returns complete token stream with EOF
- `read_number()` (lines 152-176): Numeric literal parsing with range validation (P23 compliance)
- `read_string(quote)` (lines 229-300): String literal handling with escape sequences
- `read_multiline_string(quote)` (lines 302-337): Triple-quoted string support

**Data Structures:**

- Unicode/ASCII symbol mappings (lines 461-477, 485-509)
- Escape sequence map for strings (lines 235-246)

**Constants:**

- Numeric range limits: i64 (±2^63), f64 (±3.4e38) with overflow warnings

#### Dependencies

**Inbound** (Who uses it):

- Parser subsystem (`src/pyshort/core/parser.py`)
- All compilation pipelines requiring PyShorthand input

**Outbound** (What it uses):

- Python stdlib: `dataclasses`, `enum`, `warnings`
- No internal PyShorthand dependencies (foundation module)

**External**: None (stdlib only)

#### Architectural Patterns Observed

**Design Patterns:**

- **Lexer Pattern**: Classic recursive descent tokenization with lookahead
- **Warning System**: Uses Python's `warnings` module for non-fatal issues (numeric overflow, unsupported escapes)

**Coding Conventions:**

- Stateful position tracking (line, column, pos)
- Lookahead via `peek_char(offset)` method
- Immutable token dataclasses

**Error Handling:**

- Validation at tokenization time (numeric ranges via `_validate_numeric_range()`)
- Raises `ValueError` for unterminated strings or unexpected characters
- Warnings for edge cases (octal/hex escapes, Unicode escapes)

**Notable Implementation:**

- Bidirectional symbol support: Unicode ↔ ASCII (lines 461-482)
- Escape sequence mapping with fallback for unknowns
- Multi-character operator disambiguation (e.g., `!=` vs `!`)

#### API Surface & Entry Points

**Public API:**

- `Tokenizer(source: str)` - Constructor
- `tokenize() -> list[Token]` - Main tokenization method
- `Token` and `TokenType` - Public data types

**Configuration**: None (zero-config design)

#### Testing & Quality

**Test Files**:

- Not found in scan (likely in integration tests)

**Coverage Observations**:

- Extensive token type coverage (60+ types)
- Numeric validation edge cases handled
- String escape sequences thoroughly mapped

**Known Issues/TODOs**:

- Line 262-283: Octal/hex/Unicode escape sequences emit warnings but not fully supported
- Could benefit from position-aware error messages

---

### 2. PARSER

#### Subsystem Overview

- **Name**: Recursive Descent Parser
- **Location**: `src/pyshort/core/parser.py` (1253 lines)
- **Primary Responsibility**: Transforms token stream into Abstract Syntax Tree (AST), enforcing PyShorthand grammar rules
- **Confidence Level**: High

#### Key Components

**Classes:**

- `ParseError(Exception)` (lines 88-94): Location-aware parse error with token context
- `Parser` (lines 97-1215): Main recursive descent parser with lookahead

**Key Methods:**

- `parse() -> PyShortAST` (lines 937-1066): Top-level entry point, produces complete AST
- `parse_class(line: int) -> Class` (lines 1068-1215): Class/Protocol parsing with 0.9.0-RC1 features
- `parse_function(line: int) -> Function` (lines 819-921): Function parsing with contracts
- `parse_type_spec() -> TypeSpec` (lines 292-412): Type annotation parsing (supports unions, generics, nested structures)
- `parse_expression() -> Expression` (lines 511-538): Expression parsing with operator precedence
- `parse_tag() -> Tag` (lines 450-509): Computational tag parsing (legacy support for decorators, HTTP routes, complexity)

**Data Structures:**

- `RESERVED_KEYWORDS` (lines 41-85): Python + PyShorthand reserved words
- Tag type detection via `_is_legacy_tag()` (lines 224-290)

**Constants:**

- Entity prefixes from `symbols.py` (C, D, I, M, F, P, E)

#### Dependencies

**Inbound** (Who uses it):

- Formatter subsystem
- Validator/Linter subsystem
- CLI parse command
- All downstream analysis tools

**Outbound** (What it uses):

- Tokenizer subsystem (`src/pyshort/core/tokenizer.py`)
- AST Nodes subsystem (`src/pyshort/core/ast_nodes.py`)
- Symbols subsystem (`src/pyshort/core/symbols.py`)
- Python stdlib: `re`

**External**: None (stdlib only)

#### Architectural Patterns Observed

**Design Patterns:**

- **Recursive Descent**: Classic top-down parser with lookahead
- **Error Recovery**: Continues parsing after errors, accumulates diagnostics
- **Visitor-like parsing**: Separate methods for each grammar construct

**Coding Conventions:**

- `skip_*` methods for whitespace/comment handling
- `expect(token_type)` for mandatory tokens with error reporting
- `peek(offset)` for lookahead without consuming tokens

**Error Handling:**

- Accumulates `Diagnostic` objects in AST instead of crashing
- Recovers from parse errors by skipping to next entity
- Validates reserved keywords during parsing (lines 144-157)

**Notable Implementation:**

- 0.9.0-RC1 feature support: Generics `<T>`, inheritance `◊`, protocols `[P:Name]`, abstract classes
- Legacy tag system preserved: decorators, HTTP routes, complexity, operations
- Union type support: `Type1 | Type2 | Type3`
- Reference type handling: `[Ref:Name]`
- Method signature parsing in classes with error recovery (lines 1183-1202)

#### API Surface & Entry Points

**Public Functions:**

- `parse_file(file_path: str) -> PyShortAST` (lines 1218-1234)
- `parse_string(source: str, source_name: str) -> PyShortAST` (lines 1237-1252)

**Configuration**: None

#### Testing & Quality

**Test Files**:

- `tests/unit/test_parser.py`
- `tests/unit/test_parser_v14.py`

**Coverage Observations**:

- Comprehensive grammar coverage
- Error recovery mechanisms in place
- Tag type detection sophisticated

**Known Issues/TODOs**:

- Line 902-908: Function body parsing could be more robust
- Error recovery sometimes skips too aggressively

---

### 3. AST NODES

#### Subsystem Overview

- **Name**: Abstract Syntax Tree Data Structures
- **Location**: `src/pyshort/core/ast_nodes.py` (727 lines)
- **Primary Responsibility**: Defines immutable dataclass hierarchy representing parsed PyShorthand code structure
- **Confidence Level**: High

#### Key Components

**Classes (Diagnostics):**

- `DiagnosticSeverity(Enum)` (lines 26-32): ERROR, WARNING, INFO, HINT
- `Diagnostic(frozen dataclass)` (lines 35-56): Error/warning with location and suggestion

**Classes (Metadata):**

- `Metadata(frozen dataclass)` (lines 64-92): Module-level metadata headers

**Classes (Type System):**

- `TypeSpec(frozen dataclass)` (lines 100-141): Type with shape, location, transfer, unions, generics

**Classes (Tags):**

- `Tag(frozen dataclass)` (lines 149-255): Computational tags (operation, complexity, decorator, http_route, custom)

**Classes (Expressions):**

- `Expression(ABC)` (lines 263-269): Base class for all expressions
- `Identifier`, `Literal`, `BinaryOp`, `UnaryOp`, `FunctionCall`, `IndexOp`, `AttributeAccess` (lines 272-408)

**Classes (Statements):**

- `Statement(frozen dataclass)` (lines 416-458): Single statement with tags, profiling, conditions

**Classes (Functions & State):**

- `Parameter(frozen dataclass)` (lines 493-512): Function parameter with type
- `Function(frozen dataclass)` (lines 515-553): Function with contracts, body, tags
- `StateVar(frozen dataclass)` (lines 465-485): State variable with type spec

**Classes (Entities):**

- `Entity(ABC)` (lines 560-566): Base for all entities
- `Class(frozen dataclass)` (lines 583-619): Class with state, methods, inheritance, generics (0.9.0-RC1)
- `Data`, `Interface`, `Enum`, `Module`, `Reference` (lines 622-676)

**Classes (Top-Level):**

- `PyShortAST(dataclass)` (lines 683-727): Complete AST with diagnostics

**Data Structures:**

- All frozen dataclasses with `to_dict()` serialization
- Union types via `union_types: list[str] | None`
- Generic parameters via `generic_params: list[str] | None`

#### Dependencies

**Inbound** (Who uses it):

- Parser (constructs AST)
- Formatter (reads AST, reconstructs code)
- Validator (analyzes AST)
- Decompiler (generates from Python AST)
- All analysis tools

**Outbound** (What it uses):

- Symbols subsystem (`src/pyshort/core/symbols.py`) for tag validation
- Python stdlib: `dataclasses`, `abc`, `enum`, `typing`

**External**: None (stdlib only)

#### Architectural Patterns Observed

**Design Patterns:**

- **Composite Pattern**: Expression hierarchy with recursive structure
- **Visitor Pattern**: All nodes have `to_dict()` for serialization
- **Immutability**: All nodes are frozen dataclasses

**Coding Conventions:**

- `to_dict()` methods for JSON serialization
- `__str__()` methods for PyShorthand notation
- Properties for derived values (e.g., `Tag.complexity`)

**Error Handling:**

- Tag validation in `__post_init__()` (lines 169-190)
- Raises `ValueError` for invalid tag configurations

**Notable Implementation:**

- 0.9.0-RC1 additions: `generic_params`, `base_classes`, `is_abstract`, `is_protocol` on Class
- Tag system with type discrimination: operation, complexity, decorator, http_route
- Expression tree supports indexing, attribute access, method calls
- Diagnostic system integrated into AST (lines 716-722)

#### API Surface & Entry Points

**Public Types:**

- All AST node classes (15+ types)
- `PyShortAST` as root container
- `Diagnostic` for error reporting

**Configuration**: None

#### Testing & Quality

**Test Files**:

- `tests/unit/test_ast_nodes_v14.py`
- Integration tests use AST nodes extensively

**Coverage Observations**:

- Comprehensive node types for language features
- Serialization via `to_dict()` well-supported
- Validation in dataclass constructors

**Known Issues/TODOs**:

- Line 285: Missing `Literal` class definition marker (appears to be partial)
- Some expression types could have more validation

---

### 4. VALIDATOR

#### Subsystem Overview

- **Name**: Semantic Validator & Linter
- **Location**: `src/pyshort/core/validator.py` (632 lines)
- **Primary Responsibility**: Enforces grammar constraints and semantic best practices via rule-based validation system
- **Confidence Level**: High

#### Key Components

**Classes:**

- `Rule(ABC)` (lines 27-40): Base class for validation rules
- `Linter` (lines 549-618): Main linter orchestrating all rules

**Rule Classes (14 total):**

- `MandatoryMetadataRule` (lines 43-66): Enforces [M:Name] and [Role] headers
- `ValidMetadataValuesRule` (lines 69-111): Validates metadata against allowed sets
- `DimensionConsistencyRule` (lines 114-148): Ensures dimension variables declared
- `ValidTagsRule` (lines 151-187): Validates operation tags
- `ComplexityTagValidator` (lines 190-241): Validates O(...) notation
- `DecoratorTagValidator` (lines 244-311): Validates decorator conflicts
- `HTTPRouteValidator` (lines 314-388): Validates HTTP route syntax
- `SystemMutationSafetyRule` (lines 391-412): Flags !! without [Risk:High]
- `CriticalOperationTaggingRule` (lines 415-429): Warns about untagged mutations
- `LocationInferenceRule` (lines 432-466): Validates @Location annotations
- `TypeValidityRule` (lines 469-484): Checks type names
- `ErrorSurfaceDocumentationRule` (lines 487-502): Requires [Err] for !? operations
- `GenericParametersValidityRule` (lines 505-524): Validates generic param naming (0.9.0-RC1)
- `InheritanceValidityRule` (lines 527-546): Validates base class names (0.9.0-RC1)

**Key Methods:**

- `Linter.check(ast: PyShortAST) -> list[Diagnostic]` (lines 576-601): Run all rules
- `Linter.check_file(file_path: str) -> list[Diagnostic]` (lines 603-617): Validate file

**Data Structures:**

- Rule iterator pattern via `Rule.check()` yielding diagnostics

#### Dependencies

**Inbound** (Who uses it):

- CLI lint command
- Formatter (optional validation)
- IDE integrations

**Outbound** (What it uses):

- AST Nodes subsystem (`src/pyshort/core/ast_nodes.py`)
- Symbols subsystem (`src/pyshort/core/symbols.py`)
- Enhanced Errors subsystem (`src/pyshort/core/enhanced_errors.py`)
- Parser (via `check_file()`)
- Python stdlib: `re`, `abc`, `collections.abc`

**External**: None (stdlib only)

#### Architectural Patterns Observed

**Design Patterns:**

- **Strategy Pattern**: Each rule is an independent strategy
- **Iterator Pattern**: Rules yield diagnostics
- **Template Method**: `Rule.check()` abstract method

**Coding Conventions:**

- Rules are composable and independent
- Diagnostic codes (E001-E399, W001-W399)
- "Did you mean?" suggestions via Levenshtein distance

**Error Handling:**

- Accumulates diagnostics without failing
- Strict mode upgrades warnings to errors (line 590-597)

**Notable Implementation:**

- 0.9.0-RC1 support: Generic params and inheritance validation
- Legacy tag validation: HTTP routes, decorators, complexity
- Dimension consistency checking across state variables
- Safety analysis: !! operations require [Risk:High] marker
- Suggestion system with typo detection

#### API Surface & Entry Points

**Public Functions:**

- `validate_file(file_path: str, strict: bool) -> list[Diagnostic]` (lines 620-631)
- `lint_code(source: str) -> list[Diagnostic]` (implied)

**Configuration**:

- `strict: bool` - Treat warnings as errors

#### Testing & Quality

**Test Files**:

- `tests/unit/test_validator_v14.py`

**Coverage Observations**:

- 14 independent validation rules
- Comprehensive metadata validation
- Safety-focused rules for mutations

**Known Issues/TODOs**:

- Could add more cross-entity validation
- Circular dependency detection not implemented

---

### 5. SYMBOLS

#### Subsystem Overview

- **Name**: Symbol Mapping & Constants
- **Location**: `src/pyshort/core/symbols.py` (231 lines)
- **Primary Responsibility**: Defines canonical symbol sets and provides Unicode ↔ ASCII conversion utilities
- **Confidence Level**: High

#### Key Components

**Constants (Sets):**

- `UNICODE_TO_ASCII` (lines 9-23): Mapping of Unicode symbols to ASCII equivalents
- `ASCII_TO_UNICODE` (line 26): Reverse mapping
- `VALID_TAG_BASES` (lines 29-39): Operation tag types (Lin, Thresh, Iter, Map, Stoch, IO, Sync, NN, Heur)
- `DECORATOR_TAGS` (lines 42-57): Python decorator mappings (Prop, Static, Class, Cached, Auth, etc.)
- `HTTP_METHODS` (lines 60-68): Valid HTTP verbs
- `VALID_TYPES` (lines 71-90): PyShorthand type names
- `VALID_LOCATIONS` (lines 93-103): Memory locations
- `VALID_ROLES` (line 106): Core, Glue, Script
- `VALID_LAYERS` (line 109): Domain, Infra, Adapter, Test
- `VALID_RISK_LEVELS` (line 112): High, Med, Low
- `ENTITY_PREFIXES` (lines 115-124): C, D, I, M, F, P, E

**Functions:**

- `to_ascii(text: str) -> str` (lines 127-143): Convert Unicode → ASCII
- `to_unicode(text: str) -> str` (lines 146-163): Convert ASCII → Unicode
- `is_decorator_tag(tag: str) -> bool` (lines 166-183)
- `is_complexity_tag(tag: str) -> bool` (lines 186-205): Validates O(...) pattern
- `parse_http_route(tag_string: str) -> tuple[str, str] | None` (lines 208-230): Extract HTTP method and path

#### Dependencies

**Inbound** (Who uses it):

- Parser (imports all symbol sets)
- Validator (validates against symbol sets)
- Formatter (uses conversion functions)
- AST Nodes (tag validation)
- Enhanced Errors (suggestion system)

**Outbound** (What it uses):

- Python stdlib: `re` (for complexity pattern matching)

**External**: None (stdlib only)

#### Architectural Patterns Observed

**Design Patterns:**

- **Constant Repository**: Centralized canonical definitions
- **Utility Functions**: Stateless conversion functions

**Coding Conventions:**

- All-caps for constant sets
- Type hints for function signatures
- Docstrings with examples

**Error Handling:**

- Conversion functions are total (don't fail)
- Validation functions return bool or None

**Notable Implementation:**

- Bidirectional Unicode/ASCII support via dict inversion
- HTTP route parsing with path validation
- Complexity tag validation with regex
- 0.9.0-RC1 additions: Protocol (P) and Enum (E) entity types, Abstract tag

#### API Surface & Entry Points

**Public API:**

- All `VALID_*` constant sets
- Conversion functions: `to_ascii()`, `to_unicode()`
- Validation functions: `is_decorator_tag()`, `is_complexity_tag()`, `parse_http_route()`

**Configuration**: None (constants only)

#### Testing & Quality

**Test Files**:

- Likely tested indirectly via parser/validator tests

**Coverage Observations**:

- Comprehensive symbol coverage
- Examples in docstrings
- Main block for testing (lines 189-230)

**Known Issues/TODOs**:

- Could add validation for dimension variable naming conventions
- HTTP route parsing is basic (no regex patterns in paths)

---

## BATCH 2 - TRANSFORMATION & ANALYSIS

### 6. DECOMPILER

#### Subsystem Overview

- **Name**: Python to PyShorthand Decompiler
- **Location**: `src/pyshort/decompiler/py2short.py` (1143 lines)
- **Primary Responsibility**: Reverse-engineers Python AST into PyShorthand notation with aggressive type inference and pattern detection
- **Confidence Level**: High

#### Key Components

**Classes:**

- `PyShorthandGenerator` (lines 9-1080): Main decompilation engine with type inference

**Key Methods:**

- `generate(tree: ast.Module, source_file) -> str` (lines 26-80): Top-level entry point
- `_generate_entity(cls: ast.ClassDef, tree) -> list[str]` (lines 173-254): Class → PyShorthand
- `_generate_function_signature(func) -> str` (lines 526-564): Function with legacy tags
- `_extract_function_tags(func) -> list[str]` (lines 566-597): Extract decorator/HTTP/operation/complexity tags
- `_convert_type_annotation(annotation) -> str` (lines 897-977): Python types → PyShorthand types
- `_extract_operation_tags(func) -> list[str]` (lines 723-811): Detect NN, IO, Iter patterns
- `_extract_complexity_tag(func) -> str | None` (lines 813-862): O(...) from docstring or analysis

**Data Structures:**

- Import map: `dict[str, str]` for alias → module resolution
- Local classes set for reference detection
- Type mapping: Python → PyShorthand primitives

**Constants:**

- Type mappings (lines 988-997): int→i32, float→f32, etc.

#### Dependencies

**Inbound** (Who uses it):

- Indexer subsystem (repo scanning)
- Ecosystem tools (progressive disclosure)
- CLI decompile command

**Outbound** (What it uses):

- Python stdlib: `ast`, `re`, `dataclasses`, `pathlib`

**External**: None (stdlib only)

#### Architectural Patterns Observed

**Design Patterns:**

- **Visitor Pattern**: Walks Python AST with type-specific handlers
- **Strategy Pattern**: Different type inference strategies (aggressive vs conservative)
- **Pattern Matching**: Detects frameworks (Pydantic, FastAPI, PyTorch)

**Coding Conventions:**

- Prefix methods with `_extract_`, `_generate_`, `_infer_`
- Caching via instance variables (`self.imports`, `self.local_classes`)
- Aggressive type inference optional

**Error Handling:**

- Fails gracefully on parse errors
- Returns "Unknown" for uninferable types
- Uses warnings for unsupported patterns

**Notable Implementation:**

- 0.9.0-RC1 support: Detects abstract classes, protocols, generics, inheritance
- Framework detection: Pydantic, FastAPI, Flask, Django, PyTorch
- Decorator mapping: @property → [Prop], @staticmethod → [Static]
- HTTP route extraction from decorators
- Operation tag inference: NN:∇ for torch.backward, IO:Net for requests
- Complexity inference from loop nesting depth
- Local class reference detection
- Nested structure expansion (ModuleDict, etc.)

#### API Surface & Entry Points

**Public Functions:**

- `decompile(source: str, aggressive: bool) -> str` (lines 1082-1094)
- `decompile_file(input_path, output_path, aggressive) -> str` (lines 1097-1138)
- `PyShortDecompiler` (line 1142): Backward compatibility alias

**Configuration**:

- `aggressive: bool` - Use aggressive type inference
- `with_confidence: bool` - Include confidence scores

#### Testing & Quality

**Test Files**:

- `tests/unit/test_decompiler_v14.py`

**Coverage Observations**:

- Comprehensive framework detection
- Tag inference sophisticated
- Type mapping extensive

**Known Issues/TODOs**:

- Line 1060: Could enhance type inference for complex generics
- Line 977: Fallback to "Unknown" could be more informative
- Optional/Union type handling could be more precise

---

### 7. FORMATTER

#### Subsystem Overview

- **Name**: Auto-Formatter
- **Location**: `src/pyshort/formatter/formatter.py` (417 lines)
- **Primary Responsibility**: Provides opinionated, consistent formatting for PyShorthand code with configurable style options
- **Confidence Level**: High

#### Key Components

**Classes:**

- `FormatConfig(dataclass)` (lines 13-22): Configuration options
- `Formatter` (lines 25-377): Main formatting engine

**Key Methods:**

- `format_ast(ast: PyShortAST) -> str` (lines 36-79): Format complete AST
- `_format_metadata(ast) -> list[str]` (lines 81-123): Metadata headers
- `_format_class(cls: Class) -> list[str]` (lines 126-180): Class formatting with 0.9.0-RC1 support
- `_format_function(func: Function) -> list[str]` (lines 267-316): Function with contracts
- `_format_state_variables(state_vars) -> list[str]` (lines 182-215): Aligned state vars
- `_format_tags(tags) -> str` (lines 236-265): Grouped tag formatting

**Data Structures:**

- `FormatConfig` with alignment, sorting, Unicode preference options

**Constants:**

- Default indent: 2 spaces
- Default line length: 100 characters
- Blank lines around functions/classes configurable

#### Dependencies

**Inbound** (Who uses it):

- CLI format command
- IDE integrations (planned)

**Outbound** (What it uses):

- AST Nodes subsystem
- Parser subsystem (`parse_file`, `parse_string`)
- Symbols subsystem (`to_ascii`, `to_unicode`)
- Python stdlib: `dataclasses`

**External**: None (stdlib only)

#### Architectural Patterns Observed

**Design Patterns:**

- **Visitor Pattern**: Traverses AST to produce formatted output
- **Configuration Object**: `FormatConfig` for customization
- **Template Method**: Consistent `_format_*` methods

**Coding Conventions:**

- Alignment via `ljust()` for state variables
- Sorted output for determinism
- Unicode/ASCII conversion as final step

**Error Handling:**

- Assumes valid AST (parser has already validated)
- Gracefully handles missing optional fields

**Notable Implementation:**

- 0.9.0-RC1 support: Protocols, generics, inheritance, abstract classes
- Tag grouping: Decorators → Routes → Operations → Complexity
- State variable sorting by location (GPU → CPU → Disk → Net)
- Aligned type annotations for readability
- Blank line control around entities
- Unicode preference toggle

#### API Surface & Entry Points

**Public Functions:**

- `format_string(source: str, config) -> str` (lines 380-392)
- `format_file(file_path: str, config, in_place: bool) -> str` (lines 395-416)

**Configuration**:

- `FormatConfig` with 6 options (indent, align, Unicode, sort, blank lines)

#### Testing & Quality

**Test Files**:

- `tests/unit/test_formatter.py`
- `tests/unit/test_formatter_v14.py`

**Coverage Observations**:

- Handles all AST node types
- Deterministic output
- Idempotent formatting

**Known Issues/TODOs**:

- No max line length enforcement yet
- Could add more sort options (by type, by name)

---

### 8. CONTEXT ANALYZER

#### Subsystem Overview

- **Name**: Context Pack Generator
- **Location**: `src/pyshort/analyzer/context_pack.py` (579 lines)
- **Primary Responsibility**: Generates dependency-aware context packs with F0/F1/F2 layers for LLM context windows and code understanding
- **Confidence Level**: High

#### Key Components

**Classes:**

- `ContextPack(dataclass)` (lines 21-320): Result container with filtering/export methods
- `ContextPackGenerator` (lines 322-557): Dependency graph builder and traversal engine

**Key Methods:**

- `generate_context_pack(module, target_name, max_depth, include_peers) -> ContextPack` (lines 331-396): Main entry point
- `ContextPack.to_mermaid(direction) -> str` (lines 58-126): Mermaid diagram export
- `ContextPack.to_graphviz() -> str` (lines 128-190): GraphViz DOT export
- `ContextPack.filter_by_location(location) -> ContextPack` (lines 192-227): Filter by @GPU/@CPU/etc
- `ContextPack.filter_by_pattern(pattern) -> ContextPack` (lines 229-259): Regex filtering
- `ContextPack.filter_custom(predicate) -> ContextPack` (lines 261-297): Custom predicate filtering

**Data Structures:**

- Dependency layers: F0 (target), F1 (immediate), F2 (extended)
- Bidirectional dependency graph: forward + reverse edges
- Class peer tracking for method context

**Constants:**

- F0 = target entity
- F1 = direct dependencies (1-hop)
- F2 = transitive dependencies (2-hop)

#### Dependencies

**Inbound** (Who uses it):

- Ecosystem tools (progressive disclosure)
- IDE integrations
- Documentation generators

**Outbound** (What it uses):

- AST Nodes subsystem (`Class`, `Data`, `Function`, `Module`)
- Python stdlib: `re`, `dataclasses`, `collections.abc`

**External**: None (stdlib only)

#### Architectural Patterns Observed

**Design Patterns:**

- **Builder Pattern**: `ContextPackGenerator` builds `ContextPack`
- **Fluent Interface**: Chainable filter methods
- **Graph Traversal**: BFS-style dependency exploration

**Coding Conventions:**

- Immutable filtering (returns new `ContextPack`)
- Layer-based organization (F0/F1/F2)
- Visualization export methods on result object

**Error Handling:**

- Returns `None` if target not found
- Gracefully handles missing dependencies

**Notable Implementation:**

- Three-layer dependency model perfect for LLM context windows
- Bidirectional graph: callers AND callees
- Class peer detection for method context
- State variable tracking across layers
- Filtering API for focused analysis
- Mermaid + GraphViz export with color-coding
- Type reference extraction from `[Ref:Name]` notation

#### API Surface & Entry Points

**Public Functions:**

- `generate_context_pack(module, target_name, max_depth, include_peers) -> ContextPack` (lines 559-578): Convenience function

**Configuration**:

- `max_depth: int` - How many hops (1=F1 only, 2=F1+F2)
- `include_peers: bool` - Include class peer methods

#### Testing & Quality

**Test Files**:

- `tests/test_context_pack.py`

**Coverage Observations**:

- Comprehensive filtering API
- Export formats tested
- Dependency graph construction robust

**Known Issues/TODOs**:

- Line 548: Execution analysis could be merged with this subsystem
- Circular dependency detection not implemented

---

### 9. EXECUTION ANALYZER

#### Subsystem Overview

- **Name**: Execution Flow Tracer
- **Location**: `src/pyshort/analyzer/execution_flow.py` (617 lines)
- **Primary Responsibility**: Traces runtime call paths through functions, tracking variables in scope and call graphs (unlike static dependency layers)
- **Confidence Level**: High

#### Key Components

**Classes:**

- `ExecutionStep(dataclass)` (lines 25-33): Single step with depth, scope, calls
- `ExecutionFlow(dataclass)` (lines 36-389): Complete trace with filtering/export
- `ExecutionFlowTracer` (lines 391-596): Call graph builder and tracer

**Key Methods:**

- `trace_execution(module, entry_point, max_depth, follow_calls) -> ExecutionFlow` (lines 400-438): Main entry point
- `ExecutionFlow.to_mermaid(direction) -> str` (lines 101-166): Flowchart export
- `ExecutionFlow.to_graphviz() -> str` (lines 168-230): DOT export
- `ExecutionFlow.filter_by_depth(max_depth) -> ExecutionFlow` (lines 232-260): Depth filtering
- `ExecutionFlow.filter_by_state_access(state_pattern) -> ExecutionFlow` (lines 293-326): State-based filtering
- `ExecutionFlow.get_call_chain() -> list[str]` (lines 378-388): Simple call list

**Data Structures:**

- Execution steps with depth tracking
- Variables in scope per step
- State variables accessed per step
- Call graph: entity → functions it calls

**Constants:**

- Default max depth: 10 (prevents infinite recursion)

#### Dependencies

**Inbound** (Who uses it):

- Ecosystem tools (execution tracing)
- Debuggers / profilers (planned)

**Outbound** (What it uses):

- AST Nodes subsystem (`Class`, `Function`, `Module`, `Statement`)
- Python stdlib: `re`, `dataclasses`, `collections.abc`

**External**: None (stdlib only)

#### Architectural Patterns Observed

**Design Patterns:**

- **Visitor Pattern**: Walks call graph depth-first
- **Fluent Interface**: Chainable filtering methods
- **Depth-First Search**: Call graph traversal with cycle detection

**Coding Conventions:**

- Visited set prevents infinite loops
- Depth tracking for visualization
- Immutable filtering (returns new `ExecutionFlow`)

**Error Handling:**

- Returns `None` if entry point not found
- Max depth prevents stack overflow
- Visited tracking prevents infinite loops

**Notable Implementation:**

- Runtime path vs static dependencies distinction
- Variables in scope tracking per step
- State variable access tracking
- Depth-based visualization colors
- Filtering by call pattern, depth, state access
- Call graph construction from statements
- Mermaid/GraphViz export with depth-based colors

#### API Surface & Entry Points

**Public Functions:**

- `trace_execution(module, entry_point, max_depth, follow_calls) -> ExecutionFlow` (lines 598-616): Convenience function

**Configuration**:

- `max_depth: int` - Maximum call depth (default 10)
- `follow_calls: bool` - Recursively trace or flat

#### Testing & Quality

**Test Files**:

- `tests/test_execution_flow.py`

**Coverage Observations**:

- Cycle detection working
- Export formats tested
- Filtering API comprehensive

**Known Issues/TODOs**:

- Line 489: Statement parsing for calls is simplified (could be more robust)
- Function parameter tracking incomplete

---

### 10. INDEXER

#### Subsystem Overview

- **Name**: Repository Indexer
- **Location**: `src/pyshort/indexer/repo_indexer.py` (519 lines)
- **Primary Responsibility**: Scans entire Python repositories, generating PyShorthand specs with dependency analysis and statistics
- **Confidence Level**: High

#### Key Components

**Classes:**

- `EntityInfo(dataclass)` (lines 16-27): Entity metadata (class/function)
- `ModuleInfo(dataclass)` (lines 30-38): Module-level information
- `RepositoryIndex(dataclass)` (lines 42-49): Complete repository index
- `RepositoryIndexer` (lines 56-496): Main indexing engine

**Key Methods:**

- `index_repository(verbose: bool) -> RepositoryIndex` (lines 296-339): Scan entire repo
- `index_file(file_path: Path) -> ModuleInfo` (lines 218-255): Index single file
- `build_dependency_graph()` (lines 257-277): Module-level dependencies
- `generate_report() -> str` (lines 371-414): Human-readable summary
- `generate_dependency_graph_mermaid(max_nodes) -> str` (lines 416-457): Dependency visualization
- `save_index(output_path: str)` (lines 341-369): JSON serialization

**Data Structures:**

- Entity map: fully qualified name → EntityInfo
- Dependency graph: module → dependencies
- Statistics: counts, averages, top modules

**Constants:**

- Default exclusions: venv, __pycache__, .git, node_modules, dist, build

#### Dependencies

**Inbound** (Who uses it):

- CLI index command
- Repository analysis tools
- Documentation generators

**Outbound** (What it uses):

- Decompiler subsystem (`decompile_file`)
- Python stdlib: `ast`, `json`, `pathlib`, `dataclasses`, `collections`

**External**: None (stdlib only)

#### Architectural Patterns Observed

**Design Patterns:**

- **Builder Pattern**: Incremental index construction
- **Repository Pattern**: Centralized access to repo metadata
- **Caching**: Decompiled PyShorthand cached per module

**Coding Conventions:**

- Progressive scanning with optional verbosity
- Exclusion patterns for common noise
- Module path normalization (handles src/ directories)

**Error Handling:**

- Skips files with syntax errors
- Continues on decompilation failures
- Returns `None` for unparseable files

**Notable Implementation:**

- Full repository scanning with exclusions
- Dependency graph construction (module-level)
- Statistics computation (lines, entities, classes, functions)
- PyShorthand generation for each module
- Entity extraction from Python AST
- Module path conversion (file → dotted notation)
- Top module identification
- Mermaid dependency graph export
- JSON serialization for caching

#### API Surface & Entry Points

**Public Functions:**

- `index_repository(root_path, output_path, verbose) -> RepositoryIndex` (lines 499-518): Convenience function

**Configuration**:

- `exclude_patterns: list[str]` - Patterns to skip
- `verbose: bool` - Progress output

#### Testing & Quality

**Test Files**:

- Not found in scan (integration tests likely)

**Coverage Observations**:

- Handles large repositories
- Performance optimized (O(1) lookups via sets)
- Statistics generation comprehensive

**Known Issues/TODOs**:

- Line 54: Unused `_entity` variable
- Could add cross-file dependency analysis
- Function-level dependencies not tracked (only module-level)

---

### 11. VISUALIZATION

#### Subsystem Overview

- **Name**: Mermaid Diagram Generator
- **Location**: `src/pyshort/visualization/mermaid.py` (266 lines)
- **Primary Responsibility**: Generates Mermaid syntax for documentation-friendly visualizations (flowcharts, class diagrams, graphs)
- **Confidence Level**: High

#### Key Components

**Classes:**

- `MermaidConfig(dataclass)` (lines 21-31): Diagram configuration
- `MermaidGenerator` (lines 34-238): Main diagram generator

**Key Methods:**

- `generate(ast: PyShortAST) -> str` (lines 41-48): Entry point (delegates by diagram type)
- `generate_flowchart(ast) -> str` (lines 50-100): Dataflow/dependency flowchart
- `generate_class_diagram(ast) -> str` (lines 102-135): UML-like class diagram
- `generate_graph(ast) -> str` (lines 137-156): Simple relationship graph
- `_format_class_label(cls) -> str` (lines 166-179): Class display names
- `_get_risk_color(risk: str) -> str` (lines 229-237): Color coding by risk level

**Data Structures:**

- `MermaidConfig` with 7 options
- Node ID sanitization map

**Constants:**

- Risk colors: High=#ff6b6b, Medium=#ffd93d, Low=#6bcf7f

#### Dependencies

**Inbound** (Who uses it):

- CLI viz command
- Documentation generators
- Markdown exports

**Outbound** (What it uses):

- AST Nodes subsystem (`Class`, `Function`, `PyShortAST`, `StateVar`)
- Python stdlib: `dataclasses`

**External**: None (stdlib only)

#### Architectural Patterns Observed

**Design Patterns:**

- **Strategy Pattern**: Different diagram types as strategies
- **Template Method**: Consistent `_format_*` methods
- **Configuration Object**: `MermaidConfig` for customization

**Coding Conventions:**

- Node ID sanitization (alphanumeric + underscore)
- Label truncation for readability
- Risk-based color coding

**Error Handling:**

- Assumes valid AST
- Handles missing optional fields gracefully

**Notable Implementation:**

- Three diagram types: flowchart, classDiagram, graph
- Risk-based color coding for security awareness
- Subgraph support for module organization
- Dependency edge styling (solid vs dashed)
- Label truncation to max length
- Method count display in class labels
- State variable count in labels
- Direction control (TB, LR, RL, BT)

#### API Surface & Entry Points

**Public Functions:**

- `generate_mermaid(ast, diagram_type, direction, **kwargs) -> str` (lines 240-265): Convenience function

**Configuration**:

- `MermaidConfig` with diagram type, direction, visibility toggles, color options

#### Testing & Quality

**Test Files**:

- `tests/unit/test_mermaid.py`
- `tests/test_visualization_export.py`

**Coverage Observations**:

- All diagram types covered
- Risk color coding tested
- Label formatting tested

**Known Issues/TODOs**:

- Line 122-123: Parameter access assumes `parameters` attribute (should be `params`)
- Could add more shape variety
- Sequence diagrams not yet supported

---

## BATCH 3 - INTEGRATION (CLI & Ecosystem)

### 12. CLI TOOLS

#### Subsystem Overview

- **Name**: Command-Line Interface
- **Location**: `src/pyshort/cli/*.py` (9 files)
- **Primary Responsibility**: Provides user-facing commands for parsing, linting, formatting, visualization, and decompilation
- **Confidence Level**: High

#### Key Components

**Main Entry Point:**

- `src/pyshort/cli/main.py` (90 lines): Argument parsing and command dispatch

**Command Modules:**

- `parse.py`: Parse PyShorthand → AST → JSON
- `lint.py`: Validate and lint files
- `format.py`: Auto-format files
- `viz.py`: Generate diagrams
- `py2short.py` / `decompile.py`: Python → PyShorthand conversion
- `index.py`: Repository indexing

**Key Patterns:**

- Each command has `*_command(args) -> int` function
- Uses argparse for CLI parsing
- Returns exit codes (0=success, non-zero=error)

**Data Structures:**

- Arguments from argparse
- Delegates to subsystems

#### Dependencies

**Inbound** (Who uses it):

- End users via `pyshort` CLI
- CI/CD pipelines
- Build scripts

**Outbound** (What it uses):

- Parser subsystem (`parse_file`, `parse_string`)
- Validator subsystem (`validate_file`)
- Formatter subsystem (`format_file`, `format_string`)
- Visualization subsystem (`generate_mermaid`)
- Decompiler subsystem (`decompile_file`)
- Indexer subsystem (`index_repository`)
- Python stdlib: `argparse`, `sys`

**External**: None (stdlib only)

#### Architectural Patterns Observed

**Design Patterns:**

- **Command Pattern**: Each CLI command is independent
- **Facade Pattern**: CLI facades over complex subsystems
- **Strategy Pattern**: Different output formats (JSON, text, diff)

**Coding Conventions:**

- Consistent `*_command(args)` naming
- Exit code convention (0=success)
- Lazy imports for faster startup

**Error Handling:**

- Try/except around subsystem calls
- Diagnostic output to stderr
- Exit codes for automation

**Notable Implementation:**

- Subcommand structure via argparse
- Lazy loading of subsystems
- Version command
- Format diff mode (--diff)
- Format check mode (--check)
- Strict mode for linting
- JSON output for linting
- Multiple diagram types for viz
- In-place formatting (--write)

#### API Surface & Entry Points

**CLI Commands:**

- `pyshort parse <file> [-o output.json] [--pretty]`
- `pyshort lint <file|dir> [--strict] [--json]`
- `pyshort fmt <file|dir> [-w] [--check] [--diff]`
- `pyshort viz <file> [-o output] [-t flowchart|classDiagram|graph] [-d TB|LR]`
- `pyshort version`

**Configuration**:

- Reads from `.pyshortrc` via config subsystem

#### Testing & Quality

**Test Files**:

- Integration tests exercise CLI commands

**Coverage Observations**:

- All major commands covered
- Error paths tested

**Known Issues/TODOs**:

- Could add batch processing
- Watch mode not implemented
- Language server protocol (LSP) support planned

---

### 13. ECOSYSTEM

#### Subsystem Overview

- **Name**: Progressive Disclosure System
- **Location**: `src/pyshort/ecosystem/tools.py` (699 lines)
- **Primary Responsibility**: Provides on-demand code exploration with PyShorthand overview + full Python implementation retrieval
- **Confidence Level**: High

#### Key Components

**Classes:**

- `MethodImplementation(dataclass)` (lines 22-31): Full method source + deps
- `ClassDetails(dataclass)` (lines 34-41): Detailed class info
- `CodebaseExplorer` (lines 44-688): Main exploration API

**Key Methods:**

- `get_implementation(target, include_context) -> str` (lines 62-106): Retrieve Python source
- `get_class_details(class_name, include_methods, expand_nested) -> str` (lines 108-185): Detailed class info
- `search_usage(symbol) -> list[str]` (lines 187-234): Find symbol usage
- `get_context_pack(target, max_depth, include_peers) -> dict` (lines 236-278): Dependency layers
- `trace_execution(entry_point, max_depth, follow_calls) -> dict` (lines 280-319): Call flow
- `get_module_pyshorthand() -> str` (lines 351-392): Full module as PyShorthand
- `get_class_pyshorthand(class_name) -> str` (lines 394-455): Single class as PyShorthand

**Data Structures:**

- Cache: `dict[str, str]` for implementations
- AST cache: `dict[Path, ast.Module]`

**Constants:**

- `_ADVANCED_TOOLS_AVAILABLE` flag for optional features

#### Dependencies

**Inbound** (Who uses it):

- LLM context generation
- IDE integrations
- Documentation tools

**Outbound** (What it uses):

- Context Pack subsystem (`ContextPackGenerator`)
- Execution Flow subsystem (`ExecutionFlowTracer`)
- Decompiler subsystem (`PyShorthandGenerator`)
- Parser subsystem (`parse_string`)
- Python stdlib: `ast`, `dataclasses`, `pathlib`, `re`

**External**: None (stdlib only)

#### Architectural Patterns Observed

**Design Patterns:**

- **Facade Pattern**: Unified API over multiple subsystems
- **Cache Pattern**: Implementation caching for performance
- **Progressive Disclosure**: Lightweight PyShorthand → on-demand Python
- **Factory Pattern**: Creates analyzers on demand

**Coding Conventions:**

- Public API methods without underscore
- Private helpers with underscore prefix
- Caching for expensive operations
- Graceful degradation if advanced tools unavailable

**Error Handling:**

- Returns `None` if target not found
- Skips files with parse errors
- Handles missing implementations gracefully

**Notable Implementation:**

- Two-tier system: PyShorthand overview + Python details
- Method implementation extraction from AST
- Dependency tracking in method calls
- Class detail expansion (nested ModuleDict, etc.)
- Symbol usage search across codebase
- Context pack integration (F0/F1/F2 layers)
- Execution flow tracing integration
- PyShorthand generation per-class or per-module
- Neighbor queries (callers + callees)
- Caching for performance

#### API Surface & Entry Points

**Public Methods:**

- `CodebaseExplorer(codebase_path: Path)` - Constructor
- `get_implementation(target: str) -> str | None` - Python source
- `get_class_details(class_name: str) -> str | None` - Class info
- `search_usage(symbol: str) -> list[str]` - Symbol search
- `get_context_pack(target: str) -> dict | None` - Dependency layers
- `trace_execution(entry_point: str) -> dict | None` - Call flow
- `get_neighbors(symbol: str) -> dict | None` - Direct deps
- `get_module_pyshorthand() -> str | None` - Full module
- `get_class_pyshorthand(class_name: str) -> str | None` - Single class

**Configuration**: None (codebase path only)

#### Testing & Quality

**Test Files**:

- Not found in scan (integration tests likely)

**Coverage Observations**:

- Comprehensive exploration API
- Caching implemented
- Graceful degradation

**Known Issues/TODOs**:

- Line 687: `_find_parent_context()` not implemented (TODO)
- Could add more caching strategies
- Incremental updates not supported

---

## CROSS-CUTTING ANALYSIS

### Dependency Graph Summary

```
Core Foundation:
  Symbols → (standalone)
  AST Nodes → Symbols
  Tokenizer → (standalone)

Parsing Pipeline:
  Parser → Tokenizer, AST Nodes, Symbols

Validation:
  Validator → Parser, AST Nodes, Symbols, Enhanced Errors
  Enhanced Errors → Symbols
  Error Codes → (standalone)

Transformation:
  Decompiler → (Python AST only)
  Formatter → Parser, AST Nodes, Symbols

Analysis:
  Context Analyzer → AST Nodes
  Execution Analyzer → AST Nodes
  Indexer → Decompiler
  Visualization → AST Nodes

Integration:
  CLI → All subsystems
  Ecosystem → Context, Execution, Decompiler, Parser
  Config → (standalone)
```

### Quality Observations

**Strengths:**

- Zero external dependencies (stdlib only)
- Comprehensive test coverage (14+ test files)
- Immutable data structures (frozen dataclasses)
- Consistent error handling patterns
- Good separation of concerns

**Areas for Improvement:**

- Some circular dependency potential (Parser ↔ AST)
- Test coverage could be more explicit for edge cases
- Documentation in code is good, API docs could be enhanced
- Some TODO markers indicate incomplete features

### Code Metrics Summary

**Total Lines of Code:** ~8,000+ lines

- Core Pipeline: ~3,200 lines
- Transformation & Analysis: ~4,000 lines
- Integration: ~800 lines

**Test Coverage:**

- Unit tests: 10+ files
- Integration tests: 5+ files
- Coverage appears extensive but not quantified

**Complexity:**

- Tokenizer: High (548 lines, 60+ token types)
- Parser: Very High (1253 lines, recursive descent)
- Decompiler: Very High (1143 lines, pattern matching)
- Other subsystems: Moderate to High

---

## CONCLUSION

This deep analysis reveals a well-architected PyShorthand toolchain with:

1. **Clear layering**: Core → Transformation → Analysis → Integration
2. **Strong boundaries**: Minimal coupling between subsystems
3. **Consistent patterns**: Dataclasses, visitor, strategy, fluent interfaces
4. **Quality focus**: Immutability, validation, error recovery
5. **Extensibility**: New analyzers, validators, and formatters easily added

The codebase demonstrates professional engineering with thoughtful design decisions, comprehensive error handling, and forward-looking features (0.9.0-RC1 support). The progressive disclosure system (Ecosystem) is particularly innovative for LLM integration.
