# Architecture Diagrams - PyShorthand 0.9.0-RC1

**Date:** 2025-11-24
**Diagram Notation:** C4 Model + Mermaid
**Audience:** Technical stakeholders, architects, developers
**Version:** 0.9.0-RC1

## Overview

This document provides multi-level architectural views of the PyShorthand system using the C4 model (Context, Container, Component). PyShorthand is a sophisticated intermediate representation (IR) system for Python codebases, designed to provide high-density compression optimized for LLM consumption with progressive disclosure.

**Key Architectural Characteristics:**
- **Zero-dependency core** enabling embeddability and portability
- **Immutable AST design** for thread-safety and functional programming
- **Progressive disclosure** system (93% token savings, 90% accuracy)
- **Clean 3-layer architecture** with perfect dependency direction

---

## Level 1: System Context Diagram

### Purpose
Shows how PyShorthand fits into the broader ecosystem and who/what interacts with it.

### Diagram

```mermaid
C4Context
    title System Context - PyShorthand Ecosystem

    Person(cli_user, "CLI User", "Developer using PyShorthand tools for code analysis")
    Person(llm_system, "LLM System", "AI assistant consuming compressed code representations")
    Person(ide_user, "IDE User", "Developer with PyShorthand integration in editor")

    System(pyshorthand, "PyShorthand", "Intermediate representation system for Python codebases with progressive disclosure")

    System_Ext(python_code, "Python Codebases", "Source code repositories to be analyzed and compressed")
    System_Ext(ide, "IDE/Editor", "VSCode, PyCharm with potential LSP integration")
    System_Ext(docs_gen, "Documentation Systems", "Mermaid, GraphViz diagram renderers")
    System_Ext(ci_cd, "CI/CD Pipeline", "GitHub Actions, GitLab CI for code quality checks")

    Rel(cli_user, pyshorthand, "Uses", "CLI commands: parse, lint, fmt, viz")
    Rel(llm_system, pyshorthand, "Consumes", "PyShorthand overview + on-demand Python details")
    Rel(ide_user, ide, "Edits code in")
    Rel(ide, pyshorthand, "Integrates", "Future LSP/plugin support")

    Rel(pyshorthand, python_code, "Reads & decompiles", "Python AST → PyShorthand")
    Rel(pyshorthand, docs_gen, "Generates", "Mermaid/GraphViz diagrams")
    Rel(ci_cd, pyshorthand, "Invokes", "Linting, validation, formatting checks")

    UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="2")
```

### Key Relationships

**User Interactions:**
- **CLI User → PyShorthand:** Direct command-line usage for parsing, linting, formatting, visualization, and decompilation
- **LLM System → PyShorthand:** Two-tier progressive disclosure pattern - starts with compressed PyShorthand, requests full Python implementations on-demand
- **IDE User → IDE → PyShorthand:** Future integration path for real-time linting and formatting (LSP planned)

**System Interactions:**
- **PyShorthand → Python Codebases:** Bidirectional - reads Python for decompilation, can validate and analyze existing .pys files
- **PyShorthand → Documentation Systems:** Generates Mermaid and GraphViz diagrams for architectural documentation
- **CI/CD → PyShorthand:** Automated quality gates using lint and format commands

### Context Boundaries

**PyShorthand System Boundary:**
- Core parsing and validation (zero dependencies)
- Code transformation (Python ↔ PyShorthand)
- Analysis tools (context packs, execution flow)
- Progressive disclosure API for LLM integration

**External Systems:**
- Python repositories (input)
- Visualization renderers (output)
- Development tools (integration)

---

## Level 2: Container Diagram

### Purpose
Shows the major deployable/executable components within PyShorthand and their communication patterns.

### Diagram

```mermaid
C4Container
    title Container Diagram - PyShorthand Internal Architecture

    Person(user, "User", "Developer or LLM")

    System_Boundary(pyshorthand, "PyShorthand System") {
        Container(cli, "CLI Tools", "Python", "Command-line interface: parse, lint, fmt, viz, decompile commands")

        Container(core, "Core Library", "Python 3.10+ (zero-dep)", "Tokenizer → Parser → AST → Validator pipeline")

        Container(transform, "Transformation Layer", "Python", "Decompiler (Python→PyShorthand) and Formatter")

        Container(analysis, "Analysis Tools", "Python", "Context Pack Generator, Execution Flow Tracer, Repository Indexer")

        Container(ecosystem, "Ecosystem API", "Python", "Progressive disclosure interface for LLM integration")

        Container(viz, "Visualization", "Python + Mermaid", "Diagram generation (Mermaid/GraphViz)")
    }

    System_Ext(python_src, "Python Source Files", ".py files")
    System_Ext(pys_files, "PyShorthand Files", ".pys files")
    System_Ext(llm, "LLM Context Window", "AI assistant")
    System_Ext(diagram_tools, "Diagram Renderers", "Mermaid/GraphViz")

    Rel(user, cli, "Invokes", "shell commands")

    Rel(cli, core, "Uses", "parse/validate .pys files")
    Rel(cli, transform, "Uses", "format/decompile operations")
    Rel(cli, analysis, "Uses", "index repositories")
    Rel(cli, viz, "Uses", "generate diagrams")

    Rel(transform, core, "Depends on", "AST structures")
    Rel(analysis, core, "Depends on", "AST structures")
    Rel(viz, core, "Depends on", "AST structures")

    Rel(ecosystem, core, "Uses", "parse PyShorthand")
    Rel(ecosystem, transform, "Uses", "decompile Python")
    Rel(ecosystem, analysis, "Uses", "context packs, execution flow")

    Rel(llm, ecosystem, "Calls", "get_implementation(), get_class_details()")

    Rel(core, pys_files, "Reads/Validates")
    Rel(transform, python_src, "Reads", "Decompile Python→PyShorthand")
    Rel(viz, diagram_tools, "Outputs to", "Mermaid/DOT syntax")

    UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```

### Container Descriptions

#### **CLI Tools**
- **Technology:** Python 3.10+, argparse, optional (click, rich)
- **Responsibilities:** User-facing command-line interface
- **Commands:** parse, lint, fmt, viz, py2short, pyshort-index
- **Deployment:** Installed via pip/uv as console scripts
- **Dependencies:** All other containers
- **Scale:** Single-process, invoked per-command

#### **Core Library**
- **Technology:** Python 3.10+ stdlib only (zero external dependencies)
- **Responsibilities:**
  - Lexical analysis (Tokenizer - 547 LOC)
  - Syntax analysis (Parser - 1,252 LOC)
  - AST data structures (727 LOC)
  - Semantic validation (Validator - 631 LOC)
  - Symbol mappings (Symbols - 230 LOC)
- **Key Quality:** 100% type hint coverage, immutable frozen dataclasses
- **Deployment:** Embeddable library, portable across environments
- **Performance:** O(n) parsing, single-pass tokenization

#### **Transformation Layer**
- **Technology:** Python 3.10+ with Python `ast` module
- **Responsibilities:**
  - Decompiler (Python AST → PyShorthand - 1,142 LOC)
  - Formatter (PyShorthand pretty-printing - 417 LOC)
  - Type inference and tag extraction
- **Key Features:** Framework detection (Pydantic, FastAPI, PyTorch), aggressive type inference
- **Deployment:** Library modules invoked by CLI or Ecosystem

#### **Analysis Tools**
- **Technology:** Python 3.10+ with optional networkx, graphviz
- **Responsibilities:**
  - Context Pack Generator (F0/F1/F2 dependency layers)
  - Execution Flow Tracer (runtime path analysis)
  - Repository Indexer (cross-file analysis)
- **Key Features:** Filtering APIs, Mermaid export, dependency graphs
- **Deployment:** Library modules for programmatic use

#### **Ecosystem API**
- **Technology:** Python 3.10+ facade over all layers
- **Responsibilities:** Progressive disclosure for LLM integration
- **Methods:** `get_implementation()`, `get_class_details()`, `search_usage()`, `get_context_pack()`, `trace_execution()`
- **Key Innovation:** 93% token savings vs full code, 90% accuracy (empirical)
- **Deployment:** Library facade, caching enabled

#### **Visualization**
- **Technology:** Python 3.10+ generating Mermaid/GraphViz syntax
- **Responsibilities:** Generate diagrams from AST
- **Formats:** Flowcharts, class diagrams, dependency graphs
- **Key Features:** Risk-based coloring, subgraph organization
- **Deployment:** Invoked by CLI or integrated in documentation

### Container Communication Patterns

**Dependency Flow:**
```
CLI ─────────────┐
                 ↓
Ecosystem ───────┼────→ Core (Foundation)
                 │        ↑
Transform ───────┤        │
                 │        │
Analysis ────────┤        │
                 │        │
Visualization ───┘        │
                          │
All depend on Core ───────┘
```

**Data Flow:**
1. **Parse Flow:** User → CLI → Core (Tokenizer → Parser → AST) → JSON output
2. **Decompile Flow:** User → CLI → Transform (Decompiler reads Python AST) → Core (generates PyShorthand AST) → output
3. **Progressive Disclosure Flow:** LLM → Ecosystem → Context Pack (lightweight) → on-demand → Transform (full Python)

---

## Level 3: Component Diagram - Core Library

### Purpose
Detailed view of the Core Library components and their interactions (the zero-dependency foundation).

### Diagram

```mermaid
flowchart TB
    subgraph core["Core Library Container (Zero Dependencies)"]
        direction TB

        symbols["<b>Symbols</b><br/>231 LOC<br/>━━━━━━━━━<br/>Unicode↔ASCII mappings<br/>VALID_TYPES, VALID_TAGS<br/>Constants & utilities"]

        tokenizer["<b>Tokenizer</b><br/>547 LOC<br/>━━━━━━━━━<br/>Lexical analysis<br/>60+ token types<br/>Unicode/ASCII support"]

        ast["<b>AST Nodes</b><br/>727 LOC<br/>━━━━━━━━━<br/>Frozen dataclasses<br/>PyShortAST, Entity<br/>Class, Function, StateVar<br/>Immutable design"]

        parser["<b>Parser</b><br/>1,253 LOC<br/>━━━━━━━━━<br/>Recursive descent<br/>Entity, type, expression parsing<br/>Error recovery"]

        validator["<b>Validator</b><br/>632 LOC<br/>━━━━━━━━━<br/>14 validation rules<br/>Metadata, tags, safety checks<br/>Suggestion engine"]

        errors["<b>Enhanced Errors</b><br/>102 LOC<br/>━━━━━━━━━<br/>Diagnostic dataclass<br/>Severity levels<br/>Did-you-mean suggestions"]

        config["<b>Config</b><br/>181 LOC<br/>━━━━━━━━━<br/>.pyshortrc loader<br/>Format preferences<br/>Validation strictness"]
    end

    %% Dependencies
    tokenizer -->|List[Token]| parser
    parser -->|uses| symbols
    parser -->|constructs| ast
    parser -->|raises| errors

    validator -->|analyzes| ast
    validator -->|validates against| symbols
    validator -->|generates| errors

    ast -.->|validates tags with| symbols
    errors -.->|suggests from| symbols

    config -.->|configures| validator
    config -.->|configures format| parser

    %% Styling
    style symbols fill:#e1f5e1,stroke:#2d5016,stroke-width:2px
    style tokenizer fill:#e1f5e1,stroke:#2d5016,stroke-width:2px
    style ast fill:#ffe1e1,stroke:#8b0000,stroke-width:2px
    style parser fill:#e1f5e1,stroke:#2d5016,stroke-width:2px
    style validator fill:#e1f5e1,stroke:#2d5016,stroke-width:2px
    style errors fill:#fff3cd,stroke:#856404,stroke-width:2px
    style config fill:#d1ecf1,stroke:#0c5460,stroke-width:2px
```

### Component Descriptions

#### **Symbols (Foundation)**
- **Responsibility:** Centralized symbol definitions and conversion utilities
- **Key Exports:**
  - `UNICODE_TO_ASCII`, `ASCII_TO_UNICODE` mappings
  - `VALID_TAG_BASES`, `VALID_TYPES`, `VALID_LOCATIONS`, `HTTP_METHODS`
  - `to_ascii()`, `to_unicode()` converters
  - `is_complexity_tag()`, `parse_http_route()` validators
- **Dependencies:** None (pure data + stdlib re)
- **Design Pattern:** Constant repository

#### **Tokenizer (Input Layer)**
- **Responsibility:** Convert raw text into token stream
- **Key Features:**
  - 60+ token types (keywords, operators, literals, delimiters)
  - Bidirectional Unicode/ASCII support (→ ↔ ->, ∈ ↔ IN)
  - Numeric range validation (i64: ±2^63, f64: ±3.4e38)
  - String escape sequences with multiline support
- **Output:** `List[Token]` with position tracking (line, column)
- **Dependencies:** None (stdlib only)
- **Performance:** O(n) single-pass lexing

#### **AST Nodes (Data Structures)**
- **Responsibility:** Define immutable representation of parsed code
- **Key Types:**
  - `PyShortAST` (root document with metadata, entities, diagnostics)
  - `Metadata`, `Entity`, `Class`, `Function`, `StateVar`, `Parameter`
  - `TypeSpec` (with generics, unions, references)
  - `Statement`, `Expression` (BinaryOp, FunctionCall, etc.)
  - `Tag` (operation, complexity, decorator, http_route)
  - `Diagnostic` (severity, location, suggestion)
- **Design:** Frozen dataclasses with `to_dict()` serialization
- **Benefits:** Thread-safe, hashable, GC-friendly, functional transformations
- **Dependencies:** Symbols (for tag validation)

#### **Parser (Syntax Analysis)**
- **Responsibility:** Build AST from token stream
- **Algorithm:** Recursive descent with lookahead
- **Key Methods:**
  - `parse()` - Top-level orchestration
  - `parse_class()` - Classes with generics, protocols, inheritance
  - `parse_function()` - Functions with contracts
  - `parse_type_spec()` - Complex types (unions, generics, references)
  - `parse_expression()` - Expressions with precedence
- **Error Handling:** Accumulates diagnostics, continues parsing (recovery)
- **Dependencies:** Tokenizer (input), AST Nodes (output), Symbols (constants)
- **Complexity:** 1,252 LOC, 29 methods (highest complexity in core)

#### **Validator (Semantic Analysis)**
- **Responsibility:** Enforce grammar and best practices
- **Architecture:** Rule-based system with 14 independent rules
- **Key Rules:**
  - Mandatory metadata (M:Name, Role)
  - Valid tag bases (Lin, Thresh, IO, NN, etc.)
  - Complexity notation (O(N), Θ(N²))
  - HTTP route syntax
  - Safety checks (!! requires [Risk:High])
  - Generic parameter validity
  - Inheritance validity
- **Output:** `List[Diagnostic]` with severity (ERROR, WARNING, INFO, HINT)
- **Features:** "Did you mean?" suggestions via Levenshtein distance
- **Dependencies:** AST Nodes (input), Symbols (validation sets), Enhanced Errors (diagnostics)

#### **Enhanced Errors (Quality of Life)**
- **Responsibility:** Rich error reporting
- **Features:**
  - `Diagnostic` dataclass with line, column, severity
  - Suggestion generation (`suggest_did_you_mean()`)
  - Error code system (E001-E399, W001-W399)
- **Dependencies:** Symbols (for suggestion sets)

#### **Config (Configuration)**
- **Responsibility:** Load and manage `.pyshortrc` configuration
- **Settings:**
  - Format preferences (indent, Unicode/ASCII, alignment)
  - Validation strictness
  - Linting rules enablement
- **Dependencies:** None (standalone)

### Data Flow Through Core

```mermaid
sequenceDiagram
    participant User
    participant Parser
    participant Tokenizer
    participant AST
    participant Validator
    participant Errors

    User->>Parser: parse_file("example.pys")
    Parser->>Tokenizer: tokenize(source_code)
    Tokenizer->>Tokenizer: lexical analysis
    Tokenizer-->>Parser: List[Token]

    Parser->>Parser: recursive descent parsing
    Parser->>Symbols: validate against VALID_*
    Parser->>AST: construct PyShortAST
    AST->>AST: validate tag structure
    AST-->>Parser: PyShortAST object

    Parser->>Validator: check(ast)
    Validator->>Validator: run 14 rules
    Validator->>Symbols: check against constants
    Validator->>Errors: generate Diagnostics
    Errors-->>Validator: List[Diagnostic]
    Validator-->>Parser: List[Diagnostic]

    Parser-->>User: PyShortAST + Diagnostics
```

---

## Level 3: Component Diagram - Transformation Layer

### Purpose
Detailed view of Python ↔ PyShorthand transformation components.

### Diagram

```mermaid
flowchart TB
    subgraph transform["Transformation Layer Container"]
        direction TB

        decompiler["<b>Decompiler (py2short)</b><br/>1,142 LOC<br/>━━━━━━━━━<br/>Python AST → PyShorthand<br/>Type inference engine<br/>Framework detection<br/>(Pydantic, FastAPI, PyTorch)<br/>Tag extraction (NN, IO, Iter)"]

        formatter["<b>Formatter</b><br/>417 LOC<br/>━━━━━━━━━<br/>PyShorthand pretty-printing<br/>Alignment & sorting<br/>Unicode/ASCII preference<br/>Blank line control"]

        type_inference["<b>Type Inference</b><br/>Subsystem of Decompiler<br/>━━━━━━━━━<br/>Infer types from Python AST<br/>Handle Optional, Union, List<br/>Confidence scoring"]

        tag_extractor["<b>Tag Extractor</b><br/>Subsystem of Decompiler<br/>━━━━━━━━━<br/>Decorator tags (@property)<br/>HTTP routes (@app.get)<br/>Operation tags (NN:∇, IO:File)<br/>Complexity (O(N), O(N²))"]

        framework_detector["<b>Framework Detector</b><br/>Subsystem of Decompiler<br/>━━━━━━━━━<br/>Pydantic BaseModel<br/>FastAPI routing<br/>PyTorch nn.Module<br/>Django models"]
    end

    subgraph core_ref["Core Library (Reference)"]
        parser_ref["Parser"]
        ast_ref["AST Nodes"]
        symbols_ref["Symbols"]
    end

    subgraph python_ast["Python stdlib"]
        py_ast["ast.Module<br/>Python AST"]
    end

    %% Decompiler internals
    decompiler -->|uses| type_inference
    decompiler -->|uses| tag_extractor
    decompiler -->|uses| framework_detector

    %% External dependencies
    decompiler -->|reads| py_ast
    decompiler -->|generates| ast_ref
    decompiler -.->|uses constants| symbols_ref

    formatter -->|reads| ast_ref
    formatter -->|uses| parser_ref
    formatter -.->|converts symbols| symbols_ref

    %% Styling
    style decompiler fill:#d4edff,stroke:#004085,stroke-width:2px
    style formatter fill:#d4edff,stroke:#004085,stroke-width:2px
    style type_inference fill:#e7f3ff,stroke:#004085,stroke-width:1px
    style tag_extractor fill:#e7f3ff,stroke:#004085,stroke-width:1px
    style framework_detector fill:#e7f3ff,stroke:#004085,stroke-width:1px
    style py_ast fill:#f8f9fa,stroke:#6c757d,stroke-width:1px
```

### Component Descriptions

#### **Decompiler (Python → PyShorthand)**
- **Responsibility:** Reverse-engineer Python AST into PyShorthand notation
- **Key Features:**
  - Class/function/module extraction from Python AST
  - Type annotation conversion (Python → PyShorthand types)
  - State variable extraction
  - Import dependency tracking
  - Aggressive vs conservative inference modes
  - Confidence score annotations
- **Algorithms:**
  - AST traversal with visitor pattern
  - Pattern matching for operations (loops → Iter, gradients → NN:∇)
  - Framework-specific heuristics
- **Output:** PyShorthand string or AST
- **Dependencies:** Python `ast` module, Symbols

#### **Type Inference (Subsystem)**
- **Responsibility:** Infer PyShorthand types from Python type hints and code patterns
- **Key Features:**
  - Handle `Optional[T]` → `T | None`
  - Convert `List[T]` → `[T]`
  - Map `Dict[K, V]` → `{K: V}`
  - Infer from assignments (aggressive mode)
  - Detect Pydantic field types
- **Output:** `TypeSpec` strings
- **Complexity:** 18 branches (high)

#### **Tag Extractor (Subsystem)**
- **Responsibility:** Detect and convert Python decorators/patterns to PyShorthand tags
- **Mapping:**
  - `@property` → `[Prop]`
  - `@staticmethod` → `[Static]`
  - `@app.get("/path")` → `[GET:/path]`
  - `torch.backward()` → `[NN:∇]`
  - Loop nesting → `[O(N²)]`
- **Pattern Detection:** AST pattern matching for common idioms
- **Output:** Tag strings

#### **Framework Detector (Subsystem)**
- **Responsibility:** Identify and handle framework-specific patterns
- **Supported Frameworks:**
  - **Pydantic:** BaseModel fields, validators
  - **FastAPI:** Route decorators, dependencies
  - **PyTorch:** nn.Module, layers, ModuleDict
  - **Django:** Models, querysets
  - **Flask:** Route decorators
- **Impact:** Enables framework-aware tag generation

#### **Formatter (PyShorthand Pretty-Printing)**
- **Responsibility:** Format PyShorthand code with consistent style
- **Key Features:**
  - Configurable indentation (default 2 spaces)
  - Vertical alignment of type annotations
  - State variable sorting (by location: GPU → CPU → Disk)
  - Blank line control around entities
  - Unicode/ASCII preference toggle
  - Tag grouping (Decorators → Routes → Operations → Complexity)
- **Input:** PyShortAST
- **Output:** Formatted PyShorthand string
- **Dependencies:** Parser (for re-parsing), AST Nodes, Symbols

### Decompilation Flow

```mermaid
sequenceDiagram
    participant User
    participant Decompiler
    participant PythonAST
    participant TypeInference
    participant TagExtractor
    participant FrameworkDetector
    participant PyShortAST

    User->>Decompiler: decompile_file("module.py")
    Decompiler->>PythonAST: ast.parse(source)
    PythonAST-->>Decompiler: Python AST tree

    Decompiler->>FrameworkDetector: detect_framework(ast)
    FrameworkDetector-->>Decompiler: "pydantic" / "pytorch" / None

    loop For each ClassDef
        Decompiler->>TypeInference: infer_type(field)
        TypeInference-->>Decompiler: TypeSpec

        Decompiler->>TagExtractor: extract_decorator_tags(class)
        TagExtractor-->>Decompiler: List[Tag]
    end

    loop For each FunctionDef
        Decompiler->>TypeInference: convert_annotation(func)
        TypeInference-->>Decompiler: TypeSpec

        Decompiler->>TagExtractor: extract_operation_tags(func)
        TagExtractor-->>Decompiler: List[Tag]

        Decompiler->>TagExtractor: extract_complexity_tag(func)
        TagExtractor-->>Decompiler: Tag | None
    end

    Decompiler->>PyShortAST: construct PyShorthand
    PyShortAST-->>Decompiler: PyShorthand string
    Decompiler-->>User: PyShorthand output
```

---

## Level 3: Component Diagram - Analysis Layer

### Purpose
Detailed view of analysis tools (Context Pack, Execution Flow, Indexer, Visualization).

### Diagram

```mermaid
flowchart TB
    subgraph analysis["Analysis Tools Container"]
        direction TB

        context["<b>Context Pack Generator</b><br/>579 LOC<br/>━━━━━━━━━<br/>F0/F1/F2 dependency layers<br/>Bidirectional graph<br/>Filtering API<br/>Mermaid/GraphViz export"]

        execution["<b>Execution Flow Tracer</b><br/>617 LOC<br/>━━━━━━━━━<br/>Runtime path analysis<br/>Call graph traversal<br/>Variables in scope<br/>Depth-based visualization"]

        indexer["<b>Repository Indexer</b><br/>519 LOC<br/>━━━━━━━━━<br/>Scan entire repositories<br/>Module dependency graph<br/>Entity extraction<br/>Statistics generation"]

        viz["<b>Visualization</b><br/>266 LOC<br/>━━━━━━━━━<br/>Mermaid generator<br/>Flowcharts, class diagrams<br/>Risk-based coloring<br/>Subgraph support"]
    end

    subgraph core_ref2["Core Library (Reference)"]
        ast_ref2["AST Nodes"]
    end

    subgraph transform_ref["Transformation (Reference)"]
        decompiler_ref["Decompiler"]
    end

    %% Dependencies
    context -->|analyzes| ast_ref2
    execution -->|analyzes| ast_ref2
    viz -->|generates from| ast_ref2

    indexer -->|uses| decompiler_ref
    indexer -.->|generates| ast_ref2

    %% Internal
    context -.->|exports via| viz
    execution -.->|exports via| viz

    %% Styling
    style context fill:#fff3cd,stroke:#856404,stroke-width:2px
    style execution fill:#fff3cd,stroke:#856404,stroke-width:2px
    style indexer fill:#fff3cd,stroke:#856404,stroke-width:2px
    style viz fill:#d1ecf1,stroke:#0c5460,stroke-width:2px
```

### Component Descriptions

#### **Context Pack Generator**
- **Responsibility:** Generate dependency-aware context for LLM consumption
- **Key Concepts:**
  - **F0 Layer:** Target entity (the focus)
  - **F1 Layer:** Direct dependencies (1-hop)
  - **F2 Layer:** Transitive dependencies (2-hop)
- **Key Features:**
  - Bidirectional graph (callers + callees)
  - Class peer detection (methods in same class)
  - State variable tracking across layers
  - Filtering API:
    - `filter_by_location(location)` - Filter by @GPU/@CPU
    - `filter_by_pattern(regex)` - Regex filtering
    - `filter_custom(predicate)` - Custom predicates
  - Export: Mermaid flowcharts, GraphViz DOT
- **Use Case:** Feed minimal context to LLM for targeted questions
- **Output:** `ContextPack` dataclass with F0/F1/F2 lists
- **Dependencies:** AST Nodes

#### **Execution Flow Tracer**
- **Responsibility:** Trace runtime call paths (vs static dependencies)
- **Key Features:**
  - DFS traversal with cycle detection
  - Depth tracking (max configurable, default 10)
  - Variables in scope per step
  - State variable access tracking
  - Call graph construction
  - Filtering API:
    - `filter_by_depth(max_depth)`
    - `filter_by_state_access(pattern)`
    - `filter_by_call_pattern(regex)`
  - Export: Mermaid flowcharts with depth-based colors, GraphViz
- **Use Case:** Understand actual execution paths at runtime
- **Output:** `ExecutionFlow` dataclass with steps
- **Dependencies:** AST Nodes

#### **Repository Indexer**
- **Responsibility:** Scan entire Python repositories for analysis
- **Key Features:**
  - Recursive directory scanning with exclusions (venv, __pycache__, .git)
  - Module path normalization (handles src/ directories)
  - Dependency graph construction (module-level)
  - Entity extraction (classes, functions) from each module
  - Statistics: LOC, entity counts, top modules
  - PyShorthand generation for each module
  - Mermaid dependency graph export
  - JSON serialization for caching
- **Use Case:** Repository-wide analysis, documentation generation
- **Output:** `RepositoryIndex` dataclass with module map, dependency graph, stats
- **Dependencies:** Decompiler (for Python→PyShorthand)

#### **Visualization**
- **Responsibility:** Generate Mermaid diagrams from AST
- **Diagram Types:**
  - **Flowchart:** Dataflow and dependencies
  - **Class Diagram:** UML-like structure with state/methods
  - **Graph:** Simple relationship graph
- **Key Features:**
  - Risk-based color coding (High=#ff6b6b, Med=#ffd93d, Low=#6bcf7f)
  - Subgraph support for module organization
  - Direction control (TB, LR, RL, BT)
  - Label truncation for readability
  - Method/state variable counts in labels
- **Configuration:** `MermaidConfig` with 7 options
- **Output:** Mermaid markdown syntax
- **Dependencies:** AST Nodes

### Context Pack Generation Flow

```mermaid
sequenceDiagram
    participant User
    participant ContextPackGen
    participant AST
    participant Graph
    participant Filter
    participant Mermaid

    User->>ContextPackGen: generate_context_pack(module, "TargetClass.method", depth=2)

    ContextPackGen->>AST: find entity "TargetClass.method"
    AST-->>ContextPackGen: Function node (F0)

    ContextPackGen->>Graph: build dependency graph
    Graph->>AST: traverse statements, extract calls/refs
    Graph-->>ContextPackGen: bidirectional graph

    ContextPackGen->>Graph: traverse 1-hop from F0
    Graph-->>ContextPackGen: F1 entities (direct deps)

    ContextPackGen->>Graph: traverse 1-hop from F1
    Graph-->>ContextPackGen: F2 entities (transitive deps)

    ContextPackGen-->>User: ContextPack(F0, F1, F2)

    alt User filters by GPU
        User->>Filter: filter_by_location("GPU")
        Filter->>ContextPackGen: filter entities
        Filter-->>User: Filtered ContextPack
    end

    alt User exports to Mermaid
        User->>Mermaid: to_mermaid(direction="TB")
        Mermaid->>Mermaid: generate flowchart syntax
        Mermaid-->>User: Mermaid markdown
    end
```

---

## Level 3: Component Diagram - Integration Layer

### Purpose
Detailed view of user-facing integration components (CLI, Ecosystem API).

### Diagram

```mermaid
flowchart TB
    subgraph integration["Integration Layer Container"]
        direction TB

        cli_main["<b>CLI Main</b><br/>90 LOC<br/>━━━━━━━━━<br/>Argument parser<br/>Command dispatcher<br/>Version & help"]

        cli_parse["<b>parse command</b><br/>Parse .pys → JSON/AST"]
        cli_lint["<b>lint command</b><br/>Validate & report errors"]
        cli_fmt["<b>fmt command</b><br/>Auto-format files"]
        cli_viz["<b>viz command</b><br/>Generate diagrams"]
        cli_decompile["<b>py2short command</b><br/>Decompile Python"]
        cli_index["<b>index command</b><br/>Scan repositories"]

        ecosystem["<b>Ecosystem API</b><br/>699 LOC<br/>━━━━━━━━━<br/>Progressive disclosure facade<br/>Caching layer<br/>LLM-optimized API"]

        ecosystem_methods["<b>Key Methods</b><br/>━━━━━━━━━<br/>get_implementation()<br/>get_class_details()<br/>search_usage()<br/>get_context_pack()<br/>trace_execution()"]
    end

    subgraph all_layers["All Other Containers (Reference)"]
        core_ref3["Core Library"]
        transform_ref2["Transformation"]
        analysis_ref["Analysis Tools"]
    end

    %% CLI dependencies
    cli_main -->|dispatches to| cli_parse
    cli_main -->|dispatches to| cli_lint
    cli_main -->|dispatches to| cli_fmt
    cli_main -->|dispatches to| cli_viz
    cli_main -->|dispatches to| cli_decompile
    cli_main -->|dispatches to| cli_index

    cli_parse -->|uses| core_ref3
    cli_lint -->|uses| core_ref3
    cli_fmt -->|uses| transform_ref2
    cli_viz -->|uses| analysis_ref
    cli_decompile -->|uses| transform_ref2
    cli_index -->|uses| analysis_ref

    %% Ecosystem dependencies
    ecosystem -->|contains| ecosystem_methods
    ecosystem -->|uses| core_ref3
    ecosystem -->|uses| transform_ref2
    ecosystem -->|uses| analysis_ref

    %% Styling
    style cli_main fill:#f8d7da,stroke:#721c24,stroke-width:2px
    style ecosystem fill:#d4edda,stroke:#155724,stroke-width:2px
    style ecosystem_methods fill:#e7f5e9,stroke:#155724,stroke-width:1px
    style cli_parse fill:#fef0e6,stroke:#721c24,stroke-width:1px
    style cli_lint fill:#fef0e6,stroke:#721c24,stroke-width:1px
    style cli_fmt fill:#fef0e6,stroke:#721c24,stroke-width:1px
    style cli_viz fill:#fef0e6,stroke:#721c24,stroke-width:1px
    style cli_decompile fill:#fef0e6,stroke:#721c24,stroke-width:1px
    style cli_index fill:#fef0e6,stroke:#721c24,stroke-width:1px
```

### Component Descriptions

#### **CLI Main (Command Dispatcher)**
- **Responsibility:** User-facing command-line interface
- **Technology:** argparse (stdlib) or optional click
- **Commands:**
  - `pyshort parse <file>` - Parse PyShorthand to JSON/AST
  - `pyshort lint <file>` - Validate and report errors
  - `pyshort fmt <file>` - Auto-format code
  - `pyshort viz <file>` - Generate diagrams
  - `py2short <file>` - Decompile Python to PyShorthand
  - `pyshort version` - Show version
- **Error Handling:** Exit codes (0=success, non-zero=error)
- **Dependencies:** All containers
- **Deployment:** Installed as console scripts via pip/uv

#### **Ecosystem API (Progressive Disclosure Facade)**
- **Responsibility:** Unified API for LLM integration with progressive disclosure pattern
- **Key Innovation:** Two-tier system
  - **Tier 1 (Cheap):** PyShorthand overview (894 tokens for nanoGPT)
  - **Tier 2 (On-demand):** Full Python implementation retrieval
- **Key Methods:**
  - `get_implementation(target)` → Full Python source for specific method
  - `get_class_details(class_name)` → Detailed class info with nested expansion
  - `search_usage(symbol)` → Find where symbol is used
  - `get_context_pack(target)` → F0/F1/F2 dependency layers
  - `trace_execution(entry_point)` → Call flow from entry point
  - `get_module_pyshorthand()` → Full module as PyShorthand
  - `get_class_pyshorthand(class_name)` → Single class as PyShorthand
- **Caching:** Implementation cache + AST cache for performance
- **Empirical Results:** 93% token savings, 90% accuracy vs full code
- **Dependencies:** All layers (facade pattern)
- **Use Case:** Feed PyShorthand to LLM, allow on-demand drill-down

### Progressive Disclosure Flow

```mermaid
sequenceDiagram
    participant LLM
    participant Ecosystem
    participant Parser
    participant ContextPack
    participant Decompiler
    participant Cache

    LLM->>Ecosystem: Initial request: "Explain nanoGPT"
    Ecosystem->>Parser: parse module PyShorthand
    Parser-->>Ecosystem: PyShortAST (894 tokens)
    Ecosystem-->>LLM: PyShorthand overview

    Note over LLM: LLM analyzes structure,<br/>identifies classes/methods

    LLM->>Ecosystem: "Show implementation of Block.forward()"
    Ecosystem->>Cache: check implementation cache
    Cache-->>Ecosystem: cache miss

    Ecosystem->>Decompiler: extract Python source for Block.forward
    Decompiler-->>Ecosystem: Python code (~300 tokens)

    Ecosystem->>Cache: store implementation
    Ecosystem-->>LLM: Python implementation + dependencies

    Note over LLM: LLM now has full context<br/>for specific method

    LLM->>Ecosystem: "What calls Block.forward()?"
    Ecosystem->>ContextPack: generate_context_pack("Block.forward", depth=1)
    ContextPack-->>Ecosystem: F0 + F1 (callers)
    Ecosystem-->>LLM: List of callers (~100 tokens)

    Note over Ecosystem,LLM: Total tokens: 894 + 300 + 100 = 1,294<br/>vs full code: ~18,000 tokens<br/>Savings: 93%
```

---

## Subsystem Dependency Graph

### Purpose
Shows compile-time dependencies between all 13 subsystems.

### Diagram

```mermaid
graph TB
    subgraph foundation["Layer 1: Foundation (Zero Dependencies)"]
        symbols[Symbols<br/>230 LOC]
        ast[AST Nodes<br/>727 LOC]
        tokenizer[Tokenizer<br/>547 LOC]
    end

    subgraph parsing["Layer 1: Parsing"]
        parser[Parser<br/>1,252 LOC]
    end

    subgraph validation["Layer 1: Validation"]
        validator[Validator<br/>632 LOC]
        errors[Enhanced Errors<br/>102 LOC]
        config[Config<br/>181 LOC]
    end

    subgraph transformation["Layer 2: Transformation"]
        decompiler[Decompiler<br/>1,142 LOC]
        formatter[Formatter<br/>417 LOC]
    end

    subgraph analysis["Layer 2: Analysis"]
        context[Context Pack<br/>579 LOC]
        execution[Execution Flow<br/>617 LOC]
        indexer[Indexer<br/>519 LOC]
        viz[Visualization<br/>266 LOC]
    end

    subgraph integration["Layer 3: Integration"]
        cli[CLI Tools<br/>~300 LOC]
        ecosystem[Ecosystem API<br/>699 LOC]
    end

    %% Foundation dependencies
    tokenizer -.->|uses constants| symbols
    ast -->|validates tags| symbols

    %% Parsing dependencies
    parser -->|consumes| tokenizer
    parser -->|constructs| ast
    parser -->|uses constants| symbols

    %% Validation dependencies
    validator -->|analyzes| ast
    validator -->|validates against| symbols
    validator -->|generates| errors
    errors -.->|suggests from| symbols
    config -.->|configures| validator

    %% Transformation dependencies
    formatter -->|parses via| parser
    formatter -->|reads| ast
    formatter -.->|converts| symbols

    decompiler -->|generates| ast
    decompiler -.->|uses constants| symbols

    %% Analysis dependencies
    context -->|analyzes| ast
    execution -->|analyzes| ast
    viz -->|visualizes| ast
    indexer -->|uses| decompiler

    %% Integration dependencies
    cli -->|invokes| parser
    cli -->|invokes| validator
    cli -->|invokes| formatter
    cli -->|invokes| decompiler
    cli -->|invokes| viz
    cli -->|invokes| indexer

    ecosystem -->|uses| parser
    ecosystem -->|uses| context
    ecosystem -->|uses| execution
    ecosystem -->|uses| decompiler

    %% Styling
    style symbols fill:#90ee90
    style ast fill:#90ee90
    style tokenizer fill:#90ee90
    style parser fill:#90ee90
    style validator fill:#add8e6
    style errors fill:#add8e6
    style config fill:#add8e6
    style decompiler fill:#ffd700
    style formatter fill:#ffd700
    style context fill:#ffd700
    style execution fill:#ffd700
    style indexer fill:#ffd700
    style viz fill:#ffd700
    style cli fill:#ffb6c1
    style ecosystem fill:#ffb6c1
```

### Dependency Rules

**Layering Constraints:**
1. **Layer 1 (Foundation + Core):** No dependencies on Layers 2 or 3
2. **Layer 2 (Transformation + Analysis):** Can depend on Layer 1, not Layer 3
3. **Layer 3 (Integration):** Can depend on all layers (orchestration)

**Observed Compliance:**
- **Perfect adherence:** 0 upward dependencies detected
- **No circular dependencies:** All imports follow DAG structure
- **Zero-dependency core:** Tokenizer, AST Nodes, Symbols use stdlib only

**Rationale:**
- **Embeddability:** Core can be used without CLI or analysis tools
- **Portability:** Zero external dependencies in foundation enables easy integration
- **Testability:** Each layer can be tested independently
- **Maintainability:** Clear dependency direction prevents tangled code

---

## Data Flow Diagrams

### CLI Parse Flow (User Perspective)

```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant Parser
    participant Tokenizer
    participant AST
    participant Validator
    participant JSON

    User->>CLI: $ pyshort parse example.pys
    activate CLI

    CLI->>Parser: parse_file("example.pys")
    activate Parser

    Parser->>Tokenizer: tokenize(source)
    activate Tokenizer
    Tokenizer->>Tokenizer: Lexical analysis<br/>(Unicode/ASCII, numbers, strings)
    Tokenizer-->>Parser: List[Token] + position info
    deactivate Tokenizer

    Parser->>Parser: Recursive descent parsing<br/>(metadata, entities, types)
    Parser->>AST: construct PyShortAST
    activate AST
    AST->>AST: Validate tag structure
    AST-->>Parser: PyShortAST object
    deactivate AST

    Parser-->>CLI: PyShortAST
    deactivate Parser

    opt Optional validation
        CLI->>Validator: check(ast)
        activate Validator
        Validator->>Validator: Run 14 validation rules
        Validator-->>CLI: List[Diagnostic]
        deactivate Validator
    end

    CLI->>JSON: ast.to_dict()
    JSON-->>CLI: JSON representation

    CLI->>User: Pretty-printed JSON output
    deactivate CLI
```

### Decompilation Flow (Python → PyShorthand)

```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant Decompiler
    participant PythonAST
    participant Inference
    participant TagExtractor
    participant PyShortAST
    participant Formatter

    User->>CLI: $ py2short module.py
    activate CLI

    CLI->>Decompiler: decompile_file("module.py", aggressive=True)
    activate Decompiler

    Decompiler->>PythonAST: ast.parse(source)
    activate PythonAST
    PythonAST-->>Decompiler: Python AST tree
    deactivate PythonAST

    Decompiler->>Decompiler: Detect framework<br/>(Pydantic, PyTorch, FastAPI)

    loop For each ClassDef/FunctionDef
        Decompiler->>Inference: infer_type(node)
        activate Inference
        Inference->>Inference: Analyze type hints,<br/>assignments, patterns
        Inference-->>Decompiler: TypeSpec (with confidence)
        deactivate Inference

        Decompiler->>TagExtractor: extract_tags(node)
        activate TagExtractor
        TagExtractor->>TagExtractor: Decorators → [Prop], [Static]<br/>Operations → [NN:∇], [IO:File]<br/>Complexity → [O(N)]
        TagExtractor-->>Decompiler: List[Tag]
        deactivate TagExtractor
    end

    Decompiler->>PyShortAST: construct PyShorthand AST
    activate PyShortAST
    PyShortAST-->>Decompiler: PyShorthand string
    deactivate PyShortAST

    Decompiler-->>CLI: PyShorthand output
    deactivate Decompiler

    opt Optional formatting
        CLI->>Formatter: format_string(output)
        activate Formatter
        Formatter->>Formatter: Align, sort, Unicode preference
        Formatter-->>CLI: Formatted PyShorthand
        deactivate Formatter
    end

    CLI->>User: Output to stdout or file
    deactivate CLI
```

### Progressive Disclosure Workflow (LLM Integration)

```mermaid
sequenceDiagram
    participant LLM as LLM System
    participant Ecosystem as CodebaseExplorer
    participant Parser
    participant Cache
    participant Decompiler
    participant ContextPack
    participant ExecutionFlow

    Note over LLM,Ecosystem: TIER 1: Cheap Overview

    LLM->>Ecosystem: Analyze codebase at /path/to/repo
    activate Ecosystem

    Ecosystem->>Parser: parse all .pys files
    activate Parser
    Parser-->>Ecosystem: PyShortAST for entire codebase
    deactivate Parser

    Ecosystem-->>LLM: PyShorthand overview<br/>(894 tokens for nanoGPT)

    Note over LLM: LLM analyzes structure,<br/>answers architectural questions

    Note over LLM,Ecosystem: TIER 2: On-Demand Details

    LLM->>Ecosystem: get_implementation("GPT.forward")

    Ecosystem->>Cache: Check implementation cache
    Cache-->>Ecosystem: Cache miss

    Ecosystem->>Decompiler: Extract Python source for GPT.forward
    activate Decompiler
    Decompiler-->>Ecosystem: Python code + dependencies (~300 tokens)
    deactivate Decompiler

    Ecosystem->>Cache: Store in cache
    Ecosystem-->>LLM: Python implementation

    Note over LLM: LLM analyzes implementation,<br/>asks follow-up questions

    LLM->>Ecosystem: get_context_pack("GPT.forward", depth=2)

    Ecosystem->>ContextPack: Generate F0/F1/F2 layers
    activate ContextPack
    ContextPack->>ContextPack: Build dependency graph
    ContextPack-->>Ecosystem: ContextPack(F0, F1, F2)
    deactivate ContextPack

    Ecosystem-->>LLM: Dependencies (F1: 5 entities, F2: 12 entities)

    LLM->>Ecosystem: trace_execution("GPT.forward", max_depth=3)

    Ecosystem->>ExecutionFlow: Trace call path
    activate ExecutionFlow
    ExecutionFlow->>ExecutionFlow: DFS traversal with cycle detection
    ExecutionFlow-->>Ecosystem: ExecutionFlow with steps
    deactivate ExecutionFlow

    Ecosystem-->>LLM: Call chain visualization

    deactivate Ecosystem

    Note over LLM: Total tokens: 894 + 300 + 150 = 1,344<br/>vs full code: 18,000 tokens<br/>Token savings: 93%
```

---

## Architecture Decision Records (Implicit)

### Key Architectural Decisions Observed

Based on code analysis, the following architectural decisions have been made (implicitly or explicitly):

#### 1. **Zero-Dependency Core**
- **Decision:** Core library (Tokenizer, Parser, AST, Validator, Symbols) uses Python stdlib only
- **Rationale:**
  - Embeddability in other projects without dependency conflicts
  - Portability across environments (minimal, containers, serverless)
  - Reduced security attack surface
  - Long-term stability (stdlib is stable)
- **Trade-offs:** Cannot use third-party parsing libraries (PLY, ANTLR) - must implement custom parser
- **Impact:** Excellent - Core is portable and reliable

#### 2. **Immutable AST with Frozen Dataclasses**
- **Decision:** All AST nodes are frozen dataclasses
- **Rationale:**
  - Thread-safety for parallel processing
  - Hashability enables caching and memoization
  - Functional programming style reduces bugs
  - GC-friendly (no reference cycles)
- **Trade-offs:** Cannot modify AST in-place - must create new instances
- **Impact:** Excellent - Enables caching, prevents accidental mutations

#### 3. **Rule-Based Validation Engine**
- **Decision:** Validator uses 14 independent `Rule` subclasses
- **Rationale:**
  - Extensibility without modifying core validator
  - Each rule testable in isolation
  - Easy to add custom rules for specific projects
  - Clear separation of concerns
- **Trade-offs:** Slight performance overhead from rule iteration
- **Impact:** Good - Extensible and maintainable

#### 4. **Progressive Disclosure (Two-Tier System)**
- **Decision:** PyShorthand overview (Tier 1) + on-demand Python details (Tier 2)
- **Rationale:**
  - 93% token savings for LLM context windows
  - Maintains 90% accuracy for code understanding tasks
  - Enables scaling to large codebases
  - LLM can request details only when needed
- **Trade-offs:** Requires LLM to make multiple API calls for full understanding
- **Impact:** Excellent - Empirically validated innovation

#### 5. **Unicode/ASCII Duality**
- **Decision:** Support both Unicode (→, ∈) and ASCII (->, IN) notation
- **Rationale:**
  - Platform compatibility (some terminals/editors don't render Unicode well)
  - Readability choice (users prefer different styles)
  - Lossless conversion (bidirectional mapping)
- **Trade-offs:** Tokenizer and formatter must handle both
- **Impact:** Good - Flexibility without complexity

#### 6. **Layered Architecture with Strict Dependency Direction**
- **Decision:** 3 layers (Core → Transform → Integration) with no upward dependencies
- **Rationale:**
  - Clear separation of concerns
  - Each layer testable independently
  - Prevents circular dependencies
  - Enables selective usage (can use Core without CLI)
- **Trade-offs:** More boilerplate for passing data through layers
- **Impact:** Excellent - Clean, maintainable architecture

#### 7. **Recursive Descent Parser (Not Parser Generator)**
- **Decision:** Hand-written recursive descent parser instead of ANTLR/PLY
- **Rationale:**
  - Zero external dependencies (aligns with decision #1)
  - Full control over error messages and recovery
  - Easier to debug and understand
  - No grammar file compilation step
- **Trade-offs:** More code to maintain (1,252 LOC), higher complexity
- **Impact:** Good - Flexibility and control, but high maintenance burden

#### 8. **Aggressive Type Inference in Decompiler**
- **Decision:** Offer aggressive mode for type inference from code patterns
- **Rationale:**
  - Many Python codebases lack type hints
  - Heuristic inference better than "Unknown"
  - Confidence scores allow users to validate
- **Trade-offs:** May infer incorrect types, requires validation
- **Impact:** Good - Practical for real-world Python code

#### 9. **Mermaid as Primary Visualization Format**
- **Decision:** Generate Mermaid markdown instead of images or SVG
- **Rationale:**
  - Mermaid is text-based (diffable, versionable)
  - GitHub/GitLab render Mermaid natively
  - No external rendering dependencies
  - Easy to embed in documentation
- **Trade-offs:** Limited styling compared to GraphViz, not all diagram types supported
- **Impact:** Excellent - Aligns with modern documentation workflows

#### 10. **Caching in Ecosystem API**
- **Decision:** Cache implementations and AST parses in Ecosystem
- **Rationale:**
  - LLMs may request same implementation multiple times
  - Parsing and decompilation are expensive
  - Memory trade-off acceptable for interactive use
- **Trade-offs:** Memory usage grows with cache, no cache invalidation
- **Impact:** Good - Significant performance improvement for LLM interactions

---

## Deployment Architecture

### Deployment Options

```mermaid
C4Deployment
    title Deployment Diagram - PyShorthand Usage Patterns

    Deployment_Node(dev_machine, "Developer Machine", "Laptop/Desktop") {
        Container(cli_local, "PyShorthand CLI", "Installed via pip/uv")
        Container(ide_plugin, "IDE Plugin", "Future: VSCode/PyCharm extension")
    }

    Deployment_Node(ci_server, "CI/CD Server", "GitHub Actions / GitLab CI") {
        Container(cli_ci, "PyShorthand CLI", "Linting & formatting checks")
        Container(test_runner, "Pytest", "Compliance tests")
    }

    Deployment_Node(llm_system, "LLM System", "Claude/GPT/Llama") {
        Container(llm_client, "LLM Client", "API integration")
        Container(ecosystem_client, "CodebaseExplorer", "Progressive disclosure")
    }

    Deployment_Node(docs_pipeline, "Documentation Pipeline", "MkDocs / Sphinx") {
        Container(viz_gen, "PyShorthand Viz", "Diagram generation")
        Container(index_gen, "PyShorthand Index", "Repository analysis")
    }

    Rel(dev_machine, ci_server, "Pushes code")
    Rel(llm_system, dev_machine, "Analyzes codebase")
    Rel(docs_pipeline, dev_machine, "Generates from")
```

### Deployment Characteristics

**CLI Tools:**
- **Installation:** pip install pyshorthand, uv add pyshorthand
- **Entry Points:** Console scripts (pyshort, py2short, pyshort-parse, etc.)
- **Dependencies:** Zero for core, optional for features (click, rich, networkx)
- **Platform:** Cross-platform (Linux, macOS, Windows)

**Ecosystem API:**
- **Usage:** Programmatic import in Python scripts
- **Deployment:** Library embedded in LLM integration code
- **State:** Stateful (maintains caches)
- **Thread-Safety:** Safe (immutable AST)

**CI/CD Integration:**
- **Commands:** pyshort lint --strict, pyshort fmt --check
- **Exit Codes:** 0 = success, non-zero = failure
- **Output:** JSON for machine parsing, text for humans

---

## Diagram Conventions

### Color Coding

- **Green boxes** (#e1f5e1): Foundation/core components (zero dependencies)
- **Blue boxes** (#d4edff): Transformation components (Python ↔ PyShorthand)
- **Yellow boxes** (#fff3cd): Analysis components (Context Pack, Execution Flow)
- **Pink boxes** (#f8d7da): Integration/user-facing components (CLI)
- **Light green boxes** (#d4edda): LLM integration (Ecosystem API)
- **Red boxes** (#ffe1e1): Data structures (AST Nodes)
- **Light yellow boxes** (#fff3cd): Error handling (Enhanced Errors)
- **Light blue boxes** (#d1ecf1): Configuration (Config, Visualization)
- **Gray boxes** (#f8f9fa): External systems (Python AST, Mermaid)

### Arrow Types

- **Solid arrows** (→): Direct compile-time dependencies
- **Dashed arrows** (-.->): Configuration or optional dependencies
- **Bidirectional arrows** (↔): Mutual relationships

### Box Labels

Format: `<Component Name>\n<LOC>\n━━━━━\n<Responsibilities>`

Example:
```
Parser
1,252 LOC
━━━━━━━━━
Recursive descent
Error recovery
```

### LOC Counts

LOC (Lines of Code) counts are approximate and represent source code only (excluding tests, comments, blank lines).

**Subsystem Sizes:**
- Small: <300 LOC (Symbols, Enhanced Errors, Config)
- Medium: 300-700 LOC (Tokenizer, AST Nodes, Validator, Context Pack, Execution Flow, Indexer, Ecosystem)
- Large: 700-1,500 LOC (Parser, Decompiler)

---

## Notes

### Diagram Tool Compatibility

All diagrams use **Mermaid syntax** for easy rendering in:
- GitHub README.md and markdown files
- GitLab documentation
- MkDocs with mermaid plugin
- Obsidian and other markdown editors
- VSCode with Mermaid preview extensions

### C4 Model Compliance

- **Level 1 (Context):** Uses C4Context notation with Person/System/System_Ext
- **Level 2 (Container):** Uses C4Container notation with Container/System_Boundary
- **Level 3 (Component):** Uses flowchart notation (Mermaid's C4Component support is limited)

### Dependency Graph Accuracy

The subsystem dependency graph shows **compile-time dependencies only** (import statements), not runtime method calls. For example:
- `Parser → Tokenizer` means Parser imports from Tokenizer
- Does not show that CLI calls Parser methods at runtime

### LOC Metrics Source

LOC counts derived from:
- Discovery findings document (01-discovery-findings.md)
- Subsystem catalog document (02-subsystem-catalog.md)
- Quality assessment document (05-quality-assessment.md)

Counts may vary slightly as code evolves.

---

## Future Enhancements

### Planned Diagram Additions

1. **Sequence Diagram: Error Recovery Flow**
   - Show how Parser recovers from syntax errors
   - Diagnostic accumulation and suggestion generation

2. **Component Diagram: Type System Detail**
   - Deep dive into TypeSpec, generic parameters, union types
   - Show how type inference works across layers

3. **Deployment Diagram: LSP Integration**
   - When Language Server Protocol support is added
   - Show IDE ↔ PyShorthand communication

4. **State Diagram: Validation States**
   - Show how Validator processes rules
   - State transitions for diagnostics

### Diagram Evolution Strategy

As PyShorthand evolves:
- **Update diagrams** when major architectural changes occur
- **Version diagrams** alongside code versions (e.g., 03-diagrams-v1.0.md)
- **Maintain consistency** between code and diagrams via automated checks
- **Add complexity overlays** when refactoring high-complexity components

---

## Appendix: Diagram Source Guidelines

### How to Update These Diagrams

1. **Edit Mermaid syntax** directly in this markdown file
2. **Validate rendering** using:
   - Mermaid Live Editor: https://mermaid.live
   - VSCode Mermaid Preview extension
   - GitHub markdown preview
3. **Check consistency** with code by cross-referencing:
   - 01-discovery-findings.md for architecture overview
   - 02-subsystem-catalog.md for component details
   - 05-quality-assessment.md for LOC counts
4. **Update LOC counts** when significant code changes occur
5. **Add new diagrams** in same C4 format with clear purpose statement

### Mermaid Syntax Resources

- C4 Diagrams: https://mermaid.js.org/syntax/c4.html
- Flowcharts: https://mermaid.js.org/syntax/flowchart.html
- Sequence Diagrams: https://mermaid.js.org/syntax/sequenceDiagram.html
- Graph Diagrams: https://mermaid.js.org/syntax/graph.html

---

**Document Version:** 1.0
**Last Updated:** 2025-11-24
**Author:** PyShorthand Architecture Analysis Agent
**Status:** Complete ✓
