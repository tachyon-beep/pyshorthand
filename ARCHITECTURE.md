# PyShorthand Toolchain Architecture

## Overview

The PyShorthand toolchain is a suite of tools designed to parse, validate, analyze, and visualize the PyShorthand Protocol (v1.3.1) - a high-density intermediate representation for Python codebases optimized for LLM consumption.

## Design Principles

1. **Unix Philosophy**: Each tool does one thing well and composes with others
2. **Zero-Dependency Core**: Parser and validator have no external dependencies
3. **Rich Error Messages**: Helpful diagnostics with line numbers and suggestions
4. **Performance**: Handle 10K+ line files in <1 second
5. **Extensibility**: Plugin architecture for custom rules and analysis
6. **Dual Notation**: Support both Unicode and ASCII-compatible syntax

## Package Structure

```
pyshorthand/
├── pyproject.toml              # PEP 621 package metadata
├── README.md                   # User-facing documentation
├── ARCHITECTURE.md             # This file
├── src/
│   └── pyshort/
│       ├── __init__.py
│       ├── core/               # Core library (zero dependencies)
│       │   ├── __init__.py
│       │   ├── ast_nodes.py    # AST node definitions
│       │   ├── parser.py       # PyShorthand parser
│       │   ├── validator.py    # Grammar and semantic validation
│       │   ├── grammar.py      # Grammar rules and constraints
│       │   └── symbols.py      # Symbol mapping (Unicode ↔ ASCII)
│       ├── decompiler/         # Python → PyShorthand
│       │   ├── __init__.py
│       │   ├── decompiler.py   # Main decompiler logic
│       │   ├── patterns.py     # AST pattern matching
│       │   └── inference.py    # Tag inference heuristics
│       ├── analysis/           # Analysis tools
│       │   ├── __init__.py
│       │   ├── complexity.py   # Complexity analyzer
│       │   ├── visualizer.py   # Graph visualization
│       │   ├── indexer.py      # Repository indexing
│       │   └── diff.py         # Semantic diff
│       └── cli/                # CLI tools
│           ├── __init__.py
│           ├── main.py         # Main entry point
│           ├── parse.py        # pyshort-parse command
│           ├── lint.py         # pyshort-lint command
│           ├── decompile.py    # py2short command
│           ├── complexity.py   # pyshort-complexity command
│           └── viz.py          # pyshort-viz command
├── tests/
│   ├── unit/
│   │   ├── test_parser.py
│   │   ├── test_validator.py
│   │   ├── test_decompiler.py
│   │   └── test_analysis.py
│   ├── integration/
│   │   ├── fixtures/
│   │   │   ├── vhe_canonical.pys      # VHE example from RFC
│   │   │   ├── pytorch_model.py       # Real PyTorch code
│   │   │   └── pytorch_model.pys      # Expected output
│   │   ├── test_roundtrip.py
│   │   └── test_real_world.py
│   └── performance/
│       └── test_large_files.py
└── docs/
    ├── index.md
    ├── tutorial.md
    ├── reference.md
    └── examples/
```

## Core Components

### AST Node Hierarchy

```python
@dataclass
class PyShortAST:
    metadata: Metadata
    entities: List[Entity]
    functions: List[Function]
    statements: List[Statement]
    graph: DirectedGraph

@dataclass
class Metadata:
    module_name: Optional[str]
    module_id: Optional[str]
    role: Optional[str]
    layer: Optional[str]
    risk: Optional[str]
    context: Optional[str]
    dims: Dict[str, str]
    requires: List[str]
    owner: Optional[str]

class Entity(ABC):
    pass

@dataclass
class Class(Entity):
    name: str
    state: List[StateVar]
    methods: List[Function]
    dependencies: List[Reference]

@dataclass
class StateVar:
    name: str
    type_: Optional[str]
    shape: Optional[List[str]]
    location: Optional[str]
    transfer: Optional[Tuple[str, str]]

@dataclass
class Function:
    name: str
    params: List[Parameter]
    return_type: Optional[str]
    modifiers: List[str]  # [Async], etc.
    preconditions: List[str]
    postconditions: List[str]
    errors: List[str]
    body: List[Statement]

@dataclass
class Statement:
    line: int
    type: StatementType
    lhs: Optional[str]
    op: Optional[str]  # ≡, !, !!, →, ⊳, etc.
    rhs: Optional[Expression]
    tags: List[Tag]
    profiling: Optional[str]  # ⏱16ms
```

### Parser Architecture

The parser is a hand-written recursive descent parser with these components:

1. **Lexer**: Tokenizes input, handles both Unicode and ASCII
2. **Symbol Mapper**: Translates ASCII (->>) to Unicode (⊳) and vice versa
3. **Parser**: Builds AST from token stream
4. **Error Reporter**: Rich error messages with suggestions

Key parsing phases:
- Header metadata extraction
- Entity definition parsing (Class, Data, Interface, Module)
- State architecture parsing (type ∈ Type[Shape]@Location)
- Function contract parsing ([Pre], [Post], [Err])
- Statement parsing (assignments, mutations, flow control)
- Tag and qualifier extraction

### Validator Rules

The validator enforces:

**Grammar Constraints (Section 3.8)**:
- One operation per line
- Tag position after →
- Sigil position before identifier
- Valid comment syntax

**Semantic Constraints**:
- Mandatory metadata ([M:Name], [Role])
- Dimension consistency (vars used in shapes must be declared)
- Tag coverage (critical ops should have tags)
- Location inference (validate @Location rules)
- Effect safety (!! in high-risk contexts)

**Extensible Rules**:
```python
class Rule(ABC):
    @abstractmethod
    def check(self, ast: PyShortAST) -> Iterator[Diagnostic]

class Diagnostic:
    severity: Literal["error", "warning", "info"]
    line: int
    message: str
    suggestion: Optional[str]
```

### Decompiler Strategy

The decompiler uses Python AST analysis:

1. **Extract Structure**: Classes, functions, imports
2. **Infer Tags**: Pattern matching on AST nodes
   - `for` loops → [Iter]
   - `if` statements → [Thresh]
   - `torch.*` → [NN] or [Lin]
   - `requests.*` → [IO:Net]
   - `db.*` → [IO:Disk]
3. **Detect Locations**: Type hints, tensor creation
4. **Generate Metadata**: From docstrings and module info
5. **Mark Uncertainty**: [TODO:Tag] for ambiguous cases

**Conservative Mode** (default):
- More [TODO] markers
- Minimal inference
- Human review required

**Aggressive Mode** (`--aggressive`):
- More automatic tagging
- Complexity estimation
- Shape inference from annotations

## Analysis Tools

### Complexity Analyzer

Reads tags and estimates:
- Asymptotic complexity (parse :O(N) tags, detect nesting)
- Memory transfer costs (@A→B latencies)
- Concurrency bottlenecks ([Sync] points)
- Critical path (longest ⊳ chain)

### Visualizer

Outputs:
- **Graphviz/DOT**: Static dataflow graphs
- **Mermaid**: Documentation-friendly
- **Interactive HTML**: Zoomable with tooltips

Visual encoding:
- Color by risk (!! = red, ! = yellow)
- Shape by type ([IO] = cylinder, [NN] = hexagon)
- Edge thickness by complexity
- Highlight critical path

### Repository Indexer

Scans directory tree:
- Extract all [ID:Token] declarations
- Resolve all [Ref:Token] references
- Build module dependency graph
- Detect circular dependencies
- Generate RepoIndex.yaml

## CLI Design

All CLI tools follow consistent patterns:

```bash
# Parse and validate
pyshort-parse env.pys --output env.json
pyshort-lint env.pys --strict

# Decompile
py2short src/env.py --output docs/env.pys
py2short src/ --output docs/ --recursive --aggressive

# Analyze
pyshort-complexity env.pys --function step
pyshort-viz env.pys --function step --output graph.svg
pyshort-index repo/ --output index.yaml

# Advanced
pyshort-diff v1.pys v2.pys
pyshort-coverage src/
pyshort-pack repo/ --budget 100k --focus VHE.step
```

## Testing Strategy

### Unit Tests
- Parser: Grammar compliance, error handling
- Validator: Each rule independently
- Decompiler: Pattern matching accuracy
- Analysis: Complexity calculation, graph building

### Integration Tests
- Roundtrip: parse → validate → render
- Real code: PyTorch models, FastAPI apps
- VHE canonical example from RFC

### Performance Tests
- 10K line files < 1s
- 50K LOC repository < 5s

## Performance Targets

- **Parser**: 10K lines in <1 second
- **Linter**: 50K LOC repository in <5 seconds
- **Visualizer**: Generate graphs in <10 seconds
- **Indexer**: 100-file repository in <30 seconds

## Error Handling Philosophy

All errors should:
1. Show exact line number and column
2. Highlight problematic code
3. Explain what's wrong
4. Suggest how to fix it

Example:
```
error: Invalid tag position
  --> env.pys:42:15
   |
42 |   base ≡ meters [Lin] ⊗ weights
   |                 ^^^^^
   | Tags must appear after the → operator
   |
help: Try moving the tag after the flow operator:
   |
42 |   base ≡ meters ⊗ weights →[Lin:Broad:O(N)]
   |
```

## Extensibility

### Custom Linter Rules

```python
from pyshort.lint import Linter, Rule

class NoSystemMutationInTest(Rule):
    def check(self, ast):
        if ast.metadata.layer == 'Test':
            for stmt in ast.find_all(type='mutation'):
                if stmt.op == '!!':
                    yield Warning(
                        line=stmt.line,
                        message=f"System mutation in test: {stmt}",
                        suggestion="Use ! for local mutations in tests"
                    )

linter = Linter()
linter.register(NoSystemMutationInTest())
linter.check_file("test.pys")
```

### Custom Complexity Patterns

```python
from pyshort.analysis import ComplexityAnalyzer

class TransformerComplexity:
    def analyze(self, ast):
        # Detect O(N²) attention patterns
        pass

analyzer = ComplexityAnalyzer()
analyzer.register_pattern(TransformerComplexity())
```

## Distribution Strategy

1. **PyPI Package**: `pip install pyshorthand`
2. **CLI Tools**: Installed as console scripts
3. **Documentation**: Hosted on Read the Docs or GitHub Pages
4. **VS Code Extension**: Marketplace publication
5. **GitHub Actions**: Template workflow

## Roadmap

### Phase 1: Core Infrastructure (Weeks 1-2)
- [x] Project setup
- [ ] Parser implementation
- [ ] Validator implementation
- [ ] Basic CLI tools
- [ ] Unit tests with VHE example

### Phase 2: Decompiler & Analysis (Weeks 3-4)
- [ ] Python decompiler with pattern matching
- [ ] Complexity analyzer
- [ ] Dataflow visualizer
- [ ] Integration tests on real codebases

### Phase 3: Advanced Features (Weeks 5-6)
- [ ] Repository indexer
- [ ] Differential analyzer
- [ ] Coverage reporter
- [ ] LLM context optimizer

### Phase 4: Ecosystem (Weeks 7-8)
- [ ] IDE integration (LSP)
- [ ] CI/CD templates
- [ ] Documentation generator
- [ ] PyPI publication

## Open Questions

1. Should we support incremental parsing for large files?
2. How should we handle version evolution of the PyShorthand spec?
3. Should the AST be serializable to Protobuf for cross-language support?
4. What's the best way to handle ambiguous Python patterns in decompiler?

## References

- PyShorthand RFC v1.3.1
- Python AST documentation
- Rust's Chalk for inspiration on IR design
- LLVM IR for precedent on intermediate representations
