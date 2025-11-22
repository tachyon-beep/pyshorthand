# PyShorthand Toolchain

A comprehensive toolchain for the **PyShorthand Protocol (v1.3.1)** - a high-density intermediate representation designed to maximize semantic bitrate for LLM-powered code analysis.

## What is PyShorthand?

PyShorthand is a codified framework for serializing Python codebases into a compact, semantics-first format that preserves:

- **System Topology**: Explicit dataflow and dependency graphs
- **Computational Complexity**: Big-O notation and performance characteristics
- **Memory Hierarchy**: CPU/GPU/Disk residency and transfer costs
- **Safety Properties**: Clear distinction between local (!!) and system (!!) mutations
- **Mathematical Operations**: Linear algebra, neural networks, and stochastic processes

By stripping syntactic noise while retaining architectural physics, PyShorthand enables LLMs to reason about entire systems within limited context windows.

## Features

### Core Tools

- **pyshort-parse**: Parse PyShorthand files into structured AST (JSON/YAML output)
- **pyshort-lint**: Validate grammar and enforce semantic best practices
- **py2short**: Auto-generate PyShorthand from Python source code
- **pyshort-complexity**: Analyze computational costs and identify bottlenecks
- **pyshort-viz**: Generate visual dataflow graphs (Graphviz, Mermaid, HTML)
- **pyshort-index**: Build cross-file dependency graphs for repository-scale reasoning

### Key Capabilities

- ✅ Parse both Unicode and ASCII-compatible notation
- ✅ Rich error messages with line numbers and suggestions
- ✅ Extensible linter with custom rules
- ✅ Pattern-based decompiler for PyTorch/FastAPI/Django
- ✅ Performance analysis with complexity estimation
- ✅ Interactive visualizations with risk highlighting
- ✅ Repository indexing for cross-module analysis

## Quick Start

### Installation

```bash
pip install pyshorthand[all]
```

Or for just the core parser:

```bash
pip install pyshorthand
```

### Basic Usage

**Parse a PyShorthand file:**

```bash
pyshort-parse env.pys --output env.json
```

**Validate and lint:**

```bash
pyshort-lint env.pys --strict
```

**Generate PyShorthand from Python:**

```bash
py2short src/model.py --output docs/model.pys
```

**Analyze complexity:**

```bash
pyshort-complexity env.pys --function step
```

**Visualize dataflow:**

```bash
pyshort-viz env.pys --function step --output graph.svg
```

### Example

**Input Python:**

```python
def calculate_attention(query, key, value):
    """Scaled dot-product attention."""
    scores = torch.matmul(query, key.transpose(-2, -1))
    scores = scores / math.sqrt(query.size(-1))
    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, value)
    return output
```

**Generated PyShorthand:**

```
F:calculate_attention(query, key, value)
  [Role:Core] [Risk:Med]

  query ∈ f32[B,N,D]@GPU
  key   ∈ f32[B,M,D]@GPU
  value ∈ f32[B,M,D]@GPU

  scores ≡ query ⊗ key^T →[Lin:MatMul:O(N*M*D)]
  scores ≡ scores / √D →[Lin:O(N*M)]
  attn   ≡ softmax(scores) →[Thresh:O(N*M)]
  out    ≡ attn ⊗ value →[Lin:MatMul:O(N*M*D)]

  ← out ∈ f32[B,N,D]@GPU
```

## PyShorthand Notation Primer

### State Declaration

```
name ∈ Type[Shape]@Location
```

Example:
```
batch ∈ f32[B, T, D]@GPU      # Training batch on VRAM
cache ∈ Map[1e6]@CPU          # Large lookup table in RAM
```

### Operations

- `→` Flow/pipe operator
- `⊳` Happens-after (causal dependency)
- `≡` Definition/equality
- `!` Local mutation
- `!!` System mutation (database, API, logs)
- `⊗` Tensor operation (matmul, convolution)

### Tags

```
→[Tag:Qualifier:Complexity]
```

Common tags:
- `[Lin]` - Linear/algebraic operations
- `[Iter:Hot]` - Inner loop (performance critical)
- `[IO:Net]` - Network I/O
- `[NN:∇]` - Neural network with gradients
- `[Sync:Lock]` - Synchronization point

### Metadata Headers

```python
# [M:ModuleName] [ID:UniqueID] [Role:Core] [Risk:High]
# [Context: GPU-ML] [Dims: N=batch, D=dim]
```

## Documentation

- [Architecture Guide](ARCHITECTURE.md)
- [Tutorial](docs/tutorial.md) *(coming soon)*
- [API Reference](docs/reference.md) *(coming soon)*
- [Examples](docs/examples/)

## Development

### Setup

```bash
git clone https://github.com/tachyon-beep/animated-system
cd animated-system
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest
pytest --cov=pyshort --cov-report=html
```

### Type Checking

```bash
mypy src/pyshort
```

### Linting

```bash
ruff check src/
black src/ tests/
```

## Roadmap

### Phase 1: Core Infrastructure ✅
- [x] Project setup
- [ ] Parser implementation
- [ ] Validator and linter
- [ ] Basic CLI tools

### Phase 2: Decompiler & Analysis
- [ ] Python decompiler with pattern matching
- [ ] Complexity analyzer
- [ ] Dataflow visualizer
- [ ] Integration tests on real codebases

### Phase 3: Advanced Features
- [ ] Repository indexer
- [ ] Differential analyzer
- [ ] Coverage reporter
- [ ] LLM context optimizer

### Phase 4: Ecosystem
- [ ] IDE integration (VS Code LSP)
- [ ] CI/CD templates
- [ ] Documentation generator
- [ ] PyPI publication

## Contributing

Contributions welcome! This is an early-stage project. Please:

1. Read [ARCHITECTURE.md](ARCHITECTURE.md) to understand the design
2. Check existing issues or open a new one
3. Submit PRs with tests and documentation

## License

MIT License - see LICENSE file for details

## Citation

If you use PyShorthand in research or production, please cite:

```bibtex
@misc{pyshorthand2025,
  title={PyShorthand Protocol: High-Density IR for LLM Code Analysis},
  author={PyShorthand Contributors},
  year={2025},
  howpublished={\url{https://github.com/tachyon-beep/animated-system}}
}
```

## Acknowledgments

Inspired by:
- LLVM IR for intermediate representation design
- Rust's MIR for semantic preservation
- The need for better LLM-friendly code serialization

---

**Status**: Alpha - API subject to change

**Python**: 3.10+ required

**Performance**: Targets <1s parsing for 10K line files
