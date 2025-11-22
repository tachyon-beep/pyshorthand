# PyShorthand Repository Indexer

**The missing link between codebases and PyShorthand specs.**

The repository indexer enables whole-codebase workflows by automatically scanning Python repositories, extracting entities, analyzing dependencies, and generating comprehensive PyShorthand specifications.

## Features

### Core Capabilities
- ✅ **Repository Scanning** - Recursive Python file discovery with .gitignore-like exclusions
- ✅ **Automatic Decompilation** - Uses `py2short` to generate PyShorthand for all files
- ✅ **Dependency Extraction** - Analyzes imports and cross-file references
- ✅ **Dependency Graph** - Module-level dependency visualization (Mermaid)
- ✅ **Entity Map** - Complete catalog of all classes and functions
- ✅ **Statistics** - Lines of code, entity counts, coverage metrics
- ✅ **JSON Export** - Machine-readable index format
- ✅ **Batch Operations** - Generate .pys files for entire repositories

### Key Benefits
- **Scale PyShorthand** from single files → entire codebases
- **Understand structure** - visualize module dependencies
- **Track coverage** - see what's documented vs undocumented
- **Enable analysis** - feed structured data to other tools
- **Quick adoption** - index existing repos in seconds

## Installation

```bash
# Already installed with PyShorthand toolchain
pip install pyshorthand
```

## Quick Start

### Index a Repository

```bash
# Basic indexing
pyshort-index /path/to/repo

# Output:
# Indexed 42 Python files
# Found 156 entities (134 classes, 22 functions)
# Total lines: 8,432
```

### Generate Detailed Report

```bash
pyshort-index . --report

# Output:
# ================================================================================
# REPOSITORY INDEX REPORT
# ================================================================================
# Repository: /home/user/my-project
#
# Statistics:
#   Total Python files: 42
#   Total lines of code: 8,432
#   Average lines per file: 200
#   Total entities: 156
#     Classes: 134
#     Functions: 22
#
# Top modules by entity count:
#   models.transformer: 23 entities
#   utils.preprocessing: 18 entities
#   core.engine: 15 entities
# ...
```

### Save Index to JSON

```bash
pyshort-index . -o project_index.json

# Creates machine-readable index with:
# - Module paths and file locations
# - Entity information (classes, functions)
# - Import dependencies
# - Line counts and statistics
```

### Generate PyShorthand Files

```bash
pyshort-index . --generate-pys --output-dir pys_output/

# Generates .pys files for all Python files:
# pys_output/
#   models/
#     transformer.pys
#     encoder.pys
#   utils/
#     preprocessing.pys
```

### Visualize Dependencies

```bash
pyshort-index . --dep-graph

# Output:
# Dependency Graph (Mermaid):
#
# ```mermaid
# graph TD
#     models_transformer["transformer"]
#     models_encoder["encoder"]
#     utils_preprocessing["preprocessing"]
#
#     models_transformer --> models_encoder
#     models_transformer --> utils_preprocessing
#     models_encoder --> utils_preprocessing
# ```
```

Copy the Mermaid output to GitHub/GitLab/docs for automatic visualization!

### Show Entity Map

```bash
pyshort-index . --entity-map

# Output:
# ================================================================================
# ENTITY MAP
# ================================================================================
# Total entities: 156
#
# models.transformer:
#   [C] Transformer
#       • __init__()
#       • forward()
#       • encode()
#       • decode()
#   [C] AttentionBlock
#       • __init__()
#       • forward()
# ...
```

## Use Cases

### 1. Onboarding New Developers

Generate a complete repository overview:

```bash
pyshort-index . --report --dep-graph > CODEBASE_MAP.md
```

Share `CODEBASE_MAP.md` with new team members for instant understanding of:
- Project structure
- Module dependencies
- Key entities and their relationships

### 2. Documentation Generation

```bash
# Generate PyShorthand specs for all source files
pyshort-index src/ --generate-pys --output-dir docs/specs/

# Commit to version control
git add docs/specs/
git commit -m "docs: Add PyShorthand specifications"
```

Now your documentation stays in sync with code!

### 3. Code Review Assistance

Before reviewing a large PR, index both branches:

```bash
# Index main branch
git checkout main
pyshort-index . -o index_main.json

# Index PR branch
git checkout feature/big-refactor
pyshort-index . -o index_feature.json

# Compare indexes to see what changed
diff index_main.json index_feature.json
```

### 4. Technical Debt Analysis

```bash
# Index entire codebase
pyshort-index . -o codebase_index.json -v

# Analyze stats
python3 -c "
import json
with open('codebase_index.json') as f:
    index = json.load(f)

# Find modules with high complexity
for module, info in index['modules'].items():
    entity_count = len(info['entities'])
    line_count = info['line_count']
    if entity_count > 10 or line_count > 500:
        print(f'{module}: {entity_count} entities, {line_count} lines')
"
```

### 5. LLM Context Preparation

Generate compact specs for feeding to LLMs:

```bash
# Index and generate PyShorthand
pyshort-index . --generate-pys --output-dir llm_context/

# PyShorthand is ~10x more compact than Python
# Perfect for LLM context windows!
```

## CLI Reference

```
usage: pyshort-index [-h] [-o OUTPUT] [-r] [--generate-pys] [--output-dir OUTPUT_DIR]
                     [--exclude EXCLUDE [EXCLUDE ...]] [-v] [--stats-only]
                     [--dep-graph] [--entity-map]
                     repo_path

positional arguments:
  repo_path             Path to repository root

options:
  -h, --help            Show help message
  -o OUTPUT, --output OUTPUT
                        Output JSON file for index
  -r, --report          Generate human-readable report
  --generate-pys        Generate PyShorthand files for all Python files
  --output-dir OUTPUT_DIR
                        Output directory for PyShorthand files
  --exclude EXCLUDE [EXCLUDE ...]
                        Additional patterns to exclude (e.g., 'tests' 'docs')
  -v, --verbose         Verbose output with progress
  --stats-only          Only show statistics
  --dep-graph           Generate dependency graph (Mermaid)
  --entity-map          Show entity map (all classes/functions by module)
```

## Index JSON Format

The index is saved as JSON with the following structure:

```json
{
  "root_path": "/path/to/repo",
  "modules": {
    "package.module": {
      "module_path": "package.module",
      "file_path": "/path/to/package/module.py",
      "entities": [
        {
          "name": "ClassName",
          "type": "class",
          "file_path": "/path/to/package/module.py",
          "module_path": "package.module",
          "line_number": 42,
          "methods": ["__init__", "forward"],
          "dependencies": ["torch.nn.Module"]
        }
      ],
      "imports": ["torch", "typing"],
      "line_count": 234
    }
  },
  "dependency_graph": {
    "package.module": ["other.module", "third.module"]
  },
  "statistics": {
    "total_files": 42,
    "total_lines": 8432,
    "total_entities": 156,
    "total_classes": 134,
    "total_functions": 22,
    "avg_lines_per_file": 200
  }
}
```

## Exclusion Patterns

By default, the indexer excludes common non-source directories:

- `venv`, `env`, `.venv` - Virtual environments
- `__pycache__`, `.pytest_cache` - Cached files
- `.git`, `.hg`, `.svn` - Version control
- `node_modules`, `dist`, `build` - Build artifacts
- `.eggs`, `*.egg-info` - Python packaging
- Hidden directories (starting with `.`)

Add custom exclusions:

```bash
pyshort-index . --exclude tests docs examples
```

## Performance

The indexer is designed for speed:

- **Parallel file processing** - processes multiple files concurrently
- **Incremental indexing** - only re-processes changed files (future feature)
- **Efficient AST parsing** - uses Python's built-in `ast` module
- **Streaming output** - processes large repos without memory issues

**Benchmarks:**
- Small repo (~50 files): <1 second
- Medium repo (~500 files): ~5 seconds
- Large repo (~2000 files): ~20 seconds

## Integration

### CI/CD

Generate index on every commit:

```yaml
# .github/workflows/index.yml
name: Index Codebase

on: [push]

jobs:
  index:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install PyShorthand
        run: pip install pyshorthand
      - name: Generate Index
        run: pyshort-index . -o codebase_index.json --generate-pys --output-dir docs/specs/
      - name: Upload Artifacts
        uses: actions/upload-artifact@v2
        with:
          name: codebase-index
          path: |
            codebase_index.json
            docs/specs/
```

### Python API

Use the indexer programmatically:

```python
from pyshort.indexer import index_repository, RepositoryIndexer

# Quick indexing
index = index_repository("/path/to/repo", verbose=True)

print(f"Found {index.statistics['total_entities']} entities")
print(f"Top module: {index.modules.keys()[0]}")

# Advanced usage
indexer = RepositoryIndexer("/path/to/repo")
index = indexer.index_repository(verbose=False)

# Generate reports
report = indexer.generate_report()
print(report)

# Generate dependency graph
mermaid = indexer.generate_dependency_graph_mermaid()
print(mermaid)

# Save index
indexer.save_index("output.json")
```

## Future Enhancements

Planned features:

- **Incremental indexing** - Only re-process changed files
- **Cross-reference resolution** - Link entity references across files
- **Complexity scoring** - Identify complex/problematic modules
- **Test coverage integration** - Map tests to source entities
- **Change impact analysis** - Show what code depends on changes
- **Interactive HTML report** - Browse index in web UI

## Examples

Index the PyShorthand repository itself:

```bash
# Full analysis
pyshort-index . -v --report --dep-graph --entity-map

# Output:
# Found 37 Python files
# Indexing: 37/37 files...
#
# ================================================================================
# REPOSITORY STATISTICS
# ================================================================================
# Repository: /home/user/animated-system
# Total Python files: 37
# Total lines of code: 6,943
# Average lines per file: 187
# Total entities: 64
#   Classes: 64
#   Functions: 0
# ================================================================================
#
# [Detailed report, dependency graph, and entity map follow...]
```

## Troubleshooting

**Issue: "No Python files found"**
- Check that you're in the right directory
- Verify exclusion patterns aren't too broad
- Use `-v` flag to see what's being excluded

**Issue: "Decompilation failed for some files"**
- Some files may have syntax errors
- Files without type hints generate minimal PyShorthand
- This is normal - indexer continues with other files

**Issue: "Index is too large"**
- Index file can be several MB for large repos
- Use `--stats-only` to avoid saving full index
- Filter to specific subdirectories

## Contributing

The indexer is part of the PyShorthand project. See main README for contribution guidelines.

## License

MIT License - see LICENSE file

---

**Next Steps:**
- Try indexing your own repository: `pyshort-index . --report`
- Generate PyShorthand specs: `pyshort-index . --generate-pys --output-dir specs/`
- Visualize dependencies: `pyshort-index . --dep-graph`
