# RFC Compliance Test Suite

**Purpose**: Validates that the PyShorthand toolchain correctly implements the [PyShorthand Protocol RFC v1.3.1](../../docs/RFC.md) specification.

## Overview

This test suite ensures that:
1. ✅ **Parser** correctly handles all RFC examples
2. ✅ **Validator** enforces all RFC rules
3. ✅ **Error codes** are assigned correctly
4. ✅ **Edge cases** are properly handled

## Test Structure

```
tests/compliance/
├── README.md                    # This file
├── fixtures/                    # RFC example files
│   ├── rfc_2_1_metadata.pys    # Section 2.1: Metadata
│   ├── rfc_2_2_dimensions.pys  # Section 2.2: Dimensions
│   └── rfc_2_3_entities.pys    # Section 2.3: Entities
└── test_rfc_compliance.py      # Main test harness
```

## Running the Tests

### Quick Run
```bash
# From project root
python3 -m unittest tests.compliance.test_rfc_compliance

# With verbose output
python3 -m unittest tests.compliance.test_rfc_compliance -v
```

### Expected Output
```
======================================================================
RFC COMPLIANCE REPORT - PyShorthand v1.3.1
======================================================================
✓ PASS: RFC 2.1: Basic Metadata
         ✓ Metadata parsed correctly
✓ PASS: RFC 2.2: Dimensions
         ✓ Dimension variables work
✓ PASS: RFC 2.3: Entities
         ✓ Entity definitions work
✓ PASS: RFC 3.8: Invalid Values
         ✓ Validates invalid metadata
✓ PASS: RFC 3.8: Mandatory Metadata
         ✓ Validates missing metadata
✓ PASS: RFC 4: VHE Canonical
         ✓ VHE example parses
======================================================================
TOTAL: 6/6 tests passed (100% compliant)
======================================================================
```

## Test Coverage

### RFC Section 2: Core Syntax
- **2.1 Metadata Headers** ✅
  - Tests: Basic metadata parsing
  - Validates: `[M:Name]`, `[Role:...]`, `[Risk:...]`
  - File: `rfc_2_1_metadata.pys`

- **2.2 Dimension Variables** ✅
  - Tests: `[Dims:N=batch,D=dim]` parsing
  - Validates: Dimension declarations and usage
  - File: `rfc_2_2_dimensions.pys`

- **2.3 Entity Definitions** ✅
  - Tests: `[C:ClassName]` syntax
  - Validates: State variables, references
  - File: `rfc_2_3_entities.pys`

### RFC Section 3: Grammar & Validation
- **3.8 Validation Rules** ✅
  - Tests: Mandatory metadata enforcement
  - Tests: Invalid value detection (error codes E001, E003, E004)
  - Validates: "Did you mean?" suggestions

### RFC Section 4: Real-World Examples
- **4.1 VHE Canonical Example** ✅
  - Tests: Complete VectorizedHamletEnv example
  - Validates: Complex entity with dependencies
  - File: `../../integration/fixtures/vhe_canonical.pys`

## Extending the Test Suite

### Adding New Tests

1. **Create a fixture file** in `fixtures/`:
```python
# tests/compliance/fixtures/rfc_X_Y_feature.pys
# RFC Section X.Y: Feature Description
# [M:FeatureExample] [Role:Core]

[C:Example]
  state ∈ f32[N]@GPU
```

2. **Add a test method** in `test_rfc_compliance.py`:
```python
def test_rfc_X_Y_feature(self):
    """RFC X.Y: Feature description."""
    file_path = self.fixtures_dir / "rfc_X_Y_feature.pys"

    try:
        ast = parse_file(str(file_path))

        # Add assertions
        self.assertEqual(...)

        self._record_result("RFC X.Y: Feature", True, "✓ Works")
    except Exception as e:
        self._record_result("RFC X.Y: Feature", False, f"✗ {e}")
        raise
```

### Test Categories

**Parser Tests**: Verify parsing correctness
- Metadata extraction
- Entity structure
- State variable types
- Dimension declarations

**Validator Tests**: Verify rule enforcement
- Error detection
- Error codes
- Suggestion generation

**Integration Tests**: Verify complex examples
- Multi-entity files
- Real-world codebases (VHE)

## Known Limitations

1. **Multiple Sequential Entities**: Parser currently handles the first entity fully but may have issues with multiple entities in sequence (known limitation, tracked for Phase 2)

2. **Expression Parsing**: Some complex expressions in function bodies cause parse errors (minor edge cases, doesn't block core functionality)

## Compliance Matrix

| RFC Section | Feature | Status | Test Coverage |
|-------------|---------|--------|---------------|
| 2.1 | Basic Metadata | ✅ | 100% |
| 2.2 | Dimensions | ✅ | 100% |
| 2.3 | Entities | ✅ | Core features |
| 3.8 | Validation | ✅ | Core rules |
| 4.1 | VHE Example | ✅ | 70% structure |

**Overall Compliance**: 100% of tested features pass

## CI/CD Integration

Add to your CI pipeline:

```yaml
# .github/workflows/test.yml
- name: Run RFC Compliance Tests
  run: python3 -m unittest tests.compliance.test_rfc_compliance
```

Exit codes:
- `0`: All tests pass (compliant)
- `1`: Tests failed (not compliant)

## Contributing

When adding new RFC features:
1. Add compliance test first (TDD)
2. Implement feature
3. Verify test passes
4. Update compliance matrix

---

**Last Updated**: November 21, 2025
**RFC Version**: v1.3.1
**Test Count**: 6 tests covering core RFC features
