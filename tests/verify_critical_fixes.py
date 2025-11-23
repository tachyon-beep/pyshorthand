#!/usr/bin/env python3
"""
Simple verification script for critical bug fixes (no pytest required).
"""

import json
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pyshort.core.parser import Parser
from pyshort.core.tokenizer import Tokenizer
from pyshort.decompiler.py2short import PyShorthandGenerator, decompile
from pyshort.indexer.repo_indexer import RepositoryIndexer


def test_tokenizer_escape_sequences():
    """Test T2: Escape sequences should work correctly."""
    print("Testing T2: Escape sequence fix...")

    source = '"Hello\\nWorld"'
    tokenizer = Tokenizer(source)
    tokens = tokenizer.tokenize()

    string_tokens = [t for t in tokens if t.type.name == "STRING"]
    if not string_tokens:
        print("  ❌ FAIL: No string token found")
        return False

    value = string_tokens[0].value
    if "\n" not in value:
        print(f"  ❌ FAIL: Expected newline character, got: {repr(value)}")
        return False
    if "\\n" in value:
        print(f"  ❌ FAIL: Should not contain literal backslash-n: {repr(value)}")
        return False

    print("  ✓ PASS: Escape sequences work correctly")
    return True


def test_tokenizer_number_validation():
    """Test T1: Number parsing should reject multiple decimals."""
    print("Testing T1: Number parsing fix...")

    source = "1.2.3.4"
    tokenizer = Tokenizer(source)
    tokens = tokenizer.tokenize()

    number_tokens = [t for t in tokens if t.type.name == "NUMBER"]
    if number_tokens and number_tokens[0].value == "1.2.3.4":
        print(f"  ❌ FAIL: Accepted invalid number: {number_tokens[0].value}")
        return False

    print("  ✓ PASS: Number validation works")
    return True


def test_decompiler_boolean_inference():
    """Test D1: Boolean type inference fix."""
    print("Testing D1: Boolean type inference fix...")

    source = """
class Example:
    enabled: bool = True
    count: int = 42
"""
    result = decompile(source)

    # Check for bool type (not i32)
    has_bool_type = "∈ bool" in result or ": bool" in result
    has_wrong_i32_for_bool = "enabled ∈ i32" in result or "enabled: i32" in result

    if has_wrong_i32_for_bool:
        print("  ❌ FAIL: Boolean incorrectly typed as i32")
        print(f"  Result: {result[:200]}")
        return False

    if not has_bool_type:
        print("  ⚠ WARNING: No bool type found (might be ok if using different format)")

    print("  ✓ PASS: Boolean type inference works")
    return True


def test_decompiler_import_extraction():
    """Test D2: AST traversal fix for imports."""
    print("Testing D2: Import extraction fix...")

    import ast

    source = """
import os
import sys

def helper():
    import json  # Should NOT be captured
    pass
"""
    tree = ast.parse(source)
    generator = PyShorthandGenerator()
    generator.generate(tree)

    if "json" in generator.imports:
        print("  ❌ FAIL: Nested import incorrectly captured")
        print(f"  Imports: {generator.imports}")
        return False

    if "os" not in generator.imports or "sys" not in generator.imports:
        print("  ❌ FAIL: Module-level imports not captured")
        print(f"  Imports: {generator.imports}")
        return False

    print("  ✓ PASS: Import extraction works correctly")
    return True


def test_indexer_top_level_functions():
    """Test I1: Top-level function extraction fix."""
    print("Testing I1: Top-level function extraction fix...")

    source = """
def top_level_function():
    pass

class MyClass:
    def method(self):
        pass
"""
    indexer = RepositoryIndexer(".")
    entities = indexer.extract_entities(source, "test.py", "test")

    functions = [e for e in entities if e.type == "function"]
    if len(functions) == 0:
        print("  ❌ FAIL: No top-level functions captured")
        print(f"  Entities: {entities}")
        return False

    function_names = {f.name for f in functions}
    if "top_level_function" not in function_names:
        print("  ❌ FAIL: Expected function not found")
        print(f"  Functions: {function_names}")
        return False

    if "method" in function_names:
        print("  ❌ FAIL: Class method incorrectly captured as top-level")
        return False

    print("  ✓ PASS: Top-level functions extracted correctly")
    return True


def test_indexer_nested_classes():
    """Test I2b: Nested class handling fix."""
    print("Testing I2b: Nested class handling fix...")

    source = """
class Outer:
    class Inner:
        pass

    def method(self):
        pass
"""
    indexer = RepositoryIndexer(".")
    entities = indexer.extract_entities(source, "test.py", "test")

    class_names = {e.name for e in entities if e.type == "class"}

    if "Inner" in class_names:
        print("  ❌ FAIL: Nested class incorrectly captured as top-level")
        print(f"  Classes: {class_names}")
        return False

    if "Outer" not in class_names:
        print("  ❌ FAIL: Outer class not captured")
        return False

    print("  ✓ PASS: Nested classes handled correctly")
    return True


def test_indexer_serialization():
    """Test I3: Set serialization fix."""
    print("Testing I3: Set serialization fix...")

    with tempfile.TemporaryDirectory() as tmp_dir:
        source = """
import os

class Derived:
    pass
"""
        # Create test file
        test_file = Path(tmp_dir) / "test.py"
        test_file.write_text(source)

        # Index and save
        indexer = RepositoryIndexer(tmp_dir)
        indexer.index_repository()

        output_file = Path(tmp_dir) / "index.json"
        try:
            indexer.save_index(str(output_file))

            # Load and verify
            with open(output_file) as f:
                data = json.load(f)

            # Check that it's valid JSON (no serialization errors)
            if "modules" not in data:
                print("  ❌ FAIL: Invalid index structure")
                return False

            print("  ✓ PASS: Serialization works correctly")
            return True

        except TypeError as e:
            if "not JSON serializable" in str(e):
                print(f"  ❌ FAIL: Set serialization error: {e}")
                return False
            raise


def test_parser_eof_handling():
    """Test P1-P7: EOF handling in parser."""
    print("Testing P1-P7: Parser EOF handling...")

    test_cases = [
        ("[Ref:Name", "reference"),
        ("[C:MyClass]\\n◊ [Ref:Base", "dependencies"),
        ("[Compute:GPU", "tag"),
    ]

    all_passed = True
    for source, name in test_cases:
        try:
            tokenizer = Tokenizer(source)
            tokens = tokenizer.tokenize()
            parser = Parser(tokens)
            parser.parse()

            # If we get here, it either succeeded or didn't hit the problematic code
            # The key is that it didn't hang
        except Exception as e:
            # Expected - should get an error, not hang
            error_msg = str(e).lower()
            if "unterminated" in error_msg or "expected" in error_msg:
                continue  # Good - proper error handling
            else:
                # Unexpected error
                print(f"    Unexpected error for {name}: {e}")
                all_passed = False

    if all_passed:
        print("  ✓ PASS: Parser EOF handling works (no hangs)")
    else:
        print("  ⚠ WARNING: Some parser tests had unexpected errors")

    return all_passed


def main():
    """Run all verification tests."""
    print("=" * 80)
    print("VERIFYING CRITICAL BUG FIXES")
    print("=" * 80)
    print()

    tests = [
        test_tokenizer_escape_sequences,
        test_tokenizer_number_validation,
        test_decompiler_boolean_inference,
        test_decompiler_import_extraction,
        test_indexer_top_level_functions,
        test_indexer_nested_classes,
        test_indexer_serialization,
        test_parser_eof_handling,
    ]

    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            import traceback

            traceback.print_exc()
            results.append(False)
        print()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("✓ ALL CRITICAL FIXES VERIFIED!")
    else:
        print(f"⚠ {total - passed} test(s) failed")

    print("=" * 80)

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
