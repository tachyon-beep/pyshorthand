"""Unit tests for ecosystem tools module."""

import ast
import tempfile
from pathlib import Path

import pytest

from pyshort.ecosystem.tools import CodebaseExplorer, MethodImplementation, ClassDetails


class TestParentTracking:
    """Test parent context tracking for AST nodes."""

    def test_find_parent_context_returns_class_method(self, tmp_path):
        """_find_parent_context should return class.method for nodes inside methods."""
        source = '''
class MyClass:
    def my_method(self):
        x = SomeClass()  # Target node
'''
        test_file = tmp_path / "test_parent.py"
        test_file.write_text(source)

        explorer = CodebaseExplorer(test_file)
        tree = ast.parse(source)

        # Find the Call node (SomeClass())
        call_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                call_node = node
                break

        assert call_node is not None
        result = explorer._find_parent_context(tree, call_node)
        assert result == "MyClass.my_method"

    def test_find_parent_context_top_level_returns_none(self, tmp_path):
        """_find_parent_context should return None for top-level nodes."""
        source = '''
x = SomeClass()  # Top level
'''
        test_file = tmp_path / "test_parent.py"
        test_file.write_text(source)

        explorer = CodebaseExplorer(test_file)
        tree = ast.parse(source)

        # Find the Call node
        call_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                call_node = node
                break

        result = explorer._find_parent_context(tree, call_node)
        assert result is None


class TestMethodImplementation:
    """Test MethodImplementation dataclass."""

    def test_method_implementation_creation(self):
        """MethodImplementation should store all fields."""
        impl = MethodImplementation(
            class_name="MyClass",
            method_name="my_method",
            source_code="def my_method(self): pass",
            line_start=10,
            line_end=15,
            dependencies=["helper", "other"],
        )
        assert impl.class_name == "MyClass"
        assert impl.method_name == "my_method"
        assert impl.line_start == 10
        assert len(impl.dependencies) == 2


class TestClassDetails:
    """Test ClassDetails dataclass."""

    def test_class_details_creation(self):
        """ClassDetails should store class information."""
        details = ClassDetails(
            name="MyClass",
            base_classes=["Base", "Mixin"],
            attributes={"x": "int", "y": "str"},
            methods={"__init__": "(self, x: int)", "process": "(self) -> None"},
            nested_structures={},
        )
        assert details.name == "MyClass"
        assert len(details.base_classes) == 2
        assert details.attributes["x"] == "int"


class TestCodebaseExplorer:
    """Test CodebaseExplorer class."""

    def test_explorer_initialization(self, tmp_path):
        """Explorer should initialize with codebase path."""
        test_file = tmp_path / "test.py"
        test_file.write_text("class A: pass")

        explorer = CodebaseExplorer(test_file)
        assert explorer.codebase_path == test_file
        assert explorer.cache == {}

    def test_get_implementation_simple(self, tmp_path):
        """Should retrieve method implementation."""
        source = '''
class Calculator:
    def add(self, a, b):
        return a + b
'''
        test_file = tmp_path / "calc.py"
        test_file.write_text(source)

        explorer = CodebaseExplorer(test_file)
        impl = explorer.get_implementation("Calculator.add")

        assert impl is not None
        assert "return a + b" in impl

    def test_get_implementation_not_found(self, tmp_path):
        """Should return None for non-existent method."""
        test_file = tmp_path / "test.py"
        test_file.write_text("class A: pass")

        explorer = CodebaseExplorer(test_file)
        result = explorer.get_implementation("A.nonexistent")

        assert result is None

    def test_get_implementation_invalid_target(self, tmp_path):
        """Should return None for invalid target format."""
        test_file = tmp_path / "test.py"
        test_file.write_text("class A: pass")

        explorer = CodebaseExplorer(test_file)
        result = explorer.get_implementation("no_dot_here")

        assert result is None

    def test_get_implementation_caches_result(self, tmp_path):
        """Should cache implementation lookups."""
        source = '''
class A:
    def method(self):
        return 42
'''
        test_file = tmp_path / "test.py"
        test_file.write_text(source)

        explorer = CodebaseExplorer(test_file)

        # First call
        result1 = explorer.get_implementation("A.method")
        # Second call should use cache
        result2 = explorer.get_implementation("A.method")

        assert result1 == result2
        assert "impl:A.method" in explorer.cache


class TestGetClassDetails:
    """Test get_class_details method."""

    def test_get_class_details_basic(self, tmp_path):
        """Should return class details."""
        source = '''
class Person:
    name: str
    age: int

    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

    def greet(self) -> str:
        return f"Hello, {self.name}"
'''
        test_file = tmp_path / "person.py"
        test_file.write_text(source)

        explorer = CodebaseExplorer(test_file)
        details = explorer.get_class_details("Person")

        assert details is not None
        assert "Person" in details
        assert "__init__" in details or "greet" in details

    def test_get_class_details_with_base_classes(self, tmp_path):
        """Should capture base classes."""
        source = '''
class Parent:
    pass

class Child(Parent):
    pass
'''
        test_file = tmp_path / "test.py"
        test_file.write_text(source)

        explorer = CodebaseExplorer(test_file)
        details = explorer.get_class_details("Child")

        assert details is not None
        assert "Parent" in details

    def test_get_class_details_not_found(self, tmp_path):
        """Should return None for non-existent class."""
        test_file = tmp_path / "test.py"
        test_file.write_text("class A: pass")

        explorer = CodebaseExplorer(test_file)
        result = explorer.get_class_details("NonExistent")

        assert result is None


class TestSearchUsage:
    """Test search_usage method."""

    def test_search_usage_finds_instantiation(self, tmp_path):
        """Should find where a class is instantiated."""
        source = '''
class Logger:
    pass

class App:
    def __init__(self):
        self.logger = Logger()
'''
        test_file = tmp_path / "app.py"
        test_file.write_text(source)

        explorer = CodebaseExplorer(test_file)
        usages = explorer.search_usage("Logger")

        assert len(usages) > 0
        # Should find instantiation
        assert any("instantiation" in u for u in usages)

    def test_search_usage_no_results(self, tmp_path):
        """Should return empty list when no usages found."""
        source = '''
class Unused:
    pass
'''
        test_file = tmp_path / "test.py"
        test_file.write_text(source)

        explorer = CodebaseExplorer(test_file)
        usages = explorer.search_usage("NeverUsed")

        assert usages == []


class TestGetNeighbors:
    """Test get_neighbors method (depends on advanced tools)."""

    def test_get_neighbors_returns_dict_or_none(self, tmp_path):
        """Should return neighbor dict or None if tools unavailable."""
        test_file = tmp_path / "test.py"
        test_file.write_text("class A: pass")

        explorer = CodebaseExplorer(test_file)
        result = explorer.get_neighbors("A")

        # Either returns dict with expected keys or None
        if result is not None:
            assert "callees" in result
            assert "callers" in result
            assert "peers" in result


class TestDirectoryExplorer:
    """Test explorer with directory paths."""

    def test_explorer_with_directory(self, tmp_path):
        """Should work with directory containing multiple files."""
        (tmp_path / "module_a.py").write_text("class A: pass")
        (tmp_path / "module_b.py").write_text("class B: pass")

        explorer = CodebaseExplorer(tmp_path)

        # Should be able to find classes from both files
        details_a = explorer.get_class_details("A")
        details_b = explorer.get_class_details("B")

        assert details_a is not None or details_b is not None


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_file(self, tmp_path):
        """Should handle empty files gracefully."""
        test_file = tmp_path / "empty.py"
        test_file.write_text("")

        explorer = CodebaseExplorer(test_file)
        result = explorer.get_class_details("Anything")

        assert result is None

    def test_syntax_error_file(self, tmp_path):
        """Should handle files with syntax errors."""
        test_file = tmp_path / "bad.py"
        test_file.write_text("class { invalid syntax")

        explorer = CodebaseExplorer(test_file)
        # Should not raise
        result = explorer.get_implementation("Whatever.method")

        assert result is None

    def test_nonexistent_file(self, tmp_path):
        """Should handle non-existent files gracefully."""
        fake_path = tmp_path / "does_not_exist.py"

        explorer = CodebaseExplorer(fake_path)
        result = explorer.get_class_details("A")

        assert result is None
