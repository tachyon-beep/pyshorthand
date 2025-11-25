"""Comprehensive unit tests for repository indexer."""

import json
import tempfile
from pathlib import Path

import pytest

from pyshort.indexer.repo_indexer import (
    EntityInfo,
    ModuleInfo,
    RepositoryIndex,
    RepositoryIndexer,
)


class TestEntityInfo:
    """Test EntityInfo dataclass."""

    def test_entity_info_creation(self):
        """EntityInfo should store all required fields."""
        entity = EntityInfo(
            name="MyClass",
            type="class",
            file_path="/path/to/file.py",
            module_path="module.submodule",
            line_number=42,
        )
        assert entity.name == "MyClass"
        assert entity.type == "class"
        assert entity.line_number == 42
        assert entity.methods == []
        assert entity.dependencies == set()

    def test_entity_info_with_methods(self):
        """EntityInfo should store methods list."""
        entity = EntityInfo(
            name="MyClass",
            type="class",
            file_path="test.py",
            module_path="test",
            line_number=1,
            methods=["__init__", "forward", "backward"],
        )
        assert len(entity.methods) == 3
        assert "forward" in entity.methods


class TestModuleInfo:
    """Test ModuleInfo dataclass."""

    def test_module_info_creation(self):
        """ModuleInfo should store module metadata."""
        module = ModuleInfo(
            module_path="mypackage.mymodule",
            file_path="/path/to/mymodule.py",
            line_count=100,
        )
        assert module.module_path == "mypackage.mymodule"
        assert module.line_count == 100
        assert module.entities == []
        assert module.imports == set()


class TestRepositoryIndexer:
    """Test RepositoryIndexer class."""

    def test_should_exclude_venv(self):
        """Should exclude venv directories."""
        indexer = RepositoryIndexer(".")
        assert indexer.should_exclude(Path("./venv/lib/python"))
        assert indexer.should_exclude(Path("./.venv/bin"))

    def test_should_exclude_pycache(self):
        """Should exclude __pycache__ directories."""
        indexer = RepositoryIndexer(".")
        assert indexer.should_exclude(Path("./__pycache__/module.cpython-310.pyc"))

    def test_should_exclude_dot_directories(self, tmp_path):
        """Should exclude directories starting with dot."""
        dot_dir = tmp_path / ".hidden"
        dot_dir.mkdir()
        indexer = RepositoryIndexer(str(tmp_path))
        assert indexer.should_exclude(dot_dir)

    def test_should_not_exclude_regular_dirs(self, tmp_path):
        """Should not exclude regular directories."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        indexer = RepositoryIndexer(str(tmp_path))
        assert not indexer.should_exclude(src_dir)

    def test_get_module_path_simple(self, tmp_path):
        """Should convert simple file path to module path."""
        indexer = RepositoryIndexer(str(tmp_path))
        file_path = tmp_path / "mymodule.py"
        assert indexer.get_module_path(file_path) == "mymodule"

    def test_get_module_path_nested(self, tmp_path):
        """Should convert nested file path to dotted module path."""
        indexer = RepositoryIndexer(str(tmp_path))
        nested = tmp_path / "package" / "subpackage" / "module.py"
        assert indexer.get_module_path(nested) == "package.subpackage.module"

    def test_get_module_path_strips_src(self, tmp_path):
        """Should strip 'src' prefix from module path."""
        indexer = RepositoryIndexer(str(tmp_path))
        src_file = tmp_path / "src" / "mypackage" / "module.py"
        assert indexer.get_module_path(src_file) == "mypackage.module"

    def test_get_module_path_strips_init(self, tmp_path):
        """Should strip __init__ from module path."""
        indexer = RepositoryIndexer(str(tmp_path))
        init_file = tmp_path / "mypackage" / "__init__.py"
        assert indexer.get_module_path(init_file) == "mypackage"


class TestExtractImports:
    """Test import extraction."""

    def test_extract_simple_import(self):
        """Should extract simple import statements."""
        source = "import os\nimport sys"
        indexer = RepositoryIndexer(".")
        imports = indexer.extract_imports(source)
        assert "os" in imports
        assert "sys" in imports

    def test_extract_from_import(self):
        """Should extract from...import statements."""
        source = "from pathlib import Path\nfrom typing import Optional"
        indexer = RepositoryIndexer(".")
        imports = indexer.extract_imports(source)
        assert "pathlib" in imports
        assert "typing" in imports

    def test_ignores_nested_imports(self):
        """Should not capture imports inside functions."""
        source = '''
import os

def helper():
    import json
    return json.dumps({})
'''
        indexer = RepositoryIndexer(".")
        imports = indexer.extract_imports(source)
        assert "os" in imports
        assert "json" not in imports

    def test_handles_syntax_errors(self):
        """Should return empty set on syntax errors."""
        source = "import os\nthis is not valid python!!!"
        indexer = RepositoryIndexer(".")
        imports = indexer.extract_imports(source)
        # Should not raise, returns what it can parse or empty
        assert isinstance(imports, set)


class TestExtractEntities:
    """Test entity extraction."""

    def test_extract_class(self):
        """Should extract class entities."""
        source = '''
class MyClass:
    def method(self):
        pass
'''
        indexer = RepositoryIndexer(".")
        entities = indexer.extract_entities(source, "test.py", "test")

        assert len(entities) == 1
        assert entities[0].name == "MyClass"
        assert entities[0].type == "class"
        assert "method" in entities[0].methods

    def test_extract_function(self):
        """Should extract top-level functions."""
        source = '''
def my_function(x, y):
    return x + y
'''
        indexer = RepositoryIndexer(".")
        entities = indexer.extract_entities(source, "test.py", "test")

        assert len(entities) == 1
        assert entities[0].name == "my_function"
        assert entities[0].type == "function"

    def test_extract_class_with_base(self):
        """Should extract base class dependencies."""
        source = '''
class Child(Parent):
    pass
'''
        indexer = RepositoryIndexer(".")
        entities = indexer.extract_entities(source, "test.py", "test")

        assert "Parent" in entities[0].dependencies

    def test_extract_class_with_qualified_base(self):
        """Should extract qualified base class (e.g., nn.Module)."""
        source = '''
class MyModel(torch.nn.Module):
    pass
'''
        indexer = RepositoryIndexer(".")
        entities = indexer.extract_entities(source, "test.py", "test")

        assert "torch.nn.Module" in entities[0].dependencies

    def test_ignores_nested_classes(self):
        """Should not extract nested classes as top-level."""
        source = '''
class Outer:
    class Inner:
        pass
'''
        indexer = RepositoryIndexer(".")
        entities = indexer.extract_entities(source, "test.py", "test")

        names = {e.name for e in entities}
        assert "Outer" in names
        assert "Inner" not in names


class TestIndexFile:
    """Test single file indexing."""

    def test_index_valid_file(self, tmp_path):
        """Should index a valid Python file."""
        source = '''
import os

class MyClass:
    def method(self):
        pass

def helper():
    pass
'''
        test_file = tmp_path / "test.py"
        test_file.write_text(source)

        indexer = RepositoryIndexer(str(tmp_path))
        module_info = indexer.index_file(test_file)

        assert module_info is not None
        assert module_info.module_path == "test"
        assert "os" in module_info.imports
        assert len(module_info.entities) == 2  # MyClass and helper
        assert module_info.line_count > 0

    def test_index_file_with_syntax_error(self, tmp_path):
        """Should return None for files with syntax errors."""
        test_file = tmp_path / "bad.py"
        test_file.write_text("this is not valid python {{{")

        indexer = RepositoryIndexer(str(tmp_path))
        result = indexer.index_file(test_file)

        # Should handle gracefully
        assert result is None or result.entities == []


class TestIndexRepository:
    """Test full repository indexing."""

    def test_index_simple_repo(self, tmp_path):
        """Should index a simple repository structure."""
        # Create package structure
        pkg_dir = tmp_path / "mypackage"
        pkg_dir.mkdir()

        (pkg_dir / "__init__.py").write_text("")
        (pkg_dir / "module.py").write_text('''
class MyClass:
    pass
''')

        indexer = RepositoryIndexer(str(tmp_path))
        index = indexer.index_repository()

        assert len(index.modules) >= 1
        # Entity map uses fully qualified names
        assert any("MyClass" in key for key in index.entity_map.keys())

    def test_computes_statistics(self, tmp_path):
        """Should compute repository statistics."""
        (tmp_path / "module.py").write_text('''
class A:
    pass

class B:
    pass

def func():
    pass
''')

        indexer = RepositoryIndexer(str(tmp_path))
        index = indexer.index_repository()

        assert index.statistics["total_classes"] == 2
        assert index.statistics["total_functions"] == 1
        assert index.statistics["total_entities"] == 3


class TestDependencyGraph:
    """Test dependency graph building."""

    def test_builds_internal_dependencies(self, tmp_path):
        """Should build dependency graph for internal modules."""
        (tmp_path / "module_a.py").write_text("class A: pass")
        (tmp_path / "module_b.py").write_text("import module_a\nclass B: pass")

        indexer = RepositoryIndexer(str(tmp_path))
        indexer.index_repository()

        assert "module_a" in indexer.index.dependency_graph.get("module_b", set())


class TestSaveIndex:
    """Test index serialization."""

    def test_save_and_load_index(self, tmp_path):
        """Should save index to JSON and load it back."""
        (tmp_path / "module.py").write_text('''
import os

class MyClass(Base):
    def method(self):
        pass
''')

        indexer = RepositoryIndexer(str(tmp_path))
        indexer.index_repository()

        output_file = tmp_path / "index.json"
        indexer.save_index(str(output_file))

        # Should not raise
        with open(output_file) as f:
            data = json.load(f)

        assert "modules" in data
        assert "dependency_graph" in data
        assert "statistics" in data

    def test_sets_serialized_as_lists(self, tmp_path):
        """Dependencies (sets) should serialize as lists."""
        (tmp_path / "module.py").write_text("class Child(Parent): pass")

        indexer = RepositoryIndexer(str(tmp_path))
        indexer.index_repository()

        output_file = tmp_path / "index.json"
        indexer.save_index(str(output_file))

        with open(output_file) as f:
            data = json.load(f)

        # Check dependencies are lists, not sets (JSON can't serialize sets)
        for module in data["modules"].values():
            for entity in module.get("entities", []):
                deps = entity.get("dependencies", [])
                assert isinstance(deps, list)
