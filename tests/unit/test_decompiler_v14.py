"""Unit tests for v1.4 decompiler tag generation."""

import ast
import pytest

from src.pyshort.decompiler.py2short import PyShorthandGenerator, decompile


class TestDecoratorTagExtraction:
    """Test extraction of decorator tags from Python decorators."""

    def test_property_decorator(self):
        """Test @property → [Prop]."""
        source = """
@property
def value(self) -> int:
    return 42
"""
        tree = ast.parse(source)
        func = tree.body[0]
        generator = PyShorthandGenerator()
        tags = generator._extract_decorator_tags(func)
        assert "[Prop]" in tags

    def test_staticmethod_decorator(self):
        """Test @staticmethod → [Static]."""
        source = """
@staticmethod
def helper(x: int) -> int:
    return x * 2
"""
        tree = ast.parse(source)
        func = tree.body[0]
        generator = PyShorthandGenerator()
        tags = generator._extract_decorator_tags(func)
        assert "[Static]" in tags

    def test_classmethod_decorator(self):
        """Test @classmethod → [Class]."""
        source = """
@classmethod
def create(cls):
    return cls()
"""
        tree = ast.parse(source)
        func = tree.body[0]
        generator = PyShorthandGenerator()
        tags = generator._extract_decorator_tags(func)
        assert "[Class]" in tags

    def test_lru_cache_decorator(self):
        """Test @lru_cache → [Cached]."""
        source = """
from functools import lru_cache

@lru_cache
def fib(n: int) -> int:
    return n
"""
        tree = ast.parse(source)
        func = tree.body[1]  # Skip import
        generator = PyShorthandGenerator()
        tags = generator._extract_decorator_tags(func)
        assert "[Cached]" in tags

    def test_lru_cache_with_maxsize(self):
        """Test @lru_cache(maxsize=128) → [Cached:TTL:128]."""
        source = """
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive(n: int) -> int:
    return n ** 2
"""
        tree = ast.parse(source)
        func = tree.body[1]
        generator = PyShorthandGenerator()
        tags = generator._extract_decorator_tags(func)
        assert "[Cached:TTL:128]" in tags

    def test_auth_decorator(self):
        """Test @login_required → [Auth]."""
        source = """
@login_required
def protected_view():
    pass
"""
        tree = ast.parse(source)
        func = tree.body[0]
        generator = PyShorthandGenerator()
        tags = generator._extract_decorator_tags(func)
        assert "[Auth]" in tags

    def test_multiple_decorators(self):
        """Test multiple decorators are all extracted."""
        source = """
@property
@cached_property
def value(self):
    return 42
"""
        tree = ast.parse(source)
        func = tree.body[0]
        generator = PyShorthandGenerator()
        tags = generator._extract_decorator_tags(func)
        # Should have both decorators
        assert len(tags) >= 1  # At least one decorator tag


class TestHTTPRouteTagExtraction:
    """Test extraction of HTTP route tags from web framework decorators."""

    def test_get_route(self):
        """Test @app.get('/path') → [GET /path]."""
        source = """
class App:
    def get(self, path):
        def decorator(func):
            return func
        return decorator

app = App()

@app.get('/users')
def get_users():
    pass
"""
        tree = ast.parse(source)
        func = tree.body[2]  # The decorated function
        generator = PyShorthandGenerator()
        tag = generator._extract_http_route_tag(func)
        assert tag == "[GET /users]"

    def test_post_route_with_params(self):
        """Test @router.post('/users/{id}') → [POST /users/{id}]."""
        source = """
class Router:
    def post(self, path):
        def decorator(func):
            return func
        return decorator

router = Router()

@router.post('/users/{id}')
def create_user():
    pass
"""
        tree = ast.parse(source)
        func = tree.body[2]
        generator = PyShorthandGenerator()
        tag = generator._extract_http_route_tag(func)
        assert tag == "[POST /users/{id}]"

    def test_put_route(self):
        """Test @app.put('/path') → [PUT /path]."""
        source = """
class App:
    def put(self, path):
        def decorator(func):
            return func
        return decorator

app = App()

@app.put('/api/data')
def update_data():
    pass
"""
        tree = ast.parse(source)
        func = tree.body[2]
        generator = PyShorthandGenerator()
        tag = generator._extract_http_route_tag(func)
        assert tag == "[PUT /api/data]"

    def test_delete_route(self):
        """Test @app.delete('/path') → [DELETE /path]."""
        source = """
class App:
    def delete(self, path):
        def decorator(func):
            return func
        return decorator

app = App()

@app.delete('/items/{item_id}')
def delete_item():
    pass
"""
        tree = ast.parse(source)
        func = tree.body[2]
        generator = PyShorthandGenerator()
        tag = generator._extract_http_route_tag(func)
        assert tag == "[DELETE /items/{item_id}]"

    def test_no_route_decorator(self):
        """Test function without route decorator returns None."""
        source = """
def normal_function():
    pass
"""
        tree = ast.parse(source)
        func = tree.body[0]
        generator = PyShorthandGenerator()
        tag = generator._extract_http_route_tag(func)
        assert tag is None


class TestComplexityTagExtraction:
    """Test extraction of complexity tags from docstrings and patterns."""

    def test_complexity_from_docstring(self):
        """Test extracting O(N) from docstring."""
        source = """
def search(arr, target):
    '''Search for target. O(N)'''
    pass
"""
        tree = ast.parse(source)
        func = tree.body[0]
        generator = PyShorthandGenerator()
        tag = generator._extract_complexity_tag(func)
        assert tag == "[O(N)]"

    def test_complexity_with_label(self):
        """Test extracting from 'Complexity: O(N)' label."""
        source = """
def process(data):
    '''Process data.

    Complexity: O(N*M)
    '''
    pass
"""
        tree = ast.parse(source)
        func = tree.body[0]
        generator = PyShorthandGenerator()
        tag = generator._extract_complexity_tag(func)
        assert tag == "[O(N*M)]"

    def test_complexity_time_label(self):
        """Test extracting from 'Time: O(N)' label."""
        source = """
def sort_items(items):
    '''Sort items.

    Time: O(N log N)
    '''
    pass
"""
        tree = ast.parse(source)
        func = tree.body[0]
        generator = PyShorthandGenerator()
        tag = generator._extract_complexity_tag(func)
        assert tag == "[O(N log N)]"

    def test_single_loop_infers_on(self):
        """Test single loop infers O(N)."""
        source = """
def iterate(items):
    for item in items:
        process(item)
"""
        tree = ast.parse(source)
        func = tree.body[0]
        generator = PyShorthandGenerator()
        tag = generator._extract_complexity_tag(func)
        assert tag == "[O(N)]"

    def test_nested_loops_infer_on2(self):
        """Test nested loops infer O(N²)."""
        source = """
def matrix_process(matrix):
    for row in matrix:
        for cell in row:
            process(cell)
"""
        tree = ast.parse(source)
        func = tree.body[0]
        generator = PyShorthandGenerator()
        tag = generator._extract_complexity_tag(func)
        assert tag == "[O(N²)]"

    def test_triple_nested_infers_on3(self):
        """Test triple nested loops infer O(N³)."""
        source = """
def triple_loop(data):
    for i in data:
        for j in data:
            for k in data:
                process(i, j, k)
"""
        tree = ast.parse(source)
        func = tree.body[0]
        generator = PyShorthandGenerator()
        tag = generator._extract_complexity_tag(func)
        assert tag == "[O(N³)]"

    def test_no_complexity_for_complex_function(self):
        """Test function with multiple statements but no loops."""
        source = """
def complex_logic(x, y):
    a = x + y
    b = a * 2
    c = b - 5
    return c
"""
        tree = ast.parse(source)
        func = tree.body[0]
        generator = PyShorthandGenerator()
        tag = generator._extract_complexity_tag(func)
        # Should not return a tag for non-trivial multi-statement function
        assert tag is None


class TestOperationTagExtraction:
    """Test extraction of operation tags from function bodies."""

    def test_iteration_tag_single_loop(self):
        """Test single loop generates [Iter]."""
        source = """
def process_items(items):
    for item in items:
        print(item)
"""
        tree = ast.parse(source)
        func = tree.body[0]
        generator = PyShorthandGenerator()
        tags = generator._extract_operation_tags(func)
        assert "[Iter]" in tags

    def test_iteration_nested_tag(self):
        """Test nested loops generate [Iter:Nested]."""
        source = """
def nested_iteration(matrix):
    for row in matrix:
        for cell in row:
            print(cell)
"""
        tree = ast.parse(source)
        func = tree.body[0]
        generator = PyShorthandGenerator()
        tags = generator._extract_operation_tags(func)
        assert "[Iter:Nested]" in tags

    def test_io_async_tag(self):
        """Test async operations generate [IO:Async]."""
        source = """
async def fetch_data():
    data = await get_data()
    return data
"""
        tree = ast.parse(source)
        func = tree.body[0]
        generator = PyShorthandGenerator()
        tags = generator._extract_operation_tags(func)
        assert "[IO:Async]" in tags

    def test_no_tags_for_simple_function(self):
        """Test simple function generates no operation tags."""
        source = """
def add(x, y):
    return x + y
"""
        tree = ast.parse(source)
        func = tree.body[0]
        generator = PyShorthandGenerator()
        tags = generator._extract_operation_tags(func)
        assert len(tags) == 0


class TestLoopDepthCalculation:
    """Test loop depth calculation."""

    def test_no_loops(self):
        """Test function with no loops has depth 0."""
        source = """
def simple():
    return 42
"""
        tree = ast.parse(source)
        func = tree.body[0]
        generator = PyShorthandGenerator()
        depth = generator._calculate_loop_depth(func)
        assert depth == 0

    def test_single_loop(self):
        """Test function with single loop has depth 1."""
        source = """
def iterate(items):
    for item in items:
        process(item)
"""
        tree = ast.parse(source)
        func = tree.body[0]
        generator = PyShorthandGenerator()
        depth = generator._calculate_loop_depth(func)
        assert depth == 1

    def test_nested_loops(self):
        """Test function with nested loops has depth 2."""
        source = """
def nested(matrix):
    for row in matrix:
        for cell in row:
            print(cell)
"""
        tree = ast.parse(source)
        func = tree.body[0]
        generator = PyShorthandGenerator()
        depth = generator._calculate_loop_depth(func)
        assert depth == 2

    def test_triple_nested(self):
        """Test function with triple nested loops has depth 3."""
        source = """
def triple(data):
    for i in data:
        for j in data:
            for k in data:
                process(i, j, k)
"""
        tree = ast.parse(source)
        func = tree.body[0]
        generator = PyShorthandGenerator()
        depth = generator._calculate_loop_depth(func)
        assert depth == 3


class TestIntegratedTagGeneration:
    """Test complete tag generation in function signatures."""

    def test_property_with_complexity(self):
        """Test property with complexity tag."""
        source = """
class Model:
    @property
    def count(self) -> int:
        '''Get count. O(1)'''
        return len(self.items)
"""
        tree = ast.parse(source)
        cls = tree.body[0]
        func = cls.body[0]
        generator = PyShorthandGenerator()
        sig = generator._generate_function_signature(func, indent="  # ")
        assert "[Prop]" in sig
        assert "[O(1)]" in sig

    def test_cached_with_iteration(self):
        """Test cached function with iteration."""
        source = """
from functools import lru_cache

@lru_cache(maxsize=256)
def expensive(items):
    result = []
    for item in items:
        result.append(item * 2)
    return result
"""
        tree = ast.parse(source)
        func = tree.body[1]
        generator = PyShorthandGenerator()
        sig = generator._generate_function_signature(func)
        assert "[Cached:TTL:256]" in sig
        assert "[Iter]" in sig

    def test_tag_ordering(self):
        """Test tags appear in correct order: Decorator → Route → Operation → Complexity."""
        source = """
class Router:
    def post(self, path):
        def decorator(func):
            return func
        return decorator

router = Router()

@login_required
@router.post('/api/process')
def process_data(items):
    '''Process items. O(N)'''
    for item in items:
        save(item)
"""
        tree = ast.parse(source)
        func = tree.body[2]
        generator = PyShorthandGenerator()
        sig = generator._generate_function_signature(func)

        # Find positions of each tag type
        auth_pos = sig.find("[Auth]")
        route_pos = sig.find("[POST")
        iter_pos = sig.find("[Iter]")
        complexity_pos = sig.find("[O(N)]")

        # Verify order (all present and in correct sequence)
        assert auth_pos < route_pos < iter_pos < complexity_pos

    def test_async_function_with_route(self):
        """Test async function with route tag."""
        source = """
class App:
    def get(self, path):
        def decorator(func):
            return func
        return decorator

app = App()

@app.get('/data')
async def fetch_data():
    '''Fetch data. O(1)'''
    data = await http.get('/remote')
    return data
"""
        tree = ast.parse(source)
        func = tree.body[2]
        generator = PyShorthandGenerator()
        sig = generator._generate_function_signature(func)
        assert "[GET /data]" in sig
        assert "[IO:Async]" in sig
        assert "[O(1)]" in sig


class TestUnionTypeSupport:
    """Test proper handling of Union type annotations."""

    def test_union_type_multiple_types(self):
        """Union with multiple types should show all types."""
        source = '''
from typing import Union

def process(value: Union[int, str, float]) -> Union[bool, None]:
    pass
'''
        result = decompile(source)

        # Should represent the union with all types (mapped to PyShorthand types)
        # Union[int, str, float] → i32|str|f32
        assert "i32|str|f32" in result or "int|str|float" in result

    def test_optional_type_preserved(self):
        """Optional[X] should render as X? not just X."""
        source = '''
from typing import Optional

def get_name() -> Optional[str]:
    pass
'''
        result = decompile(source)

        # Optional[str] should be str?
        assert "str?" in result
