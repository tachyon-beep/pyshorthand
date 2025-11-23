"""Unit tests for v1.4 validator enhancements."""

import pytest

from pyshort.core.ast_nodes import DiagnosticSeverity, Function, Metadata, PyShortAST, Tag
from pyshort.core.validator import (
    ComplexityTagValidator,
    DecoratorTagValidator,
    HTTPRouteValidator,
    ValidTagsRule,
)


class TestValidTagsRuleV14:
    """Test ValidTagsRule with v1.4 compatibility."""

    def test_accepts_v13_operation_tags(self):
        """Test that v1.3 operation tags are still valid."""
        tag = Tag(base="Lin", qualifiers=["MatMul"], tag_type="operation")
        func = Function(name="test", tags=[tag])
        ast = PyShortAST(functions=[func])

        rule = ValidTagsRule()
        diagnostics = list(rule.check(ast))
        assert len(diagnostics) == 0

    def test_accepts_v14_complexity_tags(self):
        """Test that v1.4 complexity tags are valid."""
        tag = Tag(base="O(N)", tag_type="complexity")
        func = Function(name="test", tags=[tag])
        ast = PyShortAST(functions=[func])

        rule = ValidTagsRule()
        diagnostics = list(rule.check(ast))
        assert len(diagnostics) == 0

    def test_accepts_v14_decorator_tags(self):
        """Test that v1.4 decorator tags are valid."""
        tag = Tag(base="Prop", tag_type="decorator")
        func = Function(name="test", tags=[tag])
        ast = PyShortAST(functions=[func])

        rule = ValidTagsRule()
        diagnostics = list(rule.check(ast))
        assert len(diagnostics) == 0

    def test_accepts_v14_http_route_tags(self):
        """Test that v1.4 HTTP route tags are valid."""
        tag = Tag(base="GET /users", tag_type="http_route", http_method="GET", http_path="/users")
        func = Function(name="test", tags=[tag])
        ast = PyShortAST(functions=[func])

        rule = ValidTagsRule()
        diagnostics = list(rule.check(ast))
        assert len(diagnostics) == 0

    def test_rejects_invalid_operation_tag(self):
        """Test that invalid operation tag base is rejected."""
        tag = Tag(base="InvalidTag", qualifiers=[], tag_type="operation")
        func = Function(name="test", tags=[tag])
        ast = PyShortAST(functions=[func])

        rule = ValidTagsRule()
        diagnostics = list(rule.check(ast))
        assert len(diagnostics) == 1
        assert diagnostics[0].severity == DiagnosticSeverity.ERROR
        assert "Invalid operation tag base" in diagnostics[0].message

    def test_accepts_custom_tags(self):
        """Test that custom tags are accepted without validation."""
        tag = Tag(base="CustomTag", tag_type="custom")
        func = Function(name="test", tags=[tag])
        ast = PyShortAST(functions=[func])

        rule = ValidTagsRule()
        diagnostics = list(rule.check(ast))
        assert len(diagnostics) == 0


class TestComplexityTagValidator:
    """Test ComplexityTagValidator."""

    def test_valid_simple_complexity(self):
        """Test valid simple complexity notation."""
        tag = Tag(base="O(N)", tag_type="complexity")
        func = Function(name="test", tags=[tag])
        ast = PyShortAST(functions=[func])

        validator = ComplexityTagValidator()
        diagnostics = list(validator.check(ast))
        assert len(diagnostics) == 0

    def test_valid_multi_variable_complexity(self):
        """Test valid multi-variable complexity."""
        tag = Tag(base="O(N*M*D)", tag_type="complexity")
        func = Function(name="test", tags=[tag])
        ast = PyShortAST(functions=[func])

        validator = ComplexityTagValidator()
        diagnostics = list(validator.check(ast))
        assert len(diagnostics) == 0

    def test_valid_batch_complexity(self):
        """Test valid batch complexity."""
        tag = Tag(base="O(B*N*D)", tag_type="complexity")
        func = Function(name="test", tags=[tag])
        ast = PyShortAST(functions=[func])

        validator = ComplexityTagValidator()
        diagnostics = list(validator.check(ast))
        assert len(diagnostics) == 0

    def test_valid_exponent_complexity(self):
        """Test valid exponent complexity."""
        tag = Tag(base="O(N²)", tag_type="complexity")
        func = Function(name="test", tags=[tag])
        ast = PyShortAST(functions=[func])

        validator = ComplexityTagValidator()
        diagnostics = list(validator.check(ast))
        assert len(diagnostics) == 0

    def test_warns_on_unusually_high_complexity(self):
        """Test warning on unusually high complexity."""
        tag = Tag(base="O(N^10)", tag_type="complexity")
        func = Function(name="test", tags=[tag])
        ast = PyShortAST(functions=[func])

        validator = ComplexityTagValidator()
        diagnostics = list(validator.check(ast))
        assert len(diagnostics) == 1
        assert diagnostics[0].severity == DiagnosticSeverity.WARNING
        assert "Unusually high complexity" in diagnostics[0].message

    def test_does_not_validate_non_complexity_tags(self):
        """Test that non-complexity tags are ignored."""
        tag = Tag(base="Lin", tag_type="operation")
        func = Function(name="test", tags=[tag])
        ast = PyShortAST(functions=[func])

        validator = ComplexityTagValidator()
        diagnostics = list(validator.check(ast))
        assert len(diagnostics) == 0


class TestDecoratorTagValidator:
    """Test DecoratorTagValidator."""

    def test_valid_property_decorator(self):
        """Test valid property decorator."""
        tag = Tag(base="Prop", tag_type="decorator")
        func = Function(name="test", tags=[tag])
        ast = PyShortAST(functions=[func])

        validator = DecoratorTagValidator()
        diagnostics = list(validator.check(ast))
        assert len(diagnostics) == 0

    def test_valid_static_decorator(self):
        """Test valid static method decorator."""
        tag = Tag(base="Static", tag_type="decorator")
        func = Function(name="test", tags=[tag])
        ast = PyShortAST(functions=[func])

        validator = DecoratorTagValidator()
        diagnostics = list(validator.check(ast))
        assert len(diagnostics) == 0

    def test_valid_cached_with_args(self):
        """Test valid cached decorator with arguments."""
        tag = Tag(base="Cached", qualifiers=["TTL", "60"], tag_type="decorator")
        func = Function(name="test", tags=[tag])
        ast = PyShortAST(functions=[func])

        validator = DecoratorTagValidator()
        diagnostics = list(validator.check(ast))
        assert len(diagnostics) == 0

    def test_detects_prop_static_conflict(self):
        """Test detection of Prop + Static conflict."""
        tag1 = Tag(base="Prop", tag_type="decorator")
        tag2 = Tag(base="Static", tag_type="decorator")
        func = Function(name="test", tags=[tag1, tag2])
        ast = PyShortAST(functions=[func])

        validator = DecoratorTagValidator()
        diagnostics = list(validator.check(ast))
        assert len(diagnostics) == 1
        assert diagnostics[0].severity == DiagnosticSeverity.ERROR
        assert "conflicting decorators" in diagnostics[0].message.lower()
        assert "Prop" in diagnostics[0].message
        assert "Static" in diagnostics[0].message

    def test_detects_prop_class_conflict(self):
        """Test detection of Prop + Class conflict."""
        tag1 = Tag(base="Prop", tag_type="decorator")
        tag2 = Tag(base="Class", tag_type="decorator")
        func = Function(name="test", tags=[tag1, tag2])
        ast = PyShortAST(functions=[func])

        validator = DecoratorTagValidator()
        diagnostics = list(validator.check(ast))
        assert len(diagnostics) == 1
        assert "conflicting decorators" in diagnostics[0].message.lower()

    def test_valid_rate_limit(self):
        """Test valid rate limit decorator."""
        tag = Tag(base="RateLimit", qualifiers=["100"], tag_type="decorator")
        func = Function(name="test", tags=[tag])
        ast = PyShortAST(functions=[func])

        validator = DecoratorTagValidator()
        diagnostics = list(validator.check(ast))
        assert len(diagnostics) == 0

    def test_invalid_rate_limit_negative(self):
        """Test invalid rate limit (negative)."""
        tag = Tag(base="RateLimit", qualifiers=["-1"], tag_type="decorator")
        func = Function(name="test", tags=[tag])
        ast = PyShortAST(functions=[func])

        validator = DecoratorTagValidator()
        diagnostics = list(validator.check(ast))
        assert len(diagnostics) == 1
        assert diagnostics[0].severity == DiagnosticSeverity.ERROR
        assert "must be positive" in diagnostics[0].message

    def test_invalid_rate_limit_zero(self):
        """Test invalid rate limit (zero)."""
        tag = Tag(base="RateLimit", qualifiers=["0"], tag_type="decorator")
        func = Function(name="test", tags=[tag])
        ast = PyShortAST(functions=[func])

        validator = DecoratorTagValidator()
        diagnostics = list(validator.check(ast))
        assert len(diagnostics) == 1
        assert "must be positive" in diagnostics[0].message

    def test_invalid_rate_limit_non_numeric(self):
        """Test invalid rate limit (non-numeric)."""
        tag = Tag(base="RateLimit", qualifiers=["abc"], tag_type="decorator")
        func = Function(name="test", tags=[tag])
        ast = PyShortAST(functions=[func])

        validator = DecoratorTagValidator()
        diagnostics = list(validator.check(ast))
        assert len(diagnostics) == 1
        assert "Invalid rate limit value" in diagnostics[0].message or "must be a number" in diagnostics[0].suggestion

    def test_does_not_validate_non_decorator_tags(self):
        """Test that non-decorator tags are ignored."""
        tag = Tag(base="Lin", tag_type="operation")
        func = Function(name="test", tags=[tag])
        ast = PyShortAST(functions=[func])

        validator = DecoratorTagValidator()
        diagnostics = list(validator.check(ast))
        assert len(diagnostics) == 0


class TestHTTPRouteValidator:
    """Test HTTPRouteValidator."""

    def test_valid_get_route(self):
        """Test valid GET route."""
        tag = Tag(base="GET /users", tag_type="http_route", http_method="GET", http_path="/users")
        func = Function(name="test", tags=[tag])
        ast = PyShortAST(functions=[func])

        validator = HTTPRouteValidator()
        diagnostics = list(validator.check(ast))
        assert len(diagnostics) == 0

    def test_valid_post_route(self):
        """Test valid POST route."""
        tag = Tag(
            base="POST /api/users",
            tag_type="http_route",
            http_method="POST",
            http_path="/api/users"
        )
        func = Function(name="test", tags=[tag])
        ast = PyShortAST(functions=[func])

        validator = HTTPRouteValidator()
        diagnostics = list(validator.check(ast))
        assert len(diagnostics) == 0

    def test_valid_route_with_params(self):
        """Test valid route with path parameters."""
        tag = Tag(
            base="GET /users/{id}",
            tag_type="http_route",
            http_method="GET",
            http_path="/users/{id}"
        )
        func = Function(name="test", tags=[tag])
        ast = PyShortAST(functions=[func])

        validator = HTTPRouteValidator()
        diagnostics = list(validator.check(ast))
        assert len(diagnostics) == 0

    def test_valid_complex_route(self):
        """Test valid complex route."""
        tag = Tag(
            base="GET /api/users/{user_id}/posts/{post_id}",
            tag_type="http_route",
            http_method="GET",
            http_path="/api/users/{user_id}/posts/{post_id}"
        )
        func = Function(name="test", tags=[tag])
        ast = PyShortAST(functions=[func])

        validator = HTTPRouteValidator()
        diagnostics = list(validator.check(ast))
        assert len(diagnostics) == 0

    def test_warns_on_multiple_routes(self):
        """Test warning on multiple HTTP route tags."""
        tag1 = Tag(base="GET /users", tag_type="http_route", http_method="GET", http_path="/users")
        tag2 = Tag(base="POST /users", tag_type="http_route", http_method="POST", http_path="/users")
        func = Function(name="test", tags=[tag1, tag2])
        ast = PyShortAST(functions=[func])

        validator = HTTPRouteValidator()
        diagnostics = list(validator.check(ast))
        assert len(diagnostics) == 1
        assert diagnostics[0].severity == DiagnosticSeverity.WARNING
        assert "multiple HTTP route tags" in diagnostics[0].message

    def test_warns_on_invalid_param_name(self):
        """Test warning on invalid parameter name."""
        tag = Tag(
            base="GET /users/{user-id}",  # Hyphens not allowed in identifiers
            tag_type="http_route",
            http_method="GET",
            http_path="/users/{user-id}"
        )
        func = Function(name="test", tags=[tag])
        ast = PyShortAST(functions=[func])

        validator = HTTPRouteValidator()
        diagnostics = list(validator.check(ast))
        assert len(diagnostics) == 1
        assert diagnostics[0].severity == DiagnosticSeverity.WARNING
        assert "Invalid parameter name" in diagnostics[0].message

    def test_does_not_validate_non_route_tags(self):
        """Test that non-route tags are ignored."""
        tag = Tag(base="Lin", tag_type="operation")
        func = Function(name="test", tags=[tag])
        ast = PyShortAST(functions=[func])

        validator = HTTPRouteValidator()
        diagnostics = list(validator.check(ast))
        assert len(diagnostics) == 0


class TestValidatorIntegration:
    """Integration tests for all v1.4 validators."""

    def test_linter_includes_all_v14_validators(self):
        """Test that Linter includes all v1.4 validators."""
        from pyshort.core.validator import Linter

        linter = Linter()
        rule_types = [type(rule).__name__ for rule in linter.rules]

        assert "ValidTagsRule" in rule_types
        assert "ComplexityTagValidator" in rule_types
        assert "DecoratorTagValidator" in rule_types
        assert "HTTPRouteValidator" in rule_types

    def test_validate_complex_function_with_all_tag_types(self):
        """Test validating function with all v1.4 tag types."""
        from pyshort.core.validator import Linter

        # Create a function with decorator, route, operation, and complexity
        tags = [
            Tag(base="Prop", tag_type="decorator"),
            Tag(base="GET /api/data", tag_type="http_route", http_method="GET", http_path="/api/data"),
            Tag(base="NN", qualifiers=["Lin"], tag_type="operation"),
            Tag(base="O(N)", tag_type="complexity"),
        ]
        func = Function(name="test", tags=tags)
        ast = PyShortAST(
            metadata=Metadata(module_name="Test"),
            functions=[func]
        )

        linter = Linter()
        diagnostics = linter.check(ast)

        # Should have no errors (all tags are valid)
        errors = [d for d in diagnostics if d.severity == DiagnosticSeverity.ERROR]
        assert len(errors) == 0

    def test_validate_catches_multiple_issues(self):
        """Test that validator catches multiple issues in one pass."""
        from pyshort.core.validator import Linter

        # Create function with conflicting decorators and invalid route
        tags = [
            Tag(base="Prop", tag_type="decorator"),
            Tag(base="Static", tag_type="decorator"),  # Conflicts with Prop
            Tag(base="RateLimit", qualifiers=["-1"], tag_type="decorator"),  # Invalid
        ]
        func = Function(name="test", tags=tags)
        ast = PyShortAST(functions=[func])

        linter = Linter()
        diagnostics = linter.check(ast)

        errors = [d for d in diagnostics if d.severity == DiagnosticSeverity.ERROR]
        # Should catch both conflicts
        assert len(errors) >= 2
