"""Validator and linter for PyShorthand.

This module enforces grammar constraints and semantic best practices
according to RFC Section 3.8 and general design principles.
"""

from abc import ABC, abstractmethod
from typing import Iterator, List

from pyshort.core.ast_nodes import Diagnostic, DiagnosticSeverity, PyShortAST, Statement
from pyshort.core.enhanced_errors import suggest_did_you_mean
from pyshort.core.symbols import (
    VALID_LAYERS,
    VALID_LOCATIONS,
    VALID_RISK_LEVELS,
    VALID_ROLES,
    VALID_TAG_BASES,
    VALID_TYPES,
)


class Rule(ABC):
    """Base class for validation rules."""

    @abstractmethod
    def check(self, ast: PyShortAST) -> Iterator[Diagnostic]:
        """Check the rule against an AST.

        Args:
            ast: PyShorthand AST to validate

        Yields:
            Diagnostic messages
        """
        pass


class MandatoryMetadataRule(Rule):
    """Enforce that all files have [M:Name] and [Role] metadata."""

    def check(self, ast: PyShortAST) -> Iterator[Diagnostic]:
        """Check for mandatory metadata."""
        if not ast.metadata.module_name:
            yield Diagnostic(
                severity=DiagnosticSeverity.ERROR,
                line=1,
                column=1,
                message="Missing mandatory [M:Name] metadata header",
                suggestion="Add '# [M:ModuleName]' at the top of the file",
                code="E004",
            )

        if not ast.metadata.role:
            yield Diagnostic(
                severity=DiagnosticSeverity.WARNING,
                line=1,
                column=1,
                message="Missing [Role] metadata header",
                suggestion="Add [Role:Core|Glue|Script] to metadata",
                code="W003",
            )


class ValidMetadataValuesRule(Rule):
    """Validate metadata values are from allowed sets."""

    def check(self, ast: PyShortAST) -> Iterator[Diagnostic]:
        """Check metadata values."""
        if ast.metadata.role and ast.metadata.role not in VALID_ROLES:
            suggestion = suggest_did_you_mean(ast.metadata.role, "role")
            if not suggestion:
                suggestion = f"Use one of: {', '.join(VALID_ROLES)}"
            yield Diagnostic(
                severity=DiagnosticSeverity.ERROR,
                line=1,
                column=1,
                message=f"Invalid role: '{ast.metadata.role}'",
                suggestion=suggestion,
                code="E001",
            )

        if ast.metadata.layer and ast.metadata.layer not in VALID_LAYERS:
            suggestion = suggest_did_you_mean(ast.metadata.layer, "layer")
            if not suggestion:
                suggestion = f"Use one of: {', '.join(VALID_LAYERS)}"
            yield Diagnostic(
                severity=DiagnosticSeverity.ERROR,
                line=1,
                column=1,
                message=f"Invalid layer: '{ast.metadata.layer}'",
                suggestion=suggestion,
                code="E002",
            )

        if ast.metadata.risk and ast.metadata.risk not in VALID_RISK_LEVELS:
            suggestion = suggest_did_you_mean(ast.metadata.risk, "risk")
            if not suggestion:
                suggestion = f"Use one of: {', '.join(VALID_RISK_LEVELS)}"
            yield Diagnostic(
                severity=DiagnosticSeverity.ERROR,
                line=1,
                column=1,
                message=f"Invalid risk level: '{ast.metadata.risk}'",
                suggestion=suggestion,
                code="E003",
            )


class DimensionConsistencyRule(Rule):
    """Ensure all dimension variables used in shapes are declared in [Dims:...]."""

    def check(self, ast: PyShortAST) -> Iterator[Diagnostic]:
        """Check dimension consistency."""
        declared_dims = set(ast.metadata.dims.keys())
        used_dims = set()

        # Collect dimensions from state variables
        for state_var in ast.state:
            if state_var.type_spec and state_var.type_spec.shape:
                for dim in state_var.type_spec.shape:
                    # Only check alphabetic dimension names (not numbers)
                    if dim.isalpha():
                        used_dims.add(dim)

        # Collect from classes
        for entity in ast.entities:
            if hasattr(entity, "state"):
                for state_var in entity.state:  # type: ignore
                    if state_var.type_spec and state_var.type_spec.shape:
                        for dim in state_var.type_spec.shape:
                            if dim.isalpha():
                                used_dims.add(dim)

        # Check for undeclared dimensions
        undeclared = used_dims - declared_dims
        if undeclared:
            yield Diagnostic(
                severity=DiagnosticSeverity.WARNING,
                line=1,
                column=1,
                message=f"Undeclared dimension variables: {', '.join(sorted(undeclared))}",
                suggestion=f"Add to metadata: [Dims:{','.join(f'{d}=description' for d in undeclared)}]",
            )


class ValidTagsRule(Rule):
    """Ensure all tags use valid base types."""

    def check(self, ast: PyShortAST) -> Iterator[Diagnostic]:
        """Check tag validity."""
        for stmt in ast.statements:
            for tag in stmt.tags:
                if tag.base not in VALID_TAG_BASES:
                    yield Diagnostic(
                        severity=DiagnosticSeverity.ERROR,
                        line=stmt.line,
                        column=1,
                        message=f"Invalid tag base: {tag.base}. Must be one of: {', '.join(VALID_TAG_BASES)}",
                        suggestion=f"Use one of: {', '.join(VALID_TAG_BASES)}",
                    )


class SystemMutationSafetyRule(Rule):
    """Flag !! operations in functions not marked as [Risk:High]."""

    def check(self, ast: PyShortAST) -> Iterator[Diagnostic]:
        """Check system mutation safety."""
        for func in ast.functions:
            has_system_mutation = any(
                stmt.is_system_mutation for stmt in func.body
            )

            if has_system_mutation:
                # Check if function or module is marked high risk
                is_high_risk = (
                    ast.metadata.risk == "High"
                    or any(tag.base == "Risk" and "High" in tag.qualifiers for tag in func.tags)
                )

                if not is_high_risk:
                    yield Diagnostic(
                        severity=DiagnosticSeverity.WARNING,
                        line=func.line,
                        column=1,
                        message=f"Function {func.name} contains system mutations (!!) but is not marked as high-risk",
                        suggestion="Consider adding [Risk:High] tag or review if !! is necessary",
                    )


class CriticalOperationTaggingRule(Rule):
    """Warn if critical operations (loops, IO, mutations) lack tags."""

    def check(self, ast: PyShortAST) -> Iterator[Diagnostic]:
        """Check critical operation tagging."""
        for stmt in ast.statements:
            # Check for mutations without tags
            if stmt.is_mutation and not stmt.tags:
                yield Diagnostic(
                    severity=DiagnosticSeverity.INFO,
                    line=stmt.line,
                    column=1,
                    message="Mutation operation lacks computational tag",
                    suggestion="Consider adding a tag like [Heur] or [Lin] to describe the operation",
                )


class LocationInferenceRule(Rule):
    """Validate @Location annotations follow inference rules."""

    def check(self, ast: PyShortAST) -> Iterator[Diagnostic]:
        """Check location inference rules."""
        for state_var in ast.state:
            if state_var.type_spec:
                loc = state_var.type_spec.location
                transfer = state_var.type_spec.transfer

                if loc and loc not in VALID_LOCATIONS:
                    yield Diagnostic(
                        severity=DiagnosticSeverity.WARNING,
                        line=state_var.line,
                        column=1,
                        message=f"Unknown location: @{loc}. Common locations: {', '.join(VALID_LOCATIONS)}",
                        suggestion=f"Consider using standard locations: {', '.join(VALID_LOCATIONS)}",
                    )

                if transfer:
                    loc1, loc2 = transfer
                    if loc1 not in VALID_LOCATIONS:
                        yield Diagnostic(
                            severity=DiagnosticSeverity.WARNING,
                            line=state_var.line,
                            column=1,
                            message=f"Unknown source location in transfer: @{loc1}",
                        )
                    if loc2 not in VALID_LOCATIONS:
                        yield Diagnostic(
                            severity=DiagnosticSeverity.WARNING,
                            line=state_var.line,
                            column=1,
                            message=f"Unknown destination location in transfer: @{loc2}",
                        )


class TypeValidityRule(Rule):
    """Check that type names are valid."""

    def check(self, ast: PyShortAST) -> Iterator[Diagnostic]:
        """Check type validity."""
        for state_var in ast.state:
            if state_var.type_spec:
                base_type = state_var.type_spec.base_type
                if base_type not in VALID_TYPES:
                    yield Diagnostic(
                        severity=DiagnosticSeverity.WARNING,
                        line=state_var.line,
                        column=1,
                        message=f"Unknown type: {base_type}. Common types: {', '.join(sorted(VALID_TYPES))}",
                        suggestion="Use standard types like f32, i64, obj, Map, etc.",
                    )


class ErrorSurfaceDocumentationRule(Rule):
    """Check that functions with !? have [Err] declarations."""

    def check(self, ast: PyShortAST) -> Iterator[Diagnostic]:
        """Check error surface documentation."""
        for func in ast.functions:
            has_error_ops = any(stmt.is_error for stmt in func.body)

            if has_error_ops and not func.errors:
                yield Diagnostic(
                    severity=DiagnosticSeverity.WARNING,
                    line=func.line,
                    column=1,
                    message=f"Function {func.name} raises errors but lacks [Err] declaration",
                    suggestion="Add [Err] section listing possible exceptions",
                )


class Linter:
    """Main linter for PyShorthand files."""

    def __init__(self, strict: bool = False) -> None:
        """Initialize linter.

        Args:
            strict: If True, warnings become errors
        """
        self.strict = strict
        self.rules: List[Rule] = [
            MandatoryMetadataRule(),
            ValidMetadataValuesRule(),
            DimensionConsistencyRule(),
            ValidTagsRule(),
            SystemMutationSafetyRule(),
            CriticalOperationTaggingRule(),
            LocationInferenceRule(),
            TypeValidityRule(),
            ErrorSurfaceDocumentationRule(),
        ]

    def register(self, rule: Rule) -> None:
        """Register a custom rule.

        Args:
            rule: Validation rule to add
        """
        self.rules.append(rule)

    def check(self, ast: PyShortAST) -> List[Diagnostic]:
        """Run all rules against an AST.

        Args:
            ast: AST to validate

        Returns:
            List of diagnostics
        """
        diagnostics = []

        for rule in self.rules:
            for diagnostic in rule.check(ast):
                # Upgrade warnings to errors in strict mode
                if self.strict and diagnostic.severity == DiagnosticSeverity.WARNING:
                    diagnostic = Diagnostic(
                        severity=DiagnosticSeverity.ERROR,
                        line=diagnostic.line,
                        column=diagnostic.column,
                        message=diagnostic.message,
                        suggestion=diagnostic.suggestion,
                        code=diagnostic.code,
                    )
                diagnostics.append(diagnostic)

        return diagnostics

    def check_file(self, file_path: str) -> List[Diagnostic]:
        """Check a PyShorthand file.

        Args:
            file_path: Path to .pys file

        Returns:
            List of diagnostics
        """
        from pyshort.core.parser import parse_file

        ast = parse_file(file_path)
        diagnostics = list(ast.diagnostics)  # Include parse errors
        diagnostics.extend(self.check(ast))
        return diagnostics


def validate_file(file_path: str, strict: bool = False) -> List[Diagnostic]:
    """Validate a PyShorthand file.

    Args:
        file_path: Path to .pys file
        strict: If True, warnings become errors

    Returns:
        List of diagnostics
    """
    linter = Linter(strict=strict)
    return linter.check_file(file_path)
