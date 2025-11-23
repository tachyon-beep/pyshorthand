"""Error codes for PyShorthand linter.

Error code format: [E|W]XXX
- E = Error (prevents successful parsing/validation)
- W = Warning (style or best practice issue)
- XXX = Three-digit number

Categories:
- E001-E099: Metadata errors
- E101-E199: Type errors
- E201-E299: Structure errors
- E301-E399: Naming errors
- W001-W099: Style warnings
- W101-W199: Best practice warnings
"""

from dataclasses import dataclass


@dataclass
class ErrorCode:
    """Error code definition."""

    code: str
    category: str
    description: str
    explanation: str | None = None


# ============================================================================
# Error Code Catalog
# ============================================================================

ERROR_CODES = {
    # Metadata Errors (E001-E099)
    "E001": ErrorCode(
        code="E001",
        category="metadata",
        description="Invalid role value",
        explanation="Role must be one of: Core, Glue, Script, Test, Util",
    ),
    "E002": ErrorCode(
        code="E002",
        category="metadata",
        description="Invalid layer value",
        explanation="Layer must be one of: Domain, Infrastructure, Adapter, Test",
    ),
    "E003": ErrorCode(
        code="E003",
        category="metadata",
        description="Invalid risk level",
        explanation="Risk must be one of: High, Medium, Low, Critical",
    ),
    "E004": ErrorCode(
        code="E004",
        category="metadata",
        description="Missing required module name",
        explanation="All PyShorthand files should have [M:ModuleName] metadata",
    ),
    # Type Errors (E101-E199)
    "E101": ErrorCode(
        code="E101",
        category="type",
        description="Invalid type specification",
        explanation="Type must be one of: f16, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64, bool, str, obj, Map, List, Set, Tuple, Any",
    ),
    "E102": ErrorCode(
        code="E102",
        category="type",
        description="Invalid location specification",
        explanation="Location must be one of: CPU, GPU, Disk, Net, TPU",
    ),
    "E103": ErrorCode(
        code="E103",
        category="type",
        description="Invalid shape dimension",
        explanation="Shape dimensions should use dimension variables (e.g., N, M, D) or integers",
    ),
    # Structure Errors (E201-E299)
    "E201": ErrorCode(
        code="E201",
        category="structure",
        description="Empty entity definition",
        explanation="Entities should have at least one state variable or method",
    ),
    "E202": ErrorCode(
        code="E202",
        category="structure",
        description="Duplicate definition",
        explanation="Entity or function names must be unique within a module",
    ),
    "E203": ErrorCode(
        code="E203",
        category="structure",
        description="Invalid dependency reference",
        explanation="Dependency references must point to valid entities",
    ),
    # Naming Errors (E301-E399)
    "E301": ErrorCode(
        code="E301",
        category="naming",
        description="Invalid naming convention",
        explanation="Names should use snake_case for variables/functions, PascalCase for classes",
    ),
    "E302": ErrorCode(
        code="E302",
        category="naming",
        description="Reserved keyword used as name",
        explanation="Cannot use Python reserved keywords as identifiers",
    ),
    # Style Warnings (W001-W099)
    "W001": ErrorCode(
        code="W001",
        category="style",
        description="Line too long",
        explanation="Lines should not exceed the configured maximum length (default: 100)",
    ),
    "W002": ErrorCode(
        code="W002",
        category="style",
        description="Inconsistent spacing",
        explanation="Use consistent spacing around operators and delimiters",
    ),
    "W003": ErrorCode(
        code="W003",
        category="style",
        description="Missing metadata",
        explanation="Consider adding metadata headers for better documentation",
    ),
    # Best Practice Warnings (W101-W199)
    "W101": ErrorCode(
        code="W101",
        category="best_practice",
        description="Complex entity",
        explanation="Entity has too many state variables or methods; consider splitting",
    ),
    "W102": ErrorCode(
        code="W102",
        category="best_practice",
        description="Unoptimized data flow",
        explanation="Consider optimizing data transfers between locations",
    ),
    "W103": ErrorCode(
        code="W103",
        category="best_practice",
        description="Missing type annotation",
        explanation="All state variables should have explicit type annotations",
    ),
    "W104": ErrorCode(
        code="W104",
        category="best_practice",
        description="Missing docstring",
        explanation="Complex entities should have docstrings explaining their purpose",
    ),
}


def get_error_code(code: str) -> ErrorCode | None:
    """Get error code definition."""
    return ERROR_CODES.get(code)


def format_diagnostic_with_code(
    severity: str, code: str, message: str, suggestion: str | None = None
) -> str:
    """Format a diagnostic message with error code."""
    parts = [f"[{severity.lower()}:{code}] {message}"]

    if suggestion:
        parts.append(f"  💡 {suggestion}")

    error_info = get_error_code(code)
    if error_info and error_info.explanation:
        parts.append(f"  ℹ️  {error_info.explanation}")

    return "\n".join(parts)


def list_error_codes(category: str | None = None) -> list:
    """List all error codes, optionally filtered by category."""
    codes = list(ERROR_CODES.values())

    if category:
        codes = [c for c in codes if c.category == category]

    return sorted(codes, key=lambda x: x.code)


def explain_error_code(code: str) -> str | None:
    """Get detailed explanation for an error code."""
    error_info = get_error_code(code)
    if not error_info:
        return None

    explanation = [
        f"Error Code: {error_info.code}",
        f"Category: {error_info.category}",
        f"Description: {error_info.description}",
    ]

    if error_info.explanation:
        explanation.append("\nExplanation:")
        explanation.append(f"  {error_info.explanation}")

    return "\n".join(explanation)


if __name__ == "__main__":
    # Test error code system
    print("PyShorthand Error Codes")
    print("=" * 60)

    for category in ["metadata", "type", "structure", "naming", "style", "best_practice"]:
        codes = list_error_codes(category)
        if codes:
            print(f"\n{category.upper()} ({len(codes)} codes)")
            print("-" * 60)
            for code_info in codes:
                print(f"  {code_info.code}: {code_info.description}")

    print("\n" + "=" * 60)
    print("\nExample diagnostic:")
    print(
        format_diagnostic_with_code("error", "E001", "Invalid role: 'Cor'", "Did you mean 'Core'?")
    )
