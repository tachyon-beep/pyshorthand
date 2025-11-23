"""Enhanced error messages with suggestions.

Provides helpful, actionable error messages for common mistakes.
"""


from pyshort.core.symbols import (
    VALID_LAYERS,
    VALID_LOCATIONS,
    VALID_RISK_LEVELS,
    VALID_ROLES,
    VALID_TAG_BASES,
    VALID_TYPES,
)


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # j+1 instead of j since previous_row and current_row are one character longer
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def find_close_matches(
    word: str, possibilities: list[str], n: int = 3, cutoff: float = 0.6
) -> list[str]:
    """Find close matches using Levenshtein distance.

    Args:
        word: Word to find matches for
        possibilities: List of possible matches
        n: Maximum number of matches to return
        cutoff: Minimum similarity ratio (0.0 to 1.0)

    Returns:
        List of close matches
    """
    if not word or not possibilities:
        return []

    # Calculate distances
    distances = []
    for possibility in possibilities:
        distance = levenshtein_distance(word.lower(), possibility.lower())
        max_len = max(len(word), len(possibility))
        similarity = 1.0 - (distance / max_len)
        if similarity >= cutoff:
            distances.append((similarity, possibility))

    # Sort by similarity and return top n
    distances.sort(reverse=True)
    return [match for _, match in distances[:n]]


def suggest_tag_for_operation(operation_name: str) -> str | None:
    """Suggest an appropriate tag for an operation.

    Args:
        operation_name: Name of the operation/function

    Returns:
        Suggested tag or None
    """
    op_lower = operation_name.lower()

    # Database operations
    if any(
        keyword in op_lower
        for keyword in [
            "query",
            "execute",
            "commit",
            "rollback",
            "insert",
            "update",
            "delete",
            "select",
        ]
    ):
        return "[IO:Disk]"

    # Network operations
    if any(
        keyword in op_lower for keyword in ["fetch", "request", "http", "api", "socket", "download"]
    ):
        return "[IO:Net]"

    # File operations
    if any(keyword in op_lower for keyword in ["read", "write", "open", "save", "load", "file"]):
        return "[IO:Disk]"

    # Mathematical operations
    if any(keyword in op_lower for keyword in ["matmul", "dot", "mult", "add", "sum", "mean"]):
        return "[Lin]"

    # Loops
    if any(keyword in op_lower for keyword in ["loop", "iterate", "scan", "foreach"]):
        return "[Iter]"

    # Conditionals
    if any(keyword in op_lower for keyword in ["filter", "where", "if", "cond", "thresh"]):
        return "[Thresh]"

    # Random/stochastic
    if any(keyword in op_lower for keyword in ["random", "sample", "shuffle", "noise"]):
        return "[Stoch]"

    return None


def suggest_location_for_type(type_name: str) -> str | None:
    """Suggest an appropriate location for a type.

    Args:
        type_name: Type name (e.g., "f32", "i64")

    Returns:
        Suggested location or None
    """
    # Tensors are usually on GPU
    if type_name.startswith("f") and any(c.isdigit() for c in type_name):
        return "@GPU"

    # Large collections might be on CPU
    if type_name in ("Map", "List", "Set", "Dict"):
        return "@CPU"

    return None


def format_error_with_context(
    message: str,
    line: int,
    column: int,
    source_line: str | None = None,
    suggestion: str | None = None,
) -> str:
    """Format an error message with context and visual indicators.

    Args:
        message: Error message
        line: Line number
        column: Column number
        source_line: Optional source code line
        suggestion: Optional suggestion for fixing

    Returns:
        Formatted error message
    """
    lines = []

    # Error header
    lines.append(f"error: {message}")
    lines.append(f"  --> line {line}, column {column}")

    # Source context
    if source_line:
        lines.append("   |")
        lines.append(f"{line:3d}|  {source_line}")
        # Add caret indicator
        spaces = " " * (column + 5)  # Account for line number
        lines.append(f"   |  {spaces}^")

    # Suggestion
    if suggestion:
        lines.append("   |")
        lines.append(f"   = help: {suggestion}")

    return "\n".join(lines)


def suggest_did_you_mean(word: str, category: str) -> str | None:
    """Generate "Did you mean?" suggestion.

    Args:
        word: Invalid word
        category: Category of word (role, layer, risk, tag, type, location)

    Returns:
        Formatted suggestion or None
    """
    possibilities_map = {
        "role": list(VALID_ROLES),
        "layer": list(VALID_LAYERS),
        "risk": list(VALID_RISK_LEVELS),
        "tag": list(VALID_TAG_BASES),
        "type": list(VALID_TYPES),
        "location": list(VALID_LOCATIONS),
    }

    possibilities = possibilities_map.get(category, [])
    if not possibilities:
        return None

    matches = find_close_matches(word, possibilities, n=3)
    if not matches:
        return None

    if len(matches) == 1:
        return f"Did you mean '{matches[0]}'?"
    else:
        options = "', '".join(matches)
        return f"Did you mean one of: '{options}'?"


def suggest_missing_tag(operation_type: str) -> str | None:
    """Suggest adding a tag to an operation.

    Args:
        operation_type: Type of operation (mutation, io, loop, etc.)

    Returns:
        Suggestion string or None
    """
    suggestions = {
        "mutation": "Consider adding a tag like [Heur] or [Lin] to describe the operation",
        "io": "I/O operations should have tags like [IO:Disk], [IO:Net], or [IO:Block]",
        "loop": "Loops should have complexity tags like [Iter:O(N)] or [Iter:Hot]",
        "database": "Database operations should have [IO:Disk] tag",
        "network": "Network operations should have [IO:Net:Async] or [IO:Net:Block] tag",
    }

    return suggestions.get(operation_type)


def suggest_complexity_tag(has_loop: bool, has_nested_loop: bool) -> str | None:
    """Suggest complexity tag based on structure.

    Args:
        has_loop: Whether code contains a loop
        has_nested_loop: Whether code contains nested loops

    Returns:
        Suggestion string or None
    """
    if has_nested_loop:
        return "Nested loops detected. Consider adding [Iter:O(N²)] or higher complexity tag"
    elif has_loop:
        return "Loop detected. Consider adding [Iter:O(N)] complexity tag"

    return None


def suggest_location_annotation(var_name: str, has_type: bool) -> str | None:
    """Suggest adding location annotation.

    Args:
        var_name: Variable name
        has_type: Whether variable already has a type

    Returns:
        Suggestion string or None
    """
    if not has_type:
        return None

    # Common patterns
    if any(keyword in var_name.lower() for keyword in ["tensor", "weight", "hidden", "embedding"]):
        return "Tensor variables should specify location: @GPU or @CPU"

    if any(keyword in var_name.lower() for keyword in ["cache", "buffer", "store"]):
        return "Cache/storage variables should specify location: @CPU, @GPU, or @Disk"

    return None


class ErrorEnhancer:
    """Enhances error messages with context and suggestions."""

    @staticmethod
    def enhance_parse_error(
        error_type: str, message: str, token_value: str, line: int, column: int
    ) -> tuple[str, str | None]:
        """Enhance a parse error with suggestions.

        Args:
            error_type: Type of error
            message: Original error message
            token_value: Token value that caused the error
            line: Line number
            column: Column number

        Returns:
            Tuple of (enhanced_message, suggestion)
        """
        suggestion = None

        if "Invalid role" in message:
            suggestion = suggest_did_you_mean(token_value, "role")
        elif "Invalid layer" in message:
            suggestion = suggest_did_you_mean(token_value, "layer")
        elif "Invalid risk" in message:
            suggestion = suggest_did_you_mean(token_value, "risk")
        elif "Invalid tag" in message:
            suggestion = suggest_did_you_mean(token_value, "tag")
        elif "Unknown type" in message:
            suggestion = suggest_did_you_mean(token_value, "type")
        elif "Unknown location" in message:
            suggestion = suggest_did_you_mean(token_value, "location")
        elif "Missing tag" in message:
            suggestion = suggest_missing_tag("mutation")
        elif "database" in message.lower():
            suggestion = suggest_missing_tag("database")
        elif "network" in message.lower():
            suggestion = suggest_missing_tag("network")

        return message, suggestion
