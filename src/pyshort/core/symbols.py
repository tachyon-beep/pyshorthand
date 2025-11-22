"""Symbol mappings for Unicode ↔ ASCII conversion.

According to RFC Section 3.7, PyShorthand supports both Unicode
and ASCII-compatible notation for portability.
"""

from typing import Dict, Set

# Unicode → ASCII mapping (canonical form → ASCII)
UNICODE_TO_ASCII: Dict[str, str] = {
    "→": "->",
    "⊳": ">>",
    "∈": "IN",
    "≡": "==",
    "⊢": "ASSERT",
    "Σ": "SUM",
    "Π": "PROD",
    "⊗": "MAT",
    "∇": "GRAD",
    "≈": "REF",
    "≜": "COPY",
    "∀": "FOR",
}

# ASCII → Unicode mapping (reverse of above)
ASCII_TO_UNICODE: Dict[str, str] = {v: k for k, v in UNICODE_TO_ASCII.items()}

# Operators (all forms)
OPERATORS: Set[str] = {
    # Flow operators
    "→",
    "->",
    "⊳",
    ">>",
    "←",
    "<-",
    # Assignment/equality
    "≡",
    "==",
    "=",
    # Mutation
    "!",
    "!!",
    "!?",
    "?!",
    # Membership
    "∈",
    "IN",
    # Logic
    "⊢",
    "ASSERT",
    "?",
    # Math
    "Σ",
    "SUM",
    "Π",
    "PROD",
    "⊗",
    "MAT",
    "∇",
    "GRAD",
    # Memory
    "≈",
    "REF",
    "≜",
    "COPY",
    # Iteration
    "∀",
    "FOR",
    # Standard operators
    "+",
    "-",
    "*",
    "/",
    "%",
    "**",
    "//",
    "&",
    "|",
    "^",
    "~",
    "<<",
    ">=",
    "<=",
    ">",
    "<",
    "!=",
}

# Flow operators (control flow)
FLOW_OPERATORS: Set[str] = {"→", "->", "⊳", ">>", "←", "<-"}

# Mutation operators
MUTATION_OPERATORS: Set[str] = {"!", "!!", "!?", "?!"}

# Assignment operators
ASSIGNMENT_OPERATORS: Set[str] = {"≡", "==", "="}

# Math operators
MATH_OPERATORS: Set[str] = {"Σ", "SUM", "Π", "PROD", "⊗", "MAT", "∇", "GRAD"}

# Valid tag base types (from RFC Section 3.5)
VALID_TAG_BASES: Set[str] = {
    "Lin",  # Linear/algebraic
    "Thresh",  # Branching/bounds
    "Iter",  # Iteration
    "Map",  # Mapping/lookup
    "Stoch",  # Stochastic
    "IO",  # Input/Output
    "Sync",  # Concurrency sync
    "NN",  # Neural net
    "Heur",  # Heuristic/business logic
}

# Common qualifiers
COMMON_QUALIFIERS: Set[str] = {
    # Complexity
    "O(1)",
    "O(N)",
    "O(N^2)",
    "O(N*M)",
    "O(N log N)",
    "O(P)",
    "Amortized",
    # Linear ops
    "Broad",
    "MatMul",
    # Iteration
    "Hot",
    "Scan",
    "Sequential",
    "Random",
    "Strided",
    # Thresholding
    "Mask",
    "Cond",
    "Clamp",
    # Mapping
    "Hash",
    "Cache",
    # Stochastic
    "Seed",
    "Dist",
    # IO
    "Net",
    "Disk",
    "Async",
    "Block",
    # Sync
    "Lock",
    "Atomic",
    "Barrier",
    "Await",
    # NN
    "∇",
    "Inf",
    # Parallel
    "||",
}

# Valid type names (from RFC Section 3.3)
VALID_TYPES: Set[str] = {
    "f32",
    "f64",
    "i8",
    "i16",
    "i32",
    "i64",
    "u8",
    "u16",
    "u32",
    "u64",
    "bool",
    "obj",
    "Map",
    "Str",
    "Any",
    "List",
    "Set",
    "Dict",
}

# Valid locations (from RFC Section 3.3)
VALID_LOCATIONS: Set[str] = {
    "CPU",
    "GPU",
    "Disk",
    "Net",
    "Cache",
    "L1",
    "L2",
    "VRAM",
    "RAM",
}

# Metadata keys (from RFC Section 3.1)
METADATA_KEYS: Set[str] = {
    "M",  # Module name
    "ID",  # Unique identifier
    "Role",  # Core|Glue|Script
    "Layer",  # Domain|Infra|Adapter|Test
    "Risk",  # High|Med|Low
    "Context",  # Domain context
    "Dims",  # Dimension variables
    "Requires",  # Dependencies
    "Owner",  # Team/person
}

# Valid roles
VALID_ROLES: Set[str] = {"Core", "Glue", "Script"}

# Valid layers
VALID_LAYERS: Set[str] = {"Domain", "Infra", "Adapter", "Test"}

# Valid risk levels
VALID_RISK_LEVELS: Set[str] = {"High", "Med", "Low"}

# Entity prefixes
ENTITY_PREFIXES: Dict[str, str] = {
    "C": "class",
    "D": "data",
    "I": "interface",
    "M": "module",
    "F": "function",
    "Ref": "reference",
}


def normalize_operator(op: str) -> str:
    """Normalize operator to canonical Unicode form.

    Args:
        op: Operator in either Unicode or ASCII form

    Returns:
        Canonical Unicode form

    Examples:
        >>> normalize_operator("->")
        '→'
        >>> normalize_operator(">>")
        '⊳'
        >>> normalize_operator("→")
        '→'
    """
    return ASCII_TO_UNICODE.get(op, op)


def to_ascii(text: str) -> str:
    """Convert Unicode symbols to ASCII equivalents.

    Args:
        text: Text with Unicode symbols

    Returns:
        Text with ASCII equivalents

    Examples:
        >>> to_ascii("x ∈ f32[N]@GPU → y")
        'x IN f32[N]@GPU -> y'
    """
    result = text
    for unicode_sym, ascii_sym in UNICODE_TO_ASCII.items():
        result = result.replace(unicode_sym, ascii_sym)
    return result


def to_unicode(text: str) -> str:
    """Convert ASCII symbols to Unicode equivalents.

    Args:
        text: Text with ASCII symbols

    Returns:
        Text with Unicode equivalents

    Examples:
        >>> to_unicode("x IN f32[N]@GPU -> y")
        'x ∈ f32[N]@GPU → y'
    """
    result = text
    # Sort by length descending to handle multi-char sequences first
    for ascii_sym in sorted(ASCII_TO_UNICODE.keys(), key=len, reverse=True):
        result = result.replace(ascii_sym, ASCII_TO_UNICODE[ascii_sym])
    return result


def is_valid_tag_base(tag: str) -> bool:
    """Check if a tag base is valid.

    Args:
        tag: Tag base name

    Returns:
        True if valid, False otherwise
    """
    return tag in VALID_TAG_BASES


def is_valid_type(type_name: str) -> bool:
    """Check if a type name is valid.

    Args:
        type_name: Type name

    Returns:
        True if valid, False otherwise
    """
    return type_name in VALID_TYPES


def is_valid_location(location: str) -> bool:
    """Check if a location is valid.

    Args:
        location: Location name

    Returns:
        True if valid, False otherwise
    """
    return location in VALID_LOCATIONS
