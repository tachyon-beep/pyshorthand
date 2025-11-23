"""Symbol mappings for Unicode ↔ ASCII conversion.

According to RFC Section 3.7, PyShorthand supports both Unicode
and ASCII-compatible notation for portability.
"""


# Unicode → ASCII mapping (canonical form → ASCII)
UNICODE_TO_ASCII: dict[str, str] = {
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
    "◊": "EXTENDS",  # 0.9.0-RC1: Inheritance
}

# ASCII → Unicode mapping (reverse of above)
ASCII_TO_UNICODE: dict[str, str] = {v: k for k, v in UNICODE_TO_ASCII.items()}

# Valid tag base types (from RFC Section 3.5)
VALID_TAG_BASES: set[str] = {
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

# Decorator tags (legacy baseline, expanded in 0.9.0-RC1)
DECORATOR_TAGS: set[str] = {
    # Standard Python decorators
    "Prop",  # @property
    "Setter",  # @name.setter
    "Deleter",  # @name.deleter
    "Cached",  # @cached_property or @lru_cache
    "Static",  # @staticmethod
    "Class",  # @classmethod
    "Abstract",  # @abstractmethod (0.9.0-RC1: also marks abstract classes/methods)
    "Protocol",  # 0.9.0-RC1: marks Protocol classes
    # Common framework decorators
    "Auth",  # Authentication required
    "Retry",  # Retry decorator
    "Timeout",  # Timeout decorator
    "RateLimit",  # Rate limiting
}

# HTTP method tags (legacy)
HTTP_METHODS: set[str] = {
    "GET",
    "POST",
    "PUT",
    "PATCH",
    "DELETE",
    "OPTIONS",
    "HEAD",
}

# Valid type names (from RFC Section 3.3)
VALID_TYPES: set[str] = {
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
VALID_LOCATIONS: set[str] = {
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

# Valid roles
VALID_ROLES: set[str] = {"Core", "Glue", "Script"}

# Valid layers
VALID_LAYERS: set[str] = {"Domain", "Infra", "Adapter", "Test"}

# Valid risk levels
VALID_RISK_LEVELS: set[str] = {"High", "Med", "Low"}

# Entity prefixes (0.9.0-RC1: added P and E)
ENTITY_PREFIXES: dict[str, str] = {
    "C": "class",
    "D": "data",
    "I": "interface",
    "M": "module",
    "F": "function",
    "P": "protocol",  # 0.9.0-RC1: Protocol (typing.Protocol)
    "E": "enum",  # 0.9.0-RC1: Enum
    "Ref": "reference",
}


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


def is_decorator_tag(tag: str) -> bool:
    """Check if a tag is a decorator tag.

    Args:
        tag: Tag name

    Returns:
        True if it's a decorator tag, False otherwise

    Examples:
        >>> is_decorator_tag("Prop")
        True
        >>> is_decorator_tag("Static")
        True
        >>> is_decorator_tag("Lin")
        False
    """
    return tag in DECORATOR_TAGS


def is_complexity_tag(tag: str) -> bool:
    """Check if a tag represents computational complexity.

    Args:
        tag: Tag string

    Returns:
        True if it's a complexity tag, False otherwise

    Examples:
        >>> is_complexity_tag("O(N)")
        True
        >>> is_complexity_tag("O(N*M)")
        True
        >>> is_complexity_tag("Lin")
        False
    """
    import re

    return bool(re.match(r"^O\(.+\)$", tag))


def parse_http_route(tag_string: str) -> tuple[str, str] | None:
    """Parse an HTTP route tag.

    Args:
        tag_string: Tag string like "GET /users/{id}"

    Returns:
        Tuple of (method, path) or None if not a valid route

    Examples:
        >>> parse_http_route("GET /users")
        ('GET', '/users')
        >>> parse_http_route("POST /api/users/{id}")
        ('POST', '/api/users/{id}')
        >>> parse_http_route("Invalid")
        None
    """
    parts = tag_string.split(None, 1)
    if len(parts) == 2:
        method, path = parts
        if method in HTTP_METHODS and path.startswith("/"):
            return (method, path)
    return None
