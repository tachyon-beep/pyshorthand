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
    "◊": "EXTENDS",  # v1.5: Inheritance
}

# ASCII → Unicode mapping (reverse of above)
ASCII_TO_UNICODE: dict[str, str] = {v: k for k, v in UNICODE_TO_ASCII.items()}

# Operators (all forms)
OPERATORS: set[str] = {
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
    # Inheritance (v1.5)
    "◊",
    "EXTENDS",
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
FLOW_OPERATORS: set[str] = {"→", "->", "⊳", ">>", "←", "<-"}

# Mutation operators
MUTATION_OPERATORS: set[str] = {"!", "!!", "!?", "?!"}

# Assignment operators
ASSIGNMENT_OPERATORS: set[str] = {"≡", "==", "="}

# Math operators
MATH_OPERATORS: set[str] = {"Σ", "SUM", "Π", "PROD", "⊗", "MAT", "∇", "GRAD"}

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

# Common qualifiers (v1.4 - expanded)
COMMON_QUALIFIERS: set[str] = {
    # Complexity (extended in v1.4)
    "O(1)",
    "O(log N)",
    "O(N)",
    "O(N log N)",
    "O(N^2)",
    "O(N²)",  # Unicode variant
    "O(N^3)",
    "O(N³)",  # Unicode variant
    "O(N*M)",
    "O(N*M*D)",
    "O(B*N*M)",
    "O(B*N*D)",
    "O(B*N²*D)",  # Attention complexity
    "O(2^N)",  # Exponential
    "O(P)",
    "Amortized",
    # Linear ops
    "Broad",
    "MatMul",
    "Reduce",
    "Transpose",
    # Iteration
    "Hot",
    "Scan",
    "Sequential",
    "Random",
    "Strided",
    # Thresholding (expanded)
    "Mask",
    "Cond",
    "Clamp",
    "Softmax",
    "ReLU",
    "GELU",
    "Sigmoid",
    "Tanh",
    # Mapping
    "Hash",
    "Cache",
    # Stochastic
    "Seed",
    "Dist",
    # IO (expanded)
    "Net",
    "Disk",
    "Async",
    "Block",
    "Stream",
    # Sync
    "Lock",
    "Atomic",
    "Barrier",
    "Await",
    # NN (expanded)
    "∇",
    "Grad",
    "Inf",
    "Train",
    # Parallel
    "||",
}

# Decorator tags (v1.4 new, v1.5 expanded)
DECORATOR_TAGS: set[str] = {
    # Standard Python decorators
    "Prop",  # @property
    "Setter",  # @name.setter
    "Deleter",  # @name.deleter
    "Cached",  # @cached_property or @lru_cache
    "Static",  # @staticmethod
    "Class",  # @classmethod
    "Abstract",  # @abstractmethod (v1.5: also marks abstract classes/methods)
    "Protocol",  # v1.5: marks Protocol classes
    # Common framework decorators
    "Auth",  # Authentication required
    "Retry",  # Retry decorator
    "Timeout",  # Timeout decorator
    "RateLimit",  # Rate limiting
}

# HTTP method tags (v1.4 new)
HTTP_METHODS: set[str] = {
    "GET",
    "POST",
    "PUT",
    "PATCH",
    "DELETE",
    "OPTIONS",
    "HEAD",
}

# All valid tag components (combined)
ALL_TAG_COMPONENTS: set[str] = VALID_TAG_BASES | COMMON_QUALIFIERS | DECORATOR_TAGS | HTTP_METHODS

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

# Metadata keys (from RFC Section 3.1)
METADATA_KEYS: set[str] = {
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
VALID_ROLES: set[str] = {"Core", "Glue", "Script"}

# Valid layers
VALID_LAYERS: set[str] = {"Domain", "Infra", "Adapter", "Test"}

# Valid risk levels
VALID_RISK_LEVELS: set[str] = {"High", "Med", "Low"}

# Entity prefixes (v1.5: added P and E)
ENTITY_PREFIXES: dict[str, str] = {
    "C": "class",
    "D": "data",
    "I": "interface",
    "M": "module",
    "F": "function",
    "P": "protocol",  # v1.5: Protocol (typing.Protocol)
    "E": "enum",  # v1.5: Enum
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


def is_http_method(tag: str) -> bool:
    """Check if a tag is an HTTP method.

    Args:
        tag: Tag name

    Returns:
        True if it's an HTTP method, False otherwise

    Examples:
        >>> is_http_method("GET")
        True
        >>> is_http_method("POST")
        True
        >>> is_http_method("Prop")
        False
    """
    return tag in HTTP_METHODS


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


def is_valid_tag_component(component: str) -> bool:
    """Check if a component is a valid tag component.

    Args:
        component: Tag component string

    Returns:
        True if valid, False otherwise

    Examples:
        >>> is_valid_tag_component("Lin")
        True
        >>> is_valid_tag_component("Prop")
        True
        >>> is_valid_tag_component("O(N)")
        True
        >>> is_valid_tag_component("InvalidTag")
        False
    """
    # Check if it's a known component
    if component in ALL_TAG_COMPONENTS:
        return True

    # Check if it's a complexity tag
    if is_complexity_tag(component):
        return True

    # Check if it's an HTTP route
    if parse_http_route(component):
        return True

    # Allow custom decorator names (alphanumeric + colon for args)
    import re

    if re.match(r"^[A-Z][a-zA-Z0-9_]*(:[a-zA-Z0-9_,]+)?$", component):
        return True

    return False
