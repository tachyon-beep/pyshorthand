"""AST node definitions for PyShorthand.

This module defines the complete Abstract Syntax Tree structure for
representing parsed PyShorthand files. All nodes are immutable dataclasses.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Union


# ============================================================================
# Diagnostic and Error Reporting
# ============================================================================


class DiagnosticSeverity(Enum):
    """Severity levels for diagnostics."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    HINT = "hint"


@dataclass(frozen=True)
class Diagnostic:
    """A diagnostic message (error, warning, or hint)."""

    severity: DiagnosticSeverity
    line: int
    column: int
    message: str
    suggestion: Optional[str] = None
    code: Optional[str] = None

    def __str__(self) -> str:
        """Format diagnostic as human-readable string."""
        severity_str = self.severity.value
        # Include error code if present
        if self.code:
            severity_str = f"{severity_str}[{self.code}]"
        location = f"line {self.line}, column {self.column}"
        result = f"{severity_str}: {self.message}\n  --> {location}"
        if self.suggestion:
            result += f"\n  help: {self.suggestion}"
        return result


# ============================================================================
# Metadata
# ============================================================================


@dataclass(frozen=True)
class Metadata:
    """Module/file-level metadata headers."""

    module_name: Optional[str] = None  # [M:Name]
    module_id: Optional[str] = None  # [ID:Token]
    role: Optional[str] = None  # [Role:Core|Glue|Script]
    layer: Optional[str] = None  # [Layer:Domain|Infra|Adapter|Test]
    risk: Optional[str] = None  # [Risk:High|Med|Low]
    context: Optional[str] = None  # [Context:GPU-RL]
    dims: Dict[str, str] = field(default_factory=dict)  # [Dims:N=agents,B=batch]
    requires: List[str] = field(default_factory=list)  # [Requires:torch>=2.0]
    owner: Optional[str] = None  # [Owner:TeamName]
    custom: Dict[str, str] = field(default_factory=dict)  # Any other metadata

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "module_name": self.module_name,
            "module_id": self.module_id,
            "role": self.role,
            "layer": self.layer,
            "risk": self.risk,
            "context": self.context,
            "dims": self.dims,
            "requires": self.requires,
            "owner": self.owner,
            "custom": self.custom,
        }


# ============================================================================
# Type System
# ============================================================================


@dataclass(frozen=True)
class TypeSpec:
    """Type specification with optional shape and location."""

    base_type: str  # f32, i64, bool, obj, Map, Str, Any
    shape: Optional[List[str]] = None  # [N, C, H, W]
    location: Optional[str] = None  # @CPU, @GPU, @Disk, @Net
    transfer: Optional[Tuple[str, str]] = None  # @CPU→GPU
    union_types: Optional[List[str]] = None  # For Union types: [i32, str, f32]

    def __str__(self) -> str:
        """Format as PyShorthand notation."""
        # Handle Union types
        if self.union_types:
            result = " | ".join(self.union_types)
        else:
            result = self.base_type

        if self.shape:
            result += f"[{', '.join(self.shape)}]"
        if self.transfer:
            result += f"@{self.transfer[0]}→{self.transfer[1]}"
        elif self.location:
            result += f"@{self.location}"
        return result


# ============================================================================
# Tags and Qualifiers
# ============================================================================


@dataclass(frozen=True)
class Tag:
    """Computational tag with qualifiers."""

    base: str  # Lin, Iter, IO, NN, Thresh, Map, Stoch, Sync, Heur
    qualifiers: List[str] = field(default_factory=list)  # O(N), Async, Hot, etc.

    def __str__(self) -> str:
        """Format as [Tag:Qual1:Qual2]."""
        if self.qualifiers:
            return f"[{self.base}:{':'.join(self.qualifiers)}]"
        return f"[{self.base}]"

    @property
    def complexity(self) -> Optional[str]:
        """Extract complexity qualifier if present."""
        for q in self.qualifiers:
            if q.startswith("O("):
                return q
        return None

    @property
    def is_io(self) -> bool:
        """Check if this is an I/O operation."""
        return self.base == "IO"

    @property
    def is_sync(self) -> bool:
        """Check if this is a synchronization point."""
        return self.base == "Sync"


# ============================================================================
# Expressions
# ============================================================================


class Expression(ABC):
    """Base class for all expressions."""

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        pass


@dataclass(frozen=True)
class Identifier(Expression):
    """Variable or function name."""

    name: str

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "identifier", "name": self.name}

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Literal(Expression):
    """Literal value."""

    value: Union[int, float, str, bool]
    type_hint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "literal", "value": self.value, "type_hint": self.type_hint}

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class BinaryOp(Expression):
    """Binary operation."""

    operator: str  # ⊗, +, -, *, /, etc.
    left: Expression
    right: Expression

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "binary_op",
            "operator": self.operator,
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
        }

    def __str__(self) -> str:
        return f"{self.left} {self.operator} {self.right}"


@dataclass(frozen=True)
class FunctionCall(Expression):
    """Function call expression."""

    function: str
    args: List[Expression] = field(default_factory=list)
    kwargs: Dict[str, Expression] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "function_call",
            "function": self.function,
            "args": [arg.to_dict() for arg in self.args],
            "kwargs": {k: v.to_dict() for k, v in self.kwargs.items()},
        }

    def __str__(self) -> str:
        args_str = ", ".join(str(arg) for arg in self.args)
        return f"{self.function}({args_str})"


@dataclass(frozen=True)
class TensorOp(Expression):
    """Tensor operation (matmul, broadcast, etc.)."""

    operation: str  # "matmul", "broadcast", "conv", etc.
    operands: List[Expression]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "tensor_op",
            "operation": self.operation,
            "operands": [op.to_dict() for op in self.operands],
        }

    def __str__(self) -> str:
        return f"⊗({', '.join(str(op) for op in self.operands)})"


@dataclass(frozen=True)
class UnaryOp(Expression):
    """Unary operation: -x, +x, !x."""

    operator: str  # -, +, !
    operand: Expression

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "unary_op",
            "operator": self.operator,
            "operand": self.operand.to_dict(),
        }

    def __str__(self) -> str:
        return f"{self.operator}{self.operand}"


@dataclass(frozen=True)
class IndexOp(Expression):
    """Array/tensor indexing operation: base[i, j, k]."""

    base: Expression
    indices: List[Expression]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "index_op",
            "base": self.base.to_dict(),
            "indices": [idx.to_dict() for idx in self.indices],
        }

    def __str__(self) -> str:
        indices_str = ", ".join(str(idx) for idx in self.indices)
        return f"{self.base}[{indices_str}]"


@dataclass(frozen=True)
class AttributeAccess(Expression):
    """Attribute access or method call: base.attr or base.method()."""

    base: Expression
    attribute: str
    call: Optional['FunctionCall'] = None  # If it's a method call

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "type": "attribute_access",
            "base": self.base.to_dict(),
            "attribute": self.attribute,
        }
        if self.call:
            result["call"] = self.call.to_dict()
        return result

    def __str__(self) -> str:
        if self.call:
            return f"{self.base}.{self.call}"
        return f"{self.base}.{self.attribute}"


# ============================================================================
# Statements
# ============================================================================


@dataclass(frozen=True)
class Statement:
    """A single statement or operation."""

    line: int
    statement_type: str  # "assignment", "mutation", "flow", "return", etc.
    lhs: Optional[Union[str, List[str]]] = None  # Left-hand side (variable name(s))
    operator: Optional[str] = None  # ≡, !, !!, →, ⊳, ←, etc.
    rhs: Optional[Expression] = None  # Right-hand side expression
    tags: List[Tag] = field(default_factory=list)
    profiling: Optional[str] = None  # ⏱16ms
    condition: Optional[Expression] = None  # For conditionals
    comment: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "line": self.line,
            "type": self.statement_type,
            "lhs": self.lhs,
            "operator": self.operator,
            "rhs": self.rhs.to_dict() if self.rhs else None,
            "tags": [str(tag) for tag in self.tags],
            "profiling": self.profiling,
            "condition": self.condition.to_dict() if self.condition else None,
            "comment": self.comment,
        }

    @property
    def is_mutation(self) -> bool:
        """Check if this is a mutation statement."""
        return self.operator in ("!", "!!")

    @property
    def is_system_mutation(self) -> bool:
        """Check if this is a system-level mutation."""
        return self.operator == "!!"

    @property
    def is_error(self) -> bool:
        """Check if this raises or handles errors."""
        return self.operator in ("!?", "?!")


# ============================================================================
# State Variables
# ============================================================================


@dataclass(frozen=True)
class StateVar:
    """State variable declaration with type and location."""

    name: str
    type_spec: Optional[TypeSpec] = None
    line: int = 0
    comment: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": str(self.type_spec) if self.type_spec else None,
            "line": self.line,
            "comment": self.comment,
        }

    def __str__(self) -> str:
        if self.type_spec:
            return f"{self.name} ∈ {self.type_spec}"
        return self.name


# ============================================================================
# Functions
# ============================================================================


@dataclass(frozen=True)
class Parameter:
    """Function parameter."""

    name: str
    type_spec: Optional[TypeSpec] = None
    default: Optional[Expression] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": str(self.type_spec) if self.type_spec else None,
            "default": self.default.to_dict() if self.default else None,
        }

    def __str__(self) -> str:
        result = self.name
        if self.type_spec:
            result += f":{self.type_spec}"
        return result


@dataclass(frozen=True)
class Function:
    """Function definition with contracts."""

    name: str
    params: List[Parameter] = field(default_factory=list)
    return_type: Optional[TypeSpec] = None
    modifiers: List[str] = field(default_factory=list)  # [Async], etc.
    preconditions: List[str] = field(default_factory=list)  # [Pre]
    postconditions: List[str] = field(default_factory=list)  # [Post]
    errors: List[str] = field(default_factory=list)  # [Err]
    body: List[Statement] = field(default_factory=list)
    line: int = 0
    profiling: Optional[str] = None  # ⏱16ms
    tags: List[Tag] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "params": [p.to_dict() for p in self.params],
            "return_type": str(self.return_type) if self.return_type else None,
            "modifiers": self.modifiers,
            "preconditions": self.preconditions,
            "postconditions": self.postconditions,
            "errors": self.errors,
            "body": [stmt.to_dict() for stmt in self.body],
            "line": self.line,
            "profiling": self.profiling,
            "tags": [str(tag) for tag in self.tags],
        }

    @property
    def is_async(self) -> bool:
        """Check if function is async."""
        return "Async" in self.modifiers

    @property
    def complexity(self) -> Optional[str]:
        """Get complexity annotation if present."""
        for tag in self.tags:
            if tag.complexity:
                return tag.complexity
        return None


# ============================================================================
# Entities (Classes, Data, Interfaces, Modules)
# ============================================================================


class Entity(ABC):
    """Base class for all entity types."""

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        pass


@dataclass(frozen=True)
class Reference(Entity):
    """Reference to an external entity."""

    ref_id: str  # [Ref:ID]
    line: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "reference", "ref_id": self.ref_id, "line": self.line}

    def __str__(self) -> str:
        return f"[Ref:{self.ref_id}]"


@dataclass(frozen=True)
class Class(Entity):
    """Class definition with state and methods."""

    name: str
    state: List[StateVar] = field(default_factory=list)
    methods: List[Function] = field(default_factory=list)
    dependencies: List[Reference] = field(default_factory=list)
    line: int = 0
    metadata: Optional[Metadata] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "class",
            "name": self.name,
            "state": [s.to_dict() for s in self.state],
            "methods": [m.to_dict() for m in self.methods],
            "dependencies": [d.to_dict() for d in self.dependencies],
            "line": self.line,
            "metadata": self.metadata.to_dict() if self.metadata else None,
        }


@dataclass(frozen=True)
class Data(Entity):
    """Data structure (dataclass, struct)."""

    name: str
    fields: List[StateVar] = field(default_factory=list)
    line: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "data",
            "name": self.name,
            "fields": [f.to_dict() for f in self.fields],
            "line": self.line,
        }


@dataclass(frozen=True)
class Interface(Entity):
    """Interface/Protocol definition."""

    name: str
    methods: List[str] = field(default_factory=list)  # Method signatures
    line: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "interface", "name": self.name, "methods": self.methods, "line": self.line}


@dataclass(frozen=True)
class Module(Entity):
    """Module/namespace definition."""

    name: str
    exports: List[str] = field(default_factory=list)
    line: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "module", "name": self.name, "exports": self.exports, "line": self.line}


# ============================================================================
# Top-Level AST
# ============================================================================


@dataclass
class PyShortAST:
    """Complete PyShorthand Abstract Syntax Tree."""

    metadata: Metadata = field(default_factory=Metadata)
    entities: List[Entity] = field(default_factory=list)
    functions: List[Function] = field(default_factory=list)
    state: List[StateVar] = field(default_factory=list)
    statements: List[Statement] = field(default_factory=list)
    diagnostics: List[Diagnostic] = field(default_factory=list)
    source_file: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire AST to dictionary."""
        return {
            "metadata": self.metadata.to_dict(),
            "entities": [e.to_dict() for e in self.entities],
            "functions": [f.to_dict() for f in self.functions],
            "state": [s.to_dict() for s in self.state],
            "statements": [s.to_dict() for s in self.statements],
            "diagnostics": [
                {
                    "severity": d.severity.value,
                    "line": d.line,
                    "column": d.column,
                    "message": d.message,
                    "suggestion": d.suggestion,
                }
                for d in self.diagnostics
            ],
            "source_file": self.source_file,
        }

    def add_diagnostic(self, diagnostic: Diagnostic) -> None:
        """Add a diagnostic message."""
        self.diagnostics.append(diagnostic)

    def has_errors(self) -> bool:
        """Check if there are any error-level diagnostics."""
        return any(d.severity == DiagnosticSeverity.ERROR for d in self.diagnostics)

    def has_warnings(self) -> bool:
        """Check if there are any warning-level diagnostics."""
        return any(d.severity == DiagnosticSeverity.WARNING for d in self.diagnostics)

    @property
    def mutations(self) -> List[Statement]:
        """Get all mutation statements."""
        return [s for s in self.statements if s.is_mutation]

    @property
    def system_mutations(self) -> List[Statement]:
        """Get all system-level mutations."""
        return [s for s in self.statements if s.is_system_mutation]

    @property
    def io_operations(self) -> List[Statement]:
        """Get all I/O operations."""
        return [s for s in self.statements if any(tag.is_io for tag in s.tags)]

    @property
    def sync_points(self) -> List[Statement]:
        """Get all synchronization points."""
        return [s for s in self.statements if any(tag.is_sync for tag in s.tags)]
