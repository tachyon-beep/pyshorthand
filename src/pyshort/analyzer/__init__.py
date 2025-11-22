"""
PyShorthand Analyzer Package

Provides static analysis tools for PyShorthand code:
- Context pack generation (F0/F1/F2 dependency layers)
- Execution flow tracing (runtime path analysis)
- Dependency graph analysis
- Code structure analysis
"""

from .context_pack import (
    ContextPack,
    ContextPackGenerator,
    generate_context_pack,
)
from .execution_flow import (
    ExecutionFlow,
    ExecutionFlowTracer,
    ExecutionStep,
    trace_execution,
)

__all__ = [
    # Context packs
    "ContextPack",
    "ContextPackGenerator",
    "generate_context_pack",
    # Execution flow tracing
    "ExecutionFlow",
    "ExecutionFlowTracer",
    "ExecutionStep",
    "trace_execution",
]
