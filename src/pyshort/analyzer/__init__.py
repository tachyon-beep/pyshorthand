"""
PyShorthand Analyzer Package

Provides static analysis tools for PyShorthand code:
- Context pack generation (F0/F1/F2 dependency layers)
- Dependency graph analysis
- Code structure analysis
"""

from .context_pack import (
    ContextPack,
    ContextPackGenerator,
    generate_context_pack,
)

__all__ = [
    "ContextPack",
    "ContextPackGenerator",
    "generate_context_pack",
]
