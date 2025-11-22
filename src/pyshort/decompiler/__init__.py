"""Python to PyShorthand decompiler.

This module provides tools to generate PyShorthand from Python source code.
"""

from pyshort.decompiler.py2short import decompile, decompile_file

__all__ = ["decompile", "decompile_file"]
