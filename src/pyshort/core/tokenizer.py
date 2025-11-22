"""Tokenizer for PyShorthand notation.

This module provides lexical analysis for PyShorthand files,
breaking input text into a stream of tokens.
"""

import warnings
from dataclasses import dataclass
from enum import Enum, auto
from typing import Iterator, List, Optional


class TokenType(Enum):
    """Token types for PyShorthand."""

    # Literals and identifiers
    IDENTIFIER = auto()
    NUMBER = auto()
    STRING = auto()

    # Operators
    ARROW = auto()  # →, ->
    HAPPENS_AFTER = auto()  # ⊳, >>
    RETURN = auto()  # ←, <-
    MEMBER_OF = auto()  # ∈, IN
    EQUALS = auto()  # ≡, ==
    ASSIGN = auto()  # =
    LOCAL_MUT = auto()  # !
    SYSTEM_MUT = auto()  # !!
    ERROR_RAISE = auto()  # !?
    ERROR_CHECK = auto()  # ?!
    QUESTION = auto()  # ?
    ASSERT = auto()  # ⊢, ASSERT
    SUM = auto()  # Σ, SUM
    PRODUCT = auto()  # Π, PROD
    TENSOR_OP = auto()  # ⊗, MAT
    GRADIENT = auto()  # ∇, GRAD
    ALIAS = auto()  # ≈, REF
    CLONE = auto()  # ≜, COPY
    FOR_ALL = auto()  # ∀, FOR

    # Delimiters
    LPAREN = auto()  # (
    RPAREN = auto()  # )
    LBRACKET = auto()  # [
    RBRACKET = auto()  # ]
    LBRACE = auto()  # {
    RBRACE = auto()  # }
    COMMA = auto()  # ,
    COLON = auto()  # :
    SEMICOLON = auto()  # ;
    DOT = auto()  # .
    PIPE = auto()  # |
    AT = auto()  # @

    # Special
    COMMENT = auto()  # //
    PROFILING = auto()  # ⏱
    NEWLINE = auto()
    EOF = auto()

    # Math operators
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    PERCENT = auto()
    POWER = auto()
    CARET = auto()

    # Comparison
    GT = auto()
    LT = auto()
    GTE = auto()
    LTE = auto()
    NE = auto()


@dataclass
class Token:
    """A single lexical token."""

    type: TokenType
    value: str
    line: int
    column: int

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, {self.line}:{self.column})"


class Tokenizer:
    """Tokenizes PyShorthand source code."""

    def __init__(self, source: str) -> None:
        """Initialize tokenizer.

        Args:
            source: Source code to tokenize
        """
        self.source = source
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens: List[Token] = []

    def current_char(self) -> Optional[str]:
        """Get current character without advancing."""
        if self.pos >= len(self.source):
            return None
        return self.source[self.pos]

    def peek_char(self, offset: int = 1) -> Optional[str]:
        """Peek at character at offset from current position."""
        pos = self.pos + offset
        if pos >= len(self.source):
            return None
        return self.source[pos]

    def advance(self) -> Optional[str]:
        """Advance position and return current character."""
        if self.pos >= len(self.source):
            return None

        char = self.source[self.pos]
        self.pos += 1

        if char == "\n":
            self.line += 1
            self.column = 1
        else:
            self.column += 1

        return char

    def skip_whitespace(self) -> None:
        """Skip whitespace except newlines."""
        while self.current_char() in (" ", "\t", "\r"):
            self.advance()

    def read_while(self, predicate) -> str:
        """Read characters while predicate is true."""
        start = self.pos
        while self.current_char() is not None and predicate(self.current_char()):
            self.advance()
        return self.source[start : self.pos]

    def read_identifier(self) -> str:
        """Read an identifier or keyword."""
        return self.read_while(lambda c: c.isalnum() or c in ("_", "-"))

    def read_number(self) -> str:
        """Read a numeric literal with range validation."""
        num = ""
        has_decimal = False

        # Read digits and at most one decimal point
        while self.current_char() and (self.current_char().isdigit() or self.current_char() == "."):
            if self.current_char() == ".":
                if has_decimal:
                    # Second decimal point - stop reading
                    break
                has_decimal = True
            num += self.advance() or ""

        # Check for scientific notation
        if self.current_char() in ("e", "E"):
            num += self.advance() or ""
            if self.current_char() in ("+", "-"):
                num += self.advance() or ""
            num += self.read_while(lambda c: c.isdigit())

        # Validate numeric range (P23)
        self._validate_numeric_range(num, has_decimal)

        return num

    def _validate_numeric_range(self, num_str: str, is_float: bool) -> None:
        """Validate that numeric literal is within reasonable range.

        Args:
            num_str: The numeric string
            is_float: True if number has decimal point or scientific notation
        """
        try:
            if is_float or 'e' in num_str.lower():
                # Float validation
                value = float(num_str)

                # Check for infinity (number too large for f64)
                if value == float('inf') or value == float('-inf'):
                    warnings.warn(
                        f"Float literal '{num_str}' at line {self.line} exceeds f64 range, will be represented as infinity",
                        SyntaxWarning
                    )
                # Warn if exceeds f32 range but not f64
                elif abs(value) > 3.4e38:
                    warnings.warn(
                        f"Float literal '{num_str}' at line {self.line} exceeds f32 range (max ±3.4e38), requires f64",
                        SyntaxWarning
                    )
            else:
                # Integer validation
                value = int(num_str)

                # Check if exceeds i64 range
                I64_MAX = 9223372036854775807  # 2^63 - 1
                I64_MIN = -9223372036854775808  # -2^63

                if value > I64_MAX or value < I64_MIN:
                    warnings.warn(
                        f"Integer literal '{num_str}' at line {self.line} exceeds i64 range ({I64_MIN} to {I64_MAX})",
                        SyntaxWarning
                    )
                # Warn if exceeds i32 range but fits in i64
                elif value > 2147483647 or value < -2147483648:
                    warnings.warn(
                        f"Integer literal '{num_str}' at line {self.line} exceeds i32 range, requires i64",
                        SyntaxWarning
                    )

        except (ValueError, OverflowError):
            # Number is malformed or too large to even parse
            warnings.warn(
                f"Numeric literal '{num_str}' at line {self.line} is malformed or extremely large",
                SyntaxWarning
            )

    def read_string(self, quote: str) -> str:
        """Read a string literal."""
        value = ""
        self.advance()  # Skip opening quote

        # Escape sequence mapping
        escape_map = {
            "n": "\n",
            "t": "\t",
            "r": "\r",
            "\\": "\\",
            quote: quote,
            "0": "\0",  # Null character
            "a": "\a",  # Bell/alert
            "b": "\b",  # Backspace
            "f": "\f",  # Form feed
            "v": "\v",  # Vertical tab
        }

        while self.current_char() != quote:
            char = self.current_char()
            if char is None:
                raise ValueError(f"Unterminated string at line {self.line}")
            if char == "\\":
                self.advance()  # Skip the backslash
                next_char = self.current_char()
                if next_char in escape_map:
                    # Add the actual escaped character
                    value += escape_map[next_char]
                    self.advance()  # Move past the escape char
                elif next_char and next_char.isdigit():
                    # Octal escape sequence (e.g., \123) - treat as literal for now
                    warnings.warn(
                        f"Octal escape sequence '\\{next_char}' at line {self.line} not supported, treating as literal",
                        SyntaxWarning
                    )
                    value += "\\" + next_char
                    self.advance()
                elif next_char == 'x':
                    # Hex escape sequence (e.g., \x41) - treat as literal for now
                    warnings.warn(
                        f"Hex escape sequence '\\x' at line {self.line} not supported, treating as literal",
                        SyntaxWarning
                    )
                    value += "\\x"
                    self.advance()
                elif next_char == 'u' or next_char == 'U':
                    # Unicode escape sequence - treat as literal for now
                    warnings.warn(
                        f"Unicode escape sequence '\\{next_char}' at line {self.line} not supported, treating as literal",
                        SyntaxWarning
                    )
                    value += "\\" + (next_char or "")
                    if next_char:
                        self.advance()
                else:
                    # Unknown escape sequence - warn and keep literal
                    if next_char:
                        warnings.warn(
                            f"Unknown escape sequence '\\{next_char}' at line {self.line}, treating as literal",
                            SyntaxWarning
                        )
                        value += "\\" + next_char
                        self.advance()
                    else:
                        value += "\\"
            else:
                value += char
                self.advance()

        self.advance()  # Skip closing quote
        return value

    def read_multiline_string(self, quote: str) -> str:
        """Read a multiline (triple-quoted) string literal.

        Args:
            quote: The quote character (either " or ')

        Returns:
            The string content (without the triple quotes)
        """
        value = ""

        # Skip opening triple quotes
        self.advance()  # First quote
        self.advance()  # Second quote
        self.advance()  # Third quote

        # Build the closing sequence we're looking for
        closing = quote * 3

        # Read until we find the closing triple quotes
        while True:
            char = self.current_char()

            if char is None:
                raise ValueError(f"Unterminated multiline string at line {self.line}")

            # Check if we've reached the closing triple quotes
            if char == quote and self.peek_char() == quote and self.peek_char(2) == quote:
                # Found closing quotes - skip them and return
                self.advance()  # First closing quote
                self.advance()  # Second closing quote
                self.advance()  # Third closing quote
                break

            # Otherwise, add the character (including newlines)
            value += char
            self.advance()

        return value

    def tokenize(self) -> List[Token]:
        """Tokenize the entire source.

        Returns:
            List of tokens
        """
        while self.pos < len(self.source):
            self.skip_whitespace()

            char = self.current_char()
            if char is None:
                break

            line = self.line
            col = self.column

            # Newline
            if char == "\n":
                self.advance()
                self.tokens.append(Token(TokenType.NEWLINE, "\\n", line, col))
                continue

            # Comments (both // and # style)
            if char == "/" and self.peek_char() == "/":
                self.advance()
                self.advance()
                comment = self.read_while(lambda c: c != "\n")
                self.tokens.append(Token(TokenType.COMMENT, comment.strip(), line, col))
                continue

            # Python-style comments (for metadata headers)
            if char == "#":
                self.advance()
                comment = self.read_while(lambda c: c != "\n")
                self.tokens.append(Token(TokenType.COMMENT, comment.strip(), line, col))
                continue

            # Numbers
            if char.isdigit():
                num = self.read_number()
                self.tokens.append(Token(TokenType.NUMBER, num, line, col))
                continue

            # Strings (check for triple-quoted strings first)
            if char in ('"', "'"):
                # Check for triple-quoted string
                if self.peek_char() == char and self.peek_char(2) == char:
                    string_val = self.read_multiline_string(char)
                    self.tokens.append(Token(TokenType.STRING, string_val, line, col))
                else:
                    string_val = self.read_string(char)
                    self.tokens.append(Token(TokenType.STRING, string_val, line, col))
                continue

            # Multi-character operators
            if char == "!" and self.peek_char() == "!":
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.SYSTEM_MUT, "!!", line, col))
                continue

            if char == "!" and self.peek_char() == "?":
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.ERROR_RAISE, "!?", line, col))
                continue

            if char == "?" and self.peek_char() == "!":
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.ERROR_CHECK, "?!", line, col))
                continue

            if char == "-" and self.peek_char() == ">":
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.ARROW, "->", line, col))
                continue

            if char == "<" and self.peek_char() == "-":
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.RETURN, "<-", line, col))
                continue

            if char == ">" and self.peek_char() == ">":
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.HAPPENS_AFTER, ">>", line, col))
                continue

            if char == "=" and self.peek_char() == "=":
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.EQUALS, "==", line, col))
                continue

            if char == ">" and self.peek_char() == "=":
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.GTE, ">=", line, col))
                continue

            if char == "<" and self.peek_char() == "=":
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.LTE, "<=", line, col))
                continue

            if char == "!" and self.peek_char() == "=":
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.NE, "!=", line, col))
                continue

            if char == "*" and self.peek_char() == "*":
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.POWER, "**", line, col))
                continue

            # Skip decorative symbols (◊ for dependencies)
            if char == "◊":
                self.advance()
                continue

            # Unicode operators
            unicode_map = {
                "→": TokenType.ARROW,
                "⊳": TokenType.HAPPENS_AFTER,
                "←": TokenType.RETURN,
                "∈": TokenType.MEMBER_OF,
                "≡": TokenType.EQUALS,
                "⊢": TokenType.ASSERT,
                "Σ": TokenType.SUM,
                "Π": TokenType.PRODUCT,
                "⊗": TokenType.TENSOR_OP,
                "∇": TokenType.GRADIENT,
                "≈": TokenType.ALIAS,
                "≜": TokenType.CLONE,
                "∀": TokenType.FOR_ALL,
                "⏱": TokenType.PROFILING,
            }

            if char in unicode_map:
                self.advance()
                self.tokens.append(Token(unicode_map[char], char, line, col))
                continue

            # Single-character tokens
            single_char_map = {
                "(": TokenType.LPAREN,
                ")": TokenType.RPAREN,
                "[": TokenType.LBRACKET,
                "]": TokenType.RBRACKET,
                "{": TokenType.LBRACE,
                "}": TokenType.RBRACE,
                ",": TokenType.COMMA,
                ":": TokenType.COLON,
                ";": TokenType.SEMICOLON,
                ".": TokenType.DOT,
                "|": TokenType.PIPE,
                "@": TokenType.AT,
                "!": TokenType.LOCAL_MUT,
                "?": TokenType.QUESTION,
                "+": TokenType.PLUS,
                "-": TokenType.MINUS,
                "*": TokenType.STAR,
                "/": TokenType.SLASH,
                "%": TokenType.PERCENT,
                "^": TokenType.CARET,
                ">": TokenType.GT,
                "<": TokenType.LT,
                "=": TokenType.ASSIGN,
            }

            if char in single_char_map:
                self.advance()
                self.tokens.append(Token(single_char_map[char], char, line, col))
                continue

            # Identifiers and keywords
            if char.isalpha() or char == "_":
                identifier = self.read_identifier()

                # Check for ASCII keywords
                keyword_map = {
                    "IN": TokenType.MEMBER_OF,
                    "ASSERT": TokenType.ASSERT,
                    "SUM": TokenType.SUM,
                    "PROD": TokenType.PRODUCT,
                    "MAT": TokenType.TENSOR_OP,
                    "GRAD": TokenType.GRADIENT,
                    "REF": TokenType.ALIAS,
                    "COPY": TokenType.CLONE,
                    "FOR": TokenType.FOR_ALL,
                }

                if identifier in keyword_map:
                    self.tokens.append(Token(keyword_map[identifier], identifier, line, col))
                else:
                    self.tokens.append(Token(TokenType.IDENTIFIER, identifier, line, col))
                continue

            # Unknown character
            raise ValueError(
                f"Unexpected character {char!r} at line {self.line}, column {self.column}"
            )

        # Add EOF token
        self.tokens.append(Token(TokenType.EOF, "", self.line, self.column))
        return self.tokens
