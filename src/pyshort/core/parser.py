"""PyShorthand parser implementation.

This module provides a recursive descent parser for PyShorthand notation,
building an AST from tokenized input.
"""

import re

from pyshort.core.ast_nodes import (
    AttributeAccess,
    BinaryOp,
    Class,
    Diagnostic,
    DiagnosticSeverity,
    Expression,
    Function,
    FunctionCall,
    Identifier,
    IndexOp,
    Literal,
    Metadata,
    Parameter,
    PyShortAST,
    Reference,
    Statement,
    StateVar,
    Tag,
    TypeSpec,
    UnaryOp,
)
from pyshort.core.symbols import (
    ENTITY_PREFIXES,
    HTTP_METHODS,
    is_complexity_tag,
    is_decorator_tag,
    parse_http_route,
)
from pyshort.core.tokenizer import Token, Tokenizer, TokenType

# Reserved keywords that cannot be used as identifiers
RESERVED_KEYWORDS = {
    # Python keywords
    "and",
    "as",
    "assert",
    "async",
    "await",
    "break",
    "class",
    "continue",
    "def",
    "del",
    "elif",
    "else",
    "except",
    "finally",
    "for",
    "from",
    "global",
    "if",
    "import",
    "in",
    "is",
    "lambda",
    "nonlocal",
    "not",
    "or",
    "pass",
    "raise",
    "return",
    "try",
    "while",
    "with",
    "yield",
    # PyShorthand reserved words
    "C",
    "F",
    "D",
    "I",
    "M",  # Entity prefixes
    "Ref",
    "GPU",
    "CPU",
    "TPU",  # Common annotations
}


class ParseError(Exception):
    """Parse error with location information."""

    def __init__(self, message: str, token: Token) -> None:
        super().__init__(message)
        self.message = message
        self.token = token


class Parser:
    """Recursive descent parser for PyShorthand."""

    def __init__(self, tokens: list[Token]) -> None:
        """Initialize parser.

        Args:
            tokens: List of tokens from tokenizer
        """
        self.tokens = tokens
        self.pos = 0
        self.current_token = self.tokens[0] if tokens else Token(TokenType.EOF, "", 0, 0)

    def advance(self) -> Token:
        """Advance to next token and return previous."""
        prev = self.current_token
        if self.pos < len(self.tokens) - 1:
            self.pos += 1
            self.current_token = self.tokens[self.pos]
        return prev

    def peek(self, offset: int = 1) -> Token | None:
        """Peek at token at offset from current position."""
        pos = self.pos + offset
        if pos < len(self.tokens):
            return self.tokens[pos]
        return None

    def expect(self, token_type: TokenType) -> Token:
        """Expect a specific token type and advance.

        Args:
            token_type: Expected token type

        Returns:
            The matched token

        Raises:
            ParseError: If token doesn't match
        """
        if self.current_token.type != token_type:
            raise ParseError(
                f"Expected {token_type.name}, got {self.current_token.type.name}",
                self.current_token,
            )
        return self.advance()

    def validate_identifier(self, name: str, token: Token) -> None:
        """Validate that identifier is not a reserved keyword.

        Args:
            name: Identifier name to validate
            token: Token for error reporting

        Raises:
            ParseError: If identifier is a reserved keyword
        """
        if name in RESERVED_KEYWORDS:
            raise ParseError(
                f"'{name}' is a reserved keyword and cannot be used as an identifier", token
            )

    def skip_newlines(self) -> None:
        """Skip any newline tokens."""
        while self.current_token.type == TokenType.NEWLINE:
            self.advance()

    def skip_comments(self) -> None:
        """Skip any comment tokens."""
        while self.current_token.type == TokenType.COMMENT:
            self.advance()

    def skip_whitespace(self) -> None:
        """Skip newlines and comments."""
        while self.current_token.type in (TokenType.NEWLINE, TokenType.COMMENT):
            self.advance()

    def parse_metadata_value(self, value_str: str) -> str:
        """Parse metadata value, handling special cases."""
        return value_str.strip()

    def parse_metadata_header(self) -> Metadata:
        """Parse metadata header from comments.

        Expected format:
        # [M:Name] [ID:ID] [Role:Core] [Layer:Domain] [Risk:High]
        # [Context:GPU-ML] [Dims:N=agents,B=batch]
        """
        metadata_dict = {}
        dims = {}
        requires = []

        # Collect all header comment lines
        while self.current_token.type == TokenType.COMMENT:
            comment = self.current_token.value
            self.advance()

            # Extract metadata tags [Key:Value]
            tags = re.findall(r"\[([^:\]]+):([^\]]+)\]", comment)

            for key, value in tags:
                key = key.strip()
                value = value.strip()

                if key == "Dims":
                    # Parse dimension definitions: N=agents, B=batch
                    for dim_def in value.split(","):
                        if "=" in dim_def:
                            dim_name, dim_desc = dim_def.split("=", 1)
                            dims[dim_name.strip()] = dim_desc.strip()
                elif key == "Requires":
                    # Parse requirements
                    requires.extend(v.strip() for v in value.split(","))
                else:
                    metadata_dict[key] = value

            self.skip_newlines()

        return Metadata(
            module_name=metadata_dict.get("M"),
            module_id=metadata_dict.get("ID"),
            role=metadata_dict.get("Role"),
            layer=metadata_dict.get("Layer"),
            risk=metadata_dict.get("Risk"),
            context=metadata_dict.get("Context"),
            dims=dims,
            requires=requires,
            owner=metadata_dict.get("Owner"),
            custom={k: v for k, v in metadata_dict.items() if k not in Metadata.__annotations__},
        )

    def _is_v14_tag(self) -> bool:
        """Check if current bracket is a v1.4 tag (not a shape parameter).

        Returns True if the bracket content looks like:
        - Decorator tag: Prop, Static, Class, Cached, Auth, etc.
        - HTTP route tag: GET /path, POST /api, etc.
        - Complexity tag: O(N), O(N²), etc.
        - Operation tag: NN:∇, Lin:MatMul, IO:Disk, etc.
        - Custom tag with qualifiers
        """
        if self.current_token.type != TokenType.LBRACKET:
            return False

        # Save current position
        saved_pos = self.pos
        saved_token = self.current_token

        try:
            self.advance()  # Skip [

            if (
                self.current_token.type == TokenType.EOF
                or self.current_token.type == TokenType.RBRACKET
            ):
                return False

            first_token = (
                self.current_token.value if self.current_token.type == TokenType.IDENTIFIER else ""
            )

            # Decorator tags
            if first_token in (
                "Prop",
                "Static",
                "Class",
                "Cached",
                "Auth",
                "Async",
                "Pure",
                "Safe",
            ):
                return True

            # HTTP methods (route tags)
            if first_token in ("GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"):
                return True

            # Complexity tags: O(...)
            if first_token == "O" and self.peek() and self.peek().type == TokenType.LPAREN:
                return True

            # Operation tags: NN, Lin, IO, Iter, Sync
            if first_token in ("NN", "Lin", "IO", "Iter", "Sync"):
                return True

            # Check for qualifiers (colon after first token suggests tag, not shape)
            if self.peek() and self.peek().type == TokenType.COLON:
                return True

            # If it's a single identifier with no comma, could be either shape or tag
            # Default to shape for backward compatibility, unless it matches known patterns
            return False

        finally:
            # Restore position
            self.pos = saved_pos
            self.current_token = saved_token

    def parse_type_spec(self) -> TypeSpec:
        """Parse type specification.

        Format: Type[Shape]@Location or Type[Shape]@Loc1→Loc2
        Or: [Ref:Name] for references
        Or: Type1 | Type2 | Type3 for unions (P16)
        """
        # Handle reference types: [Ref:Name]
        if self.current_token.type == TokenType.LBRACKET:
            # Could be a reference or a shape on a type without explicit base
            next_tok = self.peek()
            if next_tok and next_tok.type == TokenType.IDENTIFIER and next_tok.value == "Ref":
                # This is a reference type: [Ref:Name]
                # Skip the entire reference as the "type"
                ref_str = self.parse_reference_string()

                # Check for union with reference type
                union_types = None
                if self.current_token.type == TokenType.PIPE:
                    union_types = [ref_str]
                    while self.current_token.type == TokenType.PIPE:
                        self.advance()  # Skip |
                        # Next type could be another reference or a regular type
                        if self.current_token.type == TokenType.LBRACKET:
                            union_types.append(self.parse_reference_string())
                        else:
                            union_types.append(self.expect(TokenType.IDENTIFIER).value)

                return TypeSpec(
                    base_type=ref_str,
                    shape=None,
                    location=None,
                    transfer=None,
                    union_types=union_types,
                )

        base_type = self.expect(TokenType.IDENTIFIER).value

        # v1.5: Parse generic parameters <T, U> if present
        generic_params = None
        if self.current_token.type == TokenType.LT:
            self.advance()
            generic_params = []
            while self.current_token.type != TokenType.GT:
                if self.current_token.type == TokenType.IDENTIFIER:
                    generic_params.append(self.current_token.value)
                    self.advance()
                elif self.current_token.type == TokenType.ARROW:
                    # For Callable<T→U>, include the arrow
                    generic_params.append("→")
                    self.advance()
                elif self.current_token.type == TokenType.COMMA:
                    self.advance()
                elif self.current_token.type == TokenType.EOF:
                    raise ParseError("Unterminated generic parameters", self.current_token)
                else:
                    self.advance()
            self.expect(TokenType.GT)

        # v1.5: Parse nested structure { key: Type, ... } if present
        nested_structure = None
        if self.current_token.type == TokenType.LBRACE:
            self.advance()
            nested_structure = {}
            while self.current_token.type != TokenType.RBRACE:
                if self.current_token.type == TokenType.IDENTIFIER:
                    key = self.current_token.value
                    self.advance()
                    if self.current_token.type == TokenType.COLON:
                        self.advance()
                        if self.current_token.type == TokenType.IDENTIFIER:
                            value = self.current_token.value
                            self.advance()
                            nested_structure[key] = value
                if self.current_token.type == TokenType.COMMA:
                    self.advance()
                elif self.current_token.type == TokenType.EOF:
                    raise ParseError("Unterminated nested structure", self.current_token)
            self.expect(TokenType.RBRACE)

        # Check for union types (Type1 | Type2 | Type3)
        union_types = None
        if self.current_token.type == TokenType.PIPE:
            union_types = [base_type]  # Start with the base type
            while self.current_token.type == TokenType.PIPE:
                self.advance()  # Skip |
                # Next type could be a reference or a regular type
                if self.current_token.type == TokenType.LBRACKET:
                    union_types.append(self.parse_reference_string())
                else:
                    union_types.append(self.expect(TokenType.IDENTIFIER).value)

        shape = None
        if self.current_token.type == TokenType.LBRACKET:
            # Check if this bracket is a shape or a v1.4 tag
            if not self._is_v14_tag():
                shape = self.parse_shape()

        location = None
        transfer = None
        if self.current_token.type == TokenType.AT:
            self.advance()  # Skip @
            loc1 = self.expect(TokenType.IDENTIFIER).value

            # Check for transfer: @A→B
            if self.current_token.type == TokenType.ARROW:
                self.advance()
                loc2 = self.expect(TokenType.IDENTIFIER).value
                transfer = (loc1, loc2)
            else:
                location = loc1

        return TypeSpec(
            base_type=base_type,
            shape=shape,
            location=location,
            transfer=transfer,
            union_types=union_types,
            generic_params=generic_params,  # v1.5
            nested_structure=nested_structure,  # v1.5
        )

    def parse_reference_string(self) -> str:
        """Parse a reference and return it as a string: [Ref:Name] → 'Ref:Name'."""
        self.expect(TokenType.LBRACKET)
        ref_parts = []
        while self.current_token.type not in (TokenType.RBRACKET, TokenType.EOF):
            if self.current_token.type in (TokenType.IDENTIFIER, TokenType.COLON):
                ref_parts.append(self.current_token.value)
                self.advance()
            else:
                self.advance()  # Skip other tokens
        if self.current_token.type == TokenType.EOF:
            raise ParseError("Unterminated reference, expected ']'", self.current_token)
        self.expect(TokenType.RBRACKET)
        return "".join(ref_parts)

    def parse_shape(self) -> list[str]:
        """Parse shape specification [N, C, H, W]."""
        self.expect(TokenType.LBRACKET)
        dimensions = []

        while self.current_token.type not in (TokenType.RBRACKET, TokenType.EOF):
            if self.current_token.type == TokenType.IDENTIFIER or self.current_token.type == TokenType.NUMBER:
                dimensions.append(self.current_token.value)
                self.advance()
            elif self.current_token.type == TokenType.COMMA:
                self.advance()
            else:
                raise ParseError(
                    f"Unexpected token in shape: {self.current_token.value}", self.current_token
                )

        if self.current_token.type == TokenType.EOF:
            raise ParseError("Unterminated shape specification, expected ']'", self.current_token)
        self.expect(TokenType.RBRACKET)
        return dimensions

    def parse_tag(self) -> Tag:
        """Parse a computational tag [Base:Qual1:Qual2].

        Supports v1.4 tag types:
        - [Lin:MatMul] - operation tags
        - [O(N*M)] - complexity tags
        - [Prop], [Static] - decorator tags
        - [GET /path], [POST /api/users/{id}] - HTTP route tags
        """
        self.expect(TokenType.LBRACKET)

        # Collect all content inside brackets
        content = ""
        prev_token_type = None
        while self.current_token.type not in (TokenType.RBRACKET, TokenType.EOF):
            # Add space before slash if previous token was an HTTP method
            if (
                self.current_token.type == TokenType.SLASH
                and prev_token_type == TokenType.IDENTIFIER
                and content in HTTP_METHODS
            ):
                content += " "

            content += self.current_token.value
            prev_token_type = self.current_token.type
            self.advance()

        if self.current_token.type == TokenType.EOF:
            raise ParseError("Unterminated tag, expected ']'", self.current_token)
        self.expect(TokenType.RBRACKET)

        if not content:
            raise ParseError("Empty tag", self.current_token)

        # Strip whitespace
        content = content.strip()

        # Check for HTTP route tag: [GET /path], [POST /api/users/{id}]
        route = parse_http_route(content)
        if route:
            method, path = route
            return Tag(base=content, tag_type="http_route", http_method=method, http_path=path)

        # Check for complexity tag: [O(N)], [O(N*M*D)]
        if is_complexity_tag(content):
            return Tag(base=content, tag_type="complexity")

        # Parse colon-separated parts for operation/decorator tags
        parts = content.split(":")

        # Check for decorator tag: [Prop], [Static], [Cached:TTL:60]
        if is_decorator_tag(parts[0]):
            return Tag(
                base=parts[0], qualifiers=parts[1:] if len(parts) > 1 else [], tag_type="decorator"
            )

        # Default to operation tag: [Lin:MatMul], [Iter:Hot:O(N)]
        return Tag(
            base=parts[0], qualifiers=parts[1:] if len(parts) > 1 else [], tag_type="operation"
        )

    def parse_expression(self) -> Expression:
        """Parse an expression."""
        return self.parse_binary_expr()

    def parse_binary_expr(self) -> Expression:
        """Parse binary expression with operators."""
        left = self.parse_primary_expr()

        while self.current_token.type in (
            TokenType.PLUS,
            TokenType.MINUS,
            TokenType.STAR,
            TokenType.SLASH,
            TokenType.AT,  # Matrix multiplication (@)
            TokenType.TENSOR_OP,
            TokenType.CARET,
            TokenType.GT,
            TokenType.LT,
            TokenType.GTE,
            TokenType.LTE,
            TokenType.NE,
        ):
            op = self.current_token.value
            self.advance()
            right = self.parse_primary_expr()
            left = BinaryOp(operator=op, left=left, right=right)

        return left

    def parse_primary_expr(self) -> Expression:
        """Parse primary expression with postfix operators (indexing, attribute access, calls)."""
        # Parse base expression
        base_expr = self._parse_base_expr()

        # Handle postfix operators: indexing, attribute access, method calls
        while True:
            if self.current_token.type == TokenType.LBRACKET:
                # Array indexing: expr[i, j, k]
                base_expr = self._parse_indexing(base_expr)
            elif self.current_token.type == TokenType.DOT:
                # Attribute access or method call: expr.attr or expr.method()
                base_expr = self._parse_attribute_access(base_expr)
            elif self.current_token.type == TokenType.LPAREN and isinstance(base_expr, Identifier):
                # Function call: name(args)
                base_expr = self.parse_function_call(base_expr.name)
            else:
                # No more postfix operators
                break

        return base_expr

    def _parse_base_expr(self) -> Expression:
        """Parse base expression (number, string, identifier, parenthesized expr, unary ops)."""
        # Unary operators: -, +
        if self.current_token.type in (TokenType.MINUS, TokenType.PLUS):
            op = self.current_token.value
            self.advance()
            operand = self._parse_base_expr()  # Recursively parse operand
            return UnaryOp(operator=op, operand=operand)

        # Number literal
        if self.current_token.type == TokenType.NUMBER:
            value = self.current_token.value
            self.advance()
            return Literal(value=float(value) if "." in value else int(value))

        # String literal
        if self.current_token.type == TokenType.STRING:
            value = self.current_token.value
            self.advance()
            return Literal(value=value)

        # Identifier
        if self.current_token.type == TokenType.IDENTIFIER:
            name = self.current_token.value
            self.advance()
            return Identifier(name=name)

        # Parenthesized expression
        if self.current_token.type == TokenType.LPAREN:
            self.advance()
            expr = self.parse_expression()
            self.expect(TokenType.RPAREN)
            return expr

        raise ParseError(
            f"Unexpected token in expression: {self.current_token.value}", self.current_token
        )

    def _parse_indexing(self, base: Expression) -> Expression:
        """Parse array indexing: base[i, j, k]."""
        self.expect(TokenType.LBRACKET)
        indices = []

        while self.current_token.type not in (TokenType.RBRACKET, TokenType.EOF):
            indices.append(self.parse_expression())
            if self.current_token.type == TokenType.COMMA:
                self.advance()
            elif self.current_token.type not in (TokenType.RBRACKET, TokenType.EOF):
                break

        if self.current_token.type == TokenType.EOF:
            raise ParseError("Unterminated array indexing, expected ']'", self.current_token)
        self.expect(TokenType.RBRACKET)

        # Return IndexOp expression
        return IndexOp(base=base, indices=indices)

    def _parse_attribute_access(self, base: Expression) -> Expression:
        """Parse attribute access: base.attr or base.method()."""
        self.expect(TokenType.DOT)
        attr_name = self.expect(TokenType.IDENTIFIER).value

        # Check if it's a method call
        if self.current_token.type == TokenType.LPAREN:
            method_call = self.parse_function_call(attr_name)
            # Convert to AttributeAccess with method call
            return AttributeAccess(base=base, attribute=attr_name, call=method_call)
        else:
            # Just attribute access
            return AttributeAccess(base=base, attribute=attr_name, call=None)

    def parse_function_call(self, name: str) -> FunctionCall:
        """Parse function call."""
        self.expect(TokenType.LPAREN)
        args = []

        while self.current_token.type not in (TokenType.RPAREN, TokenType.EOF):
            args.append(self.parse_expression())
            if self.current_token.type == TokenType.COMMA:
                self.advance()
            elif self.current_token.type not in (TokenType.RPAREN, TokenType.EOF):
                # Expected comma or closing paren
                break

        if self.current_token.type == TokenType.EOF:
            raise ParseError(
                f"Unterminated function call '{name}', expected ')'", self.current_token
            )
        self.expect(TokenType.RPAREN)
        return FunctionCall(function=name, args=args)

    def parse_state_var(self, line: int) -> StateVar:
        """Parse state variable declaration.

        Format: name ∈ Type[Shape]@Location
        """
        name_token = self.expect(TokenType.IDENTIFIER)
        name = name_token.value

        # Validate variable name
        self.validate_identifier(name, name_token)

        type_spec = None
        if self.current_token.type == TokenType.MEMBER_OF:
            self.advance()
            type_spec = self.parse_type_spec()

        # Optional comment
        comment = None
        if self.current_token.type == TokenType.COMMENT:
            comment = self.current_token.value
            self.advance()

        return StateVar(name=name, type_spec=type_spec, line=line, comment=comment)

    def parse_statement(self, line: int) -> Statement:
        """Parse a single statement."""
        # Skip leading whitespace
        self.skip_comments()

        # Profiling annotation
        profiling = None
        if self.current_token.type == TokenType.PROFILING:
            self.advance()
            if self.current_token.type == TokenType.NUMBER:
                profiling = self.current_token.value
                self.advance()
                if self.current_token.type == TokenType.IDENTIFIER:
                    profiling += self.current_token.value  # e.g., "ms"
                    self.advance()

        # Phase markers {Phase: Name}
        if self.current_token.type == TokenType.LBRACE:
            # Skip phase markers
            while (
                self.current_token.type != TokenType.RBRACE
                and self.current_token.type != TokenType.EOF
            ):
                self.advance()
            if self.current_token.type == TokenType.RBRACE:
                self.advance()
            return Statement(line=line, statement_type="phase")

        # Happens-after dependency marker
        if self.current_token.type == TokenType.HAPPENS_AFTER:
            self.advance()
            # Parse the statement after the ⊳
            stmt = self.parse_statement(line)
            # Mark it as having causal dependency
            return Statement(
                line=line,
                statement_type="causal",
                operator="⊳",
                rhs=Identifier(name="next"),  # Placeholder
                tags=stmt.tags,
            )

        # Assertion
        if self.current_token.type == TokenType.ASSERT:
            self.advance()
            condition = self.parse_expression()
            return Statement(line=line, statement_type="assertion", operator="⊢", rhs=condition)

        # Return statement
        if self.current_token.type == TokenType.RETURN:
            self.advance()
            expr = None
            if self.current_token.type not in (TokenType.NEWLINE, TokenType.EOF):
                expr = self.parse_expression()
            return Statement(line=line, statement_type="return", operator="←", rhs=expr)

        # Conditional
        if self.current_token.type == TokenType.QUESTION:
            self.advance()
            condition = self.parse_expression()
            tags = []
            if self.current_token.type == TokenType.ARROW:
                self.advance()
                if self.current_token.type == TokenType.LBRACKET:
                    tags.append(self.parse_tag())
            return Statement(
                line=line,
                statement_type="conditional",
                operator="?",
                condition=condition,
                tags=tags,
            )

        # Assignment or mutation
        if self.current_token.type == TokenType.IDENTIFIER:
            lhs = self.current_token.value
            self.advance()

            # Check operator
            operator = None
            if self.current_token.type in (
                TokenType.EQUALS,
                TokenType.ASSIGN,
                TokenType.LOCAL_MUT,
                TokenType.SYSTEM_MUT,
                TokenType.MEMBER_OF,
            ):
                operator = self.current_token.value
                self.advance()

            # Parse RHS
            rhs = None
            if self.current_token.type not in (TokenType.NEWLINE, TokenType.EOF):
                rhs = self.parse_expression()

            # Parse tags if flow operator present
            tags = []
            if self.current_token.type == TokenType.ARROW:
                self.advance()
                if self.current_token.type == TokenType.LBRACKET:
                    tags.append(self.parse_tag())

            statement_type = "assignment"
            if operator in ("!", "!!"):
                statement_type = "mutation"

            return Statement(
                line=line,
                statement_type=statement_type,
                lhs=lhs,
                operator=operator,
                rhs=rhs,
                tags=tags,
                profiling=profiling,
            )

        # System mutation (function call with !!)
        if self.current_token.type == TokenType.SYSTEM_MUT:
            self.advance()
            if self.current_token.type == TokenType.IDENTIFIER:
                call = self.parse_function_call(self.current_token.value)
                self.advance()
                tags = []
                if self.current_token.type == TokenType.ARROW:
                    self.advance()
                    if self.current_token.type == TokenType.LBRACKET:
                        tags.append(self.parse_tag())
                return Statement(
                    line=line,
                    statement_type="mutation",
                    operator="!!",
                    rhs=call,
                    tags=tags,
                )

        # CRITICAL: Advance token to prevent infinite loop on unknown tokens
        # If we reach here, we didn't recognize the token - skip it
        if self.current_token.type != TokenType.EOF:
            self.advance()

        return Statement(line=line, statement_type="unknown")

    def parse_function(self, line: int) -> Function:
        """Parse function definition.

        Format:
        F:name(params) [modifiers]
          [Pre] conditions
          [Post] conditions
          [Err] errors
          body
        """
        # Expect F:name
        self.expect(TokenType.IDENTIFIER)  # F
        self.expect(TokenType.COLON)
        name_token = self.expect(TokenType.IDENTIFIER)
        name = name_token.value

        # Validate function name
        self.validate_identifier(name, name_token)

        # Parse parameters
        params = []
        if self.current_token.type == TokenType.LPAREN:
            self.advance()
            while self.current_token.type != TokenType.RPAREN:
                param_token = self.expect(TokenType.IDENTIFIER)
                param_name = param_token.value

                # Validate parameter name
                self.validate_identifier(param_name, param_token)

                type_spec = None
                if self.current_token.type == TokenType.COLON:
                    self.advance()
                    type_spec = self.parse_type_spec()
                params.append(Parameter(name=param_name, type_spec=type_spec))

                if self.current_token.type == TokenType.COMMA:
                    self.advance()
            self.expect(TokenType.RPAREN)

        # Parse return type if present: → Type[Shape]@Location
        return_type = None
        if self.current_token.type == TokenType.ARROW:
            self.advance()
            return_type = self.parse_type_spec()

        # Parse modifiers [Async], [Role:Core], etc.
        modifiers = []
        tags = []
        while self.current_token.type == TokenType.LBRACKET:
            tag = self.parse_tag()
            if tag.base in ("Async", "Pure", "Safe"):
                modifiers.append(tag.base)
            else:
                tags.append(tag)

        self.skip_newlines()

        # Parse contracts
        preconditions = []
        postconditions = []
        errors = []

        while self.current_token.type == TokenType.LBRACKET:
            self.advance()
            contract_type = self.expect(TokenType.IDENTIFIER).value
            self.expect(TokenType.RBRACKET)

            if contract_type == "Pre":
                # Parse precondition
                cond = self.read_until_newline()
                preconditions.append(cond)
            elif contract_type == "Post":
                cond = self.read_until_newline()
                postconditions.append(cond)
            elif contract_type == "Err":
                err = self.read_until_newline()
                errors.extend(e.strip() for e in err.split(","))

            self.skip_newlines()

        # Parse body
        body = []
        while self.current_token.type not in (TokenType.EOF,) and not self.is_next_entity():
            if self.current_token.type == TokenType.NEWLINE:
                self.advance()
                continue
            stmt = self.parse_statement(self.current_token.line)
            body.append(stmt)
            self.skip_whitespace()

        return Function(
            name=name,
            params=params,
            return_type=return_type,
            modifiers=modifiers,
            tags=tags,
            preconditions=preconditions,
            postconditions=postconditions,
            errors=errors,
            body=body,
            line=line,
        )

    def is_next_entity(self) -> bool:
        """Check if next token starts an entity definition."""
        if self.current_token.type == TokenType.IDENTIFIER:
            return self.current_token.value in ENTITY_PREFIXES
        return False

    def read_until_newline(self) -> str:
        """Read tokens until newline."""
        parts = []
        while self.current_token.type not in (TokenType.NEWLINE, TokenType.EOF):
            parts.append(self.current_token.value)
            self.advance()
        return " ".join(parts)

    def parse(self) -> PyShortAST:
        """Parse the entire PyShorthand file.

        Returns:
            Complete AST
        """
        ast = PyShortAST()

        try:
            # Skip only newlines before metadata (not comments!)
            self.skip_newlines()

            # Parse metadata header (which are comments)
            if self.current_token.type == TokenType.COMMENT:
                ast.metadata = self.parse_metadata_header()

            # Now skip both newlines and comments
            self.skip_whitespace()

            # Parse entities and statements
            while self.current_token.type != TokenType.EOF:
                self.skip_whitespace()

                if self.current_token.type == TokenType.EOF:
                    break

                line = self.current_token.line

                # Entity definition with bracket syntax: [C:Name], [D:Name], etc.
                if self.current_token.type == TokenType.LBRACKET:
                    # Peek ahead to check if this is an entity definition
                    next_tok = self.peek(1)
                    if (
                        next_tok
                        and next_tok.type == TokenType.IDENTIFIER
                        and next_tok.value in ENTITY_PREFIXES
                    ):
                        # This is an entity definition
                        self.advance()  # Skip [
                        entity_prefix = self.current_token.value

                        if entity_prefix in ("C", "P"):  # v1.5: P for Protocol
                            try:
                                entity = self.parse_class(line)
                                ast.entities.append(entity)
                            except ParseError as e:
                                # Add diagnostic but don't crash entire parse
                                diagnostic = Diagnostic(
                                    severity=DiagnosticSeverity.ERROR,
                                    line=e.token.line,
                                    column=e.token.column,
                                    message=e.message,
                                )
                                ast.add_diagnostic(diagnostic)
                                # Skip to next likely entity
                                while self.current_token.type not in (
                                    TokenType.EOF,
                                    TokenType.LBRACKET,
                                ):
                                    if (
                                        self.current_token.type == TokenType.IDENTIFIER
                                        and self.current_token.value in ENTITY_PREFIXES
                                    ):
                                        break
                                    self.advance()
                        elif entity_prefix == "F":
                            try:
                                func = self.parse_function(line)
                                ast.functions.append(func)
                            except ParseError as e:
                                diagnostic = Diagnostic(
                                    severity=DiagnosticSeverity.ERROR,
                                    line=e.token.line,
                                    column=e.token.column,
                                    message=e.message,
                                )
                                ast.add_diagnostic(diagnostic)
                        else:
                            # Other entity types - skip the rest
                            while (
                                self.current_token.type != TokenType.RBRACKET
                                and self.current_token.type != TokenType.EOF
                            ):
                                self.advance()
                            if self.current_token.type == TokenType.RBRACKET:
                                self.advance()
                    else:
                        # Not an entity, treat as statement
                        stmt = self.parse_statement(line)
                        if stmt.statement_type != "unknown":
                            ast.statements.append(stmt)

                # Entity definition without brackets (old style): C:Name, F:name, etc.
                elif self.current_token.type == TokenType.IDENTIFIER:
                    entity_prefix = self.current_token.value

                    if entity_prefix in ("C", "P"):  # v1.5: P for Protocol
                        # Class or Protocol definition
                        entity = self.parse_class(line)
                        ast.entities.append(entity)
                    elif entity_prefix == "F":
                        # Function definition
                        func = self.parse_function(line)
                        ast.functions.append(func)
                    elif entity_prefix in ENTITY_PREFIXES:
                        # Other entity types
                        self.advance()
                    else:
                        # Statement
                        stmt = self.parse_statement(line)
                        if stmt.statement_type != "unknown":
                            ast.statements.append(stmt)
                else:
                    # Statement or unknown token
                    stmt = self.parse_statement(line)
                    if stmt.statement_type != "unknown":
                        ast.statements.append(stmt)

                self.skip_whitespace()

        except ParseError as e:
            diagnostic = Diagnostic(
                severity=DiagnosticSeverity.ERROR,
                line=e.token.line,
                column=e.token.column,
                message=e.message,
            )
            ast.add_diagnostic(diagnostic)

        return ast

    def parse_class(self, line: int) -> Class:
        """Parse class definition.

        v1.5 supports:
        - [C:Foo] ◊ Bar, Baz - inheritance
        - [C:List<T>] - generic parameters
        - [C:Foo] [Abstract] - abstract class marker
        - [P:Drawable] [Protocol] - protocol marker
        """
        entity_type = self.expect(TokenType.IDENTIFIER)  # C or P
        is_protocol = entity_type.value == "P"

        self.expect(TokenType.COLON)
        name_token = self.expect(TokenType.IDENTIFIER)
        name = name_token.value

        # Validate class name
        self.validate_identifier(name, name_token)

        # v1.5: Parse generic parameters <T, U> if present
        generic_params = []
        if self.current_token.type == TokenType.LT:
            self.advance()
            while self.current_token.type != TokenType.GT:
                if self.current_token.type == TokenType.IDENTIFIER:
                    generic_params.append(self.current_token.value)
                    self.advance()
                if self.current_token.type == TokenType.COMMA:
                    self.advance()
                elif self.current_token.type == TokenType.EOF:
                    raise ParseError("Unterminated generic parameters", self.current_token)
            self.expect(TokenType.GT)

        # If using bracket syntax [C:Name], consume the closing ]
        if self.current_token.type == TokenType.RBRACKET:
            self.advance()

        # v1.5: Parse [Abstract] or [Protocol] tags if present
        is_abstract = False
        while self.current_token.type == TokenType.LBRACKET:
            peek = self.peek(1)
            if peek and peek.type == TokenType.IDENTIFIER:
                if peek.value == "Abstract":
                    self.advance()  # [
                    self.advance()  # Abstract
                    self.expect(TokenType.RBRACKET)
                    is_abstract = True
                elif peek.value == "Protocol":
                    self.advance()  # [
                    self.advance()  # Protocol
                    self.expect(TokenType.RBRACKET)
                    is_protocol = True
                else:
                    break
            else:
                break

        self.skip_newlines()

        # v1.5: Parse inheritance ◊ Base1, Base2
        base_classes = []
        if self.current_token.type == TokenType.EXTENDS:
            self.advance()
            while True:
                if self.current_token.type == TokenType.IDENTIFIER:
                    base_classes.append(self.current_token.value)
                    self.advance()
                    # Handle dotted names like nn.Module
                    while self.current_token.type == TokenType.DOT:
                        base_classes[-1] += "."
                        self.advance()
                        if self.current_token.type == TokenType.IDENTIFIER:
                            base_classes[-1] += self.current_token.value
                            self.advance()
                if self.current_token.type == TokenType.COMMA:
                    self.advance()
                    continue
                break
            self.skip_newlines()

        # Parse dependencies
        dependencies = []
        if self.current_token.type == TokenType.LBRACKET:
            # Dependencies line like: ◊ [Ref:A], [Ref:B]
            while self.current_token.type not in (TokenType.NEWLINE, TokenType.EOF):
                if self.current_token.type == TokenType.LBRACKET:
                    self.advance()
                    if self.current_token.value == "Ref":
                        self.advance()
                        self.expect(TokenType.COLON)
                        ref_id = self.expect(TokenType.IDENTIFIER).value
                        self.expect(TokenType.RBRACKET)
                        dependencies.append(Reference(ref_id=ref_id, line=line))
                else:
                    self.advance()
            self.skip_newlines()

        # Skip comments and newlines before state variables
        self.skip_whitespace()

        # Parse state variables
        state = []
        while self.current_token.type == TokenType.IDENTIFIER:
            # Check if this is a method (F:name) or state variable
            if self.peek() and self.peek().type == TokenType.COLON:  # type: ignore
                break
            state_var = self.parse_state_var(self.current_token.line)
            state.append(state_var)
            self.skip_newlines()

        # Skip comments before methods
        self.skip_whitespace()

        # Parse methods
        methods = []
        while self.current_token.type == TokenType.IDENTIFIER and self.current_token.value == "F":
            try:
                method = self.parse_function(self.current_token.line)
                methods.append(method)
                self.skip_newlines()
            except ParseError:
                # Skip this method and try to continue with next one
                # Find next method or end of class
                while self.current_token.type != TokenType.EOF:
                    if (
                        self.current_token.type == TokenType.IDENTIFIER
                        and self.current_token.value == "F"
                    ):
                        # Found next method
                        break
                    if self.current_token.type == TokenType.LBRACKET:
                        # Might be next entity
                        break
                    self.advance()
                # Don't re-raise - continue building class with what we have

        return Class(
            name=name,
            state=state,
            methods=methods,
            dependencies=dependencies,
            line=line,
            # v1.5 fields
            base_classes=base_classes,
            generic_params=generic_params,
            is_abstract=is_abstract,
            is_protocol=is_protocol,
        )


def parse_file(file_path: str) -> PyShortAST:
    """Parse a PyShorthand file.

    Args:
        file_path: Path to .pys file

    Returns:
        Parsed AST

    Raises:
        FileNotFoundError: If file doesn't exist
        ParseError: If parsing fails
    """
    with open(file_path, encoding="utf-8") as f:
        source = f.read()

    return parse_string(source, file_path)


def parse_string(source: str, source_name: str = "<string>") -> PyShortAST:
    """Parse PyShorthand source code.

    Args:
        source: Source code string
        source_name: Name for error reporting

    Returns:
        Parsed AST
    """
    tokenizer = Tokenizer(source)
    tokens = tokenizer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()
    ast.source_file = source_name
    return ast
