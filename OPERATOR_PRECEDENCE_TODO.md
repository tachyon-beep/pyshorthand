# Operator Precedence Fix - TODO

## Issue (P9 from Code Review)

The current `parse_binary_expr()` function does not implement operator precedence correctly.

**Current Behavior:**
- Expression: `a + b * c`
- Parsed as: `(a + b) * c`  ← WRONG!
- Should be: `a + (b * c)`

**Root Cause:**
The parser uses simple left-to-right association without considering operator precedence.

## Solution Required

Implement precedence climbing algorithm in `parse_binary_expr()`:

```python
# Operator precedence table (higher number = higher precedence)
PRECEDENCE = {
    TokenType.PLUS: 10,
    TokenType.MINUS: 10,
    TokenType.STAR: 20,
    TokenType.SLASH: 20,
    TokenType.AT: 20,  # Matrix multiplication
    TokenType.CARET: 30,  # Exponentiation
    # Comparison operators
    TokenType.GT: 5,
    TokenType.LT: 5,
    TokenType.GTE: 5,
    TokenType.LTE: 5,
    TokenType.NE: 5,
}

def parse_binary_expr(self, min_precedence: int = 0) -> Expression:
    """Parse binary expression with correct operator precedence."""
    left = self.parse_primary_expr()

    while self.current_token.type in PRECEDENCE:
        if PRECEDENCE[self.current_token.type] < min_precedence:
            break

        op = self.current_token.value
        op_precedence = PRECEDENCE[self.current_token.type]
        self.advance()

        # Right-associative for exponentiation, left-associative otherwise
        next_min_precedence = op_precedence + 1 if op != '^' else op_precedence
        right = self.parse_binary_expr(next_min_precedence)

        left = BinaryOp(operator=op, left=left, right=right)

    return left
```

## Impact

**Severity:** Medium (marked as critical in review, but PyShorthand is mainly for documentation, not computation)

**Priority:** Fix after all critical production blockers are resolved.

**Workaround:** For complex expressions, use parentheses to explicitly specify order of operations.

## Status

- [x] Issue identified
- [x] Solution designed
- [ ] Implementation pending
- [ ] Tests added
- [ ] Verified
