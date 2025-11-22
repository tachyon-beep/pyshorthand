# High Severity Fixes - Work Plan

**Total Issues**: 23 high-severity bugs to fix
**Priority**: Fix easiest and most impactful first

---

## BATCH 1: Quick Wins (Indexer - 4 issues)
**Estimated Time**: 1-2 hours
**Impact**: High

### I4: Incorrect Dependency Matching (Line 240)
- **Difficulty**: ⭐ Easy
- **Impact**: 🔴 High - False positives in dependency detection
- **Fix**: Change `startswith(imp)` to `== imp or startswith(imp + '.')`
- **Example**: `import py` currently matches `pyshort`, `python_utils`, `pyramid`

### I5: Empty Module Path Creates Malformed FQNs (Line 289)
- **Difficulty**: ⭐ Easy
- **Impact**: 🔴 High - Entities get FQN like `.EntityName`
- **Fix**: Check if module_path is empty before concatenating
- **Example**: `src/__init__.py` with class Foo → FQN becomes `.Foo` instead of `Foo`

### I6: Overly Broad Path Exclusion (Lines 72-74)
- **Difficulty**: ⭐⭐ Medium
- **Impact**: 🔴 High - Excludes unintended directories
- **Fix**: Match path components instead of substring in full path
- **Example**: Pattern `"test"` excludes `/home/latest/project/`

### I7: O(n×m) Performance Issue (Lines 237-241)
- **Difficulty**: ⭐⭐ Medium
- **Impact**: 🟡 Medium-High - Slow on large repos (20M iterations for 1000 modules)
- **Fix**: Use set-based lookup instead of nested loops
- **Example**: Current O(n×m), fix to O(n)

---

## BATCH 2: Decompiler Robustness (4 issues)
**Estimated Time**: 1-2 hours
**Impact**: High

### D3: Missing Exception Handling (Lines 58-62)
- **Difficulty**: ⭐ Easy
- **Impact**: 🔴 High - Crashes on malformed files
- **Fix**: Add try-except around ast.parse() and file I/O
- **Current**: Unhandled SyntaxError, IOError

### D4: Duplicate Detection Bug (Line 301)
- **Difficulty**: ⭐ Easy
- **Impact**: 🔴 High - False positives skip valid classes
- **Fix**: Use exact match instead of substring
- **Example**: Class "User" exists, "UserAdmin" incorrectly skipped

### D6: Missing File I/O Error Handling (Line 59)
- **Difficulty**: ⭐ Easy
- **Impact**: 🔴 High - No handling of missing files/permissions
- **Fix**: Add proper exception handling with clear error messages
- **Note**: May overlap with D3

### D5: Incomplete Optional Type Handling (Lines 431-436)
- **Difficulty**: ⭐⭐ Medium
- **Impact**: 🟡 Medium - Only handles `typing.Optional`, not `Optional` or `Union[X, None]`
- **Fix**: Handle all three Optional patterns

---

## BATCH 3: Parser Multi-Entity Support (1 issue)
**Estimated Time**: 1 hour
**Impact**: High

### P10: parse_entity() Doesn't Handle Multiple Entities
- **Difficulty**: ⭐⭐ Medium
- **Impact**: 🔴 High - Files with multiple classes only parse first one
- **Fix**: Loop to parse all entities in file instead of returning after first
- **Current**: Only first `[C:Name]` block parsed

---

## BATCH 4: Parser Validation & Robustness (12 issues)
**Estimated Time**: 3-4 hours
**Impact**: Medium-High

### P13: Ambiguous Grammar (Reference vs Array)
- **Difficulty**: ⭐⭐⭐ Hard
- **Impact**: 🟡 Medium - Can cause parsing confusion
- **Fix**: Better lookahead for `[Ref:Name]` vs `[N, C, H, W]`

### P14: No Validation of Identifier Names
- **Difficulty**: ⭐ Easy
- **Impact**: 🟡 Medium - Allows keywords as identifiers
- **Fix**: Check against reserved keywords list

### P15: Incorrect Handling of Nested Function Calls
- **Difficulty**: ⭐⭐ Medium
- **Impact**: 🟡 Medium - `foo(bar(x))` may not parse correctly
- **Fix**: Ensure recursive expression parsing works

### P16: Missing Support for Complex Type Unions
- **Difficulty**: ⭐⭐⭐ Hard
- **Impact**: 🟡 Medium - `Union[int, str]` not supported
- **Fix**: Add union type parsing to type specs

### P17: Incorrect Binding of Postfix Operators
- **Difficulty**: ⭐⭐ Medium
- **Impact**: 🟡 Medium - `a.b[c]()` may bind incorrectly
- **Fix**: Review postfix operator precedence

### P18: Incomplete Escape Sequence Validation
- **Difficulty**: ⭐ Easy
- **Impact**: 🟡 Medium - Unknown escapes not validated
- **Fix**: Warn on unknown escape sequences

### P19: No Handling of Unicode Identifiers
- **Difficulty**: ⭐⭐ Medium
- **Impact**: 🟢 Low-Medium - Non-ASCII identifiers fail
- **Fix**: Support Unicode in identifier parsing

### P20: Missing Validation for Circular References
- **Difficulty**: ⭐⭐⭐ Hard
- **Impact**: 🟡 Medium - `[C:A]` depends on `[C:B]` depends on `[C:A]`
- **Fix**: Build dependency graph and check for cycles

### P21: Incorrect Handling of Whitespace in Strings
- **Difficulty**: ⭐ Easy
- **Impact**: 🟡 Medium - String parsing may not preserve whitespace
- **Fix**: Review string tokenization

### P22: No Support for Multiline Strings
- **Difficulty**: ⭐⭐ Medium
- **Impact**: 🟡 Medium - Triple-quoted strings not supported
- **Fix**: Add multiline string tokenization

### P23: Missing Validation of Numeric Ranges
- **Difficulty**: ⭐ Easy
- **Impact**: 🟢 Low - Large numbers accepted without validation
- **Fix**: Add range checks for numeric literals

### P24: Incorrect Parsing of Chained Comparisons
- **Difficulty**: ⭐⭐ Medium
- **Impact**: 🟡 Medium - `a < b < c` not supported
- **Fix**: Add chained comparison support

### P25: No Validation of Method Signature Consistency
- **Difficulty**: ⭐⭐ Medium
- **Impact**: 🟡 Medium - Method signatures not validated against class
- **Fix**: Add semantic validation pass

---

## BATCH 5: Decompiler Enhancements (2 issues)
**Estimated Time**: 2-3 hours
**Impact**: Medium

### D7: Incorrect Method Signature Formatting
- **Difficulty**: ⭐⭐ Medium
- **Impact**: 🟡 Medium - Complex signatures formatted incorrectly
- **Fix**: Improve signature formatting logic

### D8: Framework Detection Too Aggressive
- **Difficulty**: ⭐⭐ Medium
- **Impact**: 🟡 Medium - False positives in framework detection
- **Fix**: More conservative pattern matching

---

## DEFERRED (Too Complex for Now)

### P11: Missing Error Recovery
- **Difficulty**: ⭐⭐⭐⭐⭐ Very Hard
- **Reason**: Requires major parser refactoring
- **Defer**: Future enhancement

### P12: No Source Location Tracking
- **Difficulty**: ⭐⭐⭐⭐ Hard
- **Reason**: Requires AST node changes throughout
- **Defer**: Future enhancement

---

## Summary

| Batch | Issues | Difficulty | Time | Priority |
|-------|--------|------------|------|----------|
| **1: Indexer** | 4 | Easy-Medium | 1-2h | ⭐⭐⭐⭐⭐ |
| **2: Decompiler** | 4 | Easy-Medium | 1-2h | ⭐⭐⭐⭐⭐ |
| **3: Parser Multi** | 1 | Medium | 1h | ⭐⭐⭐⭐ |
| **4: Parser Valid** | 12 | Mixed | 3-4h | ⭐⭐⭐ |
| **5: Decompiler Enh** | 2 | Medium | 2-3h | ⭐⭐ |
| **Deferred** | 2 | Very Hard | - | - |

**Total Fixes**: 21 issues (2 deferred)
**Estimated Total Time**: 8-14 hours
**Starting with**: Batch 1 (Indexer - quick wins)
