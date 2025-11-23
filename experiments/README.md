# PyShorthand A/B Testing Framework

**Real empirical validation** of PyShorthand's semantic preservation using LLM-based testing.

## What This Does

This framework conducts **actual A/B testing** by asking an LLM (Claude 3.5 Sonnet via OpenRouter) the same 20 questions about a codebase using two different formats:

1. **Original Python code** (331 lines)
2. **PyShorthand compressed code** (67 lines)

Then measures:
- ✅ Answer accuracy
- ⏱️ Response time
- 🎫 Token usage
- 💰 Cost efficiency
- 📊 Completeness scores

## Files

- **ab_test_framework.py** - Main test runner
- **analyze_results.py** - Results analysis and reporting
- **results/** - Output directory for test results

## Usage

### 1. Run the Experiment

```bash
# Load API key
export OPENROUTER_API_KEY=your_key_here

# Run tests (takes ~5-10 minutes for 40 API calls)
python experiments/ab_test_framework.py
```

This will:
- Ask 20 questions using original Python code
- Ask the same 20 questions using PyShorthand code
- Measure response time and token usage for each
- Save results to `experiments/results/ab_test_results_TIMESTAMP.json`

### 2. Analyze Results

```bash
python experiments/analyze_results.py experiments/results/ab_test_results_TIMESTAMP.json
```

This generates:
- Overall accuracy comparison
- Response time analysis
- Token efficiency metrics
- Question-by-question breakdown
- Areas for improvement

## Test Questions

20 questions across 4 difficulty levels:

**Easy (1-5)**: Structure and basic facts
- Class counts, decorators, configuration defaults

**Medium (6-10)**: Signatures and types
- Method parameters, return types, state variables

**Medium-Hard (11-15)**: Architecture
- Dependencies, relationships, nested structures

**Hard (16-20)**: Implementation details
- Complexity, optimizations, edge cases

## Expected Results

Based on PyShorthand's design:

✅ **Should Match Original**:
- Structural questions (100%)
- Signature questions (100%)
- Architectural questions (80-90%)

⚠️ **Expected Gaps**:
- Implementation details (20-30% - by design)
- Nested structures (may be incomplete)

## Real Benefits

This framework validates:
- 📉 Token reduction (80%+ expected)
- ⏱️ Response speed (context size matters)
- ✅ Semantic preservation (for architectural understanding)
- 💡 Areas to improve PyShorthand

## Cost Estimate

Using Claude 3.5 Sonnet via OpenRouter:
- 40 API calls (20 questions × 2 formats)
- ~$0.10-0.50 total cost
- Provides real empirical validation data
