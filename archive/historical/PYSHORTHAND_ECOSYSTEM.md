# PyShorthand Ecosystem Design

## Overview

A two-tier progressive disclosure system for code understanding:
- **Tier 1:** PyShorthand overview (cheap, always provided)
- **Tier 2:** On-demand implementation details (pay-per-use)

## Architecture

### Tier 1: PyShorthand Overview (Default Context)

**Format:** Standard PyShorthand v1.5
**Token Cost:** ~900 tokens (83% reduction)
**Answers:** Structural questions (100% accuracy on 5/20 questions)

```pyshorthand
[C:GPT] ◊ nn.Module
  config ∈ GPTConfig
  transformer ∈ ModuleDict { wte, wpe, drop, h }
  lm_head ∈ Linear

  F:forward(idx, targets) → (Tensor, Tensor?)
  F:generate(idx, max_new_tokens, temperature, top_k) → Tensor
  F:configure_optimizers(weight_decay, lr, betas) → Optimizer
  F:_init_weights(module) → None
```

### Tier 2: On-Demand Tools

#### Tool 1: `get_implementation()`

**Purpose:** Retrieve full Python implementation of a specific method

**Signature:**
```python
def get_implementation(
    target: str,  # Format: "ClassName.method_name"
    include_context: bool = True,  # Include related helper methods?
) -> str:
    """Returns the full Python implementation."""
```

**Example Usage:**
```python
# Agent wants to know what configure_optimizers() does
result = get_implementation("GPT.configure_optimizers")

# Returns:
"""
def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in self.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    ...
"""
```

**Token Cost:** Variable (~100-500 tokens depending on method complexity)

#### Tool 2: `get_class_details()`

**Purpose:** Retrieve detailed type information for a class

**Signature:**
```python
def get_class_details(
    class_name: str,
    include_methods: bool = False,  # Include method implementations?
    expand_nested: bool = True,     # Expand nested structures?
) -> str:
    """Returns detailed class information."""
```

**Example Usage:**
```python
# Agent wants to know transformer's exact structure
result = get_class_details("GPT", expand_nested=True)

# Returns:
"""
class GPT(nn.Module):
    # State Variables (with exact types):
    config: GPTConfig
    transformer: nn.ModuleDict = {
        'wte': nn.Embedding(vocab_size=50304, embedding_dim=768),
        'wpe': nn.Embedding(num_embeddings=1024, embedding_dim=768),
        'drop': nn.Dropout(p=0.0),
        'h': nn.ModuleList[Block](length=12),
        'ln_f': LayerNorm(768)
    }
    lm_head: nn.Linear(in_features=768, out_features=50304, bias=False)

    # Methods:
    forward(idx: Tensor, targets: Optional[Tensor]=None) -> Tuple[Tensor, Optional[Tensor]]
    generate(idx: Tensor, max_new_tokens: int, temperature: float=1.0, top_k: Optional[int]=None) -> Tensor
    ...
"""
```

**Token Cost:** ~200-400 tokens (much cheaper than full implementation)

#### Tool 3: `search_usage()`

**Purpose:** Find where a class/method is used

**Signature:**
```python
def search_usage(
    symbol: str,  # Class name, method name, or state variable
) -> List[str]:
    """Returns list of locations where symbol is used."""
```

**Example Usage:**
```python
# Agent wants to know what uses LayerNorm
result = search_usage("LayerNorm")

# Returns:
"""
LayerNorm is used in:
- Block.ln_1 (state variable)
- Block.ln_2 (state variable)
- GPT.transformer.ln_f (nested in ModuleDict)
"""
```

**Token Cost:** ~50-100 tokens

## Agent Workflow

### Example: Answering Q20 (Implementation Question)

**Q20:** "In configure_optimizers(), how are parameters divided into groups and why?"

**Agent reasoning:**
1. See PyShorthand: `F:configure_optimizers(weight_decay, lr, betas) → Optimizer`
2. Realize this is an implementation question
3. Call: `get_implementation("GPT.configure_optimizers")`
4. Read actual code
5. Answer: "Parameters are divided by dimensionality: 2D+ tensors (weights) get decay, 1D tensors (biases/norms) don't"

**Total tokens:** 894 (PyShorthand) + ~300 (one method) = **1,194 tokens** (78% savings vs 5,348!)

### Example: Answering Q15 (Nested Structure Question)

**Q15:** "What type is transformer and what sub-components does it contain?"

**Agent reasoning:**
1. See PyShorthand: `transformer ∈ ModuleDict { wte, wpe, drop, h }`
2. Need exact types
3. Call: `get_class_details("GPT", expand_nested=True)`
4. See full structure with types
5. Answer: "ModuleDict containing wte (Embedding), wpe (Embedding), drop (Dropout), h (ModuleList<Block>), ln_f (LayerNorm)"

**Total tokens:** 894 (PyShorthand) + ~250 (class details) = **1,144 tokens** (79% savings!)

## Expected Performance

### Token Usage Across Question Types

| Question Category | PyShorthand Only | + Tools Called | Full Code | Savings |
|-------------------|------------------|----------------|-----------|---------|
| **Easy (5)** | 894 | 894 | 5,348 | 83% |
| **Medium (5)** | 894 | 894 + 250 = 1,144 | 5,348 | 79% |
| **Med-Hard (5)** | 894 | 894 + 400 = 1,294 | 5,348 | 76% |
| **Hard (5)** | 894 | 894 + 800 = 1,694 | 5,348 | 68% |
| **WEIGHTED AVG** | 894 | **~1,400** | 5,348 | **74%** |

### Accuracy Projection

- **Easy (structural):** 5/5 = 100% ✅ (PyShorthand alone)
- **Medium (signatures):** 4/5 = 80% ✅ (PyShorthand + get_class_details)
- **Med-Hard (arch + impl):** 3/5 = 60% ✅ (PyShorthand + selective get_implementation)
- **Hard (implementation):** 4/5 = 80% ✅ (PyShorthand + get_implementation)

**Total: 16/20 = 80% accuracy** (up from 35%) with **74% token savings**!

## Implementation Strategy

### Phase 1: Core Tools (MVP)

1. **get_implementation()** - Most important, handles all implementation questions
2. **get_class_details()** - Second most important, handles signature/structure questions
3. **Basic caching** - Don't re-fetch the same methods

**Deliverable:** Working prototype with 2 tools

### Phase 2: Smart Agent Prompting

1. **System prompt addition:**
```
You have access to PyShorthand (architectural overview) and detailed code lookup tools.

STRATEGY:
1. Start with PyShorthand - it's free and fast
2. For architectural questions → answer from PyShorthand alone
3. For signature questions → use get_class_details()
4. For implementation questions → use get_implementation()
5. Be selective - each tool call costs tokens
```

2. **Few-shot examples** of good tool usage

**Deliverable:** Agent that uses tools intelligently

### Phase 3: Optimization

1. **Smart caching:** If agent requests GPT.forward, preload GPT._init_weights (often related)
2. **Batch lookups:** `get_implementations(["GPT.forward", "GPT.generate"])`
3. **Cost tracking:** Show agent its token budget and usage

**Deliverable:** Production-ready ecosystem

## Cost-Benefit Analysis

### Scenario: 1000 Question Workload (mixed difficulty)

**Full Code Approach:**
- Tokens: 5,348 × 1000 = 5,348,000 input tokens
- Cost: $18.25 (Sonnet 4.5 pricing)
- Accuracy: 35% (7/20)

**PyShorthand v1.5 Alone:**
- Tokens: 894 × 1000 = 894,000 input tokens
- Cost: $3.93
- Accuracy: 35% (7/20)
- Savings: 78%

**PyShorthand Ecosystem:**
- Tokens: ~1,400 × 1000 = 1,400,000 input tokens (PyShorthand + selective tool calls)
- Cost: $6.16
- Accuracy: **80%** (16/20)
- Savings: 66%

**ROI:**
- **vs Full Code:** Same accuracy (80% if agent reads carefully), 66% cost savings
- **vs PyShorthand Alone:** +45% accuracy (+9 questions), +57% cost (+$2.23)
- **Cost per additional correct answer:** $2.23 / 9 = **$0.25 per question** 🎯

## Integration Points

### MCP Server Implementation

```python
# PyShorthand MCP Server
class PyShorthandServer:
    def __init__(self, codebase_path: str):
        self.codebase = load_codebase(codebase_path)
        self.pyshorthand = compile_to_pyshorthand(self.codebase)
        self.cache = {}

    @tool
    def get_overview(self) -> str:
        """Get PyShorthand overview (always cached)"""
        return self.pyshorthand

    @tool
    def get_implementation(self, target: str) -> str:
        """Get full implementation of specific method"""
        if target in self.cache:
            return self.cache[target]

        impl = extract_implementation(self.codebase, target)
        self.cache[target] = impl
        return impl

    @tool
    def get_class_details(self, class_name: str, expand_nested: bool = True) -> str:
        """Get detailed class information with types"""
        return extract_class_details(self.codebase, class_name, expand_nested)
```

### Claude Desktop Integration

```json
{
  "mcpServers": {
    "pyshorthand": {
      "command": "python",
      "args": ["-m", "pyshorthand.mcp_server"],
      "env": {
        "CODEBASE_PATH": "/path/to/project"
      }
    }
  }
}
```

Agent automatically gets:
- `pyshorthand_get_overview` tool
- `pyshorthand_get_implementation` tool
- `pyshorthand_get_class_details` tool

## Success Metrics

### Alpha Test (Current nanoGPT dataset)

**Target:**
- ✅ 80% accuracy (16/20 questions)
- ✅ <2,000 tokens average per question
- ✅ Agent learns to be selective (doesn't call get_implementation for structural questions)

### Beta Test (Larger codebase)

**Target:**
- Test on larger codebase (e.g., transformers library, 100K+ LOC)
- Measure token savings at scale
- Validate caching effectiveness

### Production Metrics

**Track:**
- Tool call frequency by question type
- Token savings vs full code baseline
- Accuracy vs full code baseline
- Agent learning: does it get better at selecting which tools to use?

## Next Steps

1. ✅ Design complete (this document)
2. ⏭️ Implement core tools (get_implementation, get_class_details)
3. ⏭️ Create test harness (run 20Q test with ecosystem)
4. ⏭️ Measure empirical performance
5. ⏭️ Iterate based on results

## Open Questions

1. **Caching strategy:** Should we preload related methods? (e.g., if agent requests forward(), preload __init__?)
2. **Token budget:** Should agent have explicit token budget to encourage selective tool use?
3. **Hybrid format:** Should get_class_details() return PyShorthand or Python signatures?
4. **Cross-references:** Should get_implementation() auto-include called helper methods?

---

**Status:** Design Phase Complete
**Next:** Prototype Implementation
