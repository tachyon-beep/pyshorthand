#!/usr/bin/env python3
"""
Diagnostic Questions for PyShorthand Ecosystem

These questions require understanding across multiple files and different
levels of detail to test how GPT-5.1 reasons about tool selection.
"""

from dataclasses import dataclass


@dataclass
class DiagnosticQuestion:
    """A diagnostic question to test ecosystem reasoning."""

    id: int
    question: str
    correct_answer: str
    why_hard: str
    expected_tools: list[str]  # What tools should ideally be called
    crosses_files: int  # How many files need to be understood
    needs_implementation: bool  # Does this need actual code?


DIAGNOSTIC_QUESTIONS = [
    # Cross-file architectural question
    DiagnosticQuestion(
        id=1,
        question="Trace the data flow: When GPT.forward() is called with input tokens, which classes are invoked in order until the final logits are produced?",
        correct_answer="GPT.forward calls transformer.wte (Embedding), transformer.wpe (Embedding), transformer.drop (Dropout), then each Block in transformer.h (which calls LayerNorm, CausalSelfAttention, LayerNorm, MLP), then transformer.ln_f (LayerNorm), finally lm_head (Linear)",
        why_hard="Requires understanding nested class structure and call sequence across GPT → Block → LayerNorm/CausalSelfAttention/MLP",
        expected_tools=[
            "get_class_details(GPT)",
            "get_class_details(Block)",
            "get_implementation(GPT.forward)",
        ],
        crosses_files=3,
        needs_implementation=True,
    ),
    # Pure structural question (should answer from PyShorthand)
    DiagnosticQuestion(
        id=2,
        question="Which classes in the codebase inherit from nn.Module, and what is the inheritance hierarchy depth for each?",
        correct_answer="LayerNorm (1 level), CausalSelfAttention (1 level), MLP (1 level), Block (1 level), GPT (1 level) - all directly inherit from nn.Module",
        why_hard="Requires understanding inheritance structure, but should be visible in PyShorthand",
        expected_tools=[],  # Should answer from PyShorthand alone
        crosses_files=5,
        needs_implementation=False,
    ),
    # Implementation detail requiring multiple files
    DiagnosticQuestion(
        id=3,
        question="When CausalSelfAttention computes attention, how does it handle the case where input sequence length exceeds the bias buffer size? Walk through the specific code logic.",
        correct_answer="Checks if input sequence length T is greater than self.bias.size(-1), and if so, calls self.register_buffer to create a new larger causal mask",
        why_hard="Requires reading actual CausalSelfAttention.forward() implementation to see buffer resizing logic",
        expected_tools=["get_implementation(CausalSelfAttention.forward)"],
        crosses_files=1,
        needs_implementation=True,
    ),
    # Cross-file type propagation
    DiagnosticQuestion(
        id=4,
        question="What is the exact tensor shape transformation from input idx to output logits in GPT.forward()? Specify shapes at each major step.",
        correct_answer="Input idx: (B, T), after wte: (B, T, n_embd=768), after transformer blocks: (B, T, 768), after lm_head: (B, T, vocab_size=50304)",
        why_hard="Requires understanding tensor shapes through multiple transformations and combining config values",
        expected_tools=[
            "get_class_details(GPT)",
            "get_class_details(GPTConfig)",
            "get_implementation(GPT.forward)",
        ],
        crosses_files=2,
        needs_implementation=True,
    ),
    # Nested composition question (structural but complex)
    DiagnosticQuestion(
        id=5,
        question="The GPT model uses a ModuleDict called 'transformer'. List all its sub-components and their types, including nested structures.",
        correct_answer="transformer is nn.ModuleDict containing: wte (Embedding), wpe (Embedding), drop (Dropout), h (ModuleList of Block), ln_f (LayerNorm)",
        why_hard="Requires understanding nested module composition, may need to expand transformer structure",
        expected_tools=["get_class_details(GPT, expand_nested=True)"],
        crosses_files=1,
        needs_implementation=False,
    ),
    # Cross-file dependency chain
    DiagnosticQuestion(
        id=6,
        question="If I wanted to modify the attention computation, which classes would I need to change, and in what order should I understand them?",
        correct_answer="CausalSelfAttention (main attention logic), Block (calls CausalSelfAttention), GPT (calls Block). Understanding order: CausalSelfAttention first, then Block, then GPT",
        why_hard="Requires understanding dependency chain and which class owns what responsibility",
        expected_tools=[
            "get_class_details(CausalSelfAttention)",
            "get_class_details(Block)",
            "get_class_details(GPT)",
            "search_usage(CausalSelfAttention)",
        ],
        crosses_files=3,
        needs_implementation=False,
    ),
    # Implementation + architecture
    DiagnosticQuestion(
        id=7,
        question="How does the Block class implement residual connections? Show the exact code pattern and explain why this pattern is used.",
        correct_answer="Uses x = x + self.attn(self.ln_1(x)) and x = x + self.mlp(self.ln_2(x)) pattern. This is residual connection where original x is added to transformed version, enabling gradient flow",
        why_hard="Requires seeing actual implementation to understand the x = x + pattern",
        expected_tools=["get_implementation(Block.forward)"],
        crosses_files=1,
        needs_implementation=True,
    ),
    # Configuration propagation
    DiagnosticQuestion(
        id=8,
        question="How does the n_head configuration parameter flow through the architecture? Which classes use it and how?",
        correct_answer="GPTConfig defines n_head, passed to CausalSelfAttention which uses it to split n_embd into n_head attention heads. Each head gets n_embd/n_head dimensions",
        why_hard="Requires tracing config parameter through initialization and understanding how it affects architecture",
        expected_tools=[
            "get_class_details(GPTConfig)",
            "get_class_details(CausalSelfAttention)",
            "get_implementation(CausalSelfAttention.__init__)",
        ],
        crosses_files=2,
        needs_implementation=True,
    ),
    # Pure implementation detective work
    DiagnosticQuestion(
        id=9,
        question="The code mentions 'flash attention'. What is the exact fallback mechanism if flash attention is not available, and what are the performance implications?",
        correct_answer="Checks if hasattr(torch.nn.functional, 'scaled_dot_product_attention'), if not available falls back to manual attention computation with matmul. Flash attention is faster and more memory efficient",
        why_hard="Requires reading actual implementation to see the conditional logic",
        expected_tools=["get_implementation(CausalSelfAttention.forward)"],
        crosses_files=1,
        needs_implementation=True,
    ),
    # Architectural big picture
    DiagnosticQuestion(
        id=10,
        question="Draw the complete class dependency graph: which classes depend on which, and what is the deepest dependency chain?",
        correct_answer="GPT depends on Block, LayerNorm, Embedding, Dropout, Linear, GPTConfig. Block depends on LayerNorm, CausalSelfAttention, MLP. CausalSelfAttention depends on Linear, Dropout. MLP depends on Linear, GELU, Dropout. Deepest chain: GPT → Block → CausalSelfAttention → Linear (3 levels)",
        why_hard="Requires understanding full architecture and dependency relationships across all classes",
        expected_tools=[
            "get_class_details(GPT)",
            "get_class_details(Block)",
            "get_class_details(CausalSelfAttention)",
            "get_class_details(MLP)",
        ],
        crosses_files=4,
        needs_implementation=False,
    ),
]


def load_diagnostic_suite() -> list[DiagnosticQuestion]:
    """Load the diagnostic question suite."""
    return DIAGNOSTIC_QUESTIONS
