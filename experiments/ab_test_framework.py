#!/usr/bin/env python3
"""
Real A/B Testing Framework for PyShorthand Validation

This script conducts empirical testing by asking an LLM the same questions
about a codebase using two different formats:
1. Original Python code
2. PyShorthand compressed code

Measures:
- Response time
- Token usage
- Answer accuracy
- Completeness
"""

import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import requests


@dataclass
class Question:
    """A single test question"""

    id: int
    difficulty: str  # easy, medium, medium-hard, hard
    category: str  # structure, signature, architecture, implementation
    question: str
    correct_answer: str
    why_important: str


@dataclass
class TestResult:
    """Result from testing with one code format"""

    question_id: int
    format: str  # "original" or "pyshorthand"
    answer: str
    response_time_ms: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    is_correct: bool
    completeness_score: float  # 0.0 to 1.0
    notes: str


class OpenRouterClient:
    """Client for OpenRouter API"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = "anthropic/claude-3.5-sonnet"  # Using Sonnet for quality

    def ask_question(self, system_prompt: str, question: str, code_context: str) -> dict[str, Any]:
        """
        Ask a question given code context

        Returns:
            dict with 'answer', 'response_time_ms', 'prompt_tokens', 'completion_tokens'
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        full_prompt = f"""You are a code comprehension expert. Answer the following question about the provided codebase.

CODE CONTEXT:
```
{code_context}
```

QUESTION:
{question}

Provide a clear, concise answer. If you cannot determine the answer from the provided code, state "Cannot determine from provided code."
"""

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt},
            ],
        }

        start_time = time.time()
        response = requests.post(self.base_url, headers=headers, json=payload)
        response_time_ms = int((time.time() - start_time) * 1000)

        if response.status_code != 200:
            raise Exception(f"API error: {response.status_code} - {response.text}")

        result = response.json()

        return {
            "answer": result["choices"][0]["message"]["content"],
            "response_time_ms": response_time_ms,
            "prompt_tokens": result["usage"]["prompt_tokens"],
            "completion_tokens": result["usage"]["completion_tokens"],
            "total_tokens": result["usage"]["total_tokens"],
        }


class ExperimentRunner:
    """Runs the A/B testing experiment"""

    def __init__(self, api_key: str):
        self.client = OpenRouterClient(api_key)
        self.system_prompt = "You are an expert at reading and understanding code. Answer questions accurately and concisely."

    def run_test(self, question: Question, code_context: str, format_name: str) -> TestResult:
        """Run a single test"""
        print(f"  Testing Q{question.id} with {format_name} format...")

        try:
            result = self.client.ask_question(self.system_prompt, question.question, code_context)

            # Simple correctness check (exact match or contains)
            is_correct = self._check_correctness(result["answer"], question.correct_answer)
            completeness = self._score_completeness(result["answer"], question.correct_answer)

            return TestResult(
                question_id=question.id,
                format=format_name,
                answer=result["answer"],
                response_time_ms=result["response_time_ms"],
                prompt_tokens=result["prompt_tokens"],
                completion_tokens=result["completion_tokens"],
                total_tokens=result["total_tokens"],
                is_correct=is_correct,
                completeness_score=completeness,
                notes="",
            )

        except Exception as e:
            return TestResult(
                question_id=question.id,
                format=format_name,
                answer=f"ERROR: {str(e)}",
                response_time_ms=0,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                is_correct=False,
                completeness_score=0.0,
                notes=f"Error: {str(e)}",
            )

    def _check_correctness(self, answer: str, correct: str) -> bool:
        """Check if answer is correct (simple string matching)"""
        answer_lower = answer.lower()
        correct_lower = correct.lower()

        # If correct answer is in the response, consider it correct
        if correct_lower in answer_lower:
            return True

        # Check for "cannot determine" responses
        if "cannot determine" in answer_lower or "not shown" in answer_lower:
            return "cannot" in correct_lower or "not" in correct_lower

        return False

    def _score_completeness(self, answer: str, correct: str) -> float:
        """Score how complete the answer is (0.0 to 1.0)"""
        if "cannot determine" in answer.lower() and "cannot" not in correct.lower():
            return 0.0

        if "cannot determine" in answer.lower() and "cannot" in correct.lower():
            return 1.0

        # Simple heuristic: count how many words from correct answer appear
        correct_words = set(correct.lower().split())
        answer_words = set(answer.lower().split())

        if not correct_words:
            return 1.0

        overlap = len(correct_words & answer_words) / len(correct_words)
        return min(1.0, overlap * 1.5)  # Boost slightly


def load_test_suite() -> list[Question]:
    """Load the test questions"""
    # These are the good questions from the synthetic exam
    questions = [
        # EASY QUESTIONS (1-5)
        Question(
            1,
            "easy",
            "structure",
            "How many classes are defined in the codebase?",
            "6",
            "Tests basic code navigation",
        ),
        Question(
            2,
            "easy",
            "structure",
            "Which class is decorated with @dataclass?",
            "GPTConfig",
            "Tests decorator identification",
        ),
        Question(
            3,
            "easy",
            "architecture",
            "What is the default value for block_size in the configuration?",
            "1024",
            "Tests reading configuration defaults",
        ),
        Question(
            4,
            "easy",
            "structure",
            "Which PyTorch module does the LayerNorm class inherit from?",
            "nn.Module",
            "Tests understanding class hierarchy",
        ),
        Question(
            5,
            "easy",
            "structure",
            "How many methods does the GPT class have (excluding __init__)?",
            "7",
            "Tests method counting",
        ),
        # MEDIUM QUESTIONS (6-10)
        Question(
            6,
            "medium",
            "signature",
            "What parameters does the generate() method accept (excluding self)?",
            "idx, max_new_tokens, temperature, top_k",
            "Tests reading method signatures",
        ),
        Question(
            7,
            "medium",
            "signature",
            "Which state variables in the MLP class are of type nn.Linear?",
            "c_fc, c_proj",
            "Tests identifying typed state variables",
        ),
        Question(
            8,
            "medium",
            "signature",
            "What does the forward() method in the GPT class return?",
            "tuple of (logits, loss)",
            "Tests understanding return types",
        ),
        Question(
            9,
            "medium",
            "signature",
            "Which method is decorated with @classmethod? Provide the method name.",
            "from_pretrained",
            "Tests identifying method decorators",
        ),
        Question(
            10,
            "medium",
            "structure",
            "What are the state variables of the Block class?",
            "ln_1, attn, ln_2, mlp",
            "Tests reading class structure",
        ),
        # MEDIUM-HARD QUESTIONS (11-15)
        Question(
            11,
            "medium-hard",
            "architecture",
            "What classes does the Block class depend on (use as state variables)?",
            "LayerNorm, CausalSelfAttention, MLP",
            "Tests understanding dependencies",
        ),
        Question(
            12,
            "medium-hard",
            "signature",
            "The CausalSelfAttention class contains c_attn and c_proj. What types are they?",
            "nn.Linear",
            "Tests understanding component types",
        ),
        Question(
            13,
            "medium-hard",
            "architecture",
            "List all configuration parameters in GPTConfig with their default values.",
            "block_size=1024, vocab_size=50304, n_layer=12, n_head=12, n_embd=768, dropout=0.0, bias=True",
            "Tests complete configuration understanding",
        ),
        Question(
            14,
            "medium-hard",
            "implementation",
            "Which method in GPT class handles weight initialization, and what special treatment is given?",
            "_init_weights method, special scaled init to residual projections (c_proj.weight)",
            "Tests understanding initialization patterns",
        ),
        Question(
            15,
            "medium-hard",
            "architecture",
            "What type is the transformer state variable in GPT class, and what sub-components does it contain?",
            "nn.ModuleDict containing wte, wpe, drop, h, ln_f",
            "Tests understanding nested architecture",
        ),
        # HARD QUESTIONS (16-20)
        Question(
            16,
            "hard",
            "implementation",
            "What is the computational complexity of the generate() method?",
            "O(N)",
            "Tests complexity analysis",
        ),
        Question(
            17,
            "hard",
            "implementation",
            "What does the assertion in from_pretrained() check?",
            "model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'} and only dropout can be overridden",
            "Tests understanding validation logic",
        ),
        Question(
            18,
            "hard",
            "implementation",
            "What optimization is used in the forward() method during inference?",
            "only forward lm_head on the very last position using x[:, [-1], :]",
            "Tests understanding performance optimizations",
        ),
        Question(
            19,
            "hard",
            "implementation",
            "What happens if flash attention is NOT available in CausalSelfAttention?",
            "registers a causal mask buffer (torch.tril), uses manual attention implementation",
            "Tests understanding conditional paths",
        ),
        Question(
            20,
            "hard",
            "implementation",
            "In configure_optimizers(), how are parameters divided into groups and why?",
            "p.dim() >= 2 get weight decay (weight tensors), p.dim() < 2 no decay (biases/norms)",
            "Tests understanding optimization strategy",
        ),
    ]

    return questions


def save_results(results: list[TestResult], output_file: str):
    """Save results to JSON file"""
    with open(output_file, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\n✅ Results saved to {output_file}")


def run_experiment(original_code_path: str, pyshorthand_code_path: str, output_dir: str):
    """Run the full A/B test experiment"""

    # Load API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment")

    # Load test suite
    questions = load_test_suite()
    print(f"\n📋 Loaded {len(questions)} test questions")

    # Load code contexts
    with open(original_code_path) as f:
        original_code = f.read()

    with open(pyshorthand_code_path) as f:
        pyshorthand_code = f.read()

    print(f"📄 Original code: {len(original_code)} chars")
    print(f"📄 PyShorthand code: {len(pyshorthand_code)} chars")
    print(f"📉 Size reduction: {100 * (1 - len(pyshorthand_code) / len(original_code)):.1f}%")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Run experiments
    runner = ExperimentRunner(api_key)
    all_results = []

    print("\n" + "=" * 80)
    print("RUNNING EXPERIMENTS")
    print("=" * 80)

    for question in questions:
        print(f"\n📝 Question {question.id} ({question.difficulty} - {question.category})")
        print(f"   {question.question}")

        # Test with original code
        result_original = runner.run_test(question, original_code, "original")
        all_results.append(result_original)
        print(
            f"   ✓ Original: {result_original.response_time_ms}ms, "
            f"{result_original.total_tokens} tokens, "
            f"{'✅ Correct' if result_original.is_correct else '❌ Incorrect'}"
        )

        # Small delay to avoid rate limiting
        time.sleep(1)

        # Test with PyShorthand code
        result_pyshort = runner.run_test(question, pyshorthand_code, "pyshorthand")
        all_results.append(result_pyshort)
        print(
            f"   ✓ PyShort:  {result_pyshort.response_time_ms}ms, "
            f"{result_pyshort.total_tokens} tokens, "
            f"{'✅ Correct' if result_pyshort.is_correct else '❌ Incorrect'}"
        )

        # Small delay
        time.sleep(1)

    # Save results
    results_file = f"{output_dir}/ab_test_results_{timestamp}.json"
    save_results(all_results, results_file)

    return all_results, results_file


if __name__ == "__main__":
    # Run the experiment
    original_path = "test_repos/nanoGPT/model.py"
    pyshorthand_path = "realworld_nanogpt.pys"
    output_dir = "experiments/results"

    print("\n" + "=" * 80)
    print("PYSHORTHAND A/B TESTING FRAMEWORK")
    print("=" * 80)

    results, results_file = run_experiment(original_path, pyshorthand_path, output_dir)

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {results_file}")
    print("\nRun analyze_results.py to generate analysis report.")
