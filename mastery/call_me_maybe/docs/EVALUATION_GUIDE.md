# 42 Peer Evaluation & Defense Guide

This guide prepares you for the peer defense of **Call Me Maybe**. It outlines the evaluation workflow, essential theoretical explanations, common peer questions, and strategies for the mandatory live code modification step.

---

## 1. Evaluation Overview & Checklist

Before starting the defense:
- [ ] Working inside the delivery directory `call_me_maybe/` (or your intra clone).
- [ ] No `data/output/` directory committed (must be generated during evaluation).
- [ ] Virtual environment initialized via `uv sync` (or `make install`).
- [ ] Code passes standard linting:
  ```bash
  make lint
  # and optionally:
  make lint-strict
  ```
- [ ] Run the test pipeline:
  ```bash
  make run
  ```
- [ ] Verify `data/output/function_calling_results.json`:
  - 100% valid JSON syntax.
  - Matches required schema exactly: `prompt`, `name`, `parameters`.
  - Argument types match function definitions (`number`, `string`, `boolean`).
  - Meets the $\ge 90\%$ accuracy benchmark.
  - Runs in under 5 minutes on standard CPU hardware.

---

## 2. Key Theoretical Questions & Winning Answers

### Q1: "What is Constrained Decoding, and why not just use prompting?"
> **Answer**: Prompting an LLM (especially small models like Qwen3-0.6B) is probabilistic and unstructured. Even with detailed instructions, 0.6B models produce valid JSON only ~30% of the time due to hallucinations, missing quotes, or extra chat preamble.
> Constrained Decoding guarantees 100% structural validity mathematically by intervening at the logit level: at each token generation step, any token that would violate JSON grammar or our function schema has its logit forced to $-\infty$. This reduces the probability of invalid tokens to zero after Softmax.

### Q2: "What is the difference between Speculative Decoding and Constrained Decoding?"
> **Answer**:
> - **Speculative Decoding** (Leviathan et al., 2023) uses two models (a fast draft model and a large target model) to speed up free-form text generation via parallel verification and rejection sampling.
> - **Constrained Decoding** (Willard & Louf, 2023) enforces formal grammar constraints on a single model via logit masking.
> - **Jump-Forward Decoding** (Zheng et al., 2023 - SGLang) is the bridge: where schema characters are fixed (like `"parameters": {"a": `), an automa "speculates" them with 100% certainty, skipping expensive forward passes altogether.

### Q3: "Why don't you compute Softmax when selecting the best token?"
> **Answer**: Softmax is strictly monotonically increasing ($a > b \implies e^{a} > e^{b}$). Therefore, the token with the highest probability among the unmasked candidates is identically the token with the highest raw logit. Taking `np.argmax(masked_logits)` gives the exact same result as Softmax argmax without the expensive exponential and division operations over 151,936 tokens.

### Q4: "How does Qwen's Byte-Level BPE tokenizer handle spaces and newlines?"
> **Answer**: Qwen uses Byte-Level BPE (Radford et al., 2019). Every byte (0–255) is mapped to a unique printable Unicode character. For example:
> - Space (`0x20`) is mapped to `Ġ` (`\u0120`, 288).
> - Newline (`\n`) is mapped to `Ċ` (`\u010a`, 266).
> To inspect or match token text against schema strings, we decode the BPE characters back into standard UTF-8 characters using a reverse mapping.

### Q5: "What is Token Boundary Misalignment?"
> **Answer**: BPE tokens rarely align neatly with JSON grammatical tokens. A single token in `vocab.json` can be `", "`, `42}`, or `hello"`. The constrained decoder must look ahead at the full accumulated string rather than assuming one token equals one JSON character or delimiter.

---

## 3. Preparation for Live Code Modification (Chapter VIII)

The subject explicitly notes:
> *"During the evaluation, a brief modification of the project may occasionally be requested. This could involve a minor behaviour change, a few lines of code to write or rewrite, or an easy-to-add feature... it should be feasible within a few minutes."*

### Typical Scenarios & How to Handle Them:

#### Scenario A: Support a New Argument Type (e.g., `boolean` or `integer`)
- **Where**: In `src/fsm.py` or `src/constrained_decoder.py`.
- **Preparation**: Ensure parameter decoding delegates to clean type-specific handlers (`decode_number`, `decode_string`, `decode_boolean`).
- **If asked for `integer`**: Restrict the number regex/FSM to disallow the decimal point (`.`), e.g., `^-?(0|[1-9]\d*)$`.

#### Scenario B: Change Output Formatting or Additional CLI Flags
- **Where**: In `src/cli.py` or `src/models.py`.
- **Preparation**: Use `argparse` with structured options and Pydantic models with `model_dump()`.
- **Example**: Formatting output with indentation (`indent=4` vs compact `indent=None`).

#### Scenario C: Add Verbose / Debug Output
- **Where**: In the main generation loop.
- **Example**: Printing which tokens were masked or how many forward passes were saved by Jump-Forward.
- **Preparation**: Add a `--verbose` flag that prints token selection logs to `sys.stderr`.

#### Scenario D: Add a New Function to `functions_definition.json`
- **Verification**: Run the pipeline with a modified `functions_definition.json` (e.g. `fn_multiply_numbers` or `fn_say_bye`).
- **Expected Outcome**: Trie automatically re-indexes; no code modifications needed! This proves your code does not hardcode function names.

