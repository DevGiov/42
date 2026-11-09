# AGENTS.md — Development Guidelines for AI Assistants

This file establishes operating instructions, architectural boundaries, coding standards, and project constraints for any AI assistant (including Antigravity, Claude Code, Cursor, Copilot, etc.) working on the **42 Mastery: Call Me Maybe** repository.

---

## 1. Project Context & Repository Architecture

### 1.1 Dual Repository Structure
The workspace is split into two operational levels:
- **Outer Repository (`/`)**: Part of the personal 42 Mastery GitHub journey. Contains high-level documentation, theoretical analysis, academic references (`resources/`), and this `AGENTS.md`. These files are pushed to GitHub for portfolio and study purposes.
- **Inner Delivery Directory (`/call_me_maybe/`)**: The **exact folder** destined to be pushed to the official 42 Intra git repository (Vogsphere). It must strictly conform to the 42 subject rules (no extraneous documentation, output directories, or prohibited packages).

> **Crucial Rule:** When implementing project source code in subsequent turns, ALL functional source files must be created inside `call_me_maybe/src/`, and all delivery-related configuration files inside `call_me_maybe/`.

---

## 2. Strict Subject Constraints & Prohibitions

### 2.1 Permitted Packages
In the production source code (`call_me_maybe/src/`):
- **`pydantic` (v2)**: Mandatory for all data validation and class definitions.
- **`numpy`**: For vectorized logit masking, array operations, and argmax.
- **`json`**: Native Python JSON encoding/decoding.
- **`llm_sdk`**: Only the public methods of the `Small_LLM_Model` wrapper class.
- Standard Python libraries (e.g. `re`, `sys`, `argparse`, `pathlib`, `typing`).

### 2.2 Strictly Forbidden Packages
The subject explicitly states:
- **`torch` / `transformers` / `huggingface`**: CANNOT be imported or called directly inside `src/`. They are only utilized internally by `llm_sdk`.
- **`outlines`**, **`dspy`**, **`guidance`**, **`instructor`**, or any other high-level grammar/structured generation libraries are **COMPLETELY FORBIDDEN**.
- **No Private SDK Access**: Accessing private methods or attributes of `llm_sdk` (names starting with `_`, such as `_model` or `_tokenizer`) is strictly prohibited. Use only:
  - `model.get_logits_from_input_ids(input_ids: List[int]) -> List[float]`
  - `model.get_path_to_vocab_file() -> str`
  - `model.encode(text: str) -> Tensor`
  - `model.decode(token_ids: List[int]) -> str`

### 2.3 Prohibition of Heuristics ("Medieval Magic")
- Function selection and argument extraction **MUST be performed using LLM logits via Constrained Decoding**.
- Relying on regular expressions alone to guess the function name or parsing user input with pattern matching without consulting the LLM logits is strictly disallowed and considered a failure condition during peer evaluation.

---

## 3. Engineering & Code Quality Standards

### 3.1 Python Version & Typing
- **Python**: `>= 3.10`.
- **Static Typing**: Full type annotations on all function parameters, return types, and class attributes.
- **Mypy**: Must pass clean with flags:
  ```bash
  mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
  ```
  (Targeting `--strict` compatibility).
- **Flake8**: Must adhere strictly to PEP 8 standards with zero flake8 warnings/errors:
  ```bash
  flake8 .
  ```
- **Docstrings**: PEP 257 compliant docstrings (Google or NumPy style) on every public module, class, method, and function.

### 3.2 Robustness & Error Handling
- **No Unhandled Exceptions**: The CLI must NEVER crash unexpectedly.
- Malformed JSON input, missing files, corrupted schemas, or invalid parameters must be caught gracefully and reported with informative error messages to `sys.stderr` with a clean exit code.
- Always use context managers (`with open(...)`) for file handles and I/O.

---

## 4. Theoretical & Algorithmic Core

### 4.1 Target Model
- **Default**: `Qwen/Qwen3-0.6B` (~596M parameters).
- Because a 0.6B model fails at zero-shot JSON syntax ~70% of the time, our Constrained Decoding engine guarantees **100% syntactic validity** and **>90% semantic accuracy**.

### 4.2 Jump-Forward Constrained Decoding
To satisfy the sub-5-minute execution limit on CPU:
1. **Never query the LLM for deterministic characters**:
   - Fixed boilerplate like `{"prompt": "...", "name": "` and `, "parameters": {"` must be injected deterministically (Jump-Forward / Fast-Forward).
2. **Phase 1 — Function Name Selection**:
   - Build a Prefix Trie of valid function names from `functions_definition.json`.
   - Query LLM logits only at branching points.
   - **Trie Early-Commit**: As soon as the accumulated prefix matches a single candidate uniquely, immediately complete the token sequence without extra LLM forward passes.
3. **Phase 2 — Type-Guided Parameter FSM**:
   - For each parameter in the chosen function:
     - Inject parameter key deterministically: `"param_name": `.
     - Enforce type constraints token-by-token:
       - `number`: RFC 8259 regex/FSM (`-?(0|[1-9]\d*)(\.\d+)?`), pre-filtered against numeric token candidates.
       - `string`: Quotation-aware FSM allowing valid UTF-8 sequences and escaped quotes, terminating on closing `"`.
       - `boolean`: Binary logit comparison on tokens matching `true` vs `false`.
4. **Pre-computed Vocabulary Masks**:
   - Pre-filter `vocab.json` once during initialization into boolean lookup arrays or candidate token lists (e.g. numeric tokens, string tokens) to keep per-step masking $O(k)$ where $k \ll 151,936$.

---

## 5. Peer Evaluation Readiness (42 Defense)

During the peer evaluation at 42:
1. The student must explain the difference between:
   - **Grammar-Constrained Decoding** (Logit masking to $-\infty$, enforcing language $L$).
   - **Jump-Forward Decoding** (Compressed FSM skipping neural forward passes on deterministic strings).
2. The student may be asked to perform a **live code modification** within 5-15 minutes (e.g., adding a new type handler, supporting a new CLI flag, or modifying output formatting). Code must be clean, modular, and easy to modify.

---

## 6. Implementation Workflow for Future Turns

When instructed to begin coding:
1. Review `call_me_maybe/pyproject.toml` and lock dependencies.
2. Build modular components in `call_me_maybe/src/`:
   - `models.py`: Pydantic models for functions, inputs, outputs, CLI arguments.
   - `tokenizer.py`: Vocab loader, Byte-Level BPE mapper, and custom decoding helpers.
   - `fsm.py`: Finite state machines for JSON numbers, strings, booleans, and schema validation.
   - `constrained_decoder.py`: Logit masking engine, Trie for function names, Jump-Forward logic.
   - `generator.py`: High-level generation pipeline coordinating `llm_sdk` and the decoder.
   - `cli.py` / `__main__.py`: CLI argument parsing, input/output file handlers, and error reporting.
3. Create thorough test cases in `call_me_maybe/tests/` (unit tests and integration tests).
4. Verify execution speed, flake8 compliance, and mypy type checks.

