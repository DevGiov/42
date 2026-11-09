# Architecture & Theoretical Foundations

This document provides a deep dive into the engineering principles, mathematical formulations, and algorithmic architectures underlying the **Call Me Maybe** project.

---

## 1. Problem Statement: Small Language Models & Structured Output

Large Language Models (LLMs) are autoregressive probabilistic sequence generators trained on natural language. However, modern automated agents and API-driven systems require **deterministic, machine-parseable structured data** (such as JSON) conforming to strict schema definitions.

When small models—such as `Qwen/Qwen3-0.6B` (~596 million parameters)—are instructed via standard zero-shot or few-shot prompting to return valid JSON, empirical reliability is under **30%**. Common failure modes include:
- Extraneous conversational preamble or reasoning tokens (e.g., `<think>...</think>`, `"Sure, here is your JSON:"`).
- Malformed JSON syntax (unclosed braces, trailing commas, single quotes instead of double quotes).
- Schema violations (hallucinated parameter keys, incompatible data types, missing required arguments).

Rather than increasing model size or relying on stochastic re-prompting, **Constrained Decoding** transforms a 0.6B model into a 100% syntactically reliable function calling engine by intervening directly at the logit level during autoregressive decoding.

---

## 2. Autoregressive Decoding & Logits Mechanics

### 2.1 From Token IDs to Probability Distributions
Given an input prompt sequence of $t$ token IDs:
$$x_{1:t} = (x_1, x_2, \dots, x_t), \quad x_i \in \{0, 1, \dots, |V|-1\}$$
where $V$ is the model vocabulary ($|V| \approx 151,936$ for Qwen3), the Transformer network processes the hidden state through stacked Grouped-Query Attention (GQA) and SwiGLU layers to yield a final hidden representation $h_t \in \mathbb{R}^{d_{model}}$.

Projecting $h_t$ with the language modeling head $W_{LM} \in \mathbb{R}^{|V| \times d_{model}}$ produces the raw **unnormalized logit vector**:
$$z = W_{LM} \cdot h_t \in \mathbb{R}^{|V|}$$

In standard unconstrained generation, logit vector $z$ is passed through a Softmax function with temperature $T > 0$ to obtain next-token probabilities:
$$P(x_{t+1} = i \mid x_{1:t}) = \frac{\exp(z_i / T)}{\sum_{j=1}^{|V|} \exp(z_j / T)}$$

### 2.2 Monotonicity of Exponential & Greedy Selection
In function calling where deterministic reproducibility is critical, we use **Greedy Decoding** ($T \to 0^+$). Since the exponential function $f(u) = \exp(u)$ is strictly monotonically increasing:
$$\arg\max_{i \in V_{valid}} \frac{\exp(z_i / T)}{\sum_{j} \exp(z_j / T)} = \arg\max_{i \in V_{valid}} z_i$$
This crucial mathematical property means we **never need to compute the expensive full-vocabulary Softmax normalization**. We simply mask invalid logits and take `np.argmax(masked_logits)`.

---

## 3. Theoretical Classification: Constrained vs. Jump-Forward Decoding

A critical topic in the 42 peer evaluation is the precise distinction between:
1. **Grammar-Constrained Decoding**
2. **Jump-Forward (Compressed FSM) Decoding**

| Dimension | Constrained Decoding *(Willard & Louf, 2023)* | Jump-Forward Constrained *(Zheng et al., 2023 - SGLang)* |
| :--- | :--- | :--- |
| **Primary Goal** | Guarantee 100% adherence to grammar/schema. | Unify 100% validity with extreme execution speed. |
| **Model Requirements** | Single model + Formal Grammar / FSM. | Single model + Compressed Finite State Machine. |
| **Mechanism** | Masks invalid next-token logits ($z_v = -\infty$) based on parser state. | Injects static schema tokens deterministically; queries LLM **only** at semantic branch points. |
| **Applicability to Call Me Maybe** | **Mandatory**: Required to select function name and extract typed argument values. | **Highly recommended**: Cuts forward passes by 70–85%, ensuring completion in seconds (< 5 min limit). |

Jump-Forward is not an alternative to Constrained Decoding but an optimization of it: the same finite state machine that masks logits also detects states with a single valid continuation and fast-forwards through them.

---

## 4. Byte-Level BPE & Token Boundary Misalignment

### 4.1 Byte-Level BPE (BBPE) Encoding
Qwen3 employs Byte-Level Byte-Pair Encoding (Radford et al., 2019):
- Any arbitrary UTF-8 text is broken into individual bytes ($0 \dots 255$).
- To avoid non-printable control characters in `vocab.json`, each byte is bijectively mapped to a printable Unicode character:
  - Space (`0x20` / `' '`) $\to$ `Ġ` (`\u0120`, code point 288).
  - Newline (`0x0A` / `'\n'`) $\to$ `Ċ` (`\u010a`, code point 266).
  - Tab (`0x09` / `'\t'`) $\to$ `ĉ` (`\u0109`, code point 265).
  - Visible ASCII printable characters remain identical.

### 4.2 The Token Boundary Misalignment Problem
As highlighted in research papers on structured generation (e.g. *XGrammar*, Dong et al., 2024; *Outlines*, Willard & Louf, 2023):
1. **Boundary-Spanning Tokens**: Tokens like `",` , `":` , `"}` , or `42}` span across lexical boundaries (data literal + syntax delimiter).
2. **Sub-token Fragments**: A function name like `fn_add_numbers` does not exist as a single token; the tokenizer splits it into `["fn", "_add", "_numbers"]`.

A naive token mask that only looks at single characters will fail. The Constrained Decoder must operate on **string prefixes** and map them back to the active grammar state.

---

## 5. Algorithmic Architecture for "Call Me Maybe"

Instead of implementing a heavy generic Context-Free Grammar (CFG) pushdown automaton, our architecture leverages the structural decomposition of the known schema (**Divide and Conquer**):

```
                   Input Prompt: "What is the sum of 2 and 3?"
                                       │
                                       ▼
        ┌─────────────────────────────────────────────────────────────┐
        │  Static Prefix Jump-Forward:                                │
        │  Inject prompt context + '{"name": "'                       │
        │  (0 LLM calls)                                              │
        └──────────────────────────────┬──────────────────────────────┘
                                       │
                                       ▼
        ┌─────────────────────────────────────────────────────────────┐
        │  Phase 1: Trie-Guided Function Name Selection               │
        │  • Logit masking over valid candidate function prefixes     │
        │  • Early-Commit Jump-Forward once prefix is unique          │
        │  (1 - 2 LLM calls)                                          │
        └──────────────────────────────┬──────────────────────────────┘
                                       │
                                       ▼
        ┌─────────────────────────────────────────────────────────────┐
        │  Delimiter Jump-Forward:                                    │
        │  Inject '", "parameters": {'                                │
        │  (0 LLM calls)                                              │
        └──────────────────────────────┬──────────────────────────────┘
                                       │
                                       ▼
        ┌─────────────────────────────────────────────────────────────┐
        │  Phase 2: Type-Guided Parameter FSM                         │
        │  For each parameter (e.g. "a: number", "b: number"):        │
        │    • Inject '"param_name": '                                │
        │    • Decode value with Type Mask (RFC 8259 Number / String) │
        │    • Inject delimiter (',' or '}')                          │
        │  (1 - 3 LLM calls per parameter)                            │
        └──────────────────────────────┬──────────────────────────────┘
                                       │
                                       ▼
        ┌─────────────────────────────────────────────────────────────┐
        │  Termination Jump-Forward:                                  │
        │  Inject '}' and finalize JSON output object                 │
        │  (0 LLM calls)                                              │
        └─────────────────────────────────────────────────────────────┘
```

### 5.1 Phase 1: Function Selection via Trie & Early-Commit
- Given the set of function names $F = \{f_1, f_2, \dots, f_k\}$ from `functions_definition.json`, we construct a Prefix Tree (Trie).
- At each step $s$, the set of compatible functions is:
  $$F_{active} = \{ f \in F \mid f \text{ starts with } \text{accumulated\_string} \}$$
- A token $v \in V$ is valid if and only if $\text{accumulated\_string} + \text{str}(v)$ is a prefix of at least one $f + \text{'"'}$.
- **Trie Early-Commit Optimization**: If $|F_{active}| = 1$, we do not query the LLM for remaining sub-tokens. We directly fast-forward the remaining characters and the closing quotation mark `"`.

### 5.2 Phase 2: Schema-Guided Parameter Decoding (FSM)
For each parameter defined in the selected function's schema:
1. **Number Decoding (`type: "number"`)**:
   - RFC 8259 specifies valid JSON numbers: `^-?(0|[1-9]\d*)(\.\d+)?$`.
   - We pre-index numeric candidates: tokens consisting only of digits `0-9`, negative sign `-`, decimal point `.`, and delimiters `,` or `}`.
   - At each decoding step, we test if appending candidate token $v$ yields a valid number or a valid partial prefix of a number.
2. **String Decoding (`type: "string"`)**:
   - Deterministically open with `"`.
   - Allow UTF-8 printable tokens and proper escape sequences (`\"`, `\\`).
   - Forbid unescaped raw newlines or control characters.
   - When the closing quote `"` is reached (or followed by the argument delimiter), parameter generation finishes.
3. **Boolean Decoding (`type: "boolean"`)**:
   - Valid completions are restricted strictly to `true` and `false`.

---

## 6. Prompt Engineering for Small Models (0.6B)

While Constrained Decoding provides a mathematical guarantee of valid JSON structure, the **semantic correctness** (choosing the right function and extracting the right numbers) depends on prompt clarity and attention concentration:

1. **Short Attention Span**: Instead of dumping verbose, raw JSON definitions into the context, we serialize available functions in a clean, single-line format:
   ```text
   Available functions:
   - fn_add_numbers(a: number, b: number): Add two numbers together and return their sum.
   - fn_greet(name: string): Generate a greeting message for a person by name.
   - fn_reverse_string(s: string): Reverse a string and return the reversed result.
   ```
2. **Induction Head Placement**: In sub-billion parameter transformers, induction heads (which copy literal tokens from context to output) work most reliably when the target context is close to the generation point. We place `User prompt: "..."` immediately prior to the JSON prefix `{"name": "`.
3. **Reasoning Mode Suppression**: Qwen3 includes internal reasoning `<think>...</think>`. By forcing the generation prefix to `{"name": "`, the model immediately bypasses reasoning tokens and generates the structured target.

---

## 7. Performance & Optimization Architecture

The subject mandates that the entire test suite run in **under 5 minutes** on standard CPU hardware.

1. **Singleton Initialization**: Model weights and `vocab.json` are loaded and parsed exactly once at startup.
2. **Pre-computed Boolean Masks**: Static token filters (numeric tokens, boolean tokens, delimiter tokens) are computed once at startup as NumPy boolean vectors or integer index arrays, replacing $O(|V|)$ scans with $O(k)$ operations.
3. **Jump-Forward Speedup**: By skipping forward passes on all static schema literals (`{"name": "`, `", "parameters": {"`, `": `, `,`, `}`), total neural evaluations drop from ~35 per query down to ~4.

