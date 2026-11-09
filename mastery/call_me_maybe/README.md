# Call Me Maybe — Constrained Decoding & Function Calling on Small Language Models

![42 Mastery](https://img.shields.io/badge/42-Mastery_Curriculum-black?style=for-the-badge&logo=42)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Model](https://img.shields.io/badge/Model-Qwen3--0.6B-orange?style=for-the-badge)
![Status](https://img.shields.io/badge/Syntactic_Reliability-100%25-brightgreen?style=for-the-badge)

Welcome to the **Call Me Maybe** mastery repository, part of the 42 AI & Systems Engineering curriculum. This project demonstrates how to turn a sub-billion parameter Large Language Model (**Qwen/Qwen3-0.6B**) into an ultra-reliable, deterministic **Function Calling engine** capable of translating natural language queries into strictly typed JSON function calls.

---

## 💡 The Core Challenge: Small Models & Structured Output

Small Language Models (SLMs) such as Qwen3-0.6B (~596M parameters) are extremely lightweight and fast, but notoriously unreliable when asked to generate structured machine-readable formats. Through standard prompting alone:
- **Zero-Shot JSON Reliability**: Less than **30%** (missing delimiters, unescaped characters, hallucinated keys, conversational filler).
- **Production Standard**: Requires **100% syntactic JSON validity** and **$\ge$ 90% semantic argument accuracy**.

Rather than relying on oversized models or non-deterministic re-prompting loops, **Call Me Maybe** implements **Grammar-Constrained Decoding with Jump-Forward optimization** built directly on top of the model's raw logits.

---

## 🏛️ Two-Tier Repository Layout

This repository separates long-term engineering documentation from the official 42 submission package:

```
call_me_maybe/                               <-- GitHub Mastery Repository (Study & Documentation)
├── README.md                                <-- This document
├── AGENTS.md                                <-- AI agent instructions & coding constraints
├── docs/                                    <-- Technical documentation & guides
│   ├── ARCHITECTURE.md                      <-- Theory: Logits, BPE, Jump-Forward, FSM
│   ├── DIRECTORY_STRUCTURE.md               <-- Detailed layout & delivery mapping
│   └── EVALUATION_GUIDE.md                  <-- Peer evaluation defense & live-coding prep
├── resources/                               <-- Official subjects, PDFs, and archives
│   ├── en.subject.pdf                       <-- 42 Subject (v1.5)
│   ├── Report Tecnico e Teorico ... .pdf    <-- Comprehensive theoretical analysis
│   └── archives/                            <-- Original starter zips
│
└── call_me_maybe/                           <-- Official 42 Intra Submission Directory
    ├── Makefile                             <-- Automation rules (install, run, debug, clean, lint)
    ├── pyproject.toml                       <-- Dependency specifications (uv/pip)
    ├── uv.lock                              <-- Frozen lockfile
    ├── README.md                            <-- Mandatory 42 Intra submission README
    ├── data/input/                          <-- Evaluation input prompts & function signatures
    ├── llm_sdk/                             <-- Official 42 LLM wrapper package
    └── src/                                 <-- Core Python implementation
```

> **For 42 Submission**: The entire contents of `call_me_maybe/` are pushed to the 42 Intra (Vogsphere) repository.

---

## 🔬 Architectural & Theoretical Highlights

### 1. Logit Masking & Softmax Monotonicity
At each generation step $t$, the decoder inspects the model's unnormalized logits $z \in \mathbb{R}^{|V|}$. All tokens $v$ that would violate JSON grammar or our function schema are masked:
$$z_v = -\infty \quad \forall v \notin V_{valid}(x)$$
Because the exponential function is strictly monotonically increasing, the valid token with maximum probability is selected directly via:
$$x_{t+1} = \arg\max_{i \in V_{valid}} z_i$$
This eliminates full-vocabulary Softmax computation while guaranteeing 100% schema compliance.

### 2. Jump-Forward Decoding (SGLang Compressed FSM)
Querying a neural network to produce deterministic schema tokens (such as `{"name": "`, `", "parameters": {"`, `": `, or `}`) wastes CPU cycles. Our engine applies **Jump-Forward Decoding** (*Zheng et al., 2023*):
- Fixed schema delimiters are injected deterministically into the token context buffer.
- The neural network is queried **only at semantic branch points** (function selection and typed parameter values).
- This reduces forward passes by **70%–85%**, allowing the entire benchmark to execute in seconds on standard CPU hardware.

### 3. Trie-Based Function Selection with Early-Commit
Function names from `functions_definition.json` are indexed into a Prefix Trie. Logit masking ensures only valid function prefixes can be emitted. As soon as the accumulated prefix narrows to a single candidate, **Early-Commit** automatically completes the token sequence without extra LLM forward passes.

### 4. Type-Guided Parameter FSM (RFC 8259)
- **`number`**: Validated against RFC 8259 JSON numeric specifications (`^-?(0|[1-9]\d*)(\.\d+)?$`) over a pre-indexed numeric sub-vocabulary.
- **`string`**: Quotation-aware state machine handling UTF-8 characters and proper escape sequences (`\"`, `\\`).
- **`boolean`**: Binary decision between `true` and `false` literals.

---

## 🚀 Quick Start (Development & Evaluation)

### Setup via `uv`
```bash
cd call_me_maybe
uv sync
```

### Execution
```bash
# Run with default paths
uv run python -m src

# Run with custom input/output
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

### Code Quality & Validation
```bash
make lint
make lint-strict
```

---

## 📚 Deep Dive Documentation

- [Theory & Architecture](file:///home/deggio/Desktop/coding/42/42_github/mastery/call_me_maybe/docs/ARCHITECTURE.md)
- [Directory Structure & Delivery](file:///home/deggio/Desktop/coding/42/42_github/mastery/call_me_maybe/docs/DIRECTORY_STRUCTURE.md)
- [Evaluation & Peer Defense Guide](file:///home/deggio/Desktop/coding/42/42_github/mastery/call_me_maybe/docs/EVALUATION_GUIDE.md)
- [AI Assistants Development Guide](file:///home/deggio/Desktop/coding/42/42_github/mastery/call_me_maybe/AGENTS.md)

