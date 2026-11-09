# Project Directory Structure & Workflow

This document details the architectural layout of the **Call Me Maybe** mastery repository, explaining the separation between the GitHub tracking repository and the 42 Intra delivery directory.

---

## 1. High-Level Workspace Architecture

```
call_me_maybe/                               <-- Outer Repository Root (GitHub Portfolio)
├── AGENTS.md                                <-- Operating instructions & constraints for AI agents
├── README.md                                <-- Overview of the project, theory, and architecture
├── docs/                                    <-- In-depth technical & evaluation documentation
│   ├── ARCHITECTURE.md                      <-- Theory: Constrained & Jump-Forward Decoding
│   ├── DIRECTORY_STRUCTURE.md               <-- (This file) Repository layout and navigation
│   └── EVALUATION_GUIDE.md                  <-- Peer review preparation & live-coding defense tips
├── resources/                               <-- Reference documents & original project assets
│   ├── en.subject.pdf                       <-- Official 42 Subject (v1.5)
│   ├── Report Tecnico e Teorico ... .pdf    <-- Comprehensive theoretical analysis report
│   └── archives/                            <-- Original distributed zip packages
│       ├── data.zip
│       └── llm_sdk.zip
│
└── call_me_maybe/                           <-- Inner Submission Directory (42 Intra / Vogsphere)
    ├── .gitignore                           <-- Excludes __pycache__, .venv, .mypy_cache, output/
    ├── Makefile                             <-- Standard 42 automation: install, run, debug, clean, lint
    ├── pyproject.toml                       <-- Python package configuration & dependencies (uv/pip)
    ├── uv.lock                              <-- Deterministic lockfile for reproducible evaluation
    ├── README.md                            <-- Mandatory 42 submission README (strictly follows Ch. VI)
    ├── data/                                <-- Test benchmarks provided by 42
    │   └── input/
    │       ├── function_calling_tests.json  <-- Natural language prompts to process
    │       └── functions_definition.json    <-- Available tool signatures & argument types
    ├── llm_sdk/                             <-- Official 42 LLM wrapper package
    │   └── llm_sdk/
    │       └── __init__.py                  <-- Small_LLM_Model black-box wrapper
    └── src/                                 <-- Implementation Source Directory (Future coding phase)
        └── .gitkeep                         <-- Directory placeholder
```

---

## 2. Directory Details & Responsibility Separation

### 2.1 The Outer Workspace (GitHub Mastery Tracker)
This root folder is part of the student's 42 journey tracked on personal GitHub. It is designed for:
- Long-term knowledge archiving.
- Academic and theoretical depth (arXiv citations, transformer internals).
- Agent guidance (`AGENTS.md`) ensuring any future AI session follows strict 42 rules.
- Fast access to original subject files and technical reports in `resources/`.

### 2.2 The Inner Directory (`call_me_maybe/` — 42 Intra Submission)
This directory contains **exclusively** what the 42 peer-evaluator and automated grading system (Moulinette) expect to find.

According to **Chapter VIII (Submission and peer review)**:
1. `src/`: Contains the complete Python implementation.
2. `pyproject.toml` and `uv.lock`: Dependency definitions; the evaluator runs `uv sync`.
3. `llm_sdk/`: The provided SDK package untouched.
4. `data/input/`: The evaluation dataset.
5. `Makefile`: The required automated commands.
6. `README.md`: The submission documentation written in English, beginning with the italicized attribution line.
7. **NO `data/output/` directory**: As emphasized in Chapter VIII, output files must be dynamically generated during evaluation, never committed.

---

## 3. Workflow for 42 Submission

When the implementation in `call_me_maybe/` is complete and verified:

1. **Test inside `call_me_maybe/`**:
   ```bash
   cd call_me_maybe
   make install
   make lint
   make run
   ```

2. **Verify Git Cleanliness**:
   Ensure no cache files or `data/output/` are staged.

3. **Push to Intra**:
   - If using a separate git remote or vogsphere repository:
     ```bash
     cd call_me_maybe
     git init  # (or git remote add vogsphere <vogsphere_url>)
     git add .
     git commit -m "feat: complete call_me_maybe submission"
     git push -u origin main
     ```
   - Alternatively, copy the contents of `call_me_maybe/` directly into the cloned Vogsphere repository.

