# Makefile for 42 Mastery: Call Me Maybe
# Complies with Chapter IV.2 of the 42 subject

PYTHON := python3
UV := $(shell which uv 2>/dev/null)

.PHONY: all install run debug clean lint lint-strict test help

all: install run

install:
	@if [ -n "$(UV)" ]; then \
		echo "[*] Syncing dependencies with uv..."; \
		uv sync; \
	else \
		echo "[*] Installing dependencies with pip..."; \
		$(PYTHON) -m pip install -e . || $(PYTHON) -m pip install pydantic numpy torch transformers huggingface-hub flake8 mypy pytest; \
	fi

run:
	@if [ -n "$(UV)" ]; then \
		uv run python -m src; \
	else \
		$(PYTHON) -m src; \
	fi

debug:
	@if [ -n "$(UV)" ]; then \
		uv run python -m pdb -m src; \
	else \
		$(PYTHON) -m pdb -m src; \
	fi

clean:
	@echo "[*] Cleaning temporary files and caches..."
	rm -rf __pycache__ src/__pycache__ llm_sdk/__pycache__
	rm -rf .mypy_cache .pytest_cache .ruff_cache
	rm -rf data/output
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.py[cod]" -delete

lint:
	flake8 .
	mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	flake8 .
	mypy . --strict

test:
	@if [ -n "$(UV)" ]; then \
		uv run pytest; \
	else \
		$(PYTHON) -m pytest; \
	fi

help:
	@echo "Available Makefile rules:"
	@echo "  make install      Install project dependencies (uv or pip)"
	@echo "  make run          Execute main project via python -m src"
	@echo "  make debug        Run main script with pdb debugger"
	@echo "  make clean        Remove cache and output artifacts"
	@echo "  make lint         Run mandatory flake8 and mypy checks"
	@echo "  make lint-strict  Run strict flake8 and mypy checks"
	@echo "  make test         Run pytest test suite"

