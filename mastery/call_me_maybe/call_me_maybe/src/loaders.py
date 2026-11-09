"""Read and validate the input files, write the output file.

All I/O problems (missing file, bad permissions, invalid JSON, wrong
schema, unwritable destination...) are converted into the exceptions of
:mod:`src.errors`, so callers only ever need to handle those.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Sequence

from pydantic import TypeAdapter, ValidationError

from src.errors import InputFileError, OutputFileError
from src.models import FunctionCall, FunctionCatalog, PromptItem

_MAX_REPORTED_ERRORS = 5
_PROMPTS_ADAPTER: TypeAdapter[List[PromptItem]] = TypeAdapter(
    List[PromptItem]
)


def read_json_file(path: Path, label: str) -> Any:
    """Parse a JSON file and return its content.

    Args:
        path: File to read.
        label: Human-readable role of the file, used in error messages
            (for example ``"functions definition"``).

    Raises:
        InputFileError: If the file cannot be read or is not valid JSON.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        raise InputFileError(
            f"{label}: file not found: {path}"
        ) from None
    except IsADirectoryError:
        raise InputFileError(
            f"{label}: expected a file but {path} is a directory"
        ) from None
    except PermissionError:
        raise InputFileError(
            f"{label}: permission denied reading {path}"
        ) from None
    except UnicodeDecodeError as exc:
        raise InputFileError(
            f"{label}: {path} is not valid UTF-8 ({exc.reason})"
        ) from None
    except json.JSONDecodeError as exc:
        raise InputFileError(
            f"{label}: invalid JSON in {path} "
            f"(line {exc.lineno}, column {exc.colno}): {exc.msg}"
        ) from None
    except RecursionError:
        raise InputFileError(
            f"{label}: JSON in {path} is nested too deeply"
        ) from None
    except OSError as exc:
        raise InputFileError(
            f"{label}: cannot read {path}: {exc.strerror or exc}"
        ) from None


def format_validation_error(error: ValidationError) -> str:
    """Render a pydantic error as a short, readable multi-line message."""
    problems = error.errors()
    lines = []
    for problem in problems[:_MAX_REPORTED_ERRORS]:
        where = ".".join(str(part) for part in problem["loc"]) or "<root>"
        lines.append(f"  - {where}: {problem['msg']}")
    hidden = len(problems) - _MAX_REPORTED_ERRORS
    if hidden > 0:
        lines.append(f"  ... and {hidden} more problem(s)")
    return "\n".join(lines)


def load_function_catalog(path: Path) -> FunctionCatalog:
    """Load and validate ``functions_definition.json``.

    Raises:
        InputFileError: If the file is unreadable or does not describe a
            non-empty list of valid, uniquely named functions.
    """
    label = "functions definition"
    raw = read_json_file(path, label)
    try:
        return FunctionCatalog.model_validate({"functions": raw})
    except ValidationError as exc:
        raise InputFileError(
            f"{label}: {path} does not match the expected schema:\n"
            f"{format_validation_error(exc)}"
        ) from None


def load_prompts(path: Path) -> List[PromptItem]:
    """Load and validate the list of prompts to process.

    An empty list is valid and simply yields no results.

    Raises:
        InputFileError: If the file is unreadable or is not a list of
            objects that each carry a string ``"prompt"``.
    """
    label = "input prompts"
    raw = read_json_file(path, label)
    try:
        return _PROMPTS_ADAPTER.validate_python(raw)
    except ValidationError as exc:
        raise InputFileError(
            f"{label}: {path} does not match the expected schema:\n"
            f"{format_validation_error(exc)}"
        ) from None


def write_results(path: Path, results: Sequence[FunctionCall]) -> None:
    """Write the results as a JSON array, creating parent directories.

    The document is fully serialised before the file is opened, so a
    serialisation problem never leaves a truncated file behind.

    Raises:
        OutputFileError: If serialisation or writing fails.
    """
    try:
        text = json.dumps(
            [item.model_dump(mode="json") for item in results],
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise OutputFileError(
            f"results cannot be serialised to JSON: {exc}"
        ) from None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.write("\n")
    except OSError as exc:
        raise OutputFileError(
            f"cannot write output file {path}: {exc.strerror or exc}"
        ) from None
