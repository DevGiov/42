"""Tests for input loading and output writing."""

import json
from pathlib import Path

import pytest

from src.errors import InputFileError, OutputFileError
from src.loaders import (
    load_function_catalog,
    load_prompts,
    write_results,
)
from src.models import FunctionCall

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "input"


def _write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_provided_function_definitions_load() -> None:
    catalog = load_function_catalog(DATA_DIR / "functions_definition.json")
    assert "fn_add_numbers" in catalog.names()


def test_provided_prompts_load() -> None:
    prompts = load_prompts(DATA_DIR / "function_calling_tests.json")
    assert len(prompts) > 0
    assert all(item.prompt for item in prompts)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(InputFileError, match="file not found"):
        load_prompts(tmp_path / "absent.json")


def test_directory_instead_of_file(tmp_path: Path) -> None:
    with pytest.raises(InputFileError, match="is a directory"):
        load_prompts(tmp_path)


def test_invalid_json_reports_position(tmp_path: Path) -> None:
    path = _write(tmp_path / "bad.json", '[\n  {"prompt": "x",}\n]')
    with pytest.raises(InputFileError, match=r"line 2, column"):
        load_prompts(path)


def test_empty_file_is_invalid_json(tmp_path: Path) -> None:
    with pytest.raises(InputFileError, match="invalid JSON"):
        load_prompts(_write(tmp_path / "empty.json", ""))


def test_invalid_utf8(tmp_path: Path) -> None:
    path = tmp_path / "latin.json"
    path.write_bytes(b'["caf\xe9"]')
    with pytest.raises(InputFileError, match="not valid UTF-8"):
        load_prompts(path)


def test_deeply_nested_json_does_not_crash(tmp_path: Path) -> None:
    path = _write(tmp_path / "deep.json", "[" * 100000 + "]" * 100000)
    with pytest.raises(InputFileError):
        load_prompts(path)


def test_prompts_must_be_a_list_of_objects_with_prompt(
    tmp_path: Path,
) -> None:
    with pytest.raises(InputFileError, match="expected schema"):
        load_prompts(_write(tmp_path / "a.json", '{"prompt": "x"}'))
    with pytest.raises(InputFileError, match=r"0\.prompt"):
        load_prompts(_write(tmp_path / "b.json", '[{"text": "x"}]'))


def test_empty_prompt_list_is_valid(tmp_path: Path) -> None:
    assert load_prompts(_write(tmp_path / "e.json", "[]")) == []


def test_catalog_schema_errors_are_readable(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "f.json",
        json.dumps([{"name": "fn", "parameters": {"a": {"type": "x"}}}]),
    )
    with pytest.raises(InputFileError) as info:
        load_function_catalog(path)
    message = str(info.value)
    assert "functions.0.parameters.a.type" in message
    assert str(path) in message


def test_catalog_rejects_non_list_and_empty(tmp_path: Path) -> None:
    with pytest.raises(InputFileError):
        load_function_catalog(_write(tmp_path / "o.json", "{}"))
    with pytest.raises(InputFileError):
        load_function_catalog(_write(tmp_path / "l.json", "[]"))


def test_many_errors_are_truncated(tmp_path: Path) -> None:
    path = _write(tmp_path / "many.json", json.dumps([{}] * 20))
    with pytest.raises(InputFileError, match="more problem"):
        load_prompts(path)


def test_write_results_round_trip_and_creates_directories(
    tmp_path: Path,
) -> None:
    target = tmp_path / "nested" / "out" / "results.json"
    results = [
        FunctionCall(
            prompt="Sum 2 and 3 ✓",
            name="fn_add_numbers",
            parameters={"a": 2.0, "b": 3.0},
        )
    ]
    write_results(target, results)
    loaded = json.loads(target.read_text(encoding="utf-8"))
    assert loaded == [
        {
            "prompt": "Sum 2 and 3 ✓",
            "name": "fn_add_numbers",
            "parameters": {"a": 2.0, "b": 3.0},
        }
    ]


def test_write_empty_results(tmp_path: Path) -> None:
    target = tmp_path / "o.json"
    write_results(target, [])
    assert json.loads(target.read_text(encoding="utf-8")) == []


def test_write_to_a_directory_fails_cleanly(tmp_path: Path) -> None:
    with pytest.raises(OutputFileError, match="cannot write"):
        write_results(tmp_path, [])


def test_non_finite_numbers_never_leave_a_partial_file(
    tmp_path: Path,
) -> None:
    target = tmp_path / "o.json"
    bad = FunctionCall(
        prompt="p", name="n", parameters={"a": float("nan")}
    )
    with pytest.raises(OutputFileError, match="serialised"):
        write_results(target, [bad])
    assert not target.exists()
