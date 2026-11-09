"""Tests for the pydantic models."""

from pathlib import Path
from typing import Any, Dict, List

import pytest
from pydantic import ValidationError

from src.models import (
    CLIConfig,
    DEFAULT_OUTPUT_PATH,
    FunctionCall,
    FunctionCatalog,
)


def _function(name: str, **params: str) -> Dict[str, Any]:
    return {
        "name": name,
        "description": "d",
        "parameters": {key: {"type": kind} for key, kind in params.items()},
        "returns": {"type": "number"},
    }


def _catalog(functions: List[Dict[str, Any]]) -> FunctionCatalog:
    return FunctionCatalog.model_validate({"functions": functions})


def test_catalog_lookup_and_names() -> None:
    catalog = _catalog([_function("fn_a", x="number"), _function("fn_b")])
    assert catalog.names() == ["fn_a", "fn_b"]
    assert catalog.get("fn_b").parameters == {}


def test_catalog_unknown_name_raises_key_error() -> None:
    with pytest.raises(KeyError):
        _catalog([_function("fn_a")]).get("nope")


def test_parameter_order_follows_the_schema() -> None:
    catalog = _catalog(
        [_function("fn", z="string", a="number", m="boolean")]
    )
    assert list(catalog.get("fn").parameters) == ["z", "a", "m"]


def test_catalog_rejects_empty_list() -> None:
    with pytest.raises(ValidationError):
        _catalog([])


def test_catalog_rejects_duplicate_names() -> None:
    with pytest.raises(ValidationError, match="duplicate function name"):
        _catalog([_function("fn"), _function("fn")])


def test_catalog_rejects_unsupported_parameter_type() -> None:
    with pytest.raises(ValidationError):
        _catalog([_function("fn", x="array")])


def test_function_requires_name_and_parameters() -> None:
    with pytest.raises(ValidationError):
        _catalog([{"description": "no name", "parameters": {}}])
    with pytest.raises(ValidationError):
        _catalog([{"name": "fn"}])


def test_function_call_keeps_value_types() -> None:
    call = FunctionCall(
        prompt="p",
        name="fn",
        parameters={"a": 2.0, "b": 3, "c": "x", "d": True},
    )
    values = call.parameters
    assert isinstance(values["a"], float)
    assert isinstance(values["b"], int) and not isinstance(values["b"], bool)
    assert values["c"] == "x"
    assert values["d"] is True


def test_function_call_forbids_extra_keys() -> None:
    with pytest.raises(ValidationError):
        FunctionCall.model_validate(
            {"prompt": "p", "name": "n", "parameters": {}, "extra": 1}
        )


def test_cli_config_defaults() -> None:
    config = CLIConfig()
    assert config.output_path == DEFAULT_OUTPUT_PATH
    assert config.model_name == "Qwen/Qwen3-0.6B"


def test_cli_config_refuses_to_overwrite_an_input(tmp_path: Path) -> None:
    same = tmp_path / "in.json"
    with pytest.raises(ValidationError, match="would overwrite"):
        CLIConfig(input_path=same, output_path=same)
    with pytest.raises(ValidationError, match="would overwrite"):
        CLIConfig(functions_path=same, output_path=same)
