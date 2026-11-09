"""Pydantic models: function schema, test prompts, output and CLI config."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic import model_validator

DEFAULT_MODEL_NAME = "Qwen/Qwen3-0.6B"
DEFAULT_FUNCTIONS_PATH = Path("data/input/functions_definition.json")
DEFAULT_INPUT_PATH = Path("data/input/function_calling_tests.json")
DEFAULT_OUTPUT_PATH = Path("data/output/function_calling_results.json")

# Parameter types the constrained decoder knows how to generate.
ParamType = Literal["number", "integer", "string", "boolean"]

# Python values a generated argument can take once decoded.
ParamValue = Union[bool, int, float, str]


class ParameterSpec(BaseModel):
    """Schema of one function parameter (``{"type": "number"}``)."""

    model_config = ConfigDict(frozen=True)

    type: ParamType


class ReturnSpec(BaseModel):
    """Schema of a function return value (informational only)."""

    model_config = ConfigDict(frozen=True)

    type: str


class FunctionDefinition(BaseModel):
    """One callable function described in ``functions_definition.json``.

    ``parameters`` keeps the key order of the JSON file, so arguments are
    always generated in the order the schema declares them.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    description: str = ""
    parameters: Dict[str, ParameterSpec]
    returns: Optional[ReturnSpec] = None


class FunctionCatalog(BaseModel):
    """The non-empty set of functions the LLM can choose from."""

    model_config = ConfigDict(frozen=True)

    functions: List[FunctionDefinition] = Field(min_length=1)

    @field_validator("functions")
    @classmethod
    def _names_must_be_unique(
        cls, functions: List[FunctionDefinition]
    ) -> List[FunctionDefinition]:
        """Reject catalogs where two functions share the same name."""
        seen: set[str] = set()
        for function in functions:
            if function.name in seen:
                raise ValueError(
                    f"duplicate function name {function.name!r}"
                )
            seen.add(function.name)
        return functions

    def names(self) -> List[str]:
        """Return the function names in file order."""
        return [function.name for function in self.functions]

    def get(self, name: str) -> FunctionDefinition:
        """Return the function called ``name``.

        Raises:
            KeyError: If no function has that name.
        """
        for function in self.functions:
            if function.name == name:
                return function
        raise KeyError(name)


class PromptItem(BaseModel):
    """One natural-language request from ``function_calling_tests.json``."""

    model_config = ConfigDict(frozen=True)

    prompt: str


class FunctionCall(BaseModel):
    """One element of the output file: exactly three keys."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt: str
    name: str
    parameters: Dict[str, ParamValue]


class CLIConfig(BaseModel):
    """Validated command-line options."""

    model_config = ConfigDict(frozen=True)

    functions_path: Path = DEFAULT_FUNCTIONS_PATH
    input_path: Path = DEFAULT_INPUT_PATH
    output_path: Path = DEFAULT_OUTPUT_PATH
    model_name: str = Field(default=DEFAULT_MODEL_NAME, min_length=1)

    @model_validator(mode="after")
    def _output_must_not_overwrite_inputs(self) -> CLIConfig:
        """Refuse to clobber one of the input files with the results."""
        target = self.output_path.resolve()
        for source in (self.functions_path, self.input_path):
            if source.resolve() == target:
                raise ValueError(
                    f"output path {self.output_path} would overwrite "
                    f"the input file {source}"
                )
        return self
