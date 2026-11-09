"""Tests for the constrained decoding engine, with fake models."""

from typing import Any, List, Sequence

import pytest

from src.constrained_decoder import (
    ConstrainedDecoder,
    DecodingLimits,
    GeneratedCall,
)
from src.errors import DecodingError
from src.models import FunctionCatalog
from src.tokenizer import Vocabulary
from tests.helpers import (
    GreedyTokenizer,
    RandomModel,
    ScriptedModel,
    build_vocab_mapping,
    function,
    make_catalog,
)

PROMPT = "Task: "

ADD = function("fn_add_numbers", a="number", b="number")
GREET = function("fn_greet", name="string")
REVERSE = function("fn_reverse_string", s="string")
FLAGS = function("fn_flags", on="boolean", count="integer")
CATALOG = make_catalog(ADD, GREET, REVERSE, FLAGS)


@pytest.fixture(scope="module")
def tokenizer() -> GreedyTokenizer:
    return GreedyTokenizer(build_vocab_mapping())


def _run(
    tokenizer: GreedyTokenizer,
    catalog: FunctionCatalog,
    target: str,
    bait: Sequence[int] = (),
    limits: DecodingLimits = DecodingLimits(),
) -> GeneratedCall:
    model = ScriptedModel(tokenizer, PROMPT, target, bait=bait)
    decoder = ConstrainedDecoder(
        model=model,
        tokenizer=tokenizer,
        vocabulary=tokenizer.vocab,
        catalog=catalog,
        limits=limits,
    )
    return decoder.generate_call(tokenizer.encode(PROMPT))


def _id(tokenizer: GreedyTokenizer, text: str) -> int:
    ids = tokenizer.encode(text)
    assert len(ids) == 1, text
    return ids[0]


def test_addition(tokenizer: GreedyTokenizer) -> None:
    call = _run(
        tokenizer,
        CATALOG,
        '{"name": "fn_add_numbers", "parameters": {"a": 2, "b": 3}}',
    )
    assert call.name == "fn_add_numbers"
    assert call.parameters == {"a": 2.0, "b": 3.0}
    assert all(isinstance(v, float) for v in call.parameters.values())


def test_negative_and_decimal_numbers(tokenizer: GreedyTokenizer) -> None:
    call = _run(
        tokenizer,
        CATALOG,
        '{"name": "fn_add_numbers", "parameters": {"a": -3.14, "b": 265}}',
    )
    assert call.parameters == {"a": -3.14, "b": 265.0}


def test_string_argument_and_early_commit(tokenizer: GreedyTokenizer) -> None:
    call = _run(
        tokenizer,
        CATALOG,
        '{"name": "fn_greet", "parameters": {"name": "shrek"}}',
    )
    assert call.name == "fn_greet"
    assert call.parameters == {"name": "shrek"}


def test_string_with_quotes_unicode_and_escapes(
    tokenizer: GreedyTokenizer,
) -> None:
    call = _run(
        tokenizer,
        CATALOG,
        '{"name": "fn_reverse_string", "parameters": '
        '{"s": "café \\"x\\" \\\\ hello world"}}',
    )
    assert call.parameters == {"s": 'café "x" \\ hello world'}


def test_empty_string(tokenizer: GreedyTokenizer) -> None:
    call = _run(
        tokenizer,
        CATALOG,
        '{"name": "fn_reverse_string", "parameters": {"s": ""}}',
    )
    assert call.parameters == {"s": ""}


def test_boolean_and_integer(tokenizer: GreedyTokenizer) -> None:
    call = _run(
        tokenizer,
        CATALOG,
        '{"name": "fn_flags", "parameters": {"on": false, "count": 7}}',
    )
    assert call.parameters == {"on": False, "count": 7}
    assert isinstance(call.parameters["count"], int)
    assert call.parameters["on"] is False


def test_prefix_names_need_the_stop_quote(
    tokenizer: GreedyTokenizer,
) -> None:
    catalog = make_catalog(
        function("fn_sum", x="number"),
        function("fn_sum_all", x="number"),
    )
    short = _run(
        tokenizer, catalog,
        '{"name": "fn_sum", "parameters": {"x": 1}}',
    )
    long = _run(
        tokenizer, catalog,
        '{"name": "fn_sum_all", "parameters": {"x": 1}}',
    )
    assert (short.name, long.name) == ("fn_sum", "fn_sum_all")


def test_function_without_parameters(tokenizer: GreedyTokenizer) -> None:
    catalog = make_catalog(function("fn_ping"), function("fn_pong"))
    call = _run(
        tokenizer, catalog, '{"name": "fn_pong", "parameters": {}}'
    )
    assert (call.name, call.parameters) == ("fn_pong", {})


def test_the_model_cannot_pick_an_unknown_function(
    tokenizer: GreedyTokenizer,
) -> None:
    call = _run(
        tokenizer,
        CATALOG,
        '{"name": "fn_delete_everything", "parameters": {}}',
    )
    assert call.name in CATALOG.names()


def test_bait_tokens_are_masked(tokenizer: GreedyTokenizer) -> None:
    bait = [
        _id(tokenizer, "hello"), _id(tokenizer, '"}'),
        _id(tokenizer, "true"), _id(tokenizer, "\n\n"),
    ]
    call = _run(
        tokenizer,
        CATALOG,
        '{"name": "fn_add_numbers", "parameters": {"a": 2, "b": 3}}',
        bait=bait,
    )
    assert call.name in CATALOG.names()
    for key, value in call.parameters.items():
        assert isinstance(value, float), key


def test_leading_zero_is_never_produced(tokenizer: GreedyTokenizer) -> None:
    call = _run(
        tokenizer,
        CATALOG,
        '{"name": "fn_add_numbers", "parameters": {"a": 007, "b": 1}}',
    )
    # After "0" only "." or the end of the number are valid, so "007"
    # cannot be written whatever the model prefers.
    a = call.parameters["a"]
    assert isinstance(a, float) and 0.0 <= a < 1.0


def test_second_decimal_point_is_never_produced(
    tokenizer: GreedyTokenizer,
) -> None:
    call = _run(
        tokenizer,
        CATALOG,
        '{"name": "fn_add_numbers", "parameters": {"a": 1, "b": 1.2.3}}',
    )
    b = call.parameters["b"]
    assert isinstance(b, float) and 1.2 <= b < 1.3


def test_forward_passes_skip_the_fixed_text(
    tokenizer: GreedyTokenizer,
) -> None:
    catalog = make_catalog(function("fn_ping"))
    model = ScriptedModel(tokenizer, PROMPT, "")
    decoder = ConstrainedDecoder(
        model=model,
        tokenizer=tokenizer,
        vocabulary=tokenizer.vocab,
        catalog=catalog,
    )
    call = decoder.generate_call(tokenizer.encode(PROMPT))
    assert call.name == "fn_ping"
    # One function and no arguments: the whole call is jump-forward.
    assert decoder.forward_passes == 0


def test_forward_passes_for_addition_are_few(
    tokenizer: GreedyTokenizer,
) -> None:
    model = ScriptedModel(
        tokenizer, PROMPT,
        '{"name": "fn_add_numbers", "parameters": {"a": 2, "b": 3}}',
    )
    decoder = ConstrainedDecoder(
        model=model,
        tokenizer=tokenizer,
        vocabulary=tokenizer.vocab,
        catalog=CATALOG,
    )
    decoder.generate_call(tokenizer.encode(PROMPT))
    # name: "fn", "_add" (then early commit); a: "2", ","; b: "3", "}".
    assert decoder.forward_passes <= 8


def test_number_length_cap_forces_a_valid_number(
    tokenizer: GreedyTokenizer,
) -> None:
    call = _run(
        tokenizer,
        CATALOG,
        '{"name": "fn_add_numbers", "parameters": {"a": '
        + "9" * 100 + ', "b": 1}}',
        limits=DecodingLimits(max_number_chars=5),
    )
    assert call.parameters["a"] == 99999.0


def test_string_length_cap_still_closes_the_string(
    tokenizer: GreedyTokenizer,
) -> None:
    call = _run(
        tokenizer,
        CATALOG,
        '{"name": "fn_greet", "parameters": {"name": "' + "x" * 100 + '"}}',
        limits=DecodingLimits(max_string_tokens=4),
    )
    assert call.parameters == {"name": "xxxx"}


@pytest.mark.parametrize("seed", range(40))
def test_random_logits_always_yield_a_valid_call(
    tokenizer: GreedyTokenizer, seed: int
) -> None:
    model = RandomModel(tokenizer.vocab.size + 3, seed)
    decoder = ConstrainedDecoder(
        model=model,
        tokenizer=tokenizer,
        vocabulary=tokenizer.vocab,
        catalog=CATALOG,
        limits=DecodingLimits(max_string_tokens=12),
    )
    call = decoder.generate_call(tokenizer.encode(PROMPT))
    function_def = CATALOG.get(call.name)
    assert list(call.parameters) == list(function_def.parameters)
    for key, spec in function_def.parameters.items():
        value: Any = call.parameters[key]
        if spec.type == "number":
            assert isinstance(value, float)
        elif spec.type == "integer":
            assert isinstance(value, int) and not isinstance(value, bool)
        elif spec.type == "boolean":
            assert isinstance(value, bool)
        else:
            assert isinstance(value, str)
            value.encode("utf-8")  # must be savable as UTF-8


def test_vocabulary_without_digits_is_a_clean_error(
    tokenizer: GreedyTokenizer,
) -> None:
    mapping = {
        token: token_id
        for token, token_id in build_vocab_mapping().items()
        if not any(char in "0123456789-" for char in token)
    }
    # Re-number ids so the vocabulary has no gaps.
    mapping = {token: i for i, token in enumerate(mapping)}
    vocab = Vocabulary.from_mapping(mapping)
    small = GreedyTokenizer(mapping)
    decoder = ConstrainedDecoder(
        model=RandomModel(vocab.size, 1),
        tokenizer=small,
        vocabulary=vocab,
        catalog=make_catalog(function("fn_x", a="number")),
    )
    with pytest.raises(DecodingError):
        decoder.generate_call(small.encode(PROMPT))


class _BrokenModel:
    def get_logits_from_input_ids(self, input_ids: List[int]) -> List[float]:
        return []


def test_malformed_logits_are_a_clean_error(
    tokenizer: GreedyTokenizer,
) -> None:
    decoder = ConstrainedDecoder(
        model=_BrokenModel(),
        tokenizer=tokenizer,
        vocabulary=tokenizer.vocab,
        catalog=CATALOG,
    )
    with pytest.raises(DecodingError, match="malformed"):
        decoder.generate_call(tokenizer.encode(PROMPT))


class _NanModel:
    def get_logits_from_input_ids(self, input_ids: List[int]) -> List[float]:
        return [float("nan")] * 300


def test_nan_logits_are_a_clean_error(tokenizer: GreedyTokenizer) -> None:
    decoder = ConstrainedDecoder(
        model=_NanModel(),
        tokenizer=tokenizer,
        vocabulary=tokenizer.vocab,
        catalog=CATALOG,
    )
    with pytest.raises(DecodingError, match="malformed"):
        decoder.generate_call(tokenizer.encode(PROMPT))
