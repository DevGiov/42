"""Tests for the byte-level vocabulary and the SDK tokenizer."""

import json
from pathlib import Path
from typing import Any, List

import pytest
from pydantic import ValidationError

from src.errors import TokenizerError
from src.tokenizer import (
    SdkTokenizer,
    Tokenizer,
    Vocabulary,
    token_to_bytes,
)


def test_byte_level_mapping_of_whitespace() -> None:
    assert token_to_bytes("Ġ") == b" "
    assert token_to_bytes("Ċ") == b"\n"
    assert token_to_bytes("ĉ") == b"\t"
    assert token_to_bytes("Ġsum") == b" sum"


def test_every_byte_has_a_distinct_printable_token() -> None:
    seen = set()
    for code in range(0x20, 0x7F):
        seen.add(token_to_bytes(chr(code)))
    assert len(seen) == 0x7F - 0x20


def test_non_byte_level_token_has_no_bytes() -> None:
    assert token_to_bytes("中") is None


def test_vocabulary_decodes_text_and_flags_unusable_tokens() -> None:
    vocab = Vocabulary.from_mapping(
        {
            "hello": 0,
            "Ġsum": 1,
            "Ã©": 2,  # "é" encoded as bytes c3 a9
            "Ã": 3,  # lone first byte of a multi-byte character
            "中": 4,  # special token, not byte-level encoded
            "Ċ": 6,  # id 5 is a gap
        }
    )
    assert vocab.size == 7
    assert vocab.text_of(0) == "hello"
    assert vocab.text_of(1) == " sum"
    assert vocab.text_of(2) == "é"
    assert vocab.text_of(3) is None
    assert vocab.token_bytes[3] == b"\xc3"
    assert vocab.text_of(4) is None
    assert vocab.text_of(5) is None
    assert vocab.text_of(6) == "\n"
    assert vocab.text_token_ids() == [0, 1, 2, 6]


def test_ids_beyond_the_vocabulary_have_no_text() -> None:
    vocab = Vocabulary.from_mapping({"a": 0})
    assert vocab.text_of(151935) is None
    assert vocab.text_of(-1) is None


def test_vocabulary_rejects_bad_ids() -> None:
    with pytest.raises(TokenizerError, match="empty"):
        Vocabulary.from_mapping({})
    with pytest.raises(TokenizerError, match="out-of-range"):
        Vocabulary.from_mapping({"a": -1})
    with pytest.raises(TokenizerError, match="out-of-range"):
        Vocabulary.from_mapping({"a": 10**12})


def test_vocabulary_rejects_duplicate_ids() -> None:
    with pytest.raises(TokenizerError, match="more than once"):
        Vocabulary.from_mapping({"a": 0, "b": 0})


def test_vocabulary_from_file(tmp_path: Path) -> None:
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps({"a": 0, "Ġb": 1}), encoding="utf-8")
    vocab = Vocabulary.from_file(path)
    assert vocab.text_of(1) == " b"


def test_vocabulary_from_bad_files(tmp_path: Path) -> None:
    with pytest.raises(TokenizerError, match="file not found"):
        Vocabulary.from_file(tmp_path / "absent.json")
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    with pytest.raises(TokenizerError, match="invalid JSON"):
        Vocabulary.from_file(broken)
    for content in ('["a"]', '{"a": "x"}', '{"a": true}', '{"a": 1.5}'):
        wrong = tmp_path / "wrong.json"
        wrong.write_text(content, encoding="utf-8")
        with pytest.raises(TokenizerError, match="must be a JSON object"):
            Vocabulary.from_file(wrong)


class _FakeTensor:
    """Mimics the part of a torch tensor the tokenizer relies on."""

    def __init__(self, rows: List[List[int]]) -> None:
        self._rows = rows

    def tolist(self) -> List[List[int]]:
        return self._rows


class _FakeBackend:
    def __init__(self, result: Any = None) -> None:
        self.result = result
        self.decoded: List[Any] = []

    def encode(self, text: str) -> Any:
        if self.result is not None:
            return self.result
        return _FakeTensor([[ord(char) for char in text]])

    def decode(self, ids: Any) -> str:
        self.decoded.append(ids)
        return "".join(chr(i) for i in ids)


def test_sdk_tokenizer_flattens_the_batch_dimension() -> None:
    tokenizer: Tokenizer = SdkTokenizer(backend=_FakeBackend())
    assert tokenizer.encode("ab") == [97, 98]
    assert tokenizer.encode("") == []


def test_sdk_tokenizer_accepts_a_flat_list() -> None:
    backend = _FakeBackend(result=[1, 2, 3])
    assert SdkTokenizer(backend=backend).encode("x") == [1, 2, 3]


def test_sdk_tokenizer_decode_passes_a_plain_list() -> None:
    backend = _FakeBackend()
    tokenizer = SdkTokenizer(backend=backend)
    assert tokenizer.decode((104, 105)) == "hi"
    assert backend.decoded == [[104, 105]]


def test_sdk_tokenizer_rejects_unexpected_encode_result() -> None:
    tokenizer = SdkTokenizer(backend=_FakeBackend(result=42))
    with pytest.raises(TokenizerError, match="unexpected"):
        tokenizer.encode("x")
    tokenizer = SdkTokenizer(backend=_FakeBackend(result=[["a"]]))
    with pytest.raises(TokenizerError, match="unexpected"):
        tokenizer.encode("x")


def test_sdk_tokenizer_requires_an_encode_decode_backend() -> None:
    with pytest.raises(ValidationError):
        SdkTokenizer(backend=object())  # type: ignore[arg-type]
