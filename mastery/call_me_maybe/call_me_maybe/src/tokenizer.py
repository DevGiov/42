"""Tokenizer layer: text <-> token ids, and the id -> text vocabulary.

Two independent pieces live here:

* :class:`Tokenizer` is the interface the rest of the program uses to turn
  text into token ids and back. :class:`SdkTokenizer` implements it on top
  of the public ``encode`` / ``decode`` methods of ``Small_LLM_Model``.
  Another implementation (for the bonus, built only from ``vocab.json``)
  can be dropped in without touching the decoder.
* :class:`Vocabulary` reads ``vocab.json`` once and answers "which text
  does token ``i`` produce?", which is what logit masking needs.

Qwen3 uses byte-level BPE: every byte is stored in ``vocab.json`` as a
printable unicode character (space -> ``Ġ``, newline -> ``Ċ``...). The
mapping below undoes that to recover the real bytes of each token.
"""

from __future__ import annotations

from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

from pydantic import BaseModel, ConfigDict

from src.errors import InputFileError, TokenizerError
from src.loaders import read_json_file

# A vocabulary this large is certainly corrupted; refuse to allocate it.
_MAX_TOKEN_ID = 10_000_000


def _build_unicode_to_byte() -> Dict[str, int]:
    """Return the inverse of the GPT-2 byte -> printable-char mapping."""
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(0xA1, 0xAC + 1))
        + list(range(0xAE, 0xFF + 1))
    )
    byte_values = printable[:]
    code_points = printable[:]
    shift = 0
    for byte in range(256):
        if byte not in printable:
            byte_values.append(byte)
            code_points.append(256 + shift)
            shift += 1
    return {chr(cp): bv for bv, cp in zip(byte_values, code_points)}


_UNICODE_TO_BYTE = _build_unicode_to_byte()


def token_to_bytes(token: str) -> Optional[bytes]:
    """Convert a ``vocab.json`` key to the raw bytes it stands for.

    Returns ``None`` for keys that are not byte-level encoded (special or
    added tokens), because they never belong to generated text.
    """
    try:
        return bytes(_UNICODE_TO_BYTE[char] for char in token)
    except KeyError:
        return None


@runtime_checkable
class TokenizerBackend(Protocol):
    """The slice of ``Small_LLM_Model`` that :class:`SdkTokenizer` uses."""

    def encode(self, text: str) -> Any:
        """Return a ``[1, n]`` tensor of token ids."""

    def decode(self, ids: Any) -> str:
        """Return the text for a list of token ids."""


@runtime_checkable
class Tokenizer(Protocol):
    """Anything that converts between text and token ids."""

    def encode(self, text: str) -> List[int]:
        """Return the token ids for ``text``."""

    def decode(self, token_ids: Sequence[int]) -> str:
        """Return the text for ``token_ids``."""


class SdkTokenizer(BaseModel):
    """:class:`Tokenizer` backed by the SDK's public encode/decode."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    backend: TokenizerBackend

    def encode(self, text: str) -> List[int]:
        """Return the token ids for ``text`` as a plain list of ints.

        Raises:
            TokenizerError: If the backend returns something that is not
                a ``[1, n]`` (or flat) collection of integers.
        """
        raw = self.backend.encode(text)
        rows = raw.tolist() if hasattr(raw, "tolist") else raw
        try:
            if len(rows) > 0 and isinstance(rows[0], (list, tuple)):
                rows = rows[0]
            return [int(token_id) for token_id in rows]
        except (TypeError, ValueError):
            raise TokenizerError(
                "the model's encode() returned an unexpected value"
            ) from None

    def decode(self, token_ids: Sequence[int]) -> str:
        """Return the text for ``token_ids``."""
        return self.backend.decode(list(token_ids))


class Vocabulary(BaseModel):
    """Maps every token id to the bytes and the text it produces.

    The model's logit vector can be longer than the vocabulary file
    (special tokens are appended after it), so ``size`` is only the number
    of ids that ``vocab.json`` knows about; any larger id has no text.

    Attributes:
        token_bytes: Raw bytes of each id, ``None`` when the id is unused
            or is a special token.
        token_texts: Text of each id when its bytes are valid standalone
            UTF-8, otherwise ``None``. Tokens that carry only part of a
            multi-byte character are therefore never offered as text.
    """

    model_config = ConfigDict(frozen=True)

    token_bytes: List[Optional[bytes]]
    token_texts: List[Optional[str]]

    @classmethod
    def from_mapping(cls, raw_vocab: Mapping[str, int]) -> Vocabulary:
        """Build a vocabulary from an in-memory ``{token: id}`` mapping.

        Raises:
            TokenizerError: If ids are negative, absurdly large or
                assigned to more than one token.
        """
        if not raw_vocab:
            raise TokenizerError("the vocabulary is empty")
        max_id = max(raw_vocab.values())
        if min(raw_vocab.values()) < 0 or max_id > _MAX_TOKEN_ID:
            raise TokenizerError(
                f"the vocabulary contains out-of-range token ids "
                f"(max id {max_id})"
            )
        token_bytes: List[Optional[bytes]] = [None] * (max_id + 1)
        token_texts: List[Optional[str]] = [None] * (max_id + 1)
        assigned = [False] * (max_id + 1)
        for token, token_id in raw_vocab.items():
            if assigned[token_id]:
                raise TokenizerError(
                    f"token id {token_id} is assigned more than once"
                )
            assigned[token_id] = True
            data = token_to_bytes(token)
            token_bytes[token_id] = data
            if data is not None:
                try:
                    token_texts[token_id] = data.decode("utf-8")
                except UnicodeDecodeError:
                    pass
        return cls(token_bytes=token_bytes, token_texts=token_texts)

    @classmethod
    def from_file(cls, path: Path) -> Vocabulary:
        """Load a ``vocab.json`` file (as given by the SDK).

        Raises:
            TokenizerError: If the file is unreadable or is not a JSON
                object mapping strings to integer ids.
        """
        try:
            raw = read_json_file(path, "vocabulary")
        except InputFileError as exc:
            raise TokenizerError(str(exc)) from None
        if not isinstance(raw, dict) or not all(
            isinstance(key, str)
            and isinstance(value, int)
            and not isinstance(value, bool)
            for key, value in raw.items()
        ):
            raise TokenizerError(
                f"vocabulary: {path} must be a JSON object mapping "
                f"token strings to integer ids"
            )
        return cls.from_mapping(raw)

    @property
    def size(self) -> int:
        """Number of ids covered by the vocabulary file."""
        return len(self.token_bytes)

    def text_of(self, token_id: int) -> Optional[str]:
        """Return the text of ``token_id``, or ``None`` if it has none."""
        if 0 <= token_id < len(self.token_texts):
            return self.token_texts[token_id]
        return None

    def text_token_ids(self) -> List[int]:
        """Return every id that produces standalone, valid UTF-8 text."""
        return [
            token_id
            for token_id, text in enumerate(self.token_texts)
            if text is not None
        ]
