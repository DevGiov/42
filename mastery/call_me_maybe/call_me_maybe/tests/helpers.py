"""Test doubles: a tiny byte-level vocabulary and fake language models."""

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.models import FunctionCatalog
from src.tokenizer import Vocabulary, _UNICODE_TO_BYTE

_BYTE_TO_UNICODE: Dict[int, str] = {
    byte: char for char, byte in _UNICODE_TO_BYTE.items()
}

# Multi-byte pieces on top of the 256 single-byte tokens.
EXTRA_PIECES = [
    "fn", "_add", "_numbers", "_greet", "_reverse", "_string", "_sum",
    "true", "false", '",', '"}', "},", "hello", " world", "12", "0.5",
    "é", "\n\n", '\\"', "-", ".", ",", "}", "{", '"',
]
# Not byte-level encoded: behaves like a special token.
SPECIAL_TOKEN = "<|eos|>"


def to_byte_level(text: str) -> str:
    """Spell ``text`` the way ``vocab.json`` does (bytes as chars)."""
    return "".join(_BYTE_TO_UNICODE[b] for b in text.encode("utf-8"))


def build_vocab_mapping() -> Dict[str, int]:
    """Return ``{token: id}``: 256 byte tokens, extras, then a special."""
    mapping: Dict[str, int] = {}
    for byte in range(256):
        mapping[_BYTE_TO_UNICODE[byte]] = len(mapping)
    for piece in EXTRA_PIECES:
        mapping.setdefault(to_byte_level(piece), len(mapping))
    mapping[SPECIAL_TOKEN] = len(mapping)
    return mapping


class GreedyTokenizer:
    """Longest-match tokenizer over the synthetic vocabulary."""

    def __init__(self, mapping: Dict[str, int]) -> None:
        self.vocab = Vocabulary.from_mapping(mapping)
        self.by_bytes: Dict[bytes, int] = {}
        for data_id, data in enumerate(self.vocab.token_bytes):
            if data is not None:
                self.by_bytes[data] = data_id
        self.longest = max(len(data) for data in self.by_bytes)

    def encode(self, text: str) -> List[int]:
        data = text.encode("utf-8")
        ids: List[int] = []
        position = 0
        while position < len(data):
            for size in range(min(self.longest, len(data) - position), 0, -1):
                piece = data[position:position + size]
                if piece in self.by_bytes:
                    ids.append(self.by_bytes[piece])
                    position += size
                    break
            else:
                raise ValueError(f"cannot encode {text!r}")
        return ids

    def decode(self, token_ids: Sequence[int]) -> str:
        data = b"".join(self.vocab.token_bytes[i] or b"" for i in token_ids)
        return data.decode("utf-8", errors="replace")


class ScriptedModel:
    """A fake LLM that "wants" to write ``target`` but is a bit noisy.

    It looks at the text generated so far and prefers the token that
    continues ``target``. Random noise and optional ``bait`` tokens (with
    a huge logit) try to tempt the decoder into invalid choices.
    """

    def __init__(
        self,
        tokenizer: GreedyTokenizer,
        prompt_text: str,
        target: str,
        bait: Optional[Sequence[int]] = None,
        extra_logits: int = 3,
        seed: int = 0,
    ) -> None:
        self.tokenizer = tokenizer
        self.prompt_text = prompt_text
        self.target = target
        self.bait = list(bait or [])
        self.size = tokenizer.vocab.size + extra_logits
        self.rng = np.random.default_rng(seed)

    def get_logits_from_input_ids(self, input_ids: List[int]) -> List[float]:
        text = self.tokenizer.decode(input_ids)
        produced = text[len(self.prompt_text):]
        logits = self.rng.uniform(0.0, 1.0, size=self.size)
        if self.target.startswith(produced):
            remaining = self.target[len(produced):]
            wanted = self.tokenizer.encode(remaining)[:1]
            for token_id in wanted:
                logits[token_id] += 10.0
        for token_id in self.bait:
            logits[token_id] += 50.0
        return [float(x) for x in logits]


class RandomModel:
    """A fake LLM with meaningless logits (worst case for validity)."""

    def __init__(self, size: int, seed: int) -> None:
        self.size = size
        self.rng = np.random.default_rng(seed)

    def get_logits_from_input_ids(self, input_ids: List[int]) -> List[float]:
        return [float(x) for x in self.rng.normal(size=self.size)]


def make_catalog(*functions: Dict[str, Any]) -> FunctionCatalog:
    """Build a catalog from ``(name, {param: type})`` style dicts."""
    return FunctionCatalog.model_validate({"functions": list(functions)})


def function(name: str, /, **params: str) -> Dict[str, Any]:
    """Describe a function whose parameters have the given types."""
    return {
        "name": name,
        "description": "",
        "parameters": {key: {"type": kind} for key, kind in params.items()},
    }
