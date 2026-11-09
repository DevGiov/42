"""Constrained decoding engine with jump-forward.

For one prompt the decoder writes ``{"name": "<function>", "parameters":
{...}}`` token by token, but only the *semantic* choices are made by the
LLM:

* Fixed text (``{"name": "``, ``", "parameters": {``, ``"key": ``...) is
  injected straight into the context. No forward pass is spent on it
  (jump-forward).
* The function name is chosen by masking the logits to the tokens that
  keep the text on a path of a prefix trie of the valid names. Once a
  single name remains reachable, the rest is injected (early commit).
* Each argument is generated under the automaton of its schema type.

At every LLM step all tokens that would leave the allowed language are
discarded (their logit is effectively ``-inf``) and the best remaining
one is taken (greedy). Since ``exp`` is monotonic this is the argmax of
the masked logits, so no softmax is ever computed.
"""

from __future__ import annotations

import json
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from src.errors import DecodingError
from src.fsm import (
    NumberFSM,
    NumberState,
    PrefixTrie,
    StringFSM,
    TrieNode,
)
from src.models import FunctionCatalog, ParamValue
from src.tokenizer import Tokenizer, Vocabulary

# A number ends when the model starts what comes after it in the JSON.
NUMBER_TERMINATORS = ",}\n"
_NUMBER_CHARS = "0123456789-."
_QUOTE = '"'

IdArray = npt.NDArray[np.int64]


@runtime_checkable
class LogitsModel(Protocol):
    """The part of ``Small_LLM_Model`` the decoder needs."""

    def get_logits_from_input_ids(self, input_ids: List[int]) -> List[float]:
        """Return the next-token logits for ``input_ids``."""


class DecodingLimits(BaseModel):
    """Safety caps so a stubborn model can never loop forever."""

    model_config = ConfigDict(frozen=True)

    max_number_chars: int = Field(default=30, ge=1)
    max_string_tokens: int = Field(default=128, ge=1)


class GeneratedCall(BaseModel):
    """What the decoder produced: the chosen function and its arguments."""

    model_config = ConfigDict(frozen=True)

    name: str
    parameters: Dict[str, ParamValue]


class LiteralSet(BaseModel):
    """A closed set of strings the model must pick one of.

    Attributes:
        trie: Prefix tree of the strings.
        stop_char: If set, a token starting with this character means
            "the literal ends here". Needed when one literal is a prefix
            of another (``fn_add`` / ``fn_add_numbers``).
        candidate_ids: Sorted ids of the tokens that could ever be part of
            one of the literals; everything else is masked up front.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    trie: PrefixTrie
    stop_char: Optional[str] = None
    candidate_ids: IdArray

    @classmethod
    def build(
        cls,
        words: Sequence[str],
        vocabulary: Vocabulary,
        stop_char: Optional[str] = None,
    ) -> LiteralSet:
        """Index the vocabulary once for the strings in ``words``."""
        trie = PrefixTrie.from_words(words)
        alphabet = trie.alphabet()
        ids = [
            token_id
            for token_id, text in enumerate(vocabulary.token_texts)
            if text
            and (
                set(text) <= alphabet
                or (stop_char is not None and text.startswith(stop_char))
            )
        ]
        return cls(
            trie=trie,
            stop_char=stop_char,
            candidate_ids=np.asarray(ids, dtype=np.int64),
        )

    def accepts(self, node: TrieNode, text: str) -> bool:
        """Whether ``text`` may follow the literal prefix ``node``."""
        if self.is_stop(node, text):
            return True
        return self.trie.advance(node, text) is not None

    def is_stop(self, node: TrieNode, text: str) -> bool:
        """Whether ``text`` means the literal is complete at ``node``."""
        return (
            self.stop_char is not None
            and node.terminal
            and text.startswith(self.stop_char)
        )


class ConstrainedDecoder(BaseModel):
    """Generates one function call from a prompt, token by token.

    Attributes:
        model: The LLM, seen only through its logits.
        tokenizer: Converts the injected fixed text into token ids.
        vocabulary: Text of every token id.
        catalog: Functions the model may choose from.
        limits: Safety caps.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: LogitsModel
    tokenizer: Tokenizer
    vocabulary: Vocabulary
    catalog: FunctionCatalog
    limits: DecodingLimits = Field(default_factory=DecodingLimits)

    _names: LiteralSet = PrivateAttr()
    _booleans: LiteralSet = PrivateAttr()
    _number_ids: IdArray = PrivateAttr()
    _string_ids: IdArray = PrivateAttr()
    _handlers: Dict[str, Callable[[List[int]], ParamValue]] = PrivateAttr()
    _encoded: Dict[str, List[int]] = PrivateAttr(default_factory=dict)
    _forward_passes: int = PrivateAttr(default=0)

    def model_post_init(self, __context: object) -> None:
        """Precompute the vocabulary masks once, at construction."""
        self._names = LiteralSet.build(
            self.catalog.names(), self.vocabulary, stop_char=_QUOTE
        )
        self._booleans = LiteralSet.build(["true", "false"], self.vocabulary)
        starts = _NUMBER_CHARS + NUMBER_TERMINATORS
        self._number_ids = self._select_ids(lambda t: t[0] in starts)
        self._string_ids = self._select_ids(
            lambda t: all(ord(char) >= 0x20 for char in t)
        )
        # One handler per schema type: supporting a new type means adding
        # a ``_decode_<type>`` method and one line here.
        self._handlers = {
            "number": self._decode_number,
            "integer": self._decode_integer,
            "string": self._decode_string,
            "boolean": self._decode_boolean,
        }

    @property
    def forward_passes(self) -> int:
        """Number of LLM forward passes spent so far."""
        return self._forward_passes

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_call(self, prompt_ids: Sequence[int]) -> GeneratedCall:
        """Choose a function and generate its arguments for a prompt.

        Args:
            prompt_ids: Token ids of the instruction prompt, ending right
                before the JSON object starts.

        Raises:
            DecodingError: If no token is valid at some step (for example
                a vocabulary without digits) or the model misbehaves.
        """
        context = list(prompt_ids)
        self._inject(context, '{"name": "')
        name = self._choose_literal(context, self._names)
        function = self.catalog.get(name)
        self._inject(context, '", "parameters": {')
        values: Dict[str, ParamValue] = {}
        for index, (key, spec) in enumerate(function.parameters.items()):
            separator = ", " if index else ""
            self._inject(context, f"{separator}{json.dumps(key)}: ")
            values[key] = self._handlers[spec.type](context)
        return GeneratedCall(name=name, parameters=values)

    # ------------------------------------------------------------------
    # Value generators (one per schema type)
    # ------------------------------------------------------------------
    def _decode_number(self, context: List[int]) -> ParamValue:
        return float(self._read_number(context, NumberFSM()))

    def _decode_integer(self, context: List[int]) -> ParamValue:
        return int(self._read_number(context, NumberFSM(allow_fraction=False)))

    def _decode_boolean(self, context: List[int]) -> ParamValue:
        return self._choose_literal(context, self._booleans) == "true"

    def _decode_string(self, context: List[int]) -> ParamValue:
        fsm = StringFSM()
        self._inject(context, _QUOTE)
        parts: List[str] = []
        escaping = False
        for _ in range(self.limits.max_string_tokens):
            logits = self._logits(context)
            token_id = self._best(
                logits,
                self._string_ids,
                lambda text: fsm.scan(text, escaping) is not None,
            )
            if token_id is None:
                raise DecodingError("no valid token to continue a string")
            text = self._text(token_id)
            scan = fsm.scan(text, escaping)
            assert scan is not None  # accepted by the predicate above
            parts.append(scan.chunk)
            if scan.closed_at is not None:
                # The token holds the end of the string (and maybe more
                # that belongs to the JSON after it): keep only the part
                # inside the string and let the injected text do the rest.
                self._inject(context, text[: scan.closed_at])
                break
            context.append(token_id)
            escaping = scan.escaping
        self._inject(context, _QUOTE)
        return "".join(parts)

    # ------------------------------------------------------------------
    # Building blocks
    # ------------------------------------------------------------------
    def _read_number(self, context: List[int], fsm: NumberFSM) -> str:
        """Generate the text of a number under ``fsm``."""
        state = NumberState.START
        text = ""

        def accept(token: str) -> bool:
            new_state, used = fsm.scan(state, token)
            if len(text) + used > self.limits.max_number_chars:
                return False
            if used == len(token):
                return True
            return fsm.is_complete(new_state) and (
                token[used] in NUMBER_TERMINATORS
            )

        while len(text) < self.limits.max_number_chars:
            logits = self._logits(context)
            token_id = self._best(logits, self._number_ids, accept)
            if token_id is None:
                if not text:
                    raise DecodingError("no valid token to start a number")
                break
            token = self._text(token_id)
            state, used = fsm.scan(state, token)
            text += token[:used]
            if used < len(token):
                # A delimiter follows the digits: the number is over.
                self._inject(context, token[:used])
                return text
            context.append(token_id)
        completed = fsm.finish(state, text)
        self._inject(context, completed[len(text):])
        return completed

    def _choose_literal(self, context: List[int], literals: LiteralSet) -> str:
        """Let the LLM pick one of ``literals`` (trie and early commit)."""
        trie = literals.trie
        node = trie.root
        chosen = ""
        while True:
            rest = trie.unique_completion(node)
            if rest is not None:
                self._inject(context, rest)
                return chosen + rest
            logits = self._logits(context)
            token_id = self._best(
                logits,
                literals.candidate_ids,
                lambda text: literals.accepts(node, text),
            )
            if token_id is None:
                raise DecodingError("no valid token to continue a literal")
            text = self._text(token_id)
            if literals.is_stop(node, text):
                return chosen
            following = trie.advance(node, text)
            assert following is not None  # accepted by the predicate above
            node = following
            chosen += text
            context.append(token_id)

    def _best(
        self,
        logits: npt.NDArray[np.float64],
        candidates: IdArray,
        accept: Callable[[str], bool],
    ) -> Optional[int]:
        """Return the highest-logit candidate that ``accept`` allows.

        Candidates are visited by decreasing logit, so the first one that
        passes is the argmax over the valid tokens.
        """
        usable = candidates[: np.searchsorted(candidates, len(logits))]
        if usable.size == 0:
            return None
        scores = logits[usable]
        # Fast path: most of the time the top candidate is already valid.
        top = int(usable[np.argmax(scores)])
        if accept(self._text(top)):
            return top
        for token_id in usable[np.argsort(-scores, kind="stable")]:
            if accept(self._text(int(token_id))):
                return int(token_id)
        return None

    def _logits(self, context: List[int]) -> npt.NDArray[np.float64]:
        """Run one forward pass and return the logits as an array."""
        self._forward_passes += 1
        raw = self.model.get_logits_from_input_ids(context)
        logits = np.asarray(raw, dtype=np.float64)
        if logits.ndim != 1 or logits.size == 0 or np.isnan(logits).any():
            raise DecodingError("the model returned malformed logits")
        return logits

    def _inject(self, context: List[int], text: str) -> None:
        """Append fixed ``text`` to the context without any LLM call."""
        if not text:
            return
        ids = self._encoded.get(text)
        if ids is None:
            ids = self.tokenizer.encode(text)
            self._encoded[text] = ids
        context.extend(ids)

    def _text(self, token_id: int) -> str:
        """Text of a candidate token (candidates always have text)."""
        text = self.vocabulary.token_texts[token_id]
        assert text is not None
        return text

    def _select_ids(self, keep: Callable[[str], bool]) -> IdArray:
        """Ids of the tokens with non-empty text for which ``keep`` holds."""
        ids = [
            token_id
            for token_id, text in enumerate(self.vocabulary.token_texts)
            if text and keep(text)
        ]
        return np.asarray(ids, dtype=np.int64)
