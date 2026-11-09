"""Finite-state machines that say which text is valid at each point.

These automata know nothing about the LLM or the tokenizer: they only
answer "may this text come next?". The constrained decoder combines them
with the vocabulary to build the set of valid tokens at every step.

* :class:`PrefixTrie`: a fixed set of literals (function names, ``true`` /
  ``false``). It is the language-level equivalent of a prefix tree of the
  allowed strings.
* :class:`NumberFSM`: JSON numbers (RFC 8259, without exponent).
* :class:`StringFSM`: the content of a JSON string, with escapes.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Iterable, List, Optional, Set, Tuple

from pydantic import BaseModel, ConfigDict, Field

_DIGITS = "0123456789"
_NONZERO_DIGITS = "123456789"


# --------------------------------------------------------------------------
# Literals
# --------------------------------------------------------------------------
class TrieNode(BaseModel):
    """One node of a :class:`PrefixTrie`.

    Attributes:
        children: Next node for each possible next character.
        terminal: Whether the path to this node spells a complete literal.
        size: How many complete literals live in the subtree of this node
            (this node included).
    """

    children: Dict[str, TrieNode] = Field(default_factory=dict)
    terminal: bool = False
    size: int = 0


TrieNode.model_rebuild()


class PrefixTrie(BaseModel):
    """Prefix tree over a fixed, non-empty set of literals."""

    root: TrieNode

    @classmethod
    def from_words(cls, words: Iterable[str]) -> PrefixTrie:
        """Build the trie of ``words``.

        Raises:
            ValueError: If there are no words or one of them is empty.
        """
        root = TrieNode()
        for word in dict.fromkeys(words):
            if not word:
                raise ValueError("a literal cannot be the empty string")
            node = root
            node.size += 1
            for char in word:
                node = node.children.setdefault(char, TrieNode())
                node.size += 1
            node.terminal = True
        if root.size == 0:
            raise ValueError("a trie needs at least one literal")
        return cls(root=root)

    def advance(self, node: TrieNode, text: str) -> Optional[TrieNode]:
        """Follow ``text`` from ``node``; ``None`` if it leaves the trie."""
        for char in text:
            child = node.children.get(char)
            if child is None:
                return None
            node = child
        return node

    def unique_completion(self, node: TrieNode) -> Optional[str]:
        """Return the only way to finish a literal from ``node``.

        This is the *early commit*: when exactly one literal is still
        reachable, the rest of it is known and needs no LLM call. The
        result is ``""`` when ``node`` already spells that literal, and
        ``None`` when several literals are still possible.
        """
        if node.size != 1:
            return None
        suffix: List[str] = []
        while not node.terminal:
            char, node = next(iter(node.children.items()))
            suffix.append(char)
        return "".join(suffix)

    def alphabet(self) -> Set[str]:
        """Return every character that appears in some literal."""
        chars: Set[str] = set()
        pending = [self.root]
        while pending:
            node = pending.pop()
            for char, child in node.children.items():
                chars.add(char)
                pending.append(child)
        return chars


# --------------------------------------------------------------------------
# Numbers
# --------------------------------------------------------------------------
class NumberState(Enum):
    """States of the JSON number automaton."""

    START = "start"
    MINUS = "minus"
    ZERO = "zero"
    INT = "int"
    DOT = "dot"
    FRAC = "frac"


_ACCEPTING_NUMBER_STATES = frozenset(
    {NumberState.ZERO, NumberState.INT, NumberState.FRAC}
)


class NumberFSM(BaseModel):
    """Automaton for ``-?(0|[1-9][0-9]*)(\\.[0-9]+)?``.

    With ``allow_fraction=False`` the fractional part is rejected, which
    gives the JSON integers.
    """

    model_config = ConfigDict(frozen=True)

    allow_fraction: bool = True

    def step(self, state: NumberState, char: str) -> Optional[NumberState]:
        """Return the state after reading ``char``, or ``None``."""
        if state is NumberState.START:
            if char == "-":
                return NumberState.MINUS
            return self._first_digit(char)
        if state is NumberState.MINUS:
            return self._first_digit(char)
        if state is NumberState.ZERO:
            return self._dot(char)
        if state is NumberState.INT:
            if char in _DIGITS:
                return NumberState.INT
            return self._dot(char)
        # DOT and FRAC both need (more) digits.
        if char in _DIGITS:
            return NumberState.FRAC
        return None

    def scan(self, state: NumberState, text: str) -> Tuple[NumberState, int]:
        """Read as much of ``text`` as the automaton accepts.

        Returns:
            The state reached and the number of characters consumed. If
            fewer than ``len(text)`` were consumed, the next character
            cannot extend the number.
        """
        consumed = 0
        for char in text:
            following = self.step(state, char)
            if following is None:
                break
            state = following
            consumed += 1
        return state, consumed

    def is_complete(self, state: NumberState) -> bool:
        """Whether the text read so far is a whole, valid number."""
        return state in _ACCEPTING_NUMBER_STATES

    def finish(self, state: NumberState, text: str) -> str:
        """Complete an unfinished number so it becomes valid.

        ``"-"`` becomes ``"-0"``, ``"3."`` becomes ``"3.0"`` and the empty
        text becomes ``"0"``. Already valid numbers are returned as is.
        """
        if self.is_complete(state):
            return text
        return text + "0"

    @staticmethod
    def _first_digit(char: str) -> Optional[NumberState]:
        if char == "0":
            return NumberState.ZERO
        if char in _NONZERO_DIGITS:
            return NumberState.INT
        return None

    def _dot(self, char: str) -> Optional[NumberState]:
        if char == "." and self.allow_fraction:
            return NumberState.DOT
        return None


# --------------------------------------------------------------------------
# Strings
# --------------------------------------------------------------------------
# ``\uXXXX`` is deliberately unsupported: the model writes real unicode
# characters directly, and a lone surrogate escape would produce text that
# cannot be saved as UTF-8.
_SIMPLE_ESCAPES: Dict[str, str] = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}


class StringScan(BaseModel):
    """Result of reading a piece of text inside a JSON string.

    Attributes:
        chunk: Decoded characters (escapes resolved) read before the
            closing quote, or before the end of the text.
        escaping: True if the text ended right after a backslash.
        closed_at: Index in the text of the closing quote, if any. Text
            after it is not examined.
    """

    model_config = ConfigDict(frozen=True)

    chunk: str
    escaping: bool
    closed_at: Optional[int] = None


class StringFSM(BaseModel):
    """Automaton for the inside of a JSON string (after the open quote)."""

    model_config = ConfigDict(frozen=True)

    def scan(self, text: str, escaping: bool = False) -> Optional[StringScan]:
        """Read ``text`` as string content.

        Args:
            text: Text produced by one token.
            escaping: True if the previous token ended with a backslash.

        Returns:
            The decoded result, or ``None`` if ``text`` is invalid there:
            a raw control character or an unsupported escape sequence.
        """
        chunk: List[str] = []
        for index, char in enumerate(text):
            if escaping:
                resolved = _SIMPLE_ESCAPES.get(char)
                if resolved is None:
                    return None
                chunk.append(resolved)
                escaping = False
            elif char == "\\":
                escaping = True
            elif char == '"':
                return StringScan(
                    chunk="".join(chunk), escaping=False, closed_at=index
                )
            elif ord(char) < 0x20:
                return None
            else:
                chunk.append(char)
        return StringScan(chunk="".join(chunk), escaping=escaping)
