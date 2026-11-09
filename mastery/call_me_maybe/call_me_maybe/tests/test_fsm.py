"""Tests for the automata."""

from typing import Optional

import pytest

from src.fsm import NumberFSM, NumberState, PrefixTrie, StringFSM


def _read(fsm: NumberFSM, text: str) -> Optional[NumberState]:
    state, used = fsm.scan(NumberState.START, text)
    return state if used == len(text) else None


@pytest.mark.parametrize(
    "text", ["0", "7", "42", "-1", "-0", "3.14", "-3.14", "0.5", "10.25"]
)
def test_valid_numbers(text: str) -> None:
    fsm = NumberFSM()
    state = _read(fsm, text)
    assert state is not None and fsm.is_complete(state)


@pytest.mark.parametrize(
    "text", ["", "-", "1.", "-.", ".5", "00", "01", "--1", "1.2.3", "+1",
             "1e5", "²", "١"]
)
def test_invalid_or_incomplete_numbers(text: str) -> None:
    fsm = NumberFSM()
    state = _read(fsm, text)
    assert state is None or not fsm.is_complete(state)


def test_integers_reject_fraction() -> None:
    fsm = NumberFSM(allow_fraction=False)
    assert _read(fsm, "42") is not None
    assert _read(fsm, "4.") is None


def test_scan_stops_at_the_first_foreign_character() -> None:
    state, used = NumberFSM().scan(NumberState.START, "12,")
    assert (state, used) == (NumberState.INT, 2)


@pytest.mark.parametrize(
    "text,expected", [("", "0"), ("-", "-0"), ("3.", "3.0"), ("12", "12")]
)
def test_finish_makes_numbers_valid(text: str, expected: str) -> None:
    fsm = NumberFSM()
    state, used = fsm.scan(NumberState.START, text)
    assert used == len(text)
    assert fsm.finish(state, text) == expected


def test_string_plain_and_unicode() -> None:
    scan = StringFSM().scan("héllo 中 ✓")
    assert scan is not None
    assert (scan.chunk, scan.closed_at, scan.escaping) == (
        "héllo 中 ✓", None, False
    )


def test_string_close_quote_ignores_what_follows() -> None:
    scan = StringFSM().scan('abc", "b": \x01')
    assert scan is not None
    assert (scan.chunk, scan.closed_at) == ("abc", 3)


def test_string_escapes_are_resolved() -> None:
    scan = StringFSM().scan('say \\"hi\\"\\n\\\\')
    assert scan is not None
    assert scan.chunk == 'say "hi"\n\\'
    assert scan.closed_at is None


def test_escape_can_span_tokens() -> None:
    fsm = StringFSM()
    first = fsm.scan("a\\")
    assert first is not None and first.escaping and first.chunk == "a"
    second = fsm.scan('"b', escaping=True)
    assert second is not None
    assert (second.chunk, second.closed_at, second.escaping) == (
        '"b', None, False
    )


@pytest.mark.parametrize(
    "text", ["a\nb", "tab\t", "\x00", "bad\\x", "\\u0041"]
)
def test_string_rejects_invalid_content(text: str) -> None:
    assert StringFSM().scan(text) is None


def test_trie_walk_and_alphabet() -> None:
    trie = PrefixTrie.from_words(["fn_add", "fn_add_numbers", "fn_greet"])
    root = trie.root
    assert root.size == 3
    node = trie.advance(root, "fn_add")
    assert node is not None and node.terminal and node.size == 2
    assert trie.advance(root, "fn_x") is None
    assert trie.alphabet() == set("fn_addnumbersgret")


def test_trie_unique_completion_is_the_early_commit() -> None:
    trie = PrefixTrie.from_words(["fn_add_numbers", "fn_greet"])
    assert trie.unique_completion(trie.root) is None
    greet = trie.advance(trie.root, "fn_g")
    assert greet is not None
    assert trie.unique_completion(greet) == "reet"
    done = trie.advance(trie.root, "fn_greet")
    assert done is not None and trie.unique_completion(done) == ""


def test_trie_ambiguity_when_one_word_prefixes_another() -> None:
    trie = PrefixTrie.from_words(["fn_add", "fn_add_numbers"])
    node = trie.advance(trie.root, "fn_add")
    assert node is not None
    assert trie.unique_completion(node) is None


def test_trie_single_word_commits_immediately() -> None:
    trie = PrefixTrie.from_words(["only"])
    assert trie.unique_completion(trie.root) == "only"


def test_trie_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        PrefixTrie.from_words([])
    with pytest.raises(ValueError):
        PrefixTrie.from_words(["ok", ""])
