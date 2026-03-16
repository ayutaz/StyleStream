"""Tests for text normalization and character-level CTC tokenization.

All tests exercise ``stylestream.data.text`` and require no external data.
"""

from __future__ import annotations

import pytest

from stylestream.data.text import (
    VOCAB_SIZE,
    CharTokenizer,
    build_vocabulary,
    normalize_text,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def tok() -> CharTokenizer:
    """Return a fresh tokenizer instance."""
    return CharTokenizer()


# ---------------------------------------------------------------------------
# 1. Vocabulary
# ---------------------------------------------------------------------------


class TestVocabulary:
    def test_vocab_size(self, tok: CharTokenizer) -> None:
        """Vocab size should be 30 (matching offline.yaml asr_decoder.vocab_size)."""
        assert tok.vocab_size == VOCAB_SIZE
        assert tok.vocab_size == 30

    def test_special_token_ids(self, tok: CharTokenizer) -> None:
        """Special tokens must occupy the first four indices."""
        assert tok.blank_id == 0
        assert tok.sos_id == 1
        assert tok.eos_id == 2
        assert tok.token_to_id["<space>"] == 3

    def test_letter_range(self, tok: CharTokenizer) -> None:
        """Letters a-z should occupy indices 4-29."""
        assert tok.token_to_id["a"] == 4
        assert tok.token_to_id["z"] == 29

    def test_build_vocabulary_helper(self) -> None:
        """The standalone ``build_vocabulary`` should return a 30-entry dict."""
        vocab = build_vocabulary()
        assert isinstance(vocab, dict)
        assert len(vocab) == 30


# ---------------------------------------------------------------------------
# 2. Text normalization
# ---------------------------------------------------------------------------


class TestNormalization:
    def test_normalize_case(self, tok: CharTokenizer) -> None:
        """Text should be lowercased."""
        assert tok.normalize_text("HELLO") == "hello"
        assert tok.normalize_text("HeLLo WoRLd") == "hello world"

    def test_normalize_punctuation(self, tok: CharTokenizer) -> None:
        """Punctuation should be removed."""
        assert tok.normalize_text("Hello, World!") == "hello world"
        assert tok.normalize_text('Say "hi"...') == "say hi"
        assert tok.normalize_text("it's a test.") == "its a test"

    def test_normalize_multiple_spaces(self, tok: CharTokenizer) -> None:
        """Multiple spaces should collapse to a single space."""
        assert tok.normalize_text("hello   world") == "hello world"
        assert tok.normalize_text("  leading and trailing  ") == "leading and trailing"

    def test_normalize_non_alpha(self, tok: CharTokenizer) -> None:
        """Digits and other non-alpha characters should be removed."""
        assert tok.normalize_text("hello123world") == "helloworld"
        assert tok.normalize_text("test@#$%") == "test"

    def test_normalize_empty_string(self, tok: CharTokenizer) -> None:
        """Empty or whitespace-only input should yield empty string."""
        assert tok.normalize_text("") == ""
        assert tok.normalize_text("   ") == ""

    def test_standalone_normalize(self) -> None:
        """The standalone ``normalize_text`` function should behave identically."""
        assert normalize_text("Hello, World!") == "hello world"
        assert normalize_text("ABC  123") == "abc"


# ---------------------------------------------------------------------------
# 3. Encoding
# ---------------------------------------------------------------------------


class TestEncode:
    def test_encode_simple(self, tok: CharTokenizer) -> None:
        """Basic encoding: 'hello world' -> known IDs."""
        ids = tok.encode("hello world")
        # h=11, e=8, l=15, l=15, o=18, <space>=3, w=26, o=18, r=21, l=15, d=7
        assert ids == [11, 8, 15, 15, 18, 3, 26, 18, 21, 15, 7]

    def test_encode_with_special(self, tok: CharTokenizer) -> None:
        """Encoding with SOS/EOS wraps the token sequence."""
        ids = tok.encode("hi", add_special=True)
        # <sos>=1, h=11, i=12, <eos>=2
        assert ids == [1, 11, 12, 2]

    def test_encode_normalizes(self, tok: CharTokenizer) -> None:
        """Encoding should normalize before tokenizing."""
        assert tok.encode("Hello World") == tok.encode("hello world")
        assert tok.encode("Hello, World!") == tok.encode("hello world")

    def test_unknown_chars_skipped(self, tok: CharTokenizer) -> None:
        """Numbers, accented chars etc. should be silently skipped."""
        ids = tok.encode("abc123def")
        # Only a, b, c, d, e, f remain.
        assert ids == [4, 5, 6, 7, 8, 9]

    def test_encode_empty(self, tok: CharTokenizer) -> None:
        """Encoding an empty string returns an empty list."""
        assert tok.encode("") == []

    def test_encode_empty_with_special(self, tok: CharTokenizer) -> None:
        """Even for empty text, add_special should produce [SOS, EOS]."""
        assert tok.encode("", add_special=True) == [1, 2]


# ---------------------------------------------------------------------------
# 4. Decoding
# ---------------------------------------------------------------------------


class TestDecode:
    def test_decode_roundtrip(self, tok: CharTokenizer) -> None:
        """encode then decode should return normalized text."""
        original = "Hello World"
        ids = tok.encode(original)
        decoded = tok.decode(ids, collapse_blanks=False)
        assert decoded == "hello world"

    def test_decode_removes_special(self, tok: CharTokenizer) -> None:
        """decode with remove_special should strip <blank>/<sos>/<eos>."""
        ids = [1, 11, 12, 2]  # <sos> h i <eos>
        assert tok.decode(ids) == "hi"

    def test_decode_keeps_special(self, tok: CharTokenizer) -> None:
        """decode with remove_special=False should keep special tokens."""
        ids = [1, 11, 12, 2]  # <sos> h i <eos>
        decoded = tok.decode(ids, remove_special=False, collapse_blanks=False)
        assert "<sos>" in decoded
        assert "<eos>" in decoded


# ---------------------------------------------------------------------------
# 5. CTC decoding
# ---------------------------------------------------------------------------


class TestCTCDecode:
    def test_ctc_decode_collapse_and_blank(self, tok: CharTokenizer) -> None:
        """CTC decode should collapse repeats and remove blanks."""
        # Simulate CTC output for "hi":
        # h h h <blank> i i -> collapsed: h <blank> i -> remove blank: h i -> "hi"
        ids = [11, 11, 11, 0, 12, 12]
        assert tok.ctc_decode(ids) == "hi"

    def test_ctc_decode_blank_between_same(self, tok: CharTokenizer) -> None:
        """Blanks between identical characters should keep both."""
        # l <blank> l -> after collapse (all unique already) -> remove blank -> "ll"
        ids = [15, 0, 15]
        assert tok.ctc_decode(ids) == "ll"

    def test_ctc_decode_with_space(self, tok: CharTokenizer) -> None:
        """Spaces should survive CTC decoding."""
        # h i <space> <space> t -> collapse -> h i <space> t -> "hi t"
        ids = [11, 12, 3, 3, 23]
        assert tok.ctc_decode(ids) == "hi t"

    def test_ctc_decode_empty(self, tok: CharTokenizer) -> None:
        """CTC decode of empty list returns empty string."""
        assert tok.ctc_decode([]) == ""

    def test_ctc_decode_all_blanks(self, tok: CharTokenizer) -> None:
        """A sequence of only blanks should decode to empty string."""
        assert tok.ctc_decode([0, 0, 0]) == ""


# ---------------------------------------------------------------------------
# 6. Batch encoding
# ---------------------------------------------------------------------------


class TestBatchEncode:
    def test_batch_encode_lengths(self, tok: CharTokenizer) -> None:
        """Batch encoding should return correct lengths."""
        texts = ["hi", "hello", "a"]
        ids_list, lengths = tok.batch_encode(texts)

        assert len(ids_list) == 3
        assert len(lengths) == 3
        assert lengths == [2, 5, 1]

    def test_batch_encode_values(self, tok: CharTokenizer) -> None:
        """Each element in batch should match individual encode."""
        texts = ["abc", "hello world"]
        ids_list, lengths = tok.batch_encode(texts)

        for text, ids, length in zip(texts, ids_list, lengths):
            expected = tok.encode(text)
            assert ids == expected
            assert length == len(expected)

    def test_batch_encode_with_special(self, tok: CharTokenizer) -> None:
        """Batch with add_special should add SOS/EOS to every entry."""
        texts = ["hi", "ok"]
        ids_list, lengths = tok.batch_encode(texts, add_special=True)

        for ids in ids_list:
            assert ids[0] == tok.sos_id
            assert ids[-1] == tok.eos_id

        # "hi" -> [SOS, h, i, EOS] = length 4
        assert lengths[0] == 4

    def test_batch_encode_empty_list(self, tok: CharTokenizer) -> None:
        """Encoding an empty list of texts should return empty lists."""
        ids_list, lengths = tok.batch_encode([])
        assert ids_list == []
        assert lengths == []
