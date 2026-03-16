"""Text normalization and character-level CTC tokenization for the Destylizer ASR loss.

The Destylizer uses CTC loss to ensure content features preserve linguistic information.
This module provides character-level tokenization matching the vocabulary size of 30
defined in ``configs/destylizer/offline.yaml`` (``asr_decoder.vocab_size``).

Vocabulary layout (30 tokens)::

    0  <blank>   CTC blank
    1  <sos>     Start of sequence
    2  <eos>     End of sequence
    3  <space>   Word separator
    4-29  a-z    26 lowercase English letters

Reference: StyleStream paper, Section 10.8.
"""

from __future__ import annotations

import re
import string

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BLANK = "<blank>"  # CTC blank token (index 0)
SOS = "<sos>"  # Start of sequence (index 1)
EOS = "<eos>"  # End of sequence (index 2)
SPACE = "<space>"  # Word separator (index 3)

PAD = -1  # Padding value for batched sequences

VOCAB_SIZE = 30

# Pre-compiled regex patterns for normalization
_PUNCTUATION_RE = re.compile(r"[" + re.escape(string.punctuation) + r"]")
_MULTI_SPACE_RE = re.compile(r"\s+")
_NON_ALPHA_SPACE_RE = re.compile(r"[^a-z ]")


# ---------------------------------------------------------------------------
# CharTokenizer
# ---------------------------------------------------------------------------


class CharTokenizer:
    """Character-level tokenizer for English ASR with CTC.

    Vocabulary (30 tokens)::

        0: <blank>  (CTC blank)
        1: <sos>    (start of sequence)
        2: <eos>    (end of sequence)
        3: <space>  (word separator)
        4-29: a-z   (26 lowercase letters)
    """

    def __init__(self) -> None:
        self.vocab = self._build_vocab()
        self.token_to_id: dict[str, int] = {t: i for i, t in enumerate(self.vocab)}
        self.id_to_token: dict[int, str] = {i: t for i, t in enumerate(self.vocab)}

    # -- vocabulary --------------------------------------------------------

    def _build_vocab(self) -> list[str]:
        """Build the vocabulary list.

        Returns:
            A list of 30 tokens in canonical order.
        """
        special = [BLANK, SOS, EOS, SPACE]
        letters = [chr(ord("a") + i) for i in range(26)]
        return special + letters

    @property
    def vocab_size(self) -> int:
        """Number of tokens in the vocabulary (30)."""
        return len(self.vocab)

    @property
    def blank_id(self) -> int:
        """Token id for the CTC blank."""
        return 0

    @property
    def sos_id(self) -> int:
        """Token id for start-of-sequence."""
        return 1

    @property
    def eos_id(self) -> int:
        """Token id for end-of-sequence."""
        return 2

    # -- normalization -----------------------------------------------------

    def normalize_text(self, text: str) -> str:
        """Normalize raw text for tokenization.

        Steps:
            1. Convert to lowercase.
            2. Remove punctuation (``.,!?;:"-'()[]`` etc.).
            3. Collapse multiple whitespace characters into a single space.
            4. Strip leading/trailing whitespace.
            5. Remove any remaining non-alpha-space characters.
        """
        text = text.lower()
        text = _PUNCTUATION_RE.sub("", text)
        text = _MULTI_SPACE_RE.sub(" ", text)
        text = text.strip()
        text = _NON_ALPHA_SPACE_RE.sub("", text)
        # Re-collapse and strip in case removing characters introduced new
        # adjacent or trailing spaces (e.g. "abc 123 def" -> "abc  def").
        text = _MULTI_SPACE_RE.sub(" ", text).strip()
        return text

    # -- encoding ----------------------------------------------------------

    def encode(self, text: str, add_special: bool = False) -> list[int]:
        """Convert raw text to a list of token IDs.

        The text is normalized first (lowercased, punctuation removed, etc.).
        Characters not in the vocabulary (digits, accented characters, ...) are
        silently skipped.

        Args:
            text: Raw text string.
            add_special: If ``True``, prepend ``<sos>`` and append ``<eos>``.

        Returns:
            List of integer token IDs.

        Example::

            >>> tok = CharTokenizer()
            >>> tok.encode("Hello World")
            [11, 8, 15, 15, 18, 3, 26, 18, 21, 15, 7]
        """
        normalized = self.normalize_text(text)
        ids: list[int] = []
        for ch in normalized:
            if ch == " ":
                ids.append(self.token_to_id[SPACE])
            elif ch in self.token_to_id:
                ids.append(self.token_to_id[ch])
            # Unknown characters are silently skipped.

        if add_special:
            ids = [self.sos_id] + ids + [self.eos_id]
        return ids

    # -- decoding ----------------------------------------------------------

    def decode(
        self,
        ids: list[int],
        remove_special: bool = True,
        collapse_blanks: bool = True,
    ) -> str:
        """Convert token IDs back to text.

        Args:
            ids: List of token IDs.
            remove_special: Remove ``<blank>``, ``<sos>``, ``<eos>`` tokens.
            collapse_blanks: Collapse consecutive identical IDs (CTC decoding
                behaviour) before converting to text.

        Returns:
            Decoded text string.
        """
        if collapse_blanks:
            collapsed: list[int] = []
            prev = -1
            for tid in ids:
                if tid != prev:
                    collapsed.append(tid)
                prev = tid
            ids = collapsed

        special_ids = {self.blank_id, self.sos_id, self.eos_id}
        parts: list[str] = []
        for tid in ids:
            if remove_special and tid in special_ids:
                continue
            token = self.id_to_token.get(tid)
            if token is None:
                continue
            if token == SPACE:
                parts.append(" ")
            else:
                parts.append(token)
        return "".join(parts)

    # -- CTC greedy decode -------------------------------------------------

    def ctc_decode(self, ids: list[int]) -> str:
        """CTC greedy decoding: collapse repeated tokens then remove blanks.

        This is a convenience wrapper equivalent to
        ``decode(ids, remove_special=True, collapse_blanks=True)``.

        Args:
            ids: Raw CTC output token IDs (may contain repeats and blanks).

        Returns:
            Decoded text string.
        """
        return self.decode(ids, remove_special=True, collapse_blanks=True)

    # -- batch utilities ---------------------------------------------------

    def batch_encode(
        self,
        texts: list[str],
        add_special: bool = False,
    ) -> tuple[list[list[int]], list[int]]:
        """Encode a batch of texts.

        No padding is applied; the caller is responsible for padding if needed.

        Args:
            texts: List of raw text strings.
            add_special: If ``True``, prepend ``<sos>`` and append ``<eos>``
                to each encoded sequence.

        Returns:
            A tuple ``(token_ids_list, lengths)`` where *token_ids_list* is a
            list of integer-ID lists and *lengths* holds the length of each
            encoded sequence.
        """
        encoded = [self.encode(t, add_special=add_special) for t in texts]
        lengths = [len(e) for e in encoded]
        return encoded, lengths


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """Standalone text normalization (convenience wrapper).

    Delegates to :meth:`CharTokenizer.normalize_text`.
    """
    return CharTokenizer().normalize_text(text)


def build_vocabulary() -> dict[str, int]:
    """Build and return the character vocabulary mapping.

    Returns:
        A ``{token: id}`` dictionary with 30 entries.
    """
    return dict(CharTokenizer().token_to_id)
