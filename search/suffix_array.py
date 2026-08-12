"""Suffix-array candidate retrieval for normalized sentences."""

from bisect import bisect_right
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, order=True)
class Occurrence:
    """One exact pattern occurrence in a normalized sentence."""

    sentence_id: int
    match_offset: int


class SuffixArrayIndex:
    """One corpus and rank-doubling suffix array built during initialization."""

    separator = "\0"

    def __init__(self, normalized_sentences: Sequence[str]) -> None:
        self.normalized_sentences = list(normalized_sentences)
        if any(self.separator in sentence for sentence in self.normalized_sentences):
            raise ValueError("A normalized sentence contains the NUL separator")

        # NUL prevents matches spanning sentence boundaries. Sentence starts let
        # us map corpus positions without allocating metadata per character.
        self.sentence_starts: list[int] = []
        corpus_parts: list[str] = []
        next_start = 0
        for sentence in self.normalized_sentences:
            self.sentence_starts.append(next_start)
            corpus_parts.append(sentence)
            next_start += len(sentence) + 1

        self.corpus = self.separator.join(corpus_parts)
        self.suffix_array = self._build_rank_doubling()

    def _build_rank_doubling(self) -> list[int]:
        """Sort suffix positions using integer ranks for 1, 2, 4, ... chars."""
        corpus_length = len(self.corpus)
        if corpus_length == 0:
            return []

        suffix_array = list(range(corpus_length))
        ranks = [ord(character) for character in self.corpus]
        prefix_length = 1

        while prefix_length < corpus_length:
            suffix_array.sort(
                key=lambda position: (
                    ranks[position],
                    ranks[position + prefix_length]
                    if position + prefix_length < corpus_length
                    else -1,
                )
            )

            new_ranks = [0] * corpus_length
            rank = 0
            previous = suffix_array[0]
            for position in suffix_array[1:]:
                previous_pair = (
                    ranks[previous],
                    ranks[previous + prefix_length]
                    if previous + prefix_length < corpus_length
                    else -1,
                )
                current_pair = (
                    ranks[position],
                    ranks[position + prefix_length]
                    if position + prefix_length < corpus_length
                    else -1,
                )
                if current_pair != previous_pair:
                    rank += 1
                new_ranks[position] = rank
                previous = position

            ranks = new_ranks
            if rank == corpus_length - 1:
                break
            prefix_length *= 2

        return suffix_array

    def corpus_position_to_sentence(self, position: int) -> tuple[int, int] | None:
        """Map a corpus position to sentence ID and normalized match offset."""
        if not 0 <= position < len(self.corpus):
            raise IndexError("Corpus position is out of range")

        sentence_id = bisect_right(self.sentence_starts, position) - 1
        match_offset = position - self.sentence_starts[sentence_id]
        if match_offset >= len(self.normalized_sentences[sentence_id]):
            return None
        return sentence_id, match_offset

    def find_exact_occurrences(self, normalized_pattern: str) -> list[Occurrence]:
        """Find exact occurrences through binary search over sorted suffixes."""
        if not normalized_pattern or self.separator in normalized_pattern:
            return []

        pattern_length = len(normalized_pattern)

        def prefix_at(suffix_index: int) -> str:
            position = self.suffix_array[suffix_index]
            return self.corpus[position : position + pattern_length]

        low, high = 0, len(self.suffix_array)
        while low < high:
            middle = (low + high) // 2
            if prefix_at(middle) < normalized_pattern:
                low = middle + 1
            else:
                high = middle
        first = low

        low, high = first, len(self.suffix_array)
        while low < high:
            middle = (low + high) // 2
            if prefix_at(middle) <= normalized_pattern:
                low = middle + 1
            else:
                high = middle

        occurrences: set[Occurrence] = set()
        for suffix_index in range(first, low):
            mapped = self.corpus_position_to_sentence(
                self.suffix_array[suffix_index]
            )
            if mapped is not None:
                occurrences.add(Occurrence(*mapped))
        return sorted(occurrences)

    def find_exact_sentence_ids(self, normalized_pattern: str) -> set[int]:
        return {
            occurrence.sentence_id
            for occurrence in self.find_exact_occurrences(normalized_pattern)
        }

    def find_approximate_candidates(self, normalized_query: str) -> set[int]:
        """Return a conservative candidate superset for at most one edit."""
        if len(normalized_query) < 3:
            # One-character anchors cannot safely or usefully reduce candidates.
            return set(range(len(self.normalized_sentences)))

        split = len(normalized_query) // 2
        left = normalized_query[:split]
        right = normalized_query[split:]

        # One edit can affect one anchor, so the unaffected anchor is preserved.
        # UNION is required here; intersection would introduce false negatives.
        return self.find_exact_sentence_ids(left) | self.find_exact_sentence_ids(
            right
        )
