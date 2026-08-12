"""Production adapter for the packed native C++ suffix-array index."""

from typing import Sequence

from . import _suffix_array_cpp


class NativeSuffixArrayIndex:
    separator = "\0"

    def __init__(self, normalized_sentences: Sequence[str]) -> None:
        self.normalized_sentences = list(normalized_sentences)
        if any(self.separator in sentence for sentence in self.normalized_sentences):
            raise ValueError("A normalized sentence contains the NUL separator")
        self._native = _suffix_array_cpp.build(self.normalized_sentences)

    @property
    def corpus(self) -> str:
        return _suffix_array_cpp.corpus(self._native)

    @property
    def sentence_count(self) -> int:
        return _suffix_array_cpp.sentence_count(self._native)

    @property
    def suffix_count(self) -> int:
        return _suffix_array_cpp.suffix_count(self._native)

    @property
    def native_memory_bytes(self) -> tuple[int, int]:
        """Return persistent and estimated peak native allocation bytes."""
        return _suffix_array_cpp.memory_bytes(self._native)

    def suffix_at(self, index: int) -> int:
        """Expose one position for correctness tests without materializing the SA."""
        return _suffix_array_cpp.suffix_at(self._native, index)

    def find_exact_sentence_ids(self, normalized_pattern: str) -> set[int]:
        if not normalized_pattern or self.separator in normalized_pattern:
            return set()
        return _suffix_array_cpp.exact_sentence_ids(self._native, normalized_pattern)

    def find_approximate_candidates(self, normalized_query: str) -> set[int]:
        if len(normalized_query) < 3:
            return set(range(self.sentence_count))
        split = len(normalized_query) // 2
        return self.find_exact_sentence_ids(
            normalized_query[:split]
        ) | self.find_exact_sentence_ids(normalized_query[split:])
