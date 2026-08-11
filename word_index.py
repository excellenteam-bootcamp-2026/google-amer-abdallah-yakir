"""Build the word inverted index and word N-gram index."""

from array import array
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Set

from data_loader import SentenceRecord


@dataclass
class WordIndex:
    """
    Contains the word indexes.

    word_to_id:
        Maps every unique word to its numeric ID.

    id_to_word:
        Maps a word ID back to its original word.

    sentence_ids_by_word:
        sentence_ids_by_word[word_id] contains the IDs of all
        sentences containing that word.

    word_ids_by_ngram:
        Maps a character N-gram to the IDs of words containing it.
    """

    word_to_id: Dict[str, int]
    id_to_word: List[str]
    sentence_ids_by_word: List[array]
    word_ids_by_ngram: Dict[str, array]

    def get_sentence_ids(self, word: str) -> List[int]:
        """Return all sentence IDs containing an exact word."""

        word_id = self.word_to_id.get(word)

        if word_id is None:
            return []

        return list(self.sentence_ids_by_word[word_id])

    def get_word(self, word_id: int) -> str:
        """Return a word from its ID."""

        return self.id_to_word[word_id]


def create_character_ngrams(
    text: str,
    ngram_size: int = 3,
) -> Set[str]:
    """
    Create unique character N-grams from a word.

    Example:
        python -> {"pyt", "yth", "tho", "hon"}

    Words shorter than the N-gram size are stored as one complete
    smaller N-gram.

    Example:
        to -> {"to"}
    """

    if ngram_size <= 0:
        raise ValueError("ngram_size must be greater than zero")

    if not text:
        return set()

    if len(text) <= ngram_size:
        return {text}

    return {
        text[position : position + ngram_size]
        for position in range(len(text) - ngram_size + 1)
    }


def build_word_index(
    records: Iterable[SentenceRecord],
    ngram_size: int = 3,
) -> WordIndex:
    """
    Build the word inverted index while sentence records are being read.

    The function does not first load every sentence into another list.
    It consumes the records generator one sentence at a time.
    """

    word_to_id: Dict[str, int] = {}
    id_to_word: List[str] = []

    # One posting list for every word ID.
    sentence_ids_by_word: List[array] = []

    # array("I") stores unsigned integers using less memory
    # than an ordinary Python list of integer objects.
    word_ids_by_ngram = defaultdict(
        lambda: array("I")
    )

    for record in records:
        words = record.normalized_sentence.split()

        # dict.fromkeys removes repeated words while preserving order.
        # A sentence ID must appear only once for each word.
        unique_words = dict.fromkeys(words)

        for word in unique_words:
            word_id = word_to_id.get(word)

            # The word has never appeared before.
            if word_id is None:
                word_id = len(id_to_word)

                word_to_id[word] = word_id
                id_to_word.append(word)
                sentence_ids_by_word.append(array("I"))

                # Generate N-grams only when the word is new.
                word_ngrams = create_character_ngrams(
                    word,
                    ngram_size,
                )

                for ngram in word_ngrams:
                    word_ids_by_ngram[ngram].append(word_id)

            # Add the current sentence ID to the word posting list.
            sentence_ids_by_word[word_id].append(
                record.sentence_id
            )

    return WordIndex(
        word_to_id=word_to_id,
        id_to_word=id_to_word,
        sentence_ids_by_word=sentence_ids_by_word,
        word_ids_by_ngram=dict(word_ids_by_ngram),
    )