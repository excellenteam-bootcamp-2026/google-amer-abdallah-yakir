"""Build an index that maps each word to its sentence IDs."""

from array import array
from collections import defaultdict
from typing import Dict, Iterable

from data_loader import SentenceRecord


# Each word maps to an array of sentence IDs.
SentenceWordIndex = Dict[str, array]


def build_sentence_word_index(
    records: Iterable[SentenceRecord],
) -> SentenceWordIndex:
    """
    Build an inverted index:

        word -> sentence IDs containing that word

    The records are processed one sentence at a time, so the function
    can consume the generator returned by iter_sentence_records().
    """

    word_to_sentence_ids = defaultdict(
        lambda: array("I")
    )

    for record in records:
        words = record.normalized_sentence.split()

        # Remove repeated words from this sentence.
        #
        # Example:
        #     "python is a python language"
        #
        # Sentence ID should be added to "python" only once.
        unique_words = dict.fromkeys(words)

        for word in unique_words:
            word_to_sentence_ids[word].append(
                record.sentence_id
            )

    return dict(word_to_sentence_ids)