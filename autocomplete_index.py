"""Build all indexes needed by the autocomplete system."""

from array import array
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Union

from data_loader import SentenceRecord, iter_sentence_records


PathLike = Union[str, Path]


@dataclass
class AutocompleteIndex:
    """
    Contains the indexes created during initialization.

    word_to_id:
        Maps a word to its numeric ID.

    id_to_word:
        Maps a numeric word ID back to its word.

    sentence_ids_by_word:
        Uses a word ID to find all sentence IDs containing the word.

    word_ids_by_ngram:
        Uses a character N-gram to find candidate word IDs.
    """

    word_to_id: Dict[str, int]
    id_to_word: List[str]
    sentence_ids_by_word: List[array]
    word_ids_by_ngram: Dict[str, array]
    sentence_count: int

    def get_sentence_ids(self, word: str) -> array:
        """
        Return the sorted sentence IDs containing an exact word.
        """

        word_id = self.word_to_id.get(word)

        if word_id is None:
            return array("I")

        return self.sentence_ids_by_word[word_id]

    def get_word(self, word_id: int) -> str:
        """Return the word belonging to a word ID."""

        return self.id_to_word[word_id]

    def get_words_for_ngram(self, ngram: str) -> List[str]:
        """
        Return all words containing a particular N-gram.
        """

        word_ids = self.word_ids_by_ngram.get(ngram)

        if word_ids is None:
            return []

        return [
            self.id_to_word[word_id]
            for word_id in word_ids
        ]


def create_character_ngrams(
    text: str,
    ngram_size: int = 3,
) -> Set[str]:
    """
    Create unique character N-grams.

    Example:
        python -> {"pyt", "yth", "tho", "hon"}

    Short words are stored as one smaller N-gram:

        to -> {"to"}
    """

    if ngram_size <= 0:
        raise ValueError(
            "ngram_size must be greater than zero"
        )

    if not text:
        return set()

    if len(text) <= ngram_size:
        return {text}

    return {
        text[position : position + ngram_size]
        for position in range(
            len(text) - ngram_size + 1
        )
    }


def build_autocomplete_index_from_records(
    records: Iterable[SentenceRecord],
    ngram_size: int = 3,
    progress_every: Optional[int] = None,
) -> AutocompleteIndex:
    """
    Build both indexes using one pass through the records.

    The two indexes are:

        word -> sentence IDs
        N-gram -> word IDs
    """

    word_to_id: Dict[str, int] = {}
    id_to_word: List[str] = []

    # sentence_ids_by_word[word_id] gives all sentence IDs
    # containing that word.
    sentence_ids_by_word: List[array] = []

    # N-gram -> IDs of all words containing the N-gram.
    word_ids_by_ngram: Dict[str, array] = {}

    sentence_count = 0

    for sentence_count, record in enumerate(
        records,
        start=1,
    ):
        words = record.normalized_sentence.split()

        # Prevent the same sentence ID from being added twice
        # when a word appears multiple times in one sentence.
        unique_words = dict.fromkeys(words)

        for word in unique_words:
            word_id = word_to_id.get(word)

            # This word has never appeared before.
            if word_id is None:
                word_id = len(id_to_word)

                word_to_id[word] = word_id
                id_to_word.append(word)
                sentence_ids_by_word.append(
                    array("I")
                )

                # Generate N-grams only once, when the word
                # is first discovered.
                word_ngrams = create_character_ngrams(
                    word,
                    ngram_size,
                )

                for ngram in word_ngrams:
                    if ngram not in word_ids_by_ngram:
                        word_ids_by_ngram[ngram] = array(
                            "I"
                        )

                    word_ids_by_ngram[ngram].append(
                        word_id
                    )

            # Connect the word to the current sentence.
            sentence_ids_by_word[word_id].append(
                record.sentence_id
            )

        if (
            progress_every is not None
            and sentence_count % progress_every == 0
        ):
            print(
                f"Processed {sentence_count:,} sentences; "
                f"found {len(word_to_id):,} unique words."
            )

    return AutocompleteIndex(
        word_to_id=word_to_id,
        id_to_word=id_to_word,
        sentence_ids_by_word=sentence_ids_by_word,
        word_ids_by_ngram=word_ids_by_ngram,
        sentence_count=sentence_count,
    )


def initialize_autocomplete(
    archive_path: PathLike,
    ngram_size: int = 3,
    limit: Optional[int] = None,
) -> AutocompleteIndex:
    """
    Read the archive and build all current indexes.

    Set limit to a number when testing:

        limit=100_000

    Use limit=None to process the complete archive.
    """

    records = iter_sentence_records(
        archive_path
    )

    if limit is not None:
        records = islice(records, limit)

    return build_autocomplete_index_from_records(
        records=records,
        ngram_size=ngram_size,
        progress_every=100_000,
    )