"""Complete correctness-first implementation of Part A of the assignment."""

import argparse
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Union

from autocomplete_index import (
    AutocompleteIndex,
    build_autocomplete_index_from_records,
)
from data_loader import SentenceRecord, iter_sentence_records, normalize_text


PathLike = Union[str, Path]


@dataclass(frozen=True)
class AutoCompleteData:
    """One autocomplete result in the format required by the brief."""

    completed_sentence: str
    source_text: str
    offset: int
    score: int


def replacement_penalty(position: int) -> int:
    """Penalty for replacing the character at a zero-based position."""

    return max(1, 5 - position)


def add_or_delete_penalty(position: int) -> int:
    """Penalty for adding/deleting a character at a zero-based position."""

    return max(2, 10 - (2 * position))


def _best_replacement_score(query: str, sentence: str) -> Optional[int]:
    """Return the best score for exactly one replaced query character."""

    query_length = len(query)
    if query_length == 0 or len(sentence) < query_length:
        return None

    best_score = None
    base_score = 2 * (query_length - 1)

    for start in range(len(sentence) - query_length + 1):
        mismatch_position = -1

        for position in range(query_length):
            if query[position] == sentence[start + position]:
                continue

            if mismatch_position != -1:
                mismatch_position = -2
                break

            mismatch_position = position

        if mismatch_position >= 0:
            score = base_score - replacement_penalty(mismatch_position)
            if best_score is None or score > best_score:
                best_score = score

    return best_score


def _best_extra_character_score(query: str, sentence: str) -> Optional[int]:
    """Score a query that contains exactly one unnecessary character."""

    if not query:
        return None

    best_score = None
    base_score = 2 * (len(query) - 1)

    for position in range(len(query)):
        corrected_query = query[:position] + query[position + 1 :]

        if corrected_query in sentence:
            score = base_score - add_or_delete_penalty(position)
            if best_score is None or score > best_score:
                best_score = score

    return best_score


def _best_missing_character_score(query: str, sentence: str) -> Optional[int]:
    """Score a query that is missing exactly one sentence character."""

    corrected_length = len(query) + 1
    if len(sentence) < corrected_length:
        return None

    best_score = None
    base_score = 2 * len(query)

    for start in range(len(sentence) - corrected_length + 1):
        window = sentence[start : start + corrected_length]

        for position in range(corrected_length):
            if window[:position] != query[:position]:
                continue

            if window[position + 1 :] != query[position:]:
                continue

            score = base_score - add_or_delete_penalty(position)
            if best_score is None or score > best_score:
                best_score = score

    return best_score


def best_match_score(query: str, sentence: str) -> Optional[int]:
    """Return the highest valid Part A score, or ``None`` for no match.

    Both arguments must already be normalized.  A match can be an exact
    substring or a substring reached with one replacement, one added query
    character, or one missing query character.
    """

    if not query or not sentence:
        return None

    if query in sentence:
        return 2 * len(query)

    possible_scores = (
        _best_replacement_score(query, sentence),
        _best_extra_character_score(query, sentence),
        _best_missing_character_score(query, sentence),
    )

    valid_scores = [score for score in possible_scores if score is not None]
    return max(valid_scores) if valid_scores else None


def _result_order(result: AutoCompleteData):
    """Sort by descending score, then alphabetically as required."""

    return (
        -result.score,
        result.completed_sentence.casefold(),
        result.completed_sentence,
        result.source_text,
        result.offset,
    )


class AutoCompleteSystem:
    """Loaded sentences, initialization indexes, and Part A search."""

    def __init__(
        self,
        records: Sequence[SentenceRecord],
        indexes: AutocompleteIndex,
    ) -> None:
        self.records = records
        self.indexes = indexes

    @classmethod
    def from_archive(
        cls,
        archive_path: PathLike,
        limit: Optional[int] = None,
        progress_every: Optional[int] = 100_000,
    ) -> "AutoCompleteSystem":
        """Read the corpus and build the existing word/N-gram indexes."""

        source_records: Iterable[SentenceRecord] = iter_sentence_records(
            archive_path
        )
        if limit is not None:
            source_records = islice(source_records, limit)

        records: List[SentenceRecord] = []
        for count, record in enumerate(source_records, start=1):
            records.append(record)
            if progress_every and count % progress_every == 0:
                print(f"Loaded {count:,} sentences...")

        indexes = build_autocomplete_index_from_records(
            records,
            ngram_size=3,
            progress_every=progress_every,
        )
        return cls(records=records, indexes=indexes)

    def _candidate_word_sentence_ids(self, word: str) -> Set[int]:
        """Return sentence IDs for vocabulary words close to ``word``.

        Exact words use their posting list immediately.  If a word is not in
        the vocabulary, its character N-grams produce candidate words, which
        are then verified with the same one-edit matcher used for sentences.
        """

        exact_ids = self.indexes.get_sentence_ids(word)
        if exact_ids:
            return set(exact_ids)

        if len(word) > 5:
            grams = {
                word[position : position + 3]
                for position in range(len(word) - 2)
            }
            candidate_word_ids: Set[int] = set()
            for gram in grams:
                candidate_word_ids.update(
                    self.indexes.word_ids_by_ngram.get(gram, ())
                )
        else:
            # A short typo may share no trigram with its correction, so
            # inspect the vocabulary. It is still much smaller than corpus.
            candidate_word_ids = set(range(len(self.indexes.id_to_word)))

        sentence_ids: Set[int] = set()
        for word_id in candidate_word_ids:
            candidate_word = self.indexes.get_word(word_id)
            if best_match_score(word, candidate_word) is None:
                continue
            sentence_ids.update(self.indexes.sentence_ids_by_word[word_id])

        return sentence_ids

    def _candidate_sentence_ids(self, query: str) -> Optional[Set[int]]:
        """Use the existing word postings to reduce sentence verification.

        For a multiword query, intersect the posting groups.  We also create
        leave-one-word-out intersections, because the assignment permits one
        erroneous word.  ``None`` requests a correctness fallback over all
        stored sentences when no word can provide an anchor.
        """

        words = query.split()
        if not words:
            return set()

        word_sentence_ids = [
            self._candidate_word_sentence_ids(word) for word in words
        ]

        if len(words) == 1:
            return word_sentence_ids[0] or None

        available_count = sum(bool(ids) for ids in word_sentence_ids)
        if available_count == 0:
            return None

        candidates: Set[int] = set()

        # Strict intersection: all query words occur in the sentence.
        if available_count == len(words):
            strict = set(word_sentence_ids[0])
            for ids in word_sentence_ids[1:]:
                strict.intersection_update(ids)
            candidates.update(strict)

        # For three or more words, at most one word may be wrong. Intersecting
        # every group except one keeps the candidate set small and preserves
        # sentences where that one word needs correction.
        if len(words) >= 3:
            for omitted_position in range(len(words)):
                remaining_groups = [
                    ids
                    for position, ids in enumerate(word_sentence_ids)
                    if position != omitted_position
                ]
                if not remaining_groups or any(not ids for ids in remaining_groups):
                    continue

                relaxed = set(remaining_groups[0])
                for ids in remaining_groups[1:]:
                    relaxed.intersection_update(ids)
                candidates.update(relaxed)

        # With two words, use their intersection when possible.  If it is
        # empty, the smaller posting list is the safest practical anchor.
        elif not candidates:
            nonempty_groups = [ids for ids in word_sentence_ids if ids]
            candidates.update(min(nonempty_groups, key=len))

        return candidates or None

    def get_best_k_completions(
        self,
        prefix: str,
    ) -> List[AutoCompleteData]:
        """Return the five best completions for ``prefix``."""

        query = normalize_text(prefix)
        if not query:
            return []

        best_results: List[AutoCompleteData] = []
        candidate_ids = self._candidate_sentence_ids(query)

        if candidate_ids is None:
            candidate_records: Iterable[SentenceRecord] = self.records
        else:
            candidate_records = (
                self.records[sentence_id]
                for sentence_id in candidate_ids
                if 0 <= sentence_id < len(self.records)
            )

        for record in candidate_records:
            score = best_match_score(query, record.normalized_sentence)
            if score is None:
                continue

            result = AutoCompleteData(
                completed_sentence=record.completed_sentence,
                source_text=record.source_text,
                offset=record.offset,
                score=score,
            )

            if len(best_results) < 5:
                best_results.append(result)
                best_results.sort(key=_result_order)
                continue

            if _result_order(result) < _result_order(best_results[-1]):
                best_results[-1] = result
                best_results.sort(key=_result_order)

        return best_results


_SYSTEM: Optional[AutoCompleteSystem] = None


def initialize(
    archive_path: PathLike,
    limit: Optional[int] = None,
) -> AutoCompleteSystem:
    """Initialize the module-level system used by the required function."""

    global _SYSTEM
    _SYSTEM = AutoCompleteSystem.from_archive(archive_path, limit=limit)
    return _SYSTEM


def get_best_k_completions(prefix: str) -> List[AutoCompleteData]:
    """Assignment-compatible module-level completion function."""

    if _SYSTEM is None:
        raise RuntimeError("Call initialize(archive_path) before searching.")

    return _SYSTEM.get_best_k_completions(prefix)


def _read_terminal_text(current_text: str) -> str:
    """Read a line while prefilling the previous query when supported."""

    try:
        import readline
    except ImportError:
        return input(f"Enter full text [{current_text}]: ")

    def insert_current_text() -> None:
        readline.insert_text(current_text)
        readline.redisplay()

    readline.set_startup_hook(insert_current_text)
    try:
        return input("Enter your text: ")
    finally:
        readline.set_startup_hook()


def run_terminal(system: AutoCompleteSystem) -> None:
    """Run the online interactive Part A terminal."""

    print("The system is ready.")
    print("Type text and press Enter to receive five suggestions.")
    print("Your previous text is prefilled so you can continue typing.")
    print("Append # to show suggestions once and reset. Exit with /quit.")

    current_text = ""

    while True:
        try:
            entered_text = _read_terminal_text(current_text)
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return

        if entered_text.rstrip().endswith("/quit"):
            print("Goodbye.")
            return

        reset_after_search = entered_text.rstrip().endswith("#")
        if reset_after_search:
            entered_text = entered_text.rstrip()[:-1]

        current_text = entered_text
        if not normalize_text(current_text):
            current_text = ""
            print("Search reset. Start a new sentence.")
            continue

        print("Searching...")
        results = system.get_best_k_completions(current_text)

        if not results:
            print("No matching sentences were found.")
            continue

        print(f"Here are {len(results)} suggestions:")
        for number, result in enumerate(results, start=1):
            source_name = Path(result.source_text).stem
            print(
                f"{number}. {result.completed_sentence} "
                f"({source_name} {result.offset}, score {result.score})"
            )

        if reset_after_search:
            current_text = ""
            print("Search reset. Start a new sentence.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Part A autocomplete program")
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path(__file__).resolve().parent / "Archive",
        help="Path to the Archive directory",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Load only this many records for a quick test",
    )
    arguments = parser.parse_args()

    print("Loading files and preparing the system...")
    system = initialize(arguments.archive, limit=arguments.limit)
    print(
        f"Loaded {len(system.records):,} sentences and "
        f"{len(system.indexes.word_to_id):,} unique words."
    )
    run_terminal(system)


if __name__ == "__main__":
    main()
