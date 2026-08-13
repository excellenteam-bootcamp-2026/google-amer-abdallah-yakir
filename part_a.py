"""Complete correctness-first implementation of Part A of the assignment."""

import argparse
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
import sys
import time
from typing import Iterable, List, Optional, Sequence, Union

from data_loader import SentenceRecord, iter_sentence_records, normalize_text
from search import NativeSuffixArrayIndex, verify_one_edit_cpp


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
        search_index: NativeSuffixArrayIndex,
    ) -> None:
        self.records = list(records)
        self.search_index = search_index
        if len(self.records) != len(self.search_index.normalized_sentences):
            raise ValueError("The records and suffix-array sentences must align")
        self.offline_metrics: dict[str, float | int] = {}
        self.last_query_metrics: dict[str, float | int] = {}

    @classmethod
    def from_records(cls, records: Sequence[SentenceRecord]) -> "AutoCompleteSystem":
        """Build the offline suffix array for already-loaded records."""
        stored_records = list(records)
        if any(not record.normalized_sentence for record in stored_records):
            raise ValueError("Search records must have nonempty normalized sentences")
        if any(
            record.sentence_id != sentence_id
            for sentence_id, record in enumerate(stored_records)
        ):
            raise ValueError("Sentence IDs must be contiguous and match record order")
        search_index = NativeSuffixArrayIndex(
            [record.normalized_sentence for record in stored_records]
        )
        return cls(records=stored_records, search_index=search_index)

    @classmethod
    def from_archive(
        cls,
        archive_path: PathLike,
        limit: Optional[int] = None,
        progress_every: Optional[int] = 100_000,
    ) -> "AutoCompleteSystem":
        """Read the corpus and build the offline suffix-array index."""

        load_started = time.perf_counter()
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

        load_seconds = time.perf_counter() - load_started
        build_started = time.perf_counter()
        system = cls.from_records(records)
        build_seconds = time.perf_counter() - build_started
        native_sizes = system.search_index.native_memory_breakdown
        system.offline_metrics = {
            "load_seconds": load_seconds,
            "build_seconds": build_seconds,
            "corpus_characters": len(system.search_index.corpus),
            "corpus_bytes": sys.getsizeof(system.search_index.corpus),
            **native_sizes,
        }
        return system

    def candidate_sentence_ids(self, normalized_query: str) -> set[int]:
        """Retrieve candidates only; verification and scoring remain separate."""
        return self.search_index.find_approximate_candidates(normalized_query)

    def _rank_sentence_ids(
        self,
        normalized_query: str,
        sentence_ids: Iterable[int],
        verify_candidates: bool,
    ) -> List[AutoCompleteData]:
        """Score valid records and apply the assignment's deterministic Top-5."""
        best_results: List[AutoCompleteData] = []

        for sentence_id in sentence_ids:
            if not 0 <= sentence_id < len(self.records):
                continue
            record = self.records[sentence_id]

            if verify_candidates:
                verification = verify_one_edit_cpp(
                    normalized_query, record.normalized_sentence
                )
                if not verification.matched:
                    continue

            score = best_match_score(
                normalized_query, record.normalized_sentence
            )

            if score is None:
                if verify_candidates:
                    raise AssertionError(
                        "C++ verifier accepted a candidate rejected by the "
                        f"approved scorer: query={normalized_query!r}, "
                        f"sentence={record.normalized_sentence!r}"
                    )
                continue

            result = AutoCompleteData(
                completed_sentence=record.completed_sentence,
                source_text=record.source_text,
                offset=record.source_offset,
                score=score,
            )

            if len(best_results) < 5:
                best_results.append(result)
                best_results.sort(key=_result_order)
            elif _result_order(result) < _result_order(best_results[-1]):
                best_results[-1] = result
                best_results.sort(key=_result_order)

        return best_results

    def get_best_k_completions(
        self,
        prefix: str,
    ) -> List[AutoCompleteData]:
        """Return the five best completions for ``prefix``."""

        query = normalize_text(prefix)
        if not query:
            return []

        query_started = time.perf_counter()
        candidate_started = time.perf_counter()
        candidate_ids = self.candidate_sentence_ids(query)
        candidate_seconds = time.perf_counter() - candidate_started
        ranking_started = time.perf_counter()
        results = self._rank_sentence_ids(query, candidate_ids, verify_candidates=True)
        ranking_seconds = time.perf_counter() - ranking_started
        self.last_query_metrics = {
            "candidate_count": len(candidate_ids),
            "candidate_seconds": candidate_seconds,
            "ranking_seconds": ranking_seconds,
            "total_seconds": time.perf_counter() - query_started,
        }
        return results

    def get_best_k_completions_brute_force(
        self,
        prefix: str,
    ) -> List[AutoCompleteData]:
        """Slow correctness oracle: score every record, then return exact Top-5."""
        query = normalize_text(prefix)
        if not query:
            return []
        return self._rank_sentence_ids(
            query,
            range(len(self.records)),
            verify_candidates=False,
        )


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
        metrics = system.last_query_metrics
        print(
            "Online metrics: "
            f"{metrics['total_seconds'] * 1000:.3f} ms total, "
            f"{metrics['candidate_seconds'] * 1000:.3f} ms candidate lookup, "
            f"{metrics['ranking_seconds'] * 1000:.3f} ms verify/rank, "
            f"{metrics['candidate_count']:,} candidates."
        )

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
    try:
        system = initialize(arguments.archive, limit=arguments.limit)
    except (FileNotFoundError, NotADirectoryError) as error:
        parser.error(
            f"{error}\n"
            "Provide the corpus location with --archive, for example: "
            "python part_a.py --archive C:\\path\\to\\Archive"
        )
    print(
        f"Loaded {len(system.records):,} sentences; searchable corpus has "
        f"{len(system.search_index.corpus):,} characters."
    )
    metrics = system.offline_metrics
    if metrics:
        mib = 1024 ** 2
        print(
            "Offline metrics: "
            f"load {metrics['load_seconds']:.3f} s, "
            f"suffix-array build {metrics['build_seconds']:.3f} s."
        )
        print(
            "Index sizes: "
            f"corpus {metrics['corpus_bytes'] / mib:.1f} MiB, "
            f"packed suffix array {metrics['suffix_array'] / mib:.1f} MiB, "
            f"sentence starts {metrics['sentence_starts'] / mib:.1f} MiB, "
            f"persistent native total {metrics['persistent_total'] / mib:.1f} MiB, "
            f"estimated native build peak {metrics['build_peak'] / mib:.1f} MiB."
        )
    run_terminal(system)


if __name__ == "__main__":
    main()
