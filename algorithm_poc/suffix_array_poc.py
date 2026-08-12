"""Small proof of concept for exact substring retrieval with a suffix array."""

from bisect import bisect_right
import cProfile
from dataclasses import dataclass
import gc
import random
import re
import statistics
import time
import tracemalloc

from suffix_array_verifier_cpp import verify_one_edit_cpp as _verify_one_edit_cpp_raw


sentences = [
    "Python is a powerful programming language",
    "I like learning computer science",
    "Machine learning is very interesting",
    "Python supports object oriented programming",
    "Algorithms and data structures are important",
]


def normalize(text: str) -> str:
    """Apply the deliberately simple English-text normalization used by the POC."""
    letters_numbers_spaces = "".join(
        character for character in text.lower()
        if character.isalnum() or character.isspace()
    )
    return re.sub(r"\s+", " ", letters_numbers_spaces).strip()


@dataclass(frozen=True, order=True)
class Occurrence:
    sentence_id: int
    offset: int


@dataclass(frozen=True)
class MatchResult:
    matched: bool
    offset: int | None = None
    edit_type: str | None = None


@dataclass(frozen=True)
class BenchmarkQuery:
    text: str
    query_type: str


@dataclass(frozen=True)
class QueryMetric:
    query: str
    normalized_query: str
    query_type: str
    filtering_mode: str
    candidate_count: int
    candidate_percentage: float
    candidate_seconds: float
    verifier_seconds: float
    total_seconds: float
    match_count: int


def _one_insertion_apart(shorter: str, longer: str) -> bool:
    """Return whether inserting one character into shorter produces longer."""
    if len(longer) != len(shorter) + 1:
        return False
    left = right = 0
    skipped = False
    while left < len(shorter) and right < len(longer):
        if shorter[left] == longer[right]:
            left += 1
            right += 1
        elif skipped:
            return False
        else:
            skipped = True
            right += 1
    return True


def verify_one_edit_reference(
    normalized_query: str, normalized_sentence: str
) -> MatchResult:
    """Slice-based best-alignment oracle.

    Edit names describe the typo in the query relative to the sentence: an
    ``insertion`` is an extra query character and a ``deletion`` is a missing
    query character. At a shared offset, fewer edits are preferred.
    """
    if not normalized_query:
        return MatchResult(False)

    query_length = len(normalized_query)
    best_substitution: MatchResult | None = None
    best_insertion: MatchResult | None = None
    best_deletion: MatchResult | None = None
    for offset in range(len(normalized_sentence) + 1):
        same_length = normalized_sentence[offset : offset + query_length]
        if len(same_length) == query_length:
            if same_length == normalized_query:
                return MatchResult(True, offset, "exact")
            differences = sum(a != b for a, b in zip(normalized_query, same_length))
            if differences == 1 and best_substitution is None:
                best_substitution = MatchResult(True, offset, "substitution")

        # The query has one extra character.
        shorter_window = normalized_sentence[offset : offset + query_length - 1]
        if best_insertion is None and len(shorter_window) == query_length - 1 and _one_insertion_apart(
            shorter_window, normalized_query
        ):
            best_insertion = MatchResult(True, offset, "insertion")

        # The query is missing one character.
        longer_window = normalized_sentence[offset : offset + query_length + 1]
        if best_deletion is None and len(longer_window) == query_length + 1 and _one_insertion_apart(
            normalized_query, longer_window
        ):
            best_deletion = MatchResult(True, offset, "deletion")

    return best_substitution or best_insertion or best_deletion or MatchResult(False)


def _same_length_match_type(query: str, sentence: str, start: int) -> str | None:
    """Classify an equal-length window using indexes and at most one mismatch."""
    mismatch_seen = False
    for query_index in range(len(query)):
        if query[query_index] != sentence[start + query_index]:
            if mismatch_seen:
                return None
            mismatch_seen = True
    return "substitution" if mismatch_seen else "exact"


def _query_has_one_extra_character(query: str, sentence: str, start: int) -> bool:
    """Check whether dropping one query character equals the sentence window."""
    query_index = 0
    sentence_index = start
    sentence_end = start + len(query) - 1
    skipped = False
    while query_index < len(query) and sentence_index < sentence_end:
        if query[query_index] == sentence[sentence_index]:
            query_index += 1
            sentence_index += 1
        elif skipped:
            return False
        else:
            skipped = True
            query_index += 1
    return True


def _query_is_missing_one_character(query: str, sentence: str, start: int) -> bool:
    """Check whether dropping one sentence-window character equals the query."""
    query_index = 0
    sentence_index = start
    sentence_end = start + len(query) + 1
    skipped = False
    while query_index < len(query) and sentence_index < sentence_end:
        if query[query_index] == sentence[sentence_index]:
            query_index += 1
            sentence_index += 1
        elif skipped:
            return False
        else:
            skipped = True
            sentence_index += 1
    return True


def verify_one_edit_python(normalized_query: str, normalized_sentence: str) -> MatchResult:
    """Index-based equivalent of verify_one_edit_reference.

    Match quality is global: exact, substitution, insertion, then deletion.
    Earliest offset breaks ties within an edit type.
    """
    if not normalized_query:
        return MatchResult(False)

    query_length = len(normalized_query)
    sentence_length = len(normalized_sentence)
    best_substitution: MatchResult | None = None
    best_insertion: MatchResult | None = None
    best_deletion: MatchResult | None = None
    for offset in range(sentence_length + 1):
        if offset + query_length <= sentence_length:
            match_type = _same_length_match_type(
                normalized_query, normalized_sentence, offset
            )
            if match_type == "exact":
                return MatchResult(True, offset, "exact")
            if match_type == "substitution" and best_substitution is None:
                best_substitution = MatchResult(True, offset, "substitution")

        if best_insertion is None and offset + query_length - 1 <= sentence_length and _query_has_one_extra_character(
            normalized_query, normalized_sentence, offset
        ):
            best_insertion = MatchResult(True, offset, "insertion")

        if best_deletion is None and offset + query_length + 1 <= sentence_length and _query_is_missing_one_character(
            normalized_query, normalized_sentence, offset
        ):
            best_deletion = MatchResult(True, offset, "deletion")

    return best_substitution or best_insertion or best_deletion or MatchResult(False)


def verify_one_edit_cpp(normalized_query: str, normalized_sentence: str) -> MatchResult:
    """Adapt the minimal C++ tuple result to the POC's MatchResult type."""
    matched, offset, edit_type = _verify_one_edit_cpp_raw(
        normalized_sentence, normalized_query
    )
    return MatchResult(matched, offset, edit_type)


# Existing baseline and instrumentation continue to use the optimized Python
# verifier unless a call explicitly selects C++.
verify_one_edit = verify_one_edit_python


class SuffixArrayIndex:
    def __init__(
        self, original_sentences: list[str], builder: str = "rank_doubling"
    ) -> None:
        self.original_sentences = list(original_sentences)
        self.normalized_sentences = [normalize(text) for text in original_sentences]

        # NUL is outside normal text in this POC and blocks matches from spanning
        # two sentences. Sentence starts are enough to map positions; a metadata
        # entry for every corpus character is unnecessary.
        self.separator = "\0"
        if any(self.separator in text for text in self.normalized_sentences):
            raise ValueError("A normalized sentence contains the corpus separator")

        self.sentence_starts: list[int] = []
        corpus_parts: list[str] = []
        next_start = 0
        for text in self.normalized_sentences:
            self.sentence_starts.append(next_start)
            corpus_parts.append(text)
            next_start += len(text) + 1
        self.corpus = self.separator.join(corpus_parts)

        build_started = time.perf_counter()
        if builder == "rank_doubling":
            self.suffix_array = self.build_rank_doubling()
        elif builder == "naive":
            self.suffix_array = self.build_naive()
        else:
            raise ValueError(f"Unknown suffix-array builder: {builder}")
        self.suffix_array_build_seconds = time.perf_counter() - build_started

    def build_naive(self) -> list[int]:
        """Quadratic-memory reference builder, safe only for small comparisons."""
        return sorted(range(len(self.corpus)), key=lambda position: self.corpus[position:])

    def build_rank_doubling(self) -> list[int]:
        """Build a suffix array from integer rank pairs without copying suffixes."""
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
        """Map a corpus position to (sentence_id, offset), or None for separators."""
        if not 0 <= position < len(self.corpus):
            raise IndexError("Corpus position is out of range")

        sentence_id = bisect_right(self.sentence_starts, position) - 1
        offset = position - self.sentence_starts[sentence_id]
        if offset >= len(self.normalized_sentences[sentence_id]):
            return None
        return sentence_id, offset

    def find_occurrences(self, pattern: str) -> list[Occurrence]:
        normalized_pattern = normalize(pattern)
        if not normalized_pattern:
            return []
        if self.separator in normalized_pattern:
            return []

        pattern_length = len(normalized_pattern)

        def prefix_at(suffix_index: int) -> str:
            position = self.suffix_array[suffix_index]
            return self.corpus[position : position + pattern_length]

        # Find the half-open range of suffixes whose first characters equal the
        # pattern. Both bounds are found directly on the sorted suffix array.
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
            mapped = self.corpus_position_to_sentence(self.suffix_array[suffix_index])
            if mapped is not None:
                occurrences.add(Occurrence(*mapped))
        return sorted(occurrences)

    def find_candidate_sentences(self, pattern: str) -> set[int]:
        return {item.sentence_id for item in self.find_occurrences(pattern)}

    def baseline_search(self, query: str) -> set[int]:
        normalized_query = normalize(query)
        return {
            sentence_id
            for sentence_id, sentence in enumerate(self.normalized_sentences)
            if verify_one_edit(normalized_query, sentence).matched
        }

    def find_approximate_candidates(self, query: str) -> set[int]:
        normalized_query = normalize(query)
        # Lengths 0-2 have partitions that are empty or only one character long;
        # exact hits for those pieces rarely reduce the candidate set usefully.
        if len(normalized_query) < 3:
            return set(range(len(self.normalized_sentences)))

        split = len(normalized_query) // 2
        left, right = normalized_query[:split], normalized_query[split:]
        return self.find_candidate_sentences(left) | self.find_candidate_sentences(right)

    def filtered_search(self, query: str, verifier: str = "python") -> set[int]:
        normalized_query = normalize(query)
        verify = (
            verify_one_edit_python if verifier == "python" else verify_one_edit_cpp
        )
        if verifier not in {"python", "cpp"}:
            raise ValueError(f"Unknown verifier: {verifier}")
        return {
            sentence_id
            for sentence_id in self.find_approximate_candidates(query)
            if verify(
                normalized_query, self.normalized_sentences[sentence_id]
            ).matched
        }


def naive_occurrences(index: SuffixArrayIndex, pattern: str) -> set[Occurrence]:
    """Reference implementation used only by the correctness checks."""
    normalized_pattern = normalize(pattern)
    if not normalized_pattern:
        return set()

    matches: set[Occurrence] = set()
    for sentence_id, text in enumerate(index.normalized_sentences):
        start = 0
        while True:
            offset = text.find(normalized_pattern, start)
            if offset == -1:
                break
            matches.add(Occurrence(sentence_id, offset))
            start = offset + 1  # Include overlapping occurrences.
    return matches


def run_verifier_differential_checks(pair_count: int = 10_000) -> None:
    """Compare reference, optimized Python, and C++ semantics."""
    rng = random.Random(20260813)
    alphabet = "aaabbccdde  "  # Repetition and spaces are intentional.
    kinds = ("exact", "substitution", "insertion", "deletion", "no-match", "short")

    for case_number in range(pair_count):
        sentence_length = rng.randint(0, 45)
        sentence = "".join(rng.choice(alphabet) for _ in range(sentence_length))
        kind = kinds[case_number % len(kinds)]

        if kind == "no-match" or not sentence:
            query = f"999{case_number}"
        elif kind == "short":
            query = rng.choice("abc")
        else:
            substring_length = rng.randint(1, min(14, len(sentence)))
            position_mode = case_number % 3
            if position_mode == 0:
                start = 0
            elif position_mode == 1:
                start = (len(sentence) - substring_length) // 2
            else:
                start = len(sentence) - substring_length
            query = sentence[start : start + substring_length]
            edit_position = case_number % len(query)
            if kind == "substitution":
                replacement = "z" if query[edit_position] != "z" else "y"
                query = query[:edit_position] + replacement + query[edit_position + 1 :]
            elif kind == "insertion":
                query = query[:edit_position] + "z" + query[edit_position:]
            elif kind == "deletion":
                query = query[:edit_position] + query[edit_position + 1 :]

        reference = verify_one_edit_reference(query, sentence)
        python_result = verify_one_edit_python(query, sentence)
        cpp_result = verify_one_edit_cpp(query, sentence)
        if python_result != reference or cpp_result != python_result:
            print("VERIFIER SEMANTIC MISMATCH")
            print(f"Sentence: {sentence!r}")
            print(f"Query: {query!r}")
            print(f"Reference: {reference}")
            print(f"Python: {python_result}")
            print(f"C++: {cpp_result}")
            raise AssertionError("Python/C++ verifier semantic mismatch")


def run_correctness_checks(index: SuffixArrayIndex) -> None:
    # The quadratic builder remains a correctness oracle only for this tiny corpus.
    assert index.suffix_array == index.build_naive()

    exact_queries = [
        "python",
        "learning",
        "programming",
        "data structures",
        "computer",
        "missing phrase",
        "PYTHON!!!",
        "is",
        "  DATA,   STRUCTURES!  ",
    ]
    for query in exact_queries:
        suffix_array_results = set(index.find_occurrences(query))
        assert suffix_array_results == naive_occurrences(index, query), query

    approximate_cases = {
        "powerful": "exact",
        "xython": "substitution at first character",
        "progrxmming": "substitution in middle",
        "programminx": "substitution near the end",
        "xpython": "insertion at beginning",
        "pyxthon": "insertion in middle",
        "pythonx": "insertion near the end",
        "ython": "deletion at beginning",
        "pyhon": "deletion in middle",
        "pythn": "deletion near the end",
        "python is": "substring at sentence beginning",
        "learning computer": "substring in sentence middle",
        "programming language": "substring at sentence end",
        "MACHINE LEARNING": "uppercase differences",
        "data, structures!": "punctuation differences",
        "comp@uter sci$ence": "symbol differences",
        "machine     learning": "repeated whitespace",
        "zebra crossing": "no match",
        "pxthqn": "two edits",
        "p": "very short query",
        "machine lxearning": "edit close to partition boundary",
    }
    for query, description in approximate_cases.items():
        assert index.filtered_search(query) == index.baseline_search(query), description

    assert index.baseline_search("zebra crossing") == set()
    assert index.baseline_search("pxthqn") == set()  # Two substitutions from "python".

    # Focused verifier checks ensure all edit labels are exercised explicitly.
    verifier_cases = [
        ("cat", "cat", MatchResult(True, 0, "exact")),
        ("bat", "cat", MatchResult(True, 0, "substitution")),
        ("caat", "cat", MatchResult(True, 0, "insertion")),
        ("abcd", "abxcd", MatchResult(True, 0, "deletion")),
        # Regression: an earlier deletion must not hide a later exact match.
        (
            "data structures",
            "algorithms and data structures are important",
            MatchResult(True, 15, "exact"),
        ),
        (
            "computer science",
            "i like learning computer science",
            MatchResult(True, 16, "exact"),
        ),
        ("cat", " cat", MatchResult(True, 1, "exact")),
        # An insertion at 0 exists, but substitution at 2 has higher priority.
        ("xab", "abxac", MatchResult(True, 2, "substitution")),
        ("cat", "cat x cat", MatchResult(True, 0, "exact")),
        ("cat", "bat x dat", MatchResult(True, 0, "substitution")),
    ]
    for query, sentence, expected_result in verifier_cases:
        result = verify_one_edit(query, sentence)
        assert result == expected_result
        assert result == verify_one_edit_reference(query, sentence)
        assert result == verify_one_edit_cpp(query, sentence)

    run_verifier_differential_checks()


def demonstrate(index: SuffixArrayIndex) -> None:
    queries = [
        "python",
        "machine lerning",
        "progrxmming",
        "data, structures!",
        "comp@uter sci$ence",
        "machine     learning",
        "zebra crossing",
        "p",
    ]

    for query in queries:
        normalized_query = normalize(query)
        candidates = index.find_approximate_candidates(query)
        matches = index.filtered_search(query)
        total = len(index.normalized_sentences)
        reduction = 100 * (1 - len(candidates) / total) if total else 0
        print(f"Query: {query!r}")
        print(f"Normalized query: {normalized_query!r}")
        print(f"Total sentences: {total}")
        print(f"Candidates: {len(candidates)}")
        print(f"Candidate reduction: {reduction:.0f}%")
        print(f"Final matching sentence IDs: {sorted(matches)}")
        for sentence_id in sorted(matches):
            result = verify_one_edit(
                normalized_query, index.normalized_sentences[sentence_id]
            )
            print(f"  Match sentence {sentence_id}:")
            print(f"    offset={result.offset}")
            print(f"    edit_type={result.edit_type}")
            print(f"    text={index.original_sentences[sentence_id]}")
        print()


SYNTHETIC_WORDS = (
    "apple bridge cloud data engine forest green happy index journey language "
    "machine network object pattern python query river science search simple "
    "system table useful vector window yellow".split()
)


def generate_synthetic_sentences(count: int, seed: int) -> list[str]:
    """Generate deterministic, moderately sized English-like sentences."""
    rng = random.Random(seed)
    return [
        " ".join(rng.choices(SYNTHETIC_WORDS, k=rng.randint(3, 5))).capitalize()
        for _ in range(count)
    ]


def generate_benchmark_queries(
    normalized_sentences: list[str], count: int, seed: int
) -> list[BenchmarkQuery]:
    """Create a deterministic mixture of exact and one-edit substring queries."""
    rng = random.Random(seed)
    query_kinds = (
        ["exact"] * 35
        + ["substitution"] * 35
        + ["insertion"] * 35
        + ["deletion"] * 35
        + ["missing"] * 35
        + ["short"] * 25
    )
    queries: list[BenchmarkQuery] = []

    for kind in query_kinds[:count]:
        sentence = rng.choice(normalized_sentences)
        if kind == "short":
            short_characters = [character for character in sentence if character.isalnum()]
            queries.append(
                BenchmarkQuery(
                    rng.choice(short_characters), "short-query"
                )
            )
            continue
        if kind == "missing":
            # Synthetic sentences contain neither digits nor this marker.
            queries.append(BenchmarkQuery(f"999999{len(queries):04d}", "no-match"))
            continue

        length = rng.randint(8, min(20, len(sentence)))
        start = rng.randint(0, len(sentence) - length)
        query = sentence[start : start + length]
        edit_position = rng.randrange(len(query))
        if kind == "substitution":
            replacement = "x" if query[edit_position] != "x" else "z"
            query = query[:edit_position] + replacement + query[edit_position + 1 :]
        elif kind == "insertion":
            query = query[:edit_position] + "x" + query[edit_position:]
        elif kind == "deletion":
            query = query[:edit_position] + query[edit_position + 1 :]
        queries.append(BenchmarkQuery(query, kind))

    return queries


def measure_filtered_query(
    index: SuffixArrayIndex, benchmark_query: BenchmarkQuery, verifier: str = "python"
) -> tuple[set[int], QueryMetric]:
    """Run the unchanged two-stage search while timing each stage separately."""
    query = benchmark_query.text
    normalized_query = normalize(query)
    filtering_mode = (
        "all-sentence fallback"
        if len(normalized_query) < 3
        else "suffix-array partition filtering"
    )

    total_started = time.perf_counter()
    candidate_started = time.perf_counter()
    candidates = index.find_approximate_candidates(query)
    candidate_seconds = time.perf_counter() - candidate_started

    verifier_started = time.perf_counter()
    verify = verify_one_edit_python if verifier == "python" else verify_one_edit_cpp
    results = {
        sentence_id
        for sentence_id in candidates
        if verify(
            normalized_query, index.normalized_sentences[sentence_id]
        ).matched
    }
    verifier_seconds = time.perf_counter() - verifier_started
    total_seconds = time.perf_counter() - total_started
    total_sentences = len(index.normalized_sentences)
    metric = QueryMetric(
        query=query,
        normalized_query=normalized_query,
        query_type=benchmark_query.query_type,
        filtering_mode=filtering_mode,
        candidate_count=len(candidates),
        candidate_percentage=(100 * len(candidates) / total_sentences),
        candidate_seconds=candidate_seconds,
        verifier_seconds=verifier_seconds,
        total_seconds=total_seconds,
        match_count=len(results),
    )
    return results, metric


def _print_metric_group(label: str, metrics: list[QueryMetric]) -> None:
    print(label)
    if not metrics:
        print("  Queries: 0\n")
        return
    print(f"  Queries: {len(metrics)}")
    print(f"  Average candidates: {statistics.fmean(m.candidate_count for m in metrics):.1f}")
    print(
        "  Average candidate percentage: "
        f"{statistics.fmean(m.candidate_percentage for m in metrics):.1f}%"
    )
    print(f"  Average total time: {statistics.fmean(m.total_seconds for m in metrics) * 1000:.3f} ms")
    print(f"  Median total time: {statistics.median(m.total_seconds for m in metrics) * 1000:.3f} ms")
    print(
        "  Average candidate retrieval: "
        f"{statistics.fmean(m.candidate_seconds for m in metrics) * 1000:.3f} ms"
    )
    print(f"  Average verifier time: {statistics.fmean(m.verifier_seconds for m in metrics) * 1000:.3f} ms\n")


def _pearson(values_a: list[float], values_b: list[float]) -> float:
    mean_a = statistics.fmean(values_a)
    mean_b = statistics.fmean(values_b)
    numerator = sum((a - mean_a) * (b - mean_b) for a, b in zip(values_a, values_b))
    denominator_a = sum((a - mean_a) ** 2 for a in values_a)
    denominator_b = sum((b - mean_b) ** 2 for b in values_b)
    denominator = (denominator_a * denominator_b) ** 0.5
    return numerator / denominator if denominator else 0.0


def print_query_profile_analysis(metrics: list[QueryMetric]) -> None:
    print("50,000-sentence per-query analysis\n")
    buckets = (
        ("Length 1-2", 1, 2),
        ("Length 3-5", 3, 5),
        ("Length 6-10", 6, 10),
        ("Length 11-20", 11, 20),
        ("Length 21+", 21, float("inf")),
    )
    for label, low, high in buckets:
        group = [m for m in metrics if low <= len(m.normalized_query) <= high]
        _print_metric_group(label, group)

    _print_metric_group(
        "Suffix-array partition filtering",
        [m for m in metrics if m.filtering_mode == "suffix-array partition filtering"],
    )
    _print_metric_group(
        "All-sentence fallback",
        [m for m in metrics if m.filtering_mode == "all-sentence fallback"],
    )

    print("10 slowest filtered queries")
    for metric in sorted(metrics, key=lambda item: item.total_seconds, reverse=True)[:10]:
        print(
            f"  {metric.query!r} | length={len(metric.normalized_query)} | "
            f"type={metric.query_type} | mode={metric.filtering_mode} | "
            f"candidates={metric.candidate_count:,} | "
            f"retrieval={metric.candidate_seconds * 1000:.3f} ms | "
            f"verifier={metric.verifier_seconds * 1000:.3f} ms | "
            f"total={metric.total_seconds * 1000:.3f} ms"
        )

    length_correlation = _pearson(
        [float(len(m.normalized_query)) for m in metrics],
        [m.total_seconds for m in metrics],
    )
    candidate_correlation = _pearson(
        [float(m.candidate_count) for m in metrics],
        [m.total_seconds for m in metrics],
    )
    indexed_metrics = [
        m for m in metrics
        if m.filtering_mode == "suffix-array partition filtering"
    ]
    indexed_candidate_correlation = _pearson(
        [float(m.candidate_count) for m in indexed_metrics],
        [m.total_seconds for m in indexed_metrics],
    )
    print(f"\nPearson correlation, query length vs runtime: {length_correlation:.3f}")
    print(f"Pearson correlation, candidate count vs runtime: {candidate_correlation:.3f}\n")
    print(
        "Pearson correlation, indexed-query candidate count vs runtime: "
        f"{indexed_candidate_correlation:.3f}\n"
    )


def print_function_profile(
    index: SuffixArrayIndex, queries: list[BenchmarkQuery], verifier: str
) -> None:
    profiler = cProfile.Profile()
    profiler.enable()
    for query in queries:
        index.filtered_search(query.text, verifier=verifier)
    profiler.disable()

    poc_rows: list[tuple[float, float, int, str]] = []
    for entry in profiler.getstats():
        code = entry.code
        if isinstance(code, str):
            if "verify_one_edit_cpp" in code:
                poc_rows.append(
                    (entry.totaltime, entry.inlinetime, entry.callcount, code)
                )
        elif code.co_filename.endswith("suffix_array_poc.py"):
            poc_rows.append(
                (entry.totaltime, entry.inlinetime, entry.callcount, code.co_name)
            )

    print(f"cProfile: {verifier.upper()} filtered searches on 50,000 sentences")
    print("  cumulative s | self s | calls | function")
    for cumulative, self_time, calls, function in sorted(
        poc_rows, reverse=True
    )[:12]:
        print(f"  {cumulative:12.3f} | {self_time:6.3f} | {calls:7,} | {function}")
    print()


def benchmark_dataset(sentence_count: int, query_count: int = 200) -> bool:
    synthetic = generate_synthetic_sentences(sentence_count, seed=20260811)

    naive_time: float | None = None
    naive_peak: int | None = None
    if sentence_count == 1_000:
        gc.collect()
        tracemalloc.start()
        try:
            naive_index = SuffixArrayIndex(synthetic, builder="naive")
        except MemoryError:
            tracemalloc.stop()
            print("The 1,000-sentence naive comparison exhausted memory.")
        else:
            naive_time = naive_index.suffix_array_build_seconds
            _, naive_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            del naive_index
            gc.collect()

    gc.collect()
    tracemalloc.start()
    try:
        index = SuffixArrayIndex(synthetic, builder="rank_doubling")
    except MemoryError:
        tracemalloc.stop()
        print(f"Dataset: {sentence_count:,} sentences")
        print("STOPPED: naive suffix-array construction exhausted memory\n")
        return False
    current_memory, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    queries = generate_benchmark_queries(
        index.normalized_sentences, query_count, seed=20260812
    )
    baseline_times: list[float] = []
    filtered_times: list[float] = []
    candidate_counts: list[int] = []
    query_metrics: list[QueryMetric] = []
    cpp_metrics: list[QueryMetric] = []

    for benchmark_query in queries:
        query = benchmark_query.text
        started = time.perf_counter()
        baseline_results = index.baseline_search(query)
        baseline_times.append(time.perf_counter() - started)

        filtered_results, metric = measure_filtered_query(index, benchmark_query)
        filtered_times.append(metric.total_seconds)
        query_metrics.append(metric)

        if sentence_count == 50_000:
            cpp_results, cpp_metric = measure_filtered_query(
                index, benchmark_query, verifier="cpp"
            )
            cpp_metrics.append(cpp_metric)
            if cpp_results != filtered_results:
                print("PYTHON/C++ FILTERED RESULT MISMATCH")
                print(f"Failing query: {query!r}")
                print(f"Python results: {sorted(filtered_results)}")
                print(f"C++ results: {sorted(cpp_results)}")
                raise AssertionError("Python/C++ filtered-search mismatch")

        if filtered_results != baseline_results:
            print("CORRECTNESS MISMATCH - benchmark stopped")
            print(f"Failing query: {query!r}")
            print(f"Baseline results: {sorted(baseline_results)}")
            print(f"Filtered results: {sorted(filtered_results)}")
            missing = baseline_results - filtered_results
            extra = filtered_results - baseline_results
            print(f"False negatives: {sorted(missing)}")
            print(f"False positives: {sorted(extra)}")
            raise AssertionError(f"Approximate candidate mismatch for {query!r}")

        candidate_counts.append(metric.candidate_count)

    average_candidates = statistics.fmean(candidate_counts)
    average_reduction = 100 * (1 - average_candidates / sentence_count)
    print(f"Dataset: {sentence_count:,} sentences")
    print(f"Corpus length: {len(index.corpus):,}")
    print(f"Suffix array length: {len(index.suffix_array):,}")
    print(f"Suffix array build time: {index.suffix_array_build_seconds:.3f} s")
    print(
        "Traced memory after construction (current/peak): "
        f"{current_memory / 1024**2:.1f}/{peak_memory / 1024**2:.1f} MiB"
    )
    print(f"Queries tested: {len(queries)}")
    print("Correctness mismatches: 0")
    print(f"Baseline avg query time: {statistics.fmean(baseline_times) * 1000:.3f} ms")
    print(f"Filtered avg query time: {statistics.fmean(filtered_times) * 1000:.3f} ms")
    print(f"Filtered median query time: {statistics.median(filtered_times) * 1000:.3f} ms")
    print(f"Average candidates: {average_candidates:.1f}")
    print(f"Average candidate reduction: {average_reduction:.1f}%\n")
    if naive_time is not None and naive_peak is not None:
        print("Direct 1,000-sentence construction comparison")
        print(f"NAIVE build time: {naive_time:.3f} s")
        print(f"NAIVE peak traced memory: {naive_peak / 1024**2:.1f} MiB")
        print(
            "RANK-DOUBLING build time: "
            f"{index.suffix_array_build_seconds:.3f} s"
        )
        print(f"RANK-DOUBLING peak traced memory: {peak_memory / 1024**2:.1f} MiB\n")
    if sentence_count == 50_000:
        python_verifier_average = statistics.fmean(
            metric.verifier_seconds for metric in query_metrics
        )
        cpp_verifier_average = statistics.fmean(
            metric.verifier_seconds for metric in cpp_metrics
        )
        python_total_average = statistics.fmean(
            metric.total_seconds for metric in query_metrics
        )
        cpp_total_average = statistics.fmean(
            metric.total_seconds for metric in cpp_metrics
        )
        print("Direct PYTHON-vs-C++ verifier comparison")
        print(
            "PYTHON filtered average: "
            f"{python_total_average * 1000:.3f} ms"
        )
        print(
            "PYTHON filtered median: "
            f"{statistics.median(m.total_seconds for m in query_metrics) * 1000:.3f} ms"
        )
        print(
            "PYTHON verification average: "
            f"{python_verifier_average * 1000:.3f} ms"
        )
        print(
            f"C++ filtered average: {cpp_total_average * 1000:.3f} ms"
        )
        print(
            "C++ filtered median: "
            f"{statistics.median(m.total_seconds for m in cpp_metrics) * 1000:.3f} ms"
        )
        print(
            "C++ verification average: "
            f"{cpp_verifier_average * 1000:.3f} ms"
        )
        print(f"Verifier speedup: {python_verifier_average / cpp_verifier_average:.2f}x")
        print(f"Full filtered speedup: {python_total_average / cpp_total_average:.2f}x")
        print(
            "C++ candidate retrieval average: "
            f"{statistics.fmean(m.candidate_seconds for m in cpp_metrics) * 1000:.3f} ms\n"
        )

        overhead_calls = 200_000
        overhead_started = time.perf_counter()
        for _ in range(overhead_calls):
            _verify_one_edit_cpp_raw("", "")
        overhead_seconds = time.perf_counter() - overhead_started
        print(
            "Approximate empty-call C++ binding overhead: "
            f"{overhead_seconds / overhead_calls * 1e6:.3f} microseconds/call\n"
        )

        # Profile all 200 queries: the batch includes every generated query type
        # and both candidate modes, while remaining small enough for this POC.
        print_function_profile(index, queries, verifier="cpp")
    return True


def run_benchmarks() -> None:
    print("Benchmark and randomized correctness stress test\n")
    for sentence_count in (1_000, 10_000, 50_000):
        benchmark_dataset(sentence_count)


if __name__ == "__main__":
    suffix_index = SuffixArrayIndex(sentences)
    run_correctness_checks(suffix_index)
    print("All correctness checks passed.\n")
    demonstrate(suffix_index)
    run_benchmarks()
