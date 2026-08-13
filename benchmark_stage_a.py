"""Benchmark the integrated Stage A engine on an actual Archive directory."""

from __future__ import annotations

import argparse
from itertools import islice
from pathlib import Path
import random
import statistics
import sys
import time
import tracemalloc

from data_loader import SentenceRecord, iter_sentence_records, normalize_text
from part_a import AutoCompleteSystem, best_match_score
from search import verify_one_edit_cpp


def generate_queries(
    records: list[SentenceRecord], count: int, seed: int = 20260812
) -> list[str]:
    randomizer = random.Random(seed)
    query_types = ("exact", "substitution", "insertion", "deletion", "no-match", "short")
    queries: list[str] = []

    usable_records = [record for record in records if len(record.normalized_sentence) >= 8]
    if not usable_records:
        raise ValueError("The benchmark needs at least one sentence of length 8")

    for query_number in range(count):
        query_type = query_types[query_number % len(query_types)]
        if query_type == "no-match":
            queries.append(f"999999{query_number:06d}")
            continue

        sentence = randomizer.choice(usable_records).normalized_sentence
        if query_type == "short":
            queries.append(randomizer.choice([char for char in sentence if char.isalnum()]))
            continue

        length = randomizer.randint(8, min(20, len(sentence)))
        start = randomizer.randint(0, len(sentence) - length)
        query = sentence[start : start + length]
        position = randomizer.randrange(len(query))
        if query_type == "substitution":
            replacement = "x" if query[position] != "x" else "z"
            query = query[:position] + replacement + query[position + 1 :]
        elif query_type == "insertion":
            query = query[:position] + "x" + query[position:]
        elif query_type == "deletion":
            query = query[:position] + query[position + 1 :]
        queries.append(query)

    return queries


def benchmark_size(archive: Path, sentence_limit: int, query_count: int) -> int:
    offline_started = time.perf_counter()
    tracemalloc.start()
    load_started = time.perf_counter()
    records = list(islice(iter_sentence_records(archive), sentence_limit))
    load_seconds = time.perf_counter() - load_started
    records_current, records_peak = tracemalloc.get_traced_memory()
    if not records:
        raise ValueError(f"No searchable records found under {archive}")

    build_started = time.perf_counter()
    system = AutoCompleteSystem.from_records(records)
    build_seconds = time.perf_counter() - build_started
    current_memory, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    offline_seconds = time.perf_counter() - offline_started
    native_current, native_peak = system.search_index.native_memory_bytes
    native_sizes = system.search_index.native_memory_breakdown
    corpus_bytes = sys.getsizeof(system.search_index.corpus)

    queries = generate_queries(records, query_count)
    total_times: list[float] = []
    candidate_times: list[float] = []
    verifier_times: list[float] = []
    scoring_times: list[float] = []
    candidate_counts: list[int] = []

    for query in queries:
        normalized_query = normalize_text(query)

        started = time.perf_counter()
        candidates = system.candidate_sentence_ids(normalized_query)
        candidate_times.append(time.perf_counter() - started)
        candidate_counts.append(len(candidates))

        started = time.perf_counter()
        matched_ids = [
            sentence_id
            for sentence_id in candidates
            if verify_one_edit_cpp(
                normalized_query, records[sentence_id].normalized_sentence
            ).matched
        ]
        verifier_times.append(time.perf_counter() - started)

        started = time.perf_counter()
        for sentence_id in matched_ids:
            best_match_score(
                normalized_query, records[sentence_id].normalized_sentence
            )
        scoring_times.append(time.perf_counter() - started)

        started = time.perf_counter()
        system.get_best_k_completions(query)
        total_times.append(time.perf_counter() - started)

    # Full brute force is deliberately restricted to a manageable correctness
    # subset; it is an oracle, not part of the performance path.
    differential_count = min(20, len(queries))
    for query in queries[:differential_count]:
        indexed = system.get_best_k_completions(query)
        brute_force = system.get_best_k_completions_brute_force(query)
        if indexed != brute_force:
            print(f"DIFFERENTIAL MISMATCH for query {query!r}")
            print(f"Indexed: {indexed}")
            print(f"Brute force: {brute_force}")
            raise AssertionError("Indexed results differ from brute force")

    average_candidates = statistics.fmean(candidate_counts)
    reduction = 100 * (1 - average_candidates / len(records))
    percentile_95 = sorted(total_times)[max(0, int(len(total_times) * 0.95) - 1)]
    print(f"Dataset: {len(records):,} actual records (limit {sentence_limit:,})")
    print("OFFLINE")
    print(f"  Load/build/total: {load_seconds:.3f}/{build_seconds:.3f}/{offline_seconds:.3f} s")
    print(f"  Normalized corpus: {len(system.search_index.corpus):,} characters, "
          f"{corpus_bytes / 1024**2:.1f} MiB Python storage")
    print(f"  Packed suffix array: {native_sizes['suffix_array'] / 1024**2:.1f} MiB")
    print(f"  Sentence starts: {native_sizes['sentence_starts'] / 1024**2:.1f} MiB")
    print(f"  Temporary native build workspace: {native_sizes['build_workspace'] / 1024**2:.1f} MiB")
    print(
        "  Python traced memory after load/current/peak: "
        f"{records_current / 1024**2:.1f}/{current_memory / 1024**2:.1f}/"
        f"{peak_memory / 1024**2:.1f} MiB"
    )
    print(
        "  Native persistent/estimated build peak: "
        f"{native_current / 1024**2:.1f}/{native_peak / 1024**2:.1f} MiB"
    )
    print(f"  Combined estimated peak (Python traced + native): "
          f"{(peak_memory + native_peak) / 1024**2:.1f} MiB")
    print("ONLINE")
    print(f"  Queries: {len(queries)}; differential queries: {differential_count}")
    print(f"  Total latency avg/median/p95: {statistics.fmean(total_times) * 1000:.3f}/"
          f"{statistics.median(total_times) * 1000:.3f}/{percentile_95 * 1000:.3f} ms")
    print(f"  Candidate retrieval average: {statistics.fmean(candidate_times) * 1000:.3f} ms")
    print(f"  C++ verification average: {statistics.fmean(verifier_times) * 1000:.3f} ms")
    print(f"  Scoring average: {statistics.fmean(scoring_times) * 1000:.3f} ms")
    print(f"  Average candidates: {average_candidates:.1f}")
    print(f"  Candidate reduction: {reduction:.1f}%\n")
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[10_000, 50_000, 100_000]
    )
    arguments = parser.parse_args()

    previous_count = -1
    for size in arguments.sizes:
        record_count = benchmark_size(arguments.archive, size, arguments.queries)
        if record_count < size or record_count == previous_count:
            print("Archive exhausted; larger limits would benchmark the same records.")
            break
        previous_count = record_count


if __name__ == "__main__":
    main()
