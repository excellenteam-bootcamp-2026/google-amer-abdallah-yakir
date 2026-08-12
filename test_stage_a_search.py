import random
import tempfile
import unittest
from pathlib import Path

from data_loader import SentenceRecord, iter_sentence_records, normalize_text
from part_a import AutoCompleteSystem, best_match_score
from search import (
    MatchResult,
    NativeSuffixArrayIndex,
    SuffixArrayIndex,
    cpp_verifier_available,
    verify_one_edit_cpp,
    verify_one_edit_python,
    verify_one_edit_reference,
)


def make_records(sentences: list[str]) -> list[SentenceRecord]:
    return [
        SentenceRecord(
            sentence_id=sentence_id,
            completed_sentence=sentence,
            normalized_sentence=normalize_text(sentence),
            source_text=f"source/{sentence_id % 3}.txt",
            source_offset=100 + sentence_id,
        )
        for sentence_id, sentence in enumerate(sentences)
    ]


class NormalizationAndLoaderTests(unittest.TestCase):
    def test_normalization_is_shared_poc_behavior(self):
        cases = {
            "Hello,     WORLD!": "hello world",
            "comp@uter sci$ence": "computer science",
            "DATA, STRUCTURES!": "data structures",
            " a\t\n b ": "a b",
        }
        for source, expected in cases.items():
            self.assertEqual(normalize_text(source), expected)

    def test_recursive_loader_preserves_original_and_source_offset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "nested"
            nested.mkdir()
            (nested / "sample.txt").write_text(
                "\nHello, WORLD!\ncomp@uter sci$ence\n",
                encoding="utf-8",
            )

            records = list(iter_sentence_records(root))

        self.assertEqual(records[0].completed_sentence, "Hello, WORLD!")
        self.assertEqual(records[0].normalized_sentence, "hello world")
        self.assertEqual(records[0].source_text, "nested/sample.txt")
        self.assertEqual(records[0].source_offset, 1)
        self.assertEqual(records[1].normalized_sentence, "computer science")


class SuffixArrayTests(unittest.TestCase):
    def test_native_index_matches_python_reference(self):
        sentences = ["banana", "bandana", "ananas", "alpha beta", "beta alpha"]
        reference = SuffixArrayIndex(sentences)
        native = NativeSuffixArrayIndex(sentences)
        self.assertEqual(native.suffix_count, len(reference.suffix_array))
        self.assertTrue(
            all(
                native.suffix_at(index) == position
                for index, position in enumerate(reference.suffix_array)
            )
        )
        for pattern in ("a", "ana", "beta", "alpha", "missing"):
            self.assertEqual(
                native.find_exact_sentence_ids(pattern),
                reference.find_exact_sentence_ids(pattern),
            )
        for query in ("p", "banana", "bxnana", "alpha beta", "missing"):
            self.assertEqual(
                native.find_approximate_candidates(query),
                reference.find_approximate_candidates(query),
            )

    def test_rank_doubling_matches_naive_order_on_small_corpus(self):
        index = SuffixArrayIndex(["banana", "bandana", "ananas"])
        naive = sorted(range(len(index.corpus)), key=lambda pos: index.corpus[pos:])
        self.assertEqual(index.suffix_array, naive)

    def test_mapping_and_exact_lookup(self):
        index = SuffixArrayIndex(["alpha beta", "beta alpha"])
        occurrences = index.find_exact_occurrences("beta")
        self.assertEqual(
            {(item.sentence_id, item.match_offset) for item in occurrences},
            {(0, 6), (1, 0)},
        )

    def test_short_query_fallback_and_two_anchor_union(self):
        index = SuffixArrayIndex(["python", "jython", "unrelated"])
        self.assertEqual(index.find_approximate_candidates("p"), {0, 1, 2})
        candidates = index.find_approximate_candidates("pxthon")
        self.assertIn(0, candidates)


class VerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not cpp_verifier_available():
            raise AssertionError("Run 'python build_cpp_verifier.py' before tests")

    def assert_three_way(self, query: str, sentence: str, expected: MatchResult):
        reference = verify_one_edit_reference(query, sentence)
        python_result = verify_one_edit_python(query, sentence)
        cpp_result = verify_one_edit_cpp(query, sentence)
        self.assertEqual(reference, expected)
        self.assertEqual(python_result, reference)
        self.assertEqual(cpp_result, reference)

    def test_focused_edits_and_best_alignment(self):
        cases = [
            ("cat", "cat", MatchResult(True, 0, "exact")),
            ("bat", "cat", MatchResult(True, 0, "substitution")),
            ("caat", "cat", MatchResult(True, 0, "insertion")),
            ("abcd", "abxcd", MatchResult(True, 0, "deletion")),
            ("cat", " cat", MatchResult(True, 1, "exact")),
            ("xab", "abxac", MatchResult(True, 2, "substitution")),
            ("cat", "cat x cat", MatchResult(True, 0, "exact")),
            ("cat", "bat x dat", MatchResult(True, 0, "substitution")),
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
            ("pxthqn", "python", MatchResult(False)),
            ("café", "xx café noir", MatchResult(True, 3, "exact")),
            ("καλη", "xx καλή", MatchResult(True, 3, "substitution")),
            ("", "python", MatchResult(False)),
        ]
        for query, sentence, expected in cases:
            with self.subTest(query=query, sentence=sentence):
                self.assert_three_way(query, sentence, expected)

    def test_10000_randomized_three_way_comparisons(self):
        randomizer = random.Random(20260812)
        alphabet = "aaabbccdde  "
        edit_types = ("exact", "substitution", "insertion", "deletion", "no-match")

        for case_number in range(10_000):
            sentence = "".join(
                randomizer.choice(alphabet)
                for _ in range(randomizer.randint(0, 45))
            )
            edit_type = edit_types[case_number % len(edit_types)]
            if not sentence or edit_type == "no-match":
                query = f"999{case_number}"
            else:
                length = randomizer.randint(1, min(14, len(sentence)))
                location = case_number % 3
                if location == 0:
                    start = 0
                elif location == 1:
                    start = (len(sentence) - length) // 2
                else:
                    start = len(sentence) - length
                query = sentence[start : start + length]
                position = case_number % len(query)
                if edit_type == "substitution":
                    query = query[:position] + "z" + query[position + 1 :]
                elif edit_type == "insertion":
                    query = query[:position] + "z" + query[position:]
                elif edit_type == "deletion":
                    query = query[:position] + query[position + 1 :]

            reference = verify_one_edit_reference(query, sentence)
            python_result = verify_one_edit_python(query, sentence)
            cpp_result = verify_one_edit_cpp(query, sentence)
            if reference != python_result or reference != cpp_result:
                self.fail(
                    "Verifier mismatch:\n"
                    f"sentence={sentence!r}\nquery={query!r}\n"
                    f"reference={reference}\npython={python_result}\ncpp={cpp_result}"
                )


class EndToEndDifferentialTests(unittest.TestCase):
    def setUp(self):
        sentences = [
            "Python is a powerful programming language",
            "I like learning computer science",
            "Machine learning is very interesting",
            "Python supports object oriented programming",
            "Algorithms and data structures are important",
            "Cat cat repeated characters",
            "To be or not to be",
            "Alpha beta",
            "Bravo beta",
            "Charlie beta",
            "Delta beta",
            "Echo beta",
            "Zulu beta",
        ]
        self.system = AutoCompleteSystem.from_records(make_records(sentences))

    def test_indexed_top5_matches_true_brute_force(self):
        queries = [
            "python",
            "xython",
            "progrxmming",
            "programminx",
            "xpython",
            "pyxthon",
            "pythonx",
            "ython",
            "pyhon",
            "pythn",
            "python is",
            "learning computer",
            "programming language",
            "MACHINE LEARNING",
            "data, structures!",
            "comp@uter sci$ence",
            "machine     learning",
            "pxthqn",
            "p",
            "py",
            "machine lxearning",
            "cat",
            "zebra crossing",
        ]
        for query in queries:
            with self.subTest(query=query):
                self.assertEqual(
                    self.system.get_best_k_completions(query),
                    self.system.get_best_k_completions_brute_force(query),
                )

    def test_verifier_and_scorer_agree(self):
        randomizer = random.Random(17)
        alphabet = "abcde "
        for _ in range(2_000):
            sentence = "".join(
                randomizer.choice(alphabet)
                for _ in range(randomizer.randint(1, 30))
            )
            query = "".join(
                randomizer.choice(alphabet)
                for _ in range(randomizer.randint(0, 12))
            )
            matched = verify_one_edit_cpp(query, sentence).matched
            scored = best_match_score(query, sentence) is not None
            self.assertEqual(matched, scored, (query, sentence))

    def test_randomized_indexed_top5_matches_brute_force(self):
        randomizer = random.Random(20260813)
        words = (
            "apple bridge cloud data engine forest green happy index journey "
            "language machine network object pattern python query river science "
            "search simple system table useful vector window yellow"
        ).split()
        sentences = [
            " ".join(randomizer.choices(words, k=randomizer.randint(4, 8)))
            for _ in range(500)
        ]
        system = AutoCompleteSystem.from_records(make_records(sentences))

        queries: list[str] = ["p", "py", "999999 missing"]
        edit_types = ("exact", "substitution", "insertion", "deletion")
        for query_number in range(200):
            sentence = randomizer.choice(sentences)
            length = randomizer.randint(3, min(18, len(sentence)))
            start = randomizer.randint(0, len(sentence) - length)
            query = sentence[start : start + length]
            edit_type = edit_types[query_number % len(edit_types)]
            position = randomizer.randrange(len(query))
            if edit_type == "substitution":
                query = query[:position] + "z" + query[position + 1 :]
            elif edit_type == "insertion":
                query = query[:position] + "z" + query[position:]
            elif edit_type == "deletion":
                query = query[:position] + query[position + 1 :]
            queries.append(query)

        for query in queries:
            indexed = system.get_best_k_completions(query)
            brute_force = system.get_best_k_completions_brute_force(query)
            if indexed != brute_force:
                self.fail(
                    f"Indexed/brute-force mismatch for {query!r}:\n"
                    f"indexed={indexed}\nbrute_force={brute_force}"
                )


if __name__ == "__main__":
    unittest.main()
