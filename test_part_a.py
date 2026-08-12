import unittest

from data_loader import SentenceRecord, normalize_text
from part_a import AutoCompleteSystem, best_match_score


def make_record(sentence_id, sentence, normalized=None, source="sample.txt"):
    return SentenceRecord(
        sentence_id=sentence_id,
        completed_sentence=sentence,
        normalized_sentence=normalized or normalize_text(sentence),
        source_text=source,
        source_offset=sentence_id,
    )


class ScoreTests(unittest.TestCase):
    def test_exact_substring(self):
        self.assertEqual(
            best_match_score("or not", "to be or not to be"),
            12,
        )

    def test_one_replacement(self):
        self.assertEqual(best_match_score("to pe", "to be"), 6)

    def test_one_extra_character(self):
        self.assertEqual(best_match_score("or knot", "or not"), 8)

    def test_one_missing_character(self):
        self.assertEqual(best_match_score("or nt", "or not"), 8)

    def test_two_corrections_are_rejected(self):
        self.assertIsNone(best_match_score("zz be", "to be"))


class CompletionTests(unittest.TestCase):
    def test_returns_only_five_and_sorts_ties_alphabetically(self):
        records = [
            make_record(0, "Zulu beta", "zulu beta"),
            make_record(1, "Alpha beta", "alpha beta"),
            make_record(2, "Echo beta", "echo beta"),
            make_record(3, "Bravo beta", "bravo beta"),
            make_record(4, "Delta beta", "delta beta"),
            make_record(5, "Charlie beta", "charlie beta"),
        ]
        system = AutoCompleteSystem.from_records(records)

        results = system.get_best_k_completions("BETA")

        self.assertEqual(
            [result.completed_sentence for result in results],
            [
                "Alpha beta",
                "Bravo beta",
                "Charlie beta",
                "Delta beta",
                "Echo beta",
            ],
        )
        self.assertTrue(all(result.score == 8 for result in results))

    def test_result_preserves_source_offset_and_original_text(self):
        record = SentenceRecord(
            sentence_id=0,
            completed_sentence="To be, or not to be.",
            normalized_sentence="to be or not to be",
            source_text="nested/quote.txt",
            source_offset=7,
        )
        system = AutoCompleteSystem.from_records([record])

        result = system.get_best_k_completions("BE, OR")[0]

        self.assertEqual(result.completed_sentence, "To be, or not to be.")
        self.assertEqual(result.source_text, "nested/quote.txt")
        self.assertEqual(result.offset, 7)
        self.assertEqual(result.score, 10)

    def test_multiword_query_uses_suffix_array_candidates(self):
        records = [
            make_record(0, "How can I have this", "how can i have this"),
            make_record(1, "How things work", "how things work"),
            make_record(2, "I have this", "i have this"),
            make_record(3, "Completely unrelated", "completely unrelated"),
        ]
        system = AutoCompleteSystem.from_records(records)

        candidate_ids = system.candidate_sentence_ids("how can i have")
        results = system.get_best_k_completions("how can i have")

        self.assertIn(0, candidate_ids)
        self.assertNotIn(3, candidate_ids)
        self.assertEqual(results[0].completed_sentence, "How can I have this")


if __name__ == "__main__":
    unittest.main()
