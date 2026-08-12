from itertools import islice
from pathlib import Path

from data_loader import iter_sentence_records
from sentence_word_index import build_sentence_word_index


ARCHIVE_PATH = Path(__file__).resolve().parent / "Archive"
TEST_LIMIT = 100_000


def main():
    print(f"Reading the first {TEST_LIMIT:,} records...")

    records = iter_sentence_records(
    ARCHIVE_PATH
)

    index = build_sentence_word_index(records)

    print("Index completed.")
    print(f"Unique indexed words: {len(index):,}")

    words_to_test = [
        "the",
        "object",
        "network",
        "system",
    ]

    for word in words_to_test:
        sentence_ids = index.get(word)

        if sentence_ids is None:
            print(f"\n'{word}' was not found.")
            continue

        print(f"\nWord: {word}")
        print(f"Number of sentences: {len(sentence_ids):,}")
        print(
            "First 10 sentence IDs:",
            list(sentence_ids[:10]),
        )


if __name__ == "__main__":
    main()