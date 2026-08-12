from itertools import islice
from pathlib import Path

from data_loader import iter_sentence_records
from word_index import build_word_index


ARCHIVE_PATH = Path(__file__).resolve().parent / "Archive"


def contains_python(record):
    words = record.normalized_sentence.split()
    return "python" in words


def main():
    print("Searching for records containing 'python'...")

    all_records = iter_sentence_records(ARCHIVE_PATH)

    # Scan the dataset but keep only the first 100 records
    # that contain the complete word "python".
    python_records = islice(
        filter(contains_python, all_records),
        100,
    )

    index = build_word_index(
        python_records,
        ngram_size=3,
    )

    sentence_ids = index.get_sentence_ids("python")

    print("Word index completed.")
    print(f"Unique words: {len(index.word_to_id):,}")
    print(
        f"Sentences containing 'python': "
        f"{len(sentence_ids)}"
    )
    print(
        "First sentence IDs:",
        sentence_ids[:10],
    )

    print("\nN-grams for 'python':")

    for ngram in ["pyt", "yth", "tho", "hon"]:
        word_ids = index.word_ids_by_ngram.get(
            ngram,
            [],
        )

        words = [
            index.get_word(word_id)
            for word_id in word_ids
        ]

        print(f"{ngram}: {words[:10]}")

    if "python" in index.word_to_id:
        print("\nTEST PASSED: 'python' was indexed.")
    else:
        print("\nTEST FAILED: 'python' was not indexed.")


if __name__ == "__main__":
    main()