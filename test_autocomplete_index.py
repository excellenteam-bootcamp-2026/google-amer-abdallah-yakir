from pathlib import Path

from autocomplete_index import initialize_autocomplete


ARCHIVE_PATH = Path(__file__).resolve().parent / "Archive"


def main():
    print("Building indexes...")

    index = initialize_autocomplete(
        archive_path=ARCHIVE_PATH,
        ngram_size=3,
        limit=None,  # Set to a number for testing, e.g., limit=100_000
    )

    print("\nInitialization completed.")
    print(
        f"Sentences processed: "
        f"{index.sentence_count:,}"
    )
    print(
        f"Unique words: "
        f"{len(index.word_to_id):,}"
    )

    for word in [
        "the",
        "object",
        "network",
        "system",
    ]:
        sentence_ids = index.get_sentence_ids(
            word
        )

        print(f"\nWord: {word}")
        print(
            f"Sentence count: "
            f"{len(sentence_ids):,}"
        )
        print(
            "First 10 sentence IDs:",
            list(sentence_ids[:10]),
        )

    print("\nWords containing N-gram 'pyt':")
    print(
        index.get_words_for_ngram("pyt")[:20]
    )


if __name__ == "__main__":
    main()