from pathlib import Path

from data_loader import iter_sentence_records, iter_text_files


ARCHIVE_PATH = Path(__file__).resolve().parent / "Archive"


def main():
    text_file_count = len(list(iter_text_files(ARCHIVE_PATH)))

    sentence_count = 0
    known_example = None

    for record in iter_sentence_records(ARCHIVE_PATH):
        sentence_count += 1

        if (
            record.source_text
            == "python-3.8.4-docs-text/c-api/arg.txt"
            and record.source_offset == 332
        ):
            known_example = record

    print(f"Text files found: {text_file_count:,}")
    print(f"Searchable records found: {sentence_count:,}")

    if known_example:
        print("\nKnown assignment example:")
        print(f"Original: {known_example.completed_sentence}")
        print(f"Normalized: {known_example.normalized_sentence}")
        print(f"Source: {known_example.source_text}")
        print(f"Offset: {known_example.source_offset}")
    else:
        print("Known assignment example was not found.")

    if (
        text_file_count == 1_504
        and sentence_count == 2_392_247
        and known_example is not None
    ):
        print("\nVERIFICATION PASSED")
    else:
        print("\nVERIFICATION FAILED")


if __name__ == "__main__":
    main()
