"""Read the autocomplete corpus and prepare sentences for N-gram indexing."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Union


PathLike = Union[str, Path]


@dataclass(frozen=True)
class SentenceRecord:
    """Stores one searchable line from a text file."""

    sentence_id: int
    completed_sentence: str
    normalized_sentence: str
    source_text: str
    offset: int


def normalize_text(text: str) -> str:
    """
    Prepare text for searching and N-grams.

    - Converts uppercase letters to lowercase.
    - Replaces punctuation with spaces.
    - Replaces multiple spaces with one space.
    - Preserves letters and numbers.
    """

    normalized_characters = []
    separator_pending = False

    for character in text.casefold():
        if character.isalnum():
            if separator_pending and normalized_characters:
                normalized_characters.append(" ")

            normalized_characters.append(character)
            separator_pending = False
        else:
            separator_pending = True

    return "".join(normalized_characters)


def iter_text_files(root_directory: PathLike) -> Iterator[Path]:
    """
    Find every .txt file inside the archive and its subfolders.
    """

    root = Path(root_directory).expanduser()

    if not root.exists():
        raise FileNotFoundError(
            f"Corpus directory does not exist: {root}"
        )

    if not root.is_dir():
        raise NotADirectoryError(
            f"Corpus path is not a directory: {root}"
        )

    text_files = (
        path
        for path in root.rglob("*.txt")
        if path.is_file()
    )

    # Sorting gives every file a stable reading order.
    yield from sorted(text_files)


def iter_sentence_records(
    root_directory: PathLike,
) -> Iterator[SentenceRecord]:
    """
    Read every searchable line from all text files.

    The function is a generator, so it does not load the entire
    119 MB dataset into memory at the same time.

    The offset is the zero-based physical line number in the file.
    Empty lines and punctuation-only lines are ignored.
    """

    root = Path(root_directory).expanduser().resolve()
    sentence_id = 0

    for file_path in iter_text_files(root):
        # Store a relative path instead of the complete computer path.
        source_text = file_path.relative_to(root).as_posix()

        with file_path.open(
            mode="r",
            encoding="utf-8-sig",
            errors="replace",
            newline=None,
        ) as source_file:

            for offset, raw_line in enumerate(source_file):
                # Remove leading/trailing spaces and newline characters.
                completed_sentence = raw_line.strip()

                if not completed_sentence:
                    continue

                normalized_sentence = normalize_text(
                    completed_sentence
                )

                # Ignore lines containing punctuation but no searchable text.
                if not normalized_sentence:
                    continue

                yield SentenceRecord(
                    sentence_id=sentence_id,
                    completed_sentence=completed_sentence,
                    normalized_sentence=normalized_sentence,
                    source_text=source_text,
                    offset=offset,
                )

                sentence_id += 1


def load_sentence_records(
    root_directory: PathLike,
) -> List[SentenceRecord]:
    """
    Load every sentence into a list.

    This is convenient for small tests, but the generator
    iter_sentence_records() is safer for the complete dataset.
    """

    return list(iter_sentence_records(root_directory))