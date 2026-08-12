"""Python correctness oracles and the production C++ verifier adapter."""

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class MatchResult:
    matched: bool
    match_offset: int | None = None
    edit_type: str | None = None


def _one_insertion_apart(shorter: str, longer: str) -> bool:
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


def verify_one_edit_reference(query: str, sentence: str) -> MatchResult:
    """Readable slice-based best-alignment oracle used only by tests."""
    if not query:
        return MatchResult(False)

    query_length = len(query)
    best_substitution: MatchResult | None = None
    best_insertion: MatchResult | None = None
    best_deletion: MatchResult | None = None

    for offset in range(len(sentence) + 1):
        same_length = sentence[offset : offset + query_length]
        if len(same_length) == query_length:
            if same_length == query:
                return MatchResult(True, offset, "exact")
            if (
                best_substitution is None
                and sum(left != right for left, right in zip(query, same_length)) == 1
            ):
                best_substitution = MatchResult(True, offset, "substitution")

        shorter_window = sentence[offset : offset + query_length - 1]
        if (
            best_insertion is None
            and len(shorter_window) == query_length - 1
            and _one_insertion_apart(shorter_window, query)
        ):
            best_insertion = MatchResult(True, offset, "insertion")

        longer_window = sentence[offset : offset + query_length + 1]
        if (
            best_deletion is None
            and len(longer_window) == query_length + 1
            and _one_insertion_apart(query, longer_window)
        ):
            best_deletion = MatchResult(True, offset, "deletion")

    return best_substitution or best_insertion or best_deletion or MatchResult(False)


def _same_length_match_type(query: str, sentence: str, start: int) -> str | None:
    mismatch_seen = False
    for query_index in range(len(query)):
        if query[query_index] != sentence[start + query_index]:
            if mismatch_seen:
                return None
            mismatch_seen = True
    return "substitution" if mismatch_seen else "exact"


def _query_has_one_extra_character(query: str, sentence: str, start: int) -> bool:
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


def verify_one_edit_python(query: str, sentence: str) -> MatchResult:
    """Index-based Python verifier with global best-match semantics."""
    if not query:
        return MatchResult(False)

    query_length = len(query)
    sentence_length = len(sentence)
    best_substitution: MatchResult | None = None
    best_insertion: MatchResult | None = None
    best_deletion: MatchResult | None = None

    for offset in range(sentence_length + 1):
        if offset + query_length <= sentence_length:
            match_type = _same_length_match_type(query, sentence, offset)
            if match_type == "exact":
                return MatchResult(True, offset, "exact")
            if match_type == "substitution" and best_substitution is None:
                best_substitution = MatchResult(True, offset, "substitution")

        if (
            best_insertion is None
            and offset + query_length - 1 <= sentence_length
            and _query_has_one_extra_character(query, sentence, offset)
        ):
            best_insertion = MatchResult(True, offset, "insertion")

        if (
            best_deletion is None
            and offset + query_length + 1 <= sentence_length
            and _query_is_missing_one_character(query, sentence, offset)
        ):
            best_deletion = MatchResult(True, offset, "deletion")

    return best_substitution or best_insertion or best_deletion or MatchResult(False)


try:
    from ._verifier_cpp import verify_one_edit_cpp as _verify_one_edit_cpp_raw
except ImportError as error:
    _verify_one_edit_cpp_raw: Callable[[str, str], tuple[bool, int | None, str | None]] | None = None
    _CPP_IMPORT_ERROR = error
else:
    _CPP_IMPORT_ERROR = None


def cpp_verifier_available() -> bool:
    return _verify_one_edit_cpp_raw is not None


def verify_one_edit_cpp(query: str, sentence: str) -> MatchResult:
    """Run the production verifier; build it with build_cpp_verifier.py."""
    if _verify_one_edit_cpp_raw is None:
        raise RuntimeError(
            "The C++ verifier is not built. Run: python build_cpp_verifier.py"
        ) from _CPP_IMPORT_ERROR

    matched, match_offset, edit_type = _verify_one_edit_cpp_raw(sentence, query)
    return MatchResult(matched, match_offset, edit_type)
