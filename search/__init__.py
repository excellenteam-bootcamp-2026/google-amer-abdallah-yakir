"""Stage A candidate retrieval and verification components."""

from .suffix_array import Occurrence, SuffixArrayIndex
from .native_suffix_array import NativeSuffixArrayIndex
from .verifier import (
    MatchResult,
    cpp_verifier_available,
    verify_one_edit_cpp,
    verify_one_edit_python,
    verify_one_edit_reference,
)

__all__ = [
    "MatchResult",
    "NativeSuffixArrayIndex",
    "Occurrence",
    "SuffixArrayIndex",
    "cpp_verifier_available",
    "verify_one_edit_cpp",
    "verify_one_edit_python",
    "verify_one_edit_reference",
]
