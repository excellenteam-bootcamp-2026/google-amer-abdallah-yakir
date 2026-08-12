#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <string_view>

enum class EditType { none, exact, substitution, insertion, deletion };

struct Result {
    bool matched;
    Py_ssize_t offset;
    EditType type;
};

static EditType same_length_type(
    std::string_view query, std::string_view sentence, Py_ssize_t start
) {
    bool mismatch_seen = false;
    for (Py_ssize_t i = 0; i < static_cast<Py_ssize_t>(query.size()); ++i) {
        if (query[i] != sentence[start + i]) {
            if (mismatch_seen) return EditType::none;
            mismatch_seen = true;
        }
    }
    return mismatch_seen ? EditType::substitution : EditType::exact;
}

static bool query_has_extra(
    std::string_view query, std::string_view sentence, Py_ssize_t start
) {
    Py_ssize_t qi = 0;
    Py_ssize_t si = start;
    const Py_ssize_t end = start + static_cast<Py_ssize_t>(query.size()) - 1;
    bool skipped = false;
    while (qi < static_cast<Py_ssize_t>(query.size()) && si < end) {
        if (query[qi] == sentence[si]) {
            ++qi;
            ++si;
        } else if (skipped) {
            return false;
        } else {
            skipped = true;
            ++qi;
        }
    }
    return true;
}

static bool query_is_missing(
    std::string_view query, std::string_view sentence, Py_ssize_t start
) {
    Py_ssize_t qi = 0;
    Py_ssize_t si = start;
    const Py_ssize_t end = start + static_cast<Py_ssize_t>(query.size()) + 1;
    bool skipped = false;
    while (qi < static_cast<Py_ssize_t>(query.size()) && si < end) {
        if (query[qi] == sentence[si]) {
            ++qi;
            ++si;
        } else if (skipped) {
            return false;
        } else {
            skipped = true;
            ++si;
        }
    }
    return true;
}

static Result verify(std::string_view sentence, std::string_view query) {
    if (query.empty()) return {false, -1, EditType::none};

    const auto query_length = static_cast<Py_ssize_t>(query.size());
    const auto sentence_length = static_cast<Py_ssize_t>(sentence.size());
    Py_ssize_t substitution_offset = -1;
    Py_ssize_t insertion_offset = -1;
    Py_ssize_t deletion_offset = -1;
    for (Py_ssize_t offset = 0; offset <= sentence_length; ++offset) {
        if (offset + query_length <= sentence_length) {
            const EditType type = same_length_type(query, sentence, offset);
            if (type == EditType::exact) return {true, offset, type};
            if (type == EditType::substitution && substitution_offset < 0) {
                substitution_offset = offset;
            }
        }
        if (insertion_offset < 0 && offset + query_length - 1 <= sentence_length &&
            query_has_extra(query, sentence, offset)) {
            insertion_offset = offset;
        }
        if (deletion_offset < 0 && offset + query_length + 1 <= sentence_length &&
            query_is_missing(query, sentence, offset)) {
            deletion_offset = offset;
        }
    }
    if (substitution_offset >= 0) {
        return {true, substitution_offset, EditType::substitution};
    }
    if (insertion_offset >= 0) {
        return {true, insertion_offset, EditType::insertion};
    }
    if (deletion_offset >= 0) {
        return {true, deletion_offset, EditType::deletion};
    }
    return {false, -1, EditType::none};
}

static PyObject* verify_one_edit_cpp(PyObject*, PyObject* args) {
    PyObject* sentence_object;
    PyObject* query_object;
    if (!PyArg_ParseTuple(args, "UU", &sentence_object, &query_object)) return nullptr;

    Py_ssize_t sentence_size;
    Py_ssize_t query_size;
    const char* sentence = PyUnicode_AsUTF8AndSize(sentence_object, &sentence_size);
    const char* query = PyUnicode_AsUTF8AndSize(query_object, &query_size);
    if (sentence == nullptr || query == nullptr) return nullptr;

    const Result result = verify(
        std::string_view(sentence, static_cast<size_t>(sentence_size)),
        std::string_view(query, static_cast<size_t>(query_size))
    );
    if (!result.matched) return Py_BuildValue("(OOO)", Py_False, Py_None, Py_None);

    const char* type = nullptr;
    switch (result.type) {
        case EditType::exact: type = "exact"; break;
        case EditType::substitution: type = "substitution"; break;
        case EditType::insertion: type = "insertion"; break;
        case EditType::deletion: type = "deletion"; break;
        default: Py_UNREACHABLE();
    }
    return Py_BuildValue("(Ons)", Py_True, result.offset, type);
}

static PyMethodDef methods[] = {
    {"verify_one_edit_cpp", verify_one_edit_cpp, METH_VARARGS,
     "Verify an English-text substring match with at most one edit."},
    {nullptr, nullptr, 0, nullptr},
};

static PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "suffix_array_verifier_cpp",
    "Minimal C++ one-edit verifier.",
    -1,
    methods,
};

PyMODINIT_FUNC PyInit_suffix_array_verifier_cpp() {
    return PyModule_Create(&module);
}
