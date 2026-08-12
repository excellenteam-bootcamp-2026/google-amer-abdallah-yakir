#define PY_SSIZE_T_CLEAN
#include <Python.h>

enum class EditType { none, exact, substitution, insertion, deletion };

struct Result {
    bool matched;
    Py_ssize_t match_offset;
    EditType type;
};

struct UnicodeView {
    int kind;
    const void* data;
    Py_ssize_t length;

    Py_UCS4 operator[](Py_ssize_t index) const {
        return PyUnicode_READ(kind, data, index);
    }
};

static EditType same_length_type(
    const UnicodeView& query, const UnicodeView& sentence, Py_ssize_t start
) {
    bool mismatch_seen = false;
    for (Py_ssize_t index = 0; index < query.length; ++index) {
        if (query[index] != sentence[start + index]) {
            if (mismatch_seen) return EditType::none;
            mismatch_seen = true;
        }
    }
    return mismatch_seen ? EditType::substitution : EditType::exact;
}

static bool query_has_extra(
    const UnicodeView& query, const UnicodeView& sentence, Py_ssize_t start
) {
    Py_ssize_t query_index = 0;
    Py_ssize_t sentence_index = start;
    const Py_ssize_t sentence_end =
        start + query.length - 1;
    bool skipped = false;

    while (
        query_index < query.length &&
        sentence_index < sentence_end
    ) {
        if (query[query_index] == sentence[sentence_index]) {
            ++query_index;
            ++sentence_index;
        } else if (skipped) {
            return false;
        } else {
            skipped = true;
            ++query_index;
        }
    }
    return true;
}

static bool query_is_missing(
    const UnicodeView& query, const UnicodeView& sentence, Py_ssize_t start
) {
    Py_ssize_t query_index = 0;
    Py_ssize_t sentence_index = start;
    const Py_ssize_t sentence_end =
        start + query.length + 1;
    bool skipped = false;

    while (
        query_index < query.length &&
        sentence_index < sentence_end
    ) {
        if (query[query_index] == sentence[sentence_index]) {
            ++query_index;
            ++sentence_index;
        } else if (skipped) {
            return false;
        } else {
            skipped = true;
            ++sentence_index;
        }
    }
    return true;
}

static Result verify(const UnicodeView& sentence, const UnicodeView& query) {
    if (query.length == 0) return {false, -1, EditType::none};

    const Py_ssize_t query_length = query.length;
    const Py_ssize_t sentence_length = sentence.length;
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
        if (
            insertion_offset < 0 &&
            offset + query_length - 1 <= sentence_length &&
            query_has_extra(query, sentence, offset)
        ) {
            insertion_offset = offset;
        }
        if (
            deletion_offset < 0 &&
            offset + query_length + 1 <= sentence_length &&
            query_is_missing(query, sentence, offset)
        ) {
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

static PyObject* verify_one_edit_cpp(PyObject*, PyObject* arguments) {
    PyObject* sentence_object;
    PyObject* query_object;
    if (!PyArg_ParseTuple(
            arguments, "UU", &sentence_object, &query_object
        )) {
        return nullptr;
    }

    const UnicodeView sentence = {
        static_cast<int>(PyUnicode_KIND(sentence_object)),
        PyUnicode_DATA(sentence_object),
        PyUnicode_GET_LENGTH(sentence_object),
    };
    const UnicodeView query = {
        static_cast<int>(PyUnicode_KIND(query_object)),
        PyUnicode_DATA(query_object),
        PyUnicode_GET_LENGTH(query_object),
    };

    const Result result = verify(sentence, query);
    if (!result.matched) {
        return Py_BuildValue("(OOO)", Py_False, Py_None, Py_None);
    }

    const char* edit_type = nullptr;
    switch (result.type) {
        case EditType::exact: edit_type = "exact"; break;
        case EditType::substitution: edit_type = "substitution"; break;
        case EditType::insertion: edit_type = "insertion"; break;
        case EditType::deletion: edit_type = "deletion"; break;
        default: Py_UNREACHABLE();
    }
    return Py_BuildValue("(Ons)", Py_True, result.match_offset, edit_type);
}

static PyMethodDef methods[] = {
    {
        "verify_one_edit_cpp",
        verify_one_edit_cpp,
        METH_VARARGS,
        "Return (matched, match_offset, edit_type) for at most one edit."
    },
    {nullptr, nullptr, 0, nullptr},
};

static PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_verifier_cpp",
    "Minimal C++ one-edit verifier.",
    -1,
    methods,
};

PyMODINIT_FUNC PyInit__verifier_cpp() {
    return PyModule_Create(&module);
}
