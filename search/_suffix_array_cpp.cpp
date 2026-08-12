#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <new>
#include <vector>

struct NativeIndex {
    PyObject* corpus = nullptr;
    std::vector<uint32_t> suffix_array;
    std::vector<uint32_t> sentence_starts;
    uint64_t build_peak_native_bytes = 0;

    ~NativeIndex() { Py_XDECREF(corpus); }
};

static const char* capsule_name = "search.NativeSuffixArrayIndex";

static NativeIndex* get_index(PyObject* capsule) {
    return static_cast<NativeIndex*>(PyCapsule_GetPointer(capsule, capsule_name));
}

static void destroy_index(PyObject* capsule) {
    void* pointer = PyCapsule_GetPointer(capsule, capsule_name);
    if (pointer != nullptr) delete static_cast<NativeIndex*>(pointer);
    else PyErr_Clear();
}

static PyObject* build(PyObject*, PyObject* sentences_object) {
    PyObject* sentences = PySequence_Fast(sentences_object, "sentences must be a sequence");
    if (sentences == nullptr) return nullptr;

    const Py_ssize_t sentence_count = PySequence_Fast_GET_SIZE(sentences);
    PyObject* separator = PyUnicode_FromStringAndSize("\0", 1);
    if (separator == nullptr) { Py_DECREF(sentences); return nullptr; }
    PyObject* corpus = PyUnicode_Join(separator, sentences);
    Py_DECREF(separator);
    if (corpus == nullptr) { Py_DECREF(sentences); return nullptr; }

    const Py_ssize_t corpus_length_signed = PyUnicode_GET_LENGTH(corpus);
    if (corpus_length_signed < 0 ||
        static_cast<uint64_t>(corpus_length_signed) > std::numeric_limits<uint32_t>::max()) {
        Py_DECREF(corpus); Py_DECREF(sentences);
        PyErr_SetString(PyExc_OverflowError, "corpus exceeds uint32_t suffix positions");
        return nullptr;
    }
    const uint32_t corpus_length = static_cast<uint32_t>(corpus_length_signed);

    NativeIndex* index = new (std::nothrow) NativeIndex();
    if (index == nullptr) { Py_DECREF(corpus); Py_DECREF(sentences); return PyErr_NoMemory(); }
    index->corpus = corpus;

    try {
        index->sentence_starts.reserve(static_cast<size_t>(sentence_count));
        uint64_t next_start = 0;
        for (Py_ssize_t sentence_id = 0; sentence_id < sentence_count; ++sentence_id) {
            PyObject* sentence = PySequence_Fast_GET_ITEM(sentences, sentence_id);
            if (!PyUnicode_Check(sentence)) {
                Py_DECREF(sentences); delete index;
                PyErr_SetString(PyExc_TypeError, "every sentence must be str");
                return nullptr;
            }
            if (next_start > std::numeric_limits<uint32_t>::max()) {
                Py_DECREF(sentences); delete index;
                PyErr_SetString(PyExc_OverflowError, "sentence start exceeds uint32_t");
                return nullptr;
            }
            index->sentence_starts.push_back(static_cast<uint32_t>(next_start));
            next_start += static_cast<uint64_t>(PyUnicode_GET_LENGTH(sentence)) + 1;
        }

        std::vector<uint32_t> text(corpus_length);
        index->suffix_array.resize(corpus_length);
        std::vector<uint32_t> ranks(corpus_length);
        std::vector<uint32_t> new_ranks(corpus_length);
        index->build_peak_native_bytes =
            static_cast<uint64_t>(corpus_length) * sizeof(uint32_t) * 4 +
            static_cast<uint64_t>(index->sentence_starts.capacity()) * sizeof(uint32_t);

        const int kind = PyUnicode_KIND(corpus);
        const void* data = PyUnicode_DATA(corpus);
        for (uint32_t position = 0; position < corpus_length; ++position) {
            text[position] = PyUnicode_READ(kind, data, position);
            index->suffix_array[position] = position;
            ranks[position] = text[position];
        }

        Py_BEGIN_ALLOW_THREADS
        for (uint32_t prefix_length = 1; prefix_length < corpus_length;) {
            auto second_rank = [&](uint32_t position) -> int64_t {
                return position + prefix_length < corpus_length
                    ? static_cast<int64_t>(ranks[position + prefix_length])
                    : static_cast<int64_t>(-1);
            };
            std::sort(
                index->suffix_array.begin(), index->suffix_array.end(),
                [&](uint32_t left, uint32_t right) {
                    if (ranks[left] != ranks[right]) return ranks[left] < ranks[right];
                    return second_rank(left) < second_rank(right);
                }
            );

            uint32_t rank = 0;
            new_ranks[index->suffix_array[0]] = 0;
            for (size_t suffix_index = 1; suffix_index < index->suffix_array.size(); ++suffix_index) {
                const uint32_t previous = index->suffix_array[suffix_index - 1];
                const uint32_t current = index->suffix_array[suffix_index];
                if (ranks[previous] != ranks[current] ||
                    second_rank(previous) != second_rank(current)) ++rank;
                new_ranks[current] = rank;
            }
            ranks.swap(new_ranks);
            if (rank == corpus_length - 1) break;
            if (prefix_length > corpus_length / 2) break;
            prefix_length *= 2;
        }
        Py_END_ALLOW_THREADS
    } catch (const std::bad_alloc&) {
        Py_DECREF(sentences); delete index; return PyErr_NoMemory();
    }

    Py_DECREF(sentences);
    PyObject* capsule = PyCapsule_New(index, capsule_name, destroy_index);
    if (capsule == nullptr) delete index;
    return capsule;
}

static int compare_suffix_prefix(const NativeIndex* index, uint32_t position, PyObject* pattern) {
    const Py_ssize_t corpus_length = PyUnicode_GET_LENGTH(index->corpus);
    const Py_ssize_t pattern_length = PyUnicode_GET_LENGTH(pattern);
    const int corpus_kind = PyUnicode_KIND(index->corpus);
    const void* corpus_data = PyUnicode_DATA(index->corpus);
    const int pattern_kind = PyUnicode_KIND(pattern);
    const void* pattern_data = PyUnicode_DATA(pattern);
    for (Py_ssize_t offset = 0; offset < pattern_length; ++offset) {
        if (static_cast<Py_ssize_t>(position) + offset >= corpus_length) return -1;
        const Py_UCS4 left = PyUnicode_READ(corpus_kind, corpus_data, position + offset);
        const Py_UCS4 right = PyUnicode_READ(pattern_kind, pattern_data, offset);
        if (left < right) return -1;
        if (left > right) return 1;
    }
    return 0;
}

static PyObject* exact_sentence_ids(PyObject*, PyObject* arguments) {
    PyObject* capsule; PyObject* pattern;
    if (!PyArg_ParseTuple(arguments, "OU", &capsule, &pattern)) return nullptr;
    NativeIndex* index = get_index(capsule);
    if (index == nullptr) return nullptr;
    if (PyUnicode_GET_LENGTH(pattern) == 0) return PySet_New(nullptr);

    size_t low = 0, high = index->suffix_array.size();
    while (low < high) {
        const size_t middle = low + (high - low) / 2;
        if (compare_suffix_prefix(index, index->suffix_array[middle], pattern) < 0) low = middle + 1;
        else high = middle;
    }
    const size_t first = low;
    high = index->suffix_array.size();
    while (low < high) {
        const size_t middle = low + (high - low) / 2;
        if (compare_suffix_prefix(index, index->suffix_array[middle], pattern) <= 0) low = middle + 1;
        else high = middle;
    }

    PyObject* result = PySet_New(nullptr);
    if (result == nullptr) return nullptr;
    for (size_t suffix_index = first; suffix_index < low; ++suffix_index) {
        const uint32_t position = index->suffix_array[suffix_index];
        auto iterator = std::upper_bound(
            index->sentence_starts.begin(), index->sentence_starts.end(), position
        );
        if (iterator == index->sentence_starts.begin()) continue;
        const uint32_t sentence_id = static_cast<uint32_t>(iterator - index->sentence_starts.begin() - 1);
        PyObject* value = PyLong_FromUnsignedLong(sentence_id);
        if (value == nullptr || PySet_Add(result, value) < 0) {
            Py_XDECREF(value); Py_DECREF(result); return nullptr;
        }
        Py_DECREF(value);
    }
    return result;
}

static PyObject* corpus(PyObject*, PyObject* capsule) {
    NativeIndex* index = get_index(capsule); if (index == nullptr) return nullptr;
    return Py_NewRef(index->corpus);
}
static PyObject* sentence_count(PyObject*, PyObject* capsule) {
    NativeIndex* index = get_index(capsule); if (index == nullptr) return nullptr;
    return PyLong_FromSize_t(index->sentence_starts.size());
}
static PyObject* suffix_count(PyObject*, PyObject* capsule) {
    NativeIndex* index = get_index(capsule); if (index == nullptr) return nullptr;
    return PyLong_FromSize_t(index->suffix_array.size());
}
static PyObject* suffix_at(PyObject*, PyObject* arguments) {
    PyObject* capsule; Py_ssize_t offset;
    if (!PyArg_ParseTuple(arguments, "On", &capsule, &offset)) return nullptr;
    NativeIndex* index = get_index(capsule); if (index == nullptr) return nullptr;
    if (offset < 0 || static_cast<size_t>(offset) >= index->suffix_array.size()) {
        PyErr_SetString(PyExc_IndexError, "suffix index out of range"); return nullptr;
    }
    return PyLong_FromUnsignedLong(index->suffix_array[static_cast<size_t>(offset)]);
}
static PyObject* memory_bytes(PyObject*, PyObject* capsule) {
    NativeIndex* index = get_index(capsule); if (index == nullptr) return nullptr;
    const uint64_t persistent =
        static_cast<uint64_t>(index->suffix_array.capacity()) * sizeof(uint32_t) +
        static_cast<uint64_t>(index->sentence_starts.capacity()) * sizeof(uint32_t);
    return Py_BuildValue("(KK)",
        static_cast<unsigned long long>(persistent),
        static_cast<unsigned long long>(index->build_peak_native_bytes));
}

static PyMethodDef methods[] = {
    {"build", (PyCFunction)build, METH_O, "Build a packed native suffix array."},
    {"exact_sentence_ids", exact_sentence_ids, METH_VARARGS, "Return exact matching sentence IDs."},
    {"corpus", (PyCFunction)corpus, METH_O, "Return the retained corpus."},
    {"sentence_count", (PyCFunction)sentence_count, METH_O, "Return sentence count."},
    {"suffix_count", (PyCFunction)suffix_count, METH_O, "Return suffix count."},
    {"suffix_at", suffix_at, METH_VARARGS, "Return one suffix position for tests."},
    {"memory_bytes", (PyCFunction)memory_bytes, METH_O, "Return persistent and peak native bytes."},
    {nullptr, nullptr, 0, nullptr},
};
static PyModuleDef module = {PyModuleDef_HEAD_INIT, "_suffix_array_cpp", nullptr, -1, methods};
PyMODINIT_FUNC PyInit__suffix_array_cpp() { return PyModule_Create(&module); }
