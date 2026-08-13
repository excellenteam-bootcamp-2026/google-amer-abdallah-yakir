**Project Overview**
- **Description:** A correctness-first autocomplete system for the assignment. It builds a suffix-array index over the provided `Archive` text corpus and serves part-A autocomplete queries.
- **Language:** Python 3.14 (tested), with an optional native C++ extension for the suffix-array in `search/_suffix_array_cpp.*`.

**Quick Start**
- **Prepare:** Ensure you have Python 3.14 available.
- **Run once to build the index:**

  `python3.14 part_a.py`

  The first run builds the suffix-array index and writes a cache file at `Archive/.autocomplete_cache.pkl` for subsequent fast startup.

**Cache behavior**
- **Cache file:** `Archive/.autocomplete_cache.pkl` — the program saves a pickled index so later runs load quickly instead of rebuilding.
- **Invalidation:** The cache is invalidated automatically when the corpus directory changes (file count or latest mtime), or when you change the `--limit` used while building.
- **Force rebuild:** Delete the cache and re-run:

  `rm Archive/.autocomplete_cache.pkl && python3.14 part_a.py`

**Notes on performance & native extension**
- If the native C++ extension (`search/_suffix_array_cpp.*`) is available, the code prefers it for the initial build (much faster). The native build is then converted into a small, picklable Python index and cached.
- Measured rough first-run times on this machine (extrapolated):
  - With native extension: ~2 minutes for full Archive (approximate).
  - Without native extension (pure-Python build): slower (previously ~5 minutes extrapolated).

**How to run queries programmatically**
- Initialize in Python:

  ```python
  from part_a import initialize, get_best_k_completions
  system = initialize('Archive')
  results = get_best_k_completions('prefix text')
  for r in results:
      print(r.completed_sentence, r.score)
  ```

**Files of interest**
- **Main runner:** [part_a.py](part_a.py#L1)
- **Corpus loader:** [data_loader.py](data_loader.py#L1)
- **Pure-Python suffix array:** [search/suffix_array.py](search/suffix_array.py#L1)
- **Native extension wrapper:** [search/_suffix_array_cpp.cpp](search/_suffix_array_cpp.cpp#L1)

**Tests**
- Run the test suite with:

  `pytest -q`

**Submitting / Pushing**
- Commit your changes and push to GitHub as usual:

  `git add -A && git commit -m "final submission" && git push`

**Contact / Next steps**
- If you want stricter cache invalidation (per-file hashes), a CLI flag `--rebuild-cache`, or progress logging during the build, I can add those improvements.

---
Generated README for submission.
# Stage A autocomplete

The runtime loads `.txt` files recursively, preserves each original line and
source-line offset, normalizes one searchable copy, and builds one rank-doubling
Suffix Array. Online queries use two exact anchors to retrieve a conservative
candidate union, the C++ one-edit verifier to reject invalid candidates, and the
mentor-approved Python scorer to rank the final Top 5.

`algorithm_poc/` is retained as experiment history. Production code does not
import it. The older word, N-gram, and trie modules are also retained for
reference but are not used by `part_a.py`.

## Build the C++ verifier

This POC uses the CPython C API and the installed Windows/MSVC toolchain; it has
no third-party dependencies. From a Visual Studio-capable Windows machine run:

```powershell
python build_cpp_verifier.py
```

The script discovers Visual Studio Build Tools and the active Python headers,
then writes `search/_verifier_cpp<python-extension-suffix>.pyd`. Generated
binaries and compiler intermediates are ignored by Git and should be rebuilt
for each Python/OS environment.

## Test and run

```powershell
python -m unittest discover -v
python part_a.py --archive C:\path\to\Archive
python benchmark_stage_a.py --archive C:\path\to\Archive
```

Use `--limit 10000` for a smaller initialization check. Normal startup performs
no synthetic benchmarks or randomized tests.
