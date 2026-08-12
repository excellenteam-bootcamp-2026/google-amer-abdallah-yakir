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
