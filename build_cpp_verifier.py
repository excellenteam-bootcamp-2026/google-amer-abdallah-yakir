"""Build the production CPython C++ search extensions on Windows/MSVC."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile


ROOT = Path(__file__).resolve().parent
SOURCES = (
    ROOT / "search" / "_verifier_cpp.cpp",
    ROOT / "search" / "_suffix_array_cpp.cpp",
)


def _visual_studio_environment_script() -> Path:
    program_files_x86 = Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)"))
    vswhere = program_files_x86 / "Microsoft Visual Studio/Installer/vswhere.exe"
    if not vswhere.exists():
        raise RuntimeError("Visual Studio Build Tools were not found (vswhere.exe is missing).")

    installation = subprocess.check_output(
        [
            str(vswhere),
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ],
        text=True,
    ).strip()
    if not installation:
        raise RuntimeError("A Visual Studio C++ x64 toolchain was not found.")
    return Path(installation) / "Common7/Tools/VsDevCmd.bat"


def main() -> None:
    if os.name != "nt":
        raise RuntimeError("This minimal POC build script currently supports Windows/MSVC only.")

    include_directory = Path(sysconfig.get_path("include"))
    library_directory = Path(sys.base_prefix) / "libs"
    extension_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not extension_suffix:
        raise RuntimeError("Python did not report an extension-module suffix.")

    python_library = f"python{sys.version_info.major}{sys.version_info.minor}.lib"
    environment_script = _visual_studio_environment_script()
    commands = []
    outputs = []
    for source in SOURCES:
        output = source.with_name(f"{source.stem}{extension_suffix}")
        outputs.append(output)
        commands.append(
            f'cl /nologo /O2 /EHsc /std:c++17 /LD "{source}" '
            f'/I "{include_directory}" /link /LIBPATH:"{library_directory}" '
            f'{python_library} /OUT:"{output}"'
        )
    command = (
        f'call "{environment_script}" -arch=x64 -host_arch=x64 && '
        + " && ".join(commands)
    )

    command_file: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".cmd",
            dir=ROOT,
            delete=False,
            encoding="utf-8",
        ) as script:
            script.write("@echo off\n")
            script.write(command)
            script.write("\n")
            command_file = Path(script.name)
        try:
            subprocess.run(
                ["cmd.exe", "/d", "/c", str(command_file)],
                cwd=ROOT,
                check=True,
            )
        except subprocess.CalledProcessError as error:
            raise SystemExit(
                "Native extension build failed. On Windows, LNK1104 for a "
                "search/*.pyd file usually means that a running Python process "
                "has loaded and locked it. Close running 'python part_a.py' "
                "sessions, then run this build command again."
            ) from None
    finally:
        if command_file is not None:
            command_file.unlink(missing_ok=True)
    for output in outputs:
        print(f"Built {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
