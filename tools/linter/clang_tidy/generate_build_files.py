from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


BUILD_DIR = "build/clang-tidy"


def run_cmd(cmd: list[str]) -> None:
    print(f"Running: {cmd}")
    result = subprocess.run(
        cmd,
        capture_output=True,
    )
    stdout, stderr = (
        result.stdout.decode("utf-8").strip(),
        result.stderr.decode("utf-8").strip(),
    )
    print(stdout)
    print(stderr)
    if result.returncode != 0:
        print(f"Failed to run {cmd}")
        sys.exit(1)


def normalize_gcc_commands() -> None:
    path = Path(BUILD_DIR, "compile_commands.json")
    commands = json.loads(path.read_text())
    header_commands = []
    for command in commands:
        args = [arg for arg in shlex.split(command["command"]) if arg != "-fopenmp"]
        command["command"] = shlex.join(args)

        header_args = []
        i = 0
        while i < len(args):
            if args[i] == "-Winvalid-pch":
                i += 1
                continue
            if args[i] == "-include" and "cmake_pch.hxx" in args[i + 1]:
                i += 2
                continue
            header_args.append(args[i])
            i += 1
        header_commands.append({**command, "command": shlex.join(header_args)})

    path.write_text(json.dumps(commands))
    header_dir = Path(BUILD_DIR, "headers")
    header_dir.mkdir(exist_ok=True)
    Path(header_dir, "compile_commands.json").write_text(json.dumps(header_commands))

    install_dir = subprocess.check_output(
        ["g++", "-print-search-dirs"], text=True
    ).splitlines()[0]
    toolchain = Path(install_dir.removeprefix("install: ")).resolve().parents[3]
    builtin_include = subprocess.check_output(
        ["g++", "-print-file-name=include"], text=True
    ).strip()
    Path(BUILD_DIR, "gcc_clang_tidy_args.json").write_text(
        json.dumps(
            [
                f"--gcc-toolchain={toolchain}",
                f"-I{builtin_include}",
                "-Wno-invalid-constexpr",
            ]
        )
    )


def update_submodules() -> None:
    run_cmd(["git", "submodule", "update", "--init", "--recursive"])


def gen_compile_commands() -> None:
    """Configure cmake to produce compile_commands.json for clang-tidy.

    Configure-only invocation; does not run the build step. The repo-level
    cmake/EnvVarForwarding.cmake forwards BUILD_*/USE_* environment
    variables to the corresponding CMake cache variables, so setting them
    in os.environ before this call propagates them through to CMake.
    """
    os.environ["USE_NCCL"] = "0"
    cc = os.environ.get("CLANGTIDY_CC")
    cxx = os.environ.get("CLANGTIDY_CXX")
    if cc:
        os.environ["CC"] = cc
    if cxx:
        os.environ["CXX"] = cxx
    os.environ["USE_PRECOMPILED_HEADERS"] = "1"
    run_cmd(["cmake", "-S", ".", "-B", BUILD_DIR, "-G", "Ninja"])
    if cc != "clang":
        normalize_gcc_commands()


def run_autogen() -> None:
    run_cmd(
        [
            sys.executable,
            "-m",
            "torchgen.gen",
            "-s",
            "aten/src/ATen",
            "-d",
            f"{BUILD_DIR}/aten/src/ATen",
            "--per-operator-headers",
        ]
    )

    run_cmd(
        [
            sys.executable,
            "tools/setup_helpers/generate_code.py",
            "--native-functions-path",
            "aten/src/ATen/native/native_functions.yaml",
            "--tags-path",
            "aten/src/ATen/native/tags.yaml",
            "--gen-lazy-ts-backend",
        ]
    )


def generate_build_files() -> None:
    update_submodules()
    gen_compile_commands()
    run_autogen()


if __name__ == "__main__":
    generate_build_files()
