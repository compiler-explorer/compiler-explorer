# Copyright (c) 2026, Compiler Explorer Authors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import argparse
import contextlib
import io
import os
import re
import runpy
import sys
from pathlib import Path


def add_python_paths(raw_paths: str) -> None:
    if not raw_paths:
        return
    for raw_path in reversed(raw_paths.split(os.pathsep)):
        if raw_path:
            sys.path.insert(0, raw_path)


def patch_runtime_calls() -> None:
    """Prevent example scripts from launching kernels after cute.compile."""
    try:
        from cutlass.base_dsl import jit_executor

        def no_run(*_args, **_kwargs):
            return None

        jit_executor.JitCompiledFunction.__call__ = no_run
        jit_executor.JitExecutor.__call__ = no_run
    except Exception:
        pass

    try:
        from cutlass.cutlass_dsl import tvm_ffi_provider

        def no_run_tvm_ffi(*_args, **_kwargs):
            return None

        tvm_ffi_provider.TVMFFIJitCompiledFunction.__call__ = no_run_tvm_ffi
        tvm_ffi_provider.TVMFFIJitCompiledFunctionWithKwargs.__call__ = no_run_tvm_ffi
    except Exception:
        pass


def has_compile_option(options: str, option_name: str) -> bool:
    for token in options.split():
        if token == option_name or token.startswith(f"{option_name}="):
            return True
    return False


def merge_compile_options(options: object, default_options: str) -> object:
    if options is None or options == "":
        return default_options
    if not isinstance(options, str):
        return options

    merged = options
    for token in default_options.split():
        option_name = token.split("=", 1)[0]
        if not has_compile_option(merged, option_name):
            merged = f"{merged} {token}"
    return merged


def is_unsupported_option_error(error: Exception) -> bool:
    error_text = str(error)
    return "Invalid compile options" in error_text or "unrecognized arguments" in error_text


def safe_name(raw_name: object, fallback: str) -> str:
    name = str(raw_name or fallback)
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
    return name or fallback


def flush_stream(stream: object) -> None:
    try:
        stream.flush()
    except Exception:
        pass


@contextlib.contextmanager
def suppress_output_fds():
    stdout_copy = os.dup(1)
    stderr_copy = os.dup(2)
    flush_stream(sys.stdout)
    flush_stream(sys.stderr)
    try:
        with open(os.devnull, "wb") as devnull:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        os.dup2(stdout_copy, 1)
        os.dup2(stderr_copy, 2)
        os.close(stdout_copy)
        os.close(stderr_copy)


def dump_compile_result(result: object, dump_dir: Path, fallback_name: str) -> None:
    ir_module = getattr(result, "ir_module", None)
    if ir_module is not None and not any(dump_dir.rglob("*.mlir")):
        name = safe_name(getattr(result, "function_name", None), fallback_name)
        (dump_dir / f"{name}.mlir").write_text(str(ir_module))

    if ir_module is None or any(dump_dir.rglob("*.cubin")):
        return

    walk_module = getattr(result, "walk_module_and_get_cubin_data", None)
    if not callable(walk_module):
        return

    seen_data: set[bytes] = set()

    def write_cubin(sym, func_sym, cubin_data):
        name = safe_name(func_sym or sym, fallback_name)
        if cubin_data in seen_data:
            return
        seen_data.add(cubin_data)
        (dump_dir / f"{name}.cubin").write_bytes(cubin_data)

    for sym in [getattr(result, "function_name", None), "kernels"]:
        if not sym:
            continue
        try:
            walk_module(ir_module, str(sym), write_cubin)
        except Exception:
            pass


def patch_compile_defaults(cute_module, arch: str, dump_dir: Path) -> None:
    """Inject file-dump options for CuTe DSL versions that do not read env vars."""
    default_options = f"--gpu-arch={arch} --dump-dir={dump_dir} --keep-ptx --keep-cubin"
    original_compile = cute_module.compile

    class CompileWithDefaults:
        def __init__(self, delegate):
            self.delegate = delegate

        def __call__(self, *args, **kwargs):
            fallback_name = getattr(args[0], "__name__", "cutedsl") if args else "cutedsl"
            original_kwargs = dict(kwargs)
            kwargs["options"] = merge_compile_options(kwargs.get("options"), default_options)
            stderr_buffer = io.StringIO()
            try:
                with contextlib.redirect_stderr(stderr_buffer):
                    result = self.delegate(*args, **kwargs)
            except Exception as exc:
                if not is_unsupported_option_error(exc):
                    sys.stderr.write(stderr_buffer.getvalue())
                    raise
                result = self.delegate(*args, **original_kwargs)
            dump_compile_result(result, dump_dir, fallback_name)
            return result

        def __getitem__(self, options):
            return CompileWithDefaults(self.delegate[merge_compile_options(options, default_options)])

        def __getattr__(self, name):
            return getattr(self.delegate, name)

    cute_module.compile = CompileWithDefaults(original_compile)


def write_sass_outputs(dump_dir: Path) -> None:
    try:
        from triton.tools import disasm
    except Exception:
        return

    for cubin_file in dump_dir.rglob("*.cubin"):
        sass_file = cubin_file.with_suffix(".sass")
        try:
            with suppress_output_fds():
                sass_file.write_text(disasm.get_sass(cubin_file.read_bytes()))
        except Exception:
            sass_file.unlink(missing_ok=True)


def write_primary_output(output_file: Path, dump_dir: Path) -> None:
    write_sass_outputs(dump_dir)
    ptx_files = [
        path
        for path in sorted(dump_dir.rglob("*.ptx"), key=lambda item: (item.stat().st_mtime_ns, str(item)))
        if path.resolve() != output_file.resolve()
    ]
    mlir_files = sorted(dump_dir.rglob("*.mlir"), key=lambda item: (item.stat().st_mtime_ns, str(item)))

    if ptx_files:
        with output_file.open("w") as output:
            for path in ptx_files:
                output.write(f"// {path.name}\n")
                output.write(path.read_text(errors="replace"))
                output.write("\n\n")
        return

    message = [
        "// No CuTe DSL PTX was generated.",
        "// Make sure the source imports cutlass.cute and calls cute.compile(...).",
    ]
    if mlir_files:
        message.append("// Generated MLIR is available in the device-code view.")
    output_file.write_text("\n".join(message) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="CuTe DSL Compiler Explorer wrapper")
    parser.add_argument("input_file", type=Path, help="Path to the input Python file")
    parser.add_argument("--output_file", type=Path, required=True, help="Path to the output PTX file")
    parser.add_argument("--arch", default="sm_90a", help="Value for CUTE_DSL_ARCH")
    parser.add_argument("--keep", default=None, help="Value for CUTE_DSL_KEEP")
    parser.add_argument("--python_path", default="", help="Extra PYTHONPATH entries separated by os.pathsep")
    parser.add_argument("--no_runtime_patch", action="store_true", help="Allow compiled kernels to run")
    args, source_args = parser.parse_known_args()

    output_file = args.output_file
    dump_dir = output_file.parent
    dump_dir.mkdir(parents=True, exist_ok=True)
    output_file.unlink(missing_ok=True)

    add_python_paths(args.python_path)
    add_python_paths(os.environ.get("CUTE_DSL_PYTHONPATH", ""))

    os.environ["CUTE_DSL_ARCH"] = args.arch
    os.environ["CUTE_DSL_DUMP_DIR"] = str(dump_dir)
    os.environ["CUTE_DSL_KEEP"] = args.keep or os.environ.get("CUTE_DSL_KEEP", "ir,ptx,cubin")
    os.environ.setdefault("CUTE_DSL_KEEP_IR", "1")
    os.environ.setdefault("CUTE_DSL_KEEP_PTX", "1")
    os.environ.setdefault("CUTE_DSL_KEEP_CUBIN", "1")
    os.environ.setdefault("CUTE_DSL_NO_CACHE", "1")
    os.environ.setdefault("CUTE_DSL_DISABLE_FILE_CACHING", "1")
    os.environ.setdefault("CUDA_TOOLKIT_PATH", "/usr/local")

    try:
        import cutlass.cute as cute
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "CuTe DSL is not importable. Use a Python with cutlass.cute installed, "
            f"or pass --python_path /path/to/cutlass/python/CuTeDSL. Missing module: {exc.name}."
        ) from exc

    patch_compile_defaults(cute, args.arch, dump_dir)

    if not args.no_runtime_patch:
        patch_runtime_calls()

    old_argv = sys.argv
    try:
        sys.argv = [str(args.input_file), *source_args]
        runpy.run_path(str(args.input_file), run_name="__main__")
    finally:
        sys.argv = old_argv

    write_primary_output(output_file, dump_dir)


if __name__ == "__main__":
    main()
