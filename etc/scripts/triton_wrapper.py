import argparse
import importlib.util
import inspect
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, TYPE_CHECKING, Union

import triton


def patch_triton(output_dir: Path, backend: str):
    """
    Patch Triton to dump the compiled kernels to output dir without actually running them.

    This is needed because
    1. Triton does not easily support such use case. There exists an AOT compiler at
       https://github.com/triton-lang/triton/blob/main/python/triton/tools/compile.py,
       but it requires a bunch of boilerplate code and also requires additional user
       input to specify the kernel name, signature, etc.
    2. Even if Triton adds such support, older versions of Triton (e.g., v2.3.x) still
       requirs such patching to work.

    This function is a collection of hacks. It has been tested to work with Triton
    2.3.0, 2.3.1, 3.0.0, 3.1.0, 3.2.0, 3.3.0, 3.3.1.
    """

    # We mock a GPU driver to avoid the need to initialize CUDA/ROCm
    # The driver is only used in runtime instead of compile time,
    # so it's safe to do this.
    class GPUDriver:
        def get_current_device(self):
            return 0

        def get_current_stream(self, device):
            return 0

        def get_current_target(self):
            try:
                from triton.compiler.compiler import GPUTarget

                if backend == "cuda":
                    return GPUTarget(backend="cuda", arch=89, warp_size=32)
                elif backend == "hip":
                    return GPUTarget(backend="hip", arch="gfx942", warp_size=32)
            except ImportError:
                # For Triton v2.3.x, we don't have GPUTarget
                return ("cuda", 89)

        @property
        def binary_ext(self):
            # For Triton v2.3.x
            return "cubin"

    try:
        from triton.runtime.driver import DriverConfig

        triton.runtime.driver.set_active(GPUDriver())
    except ImportError:
        # For Triton v2.3.x, we don't have set_active
        triton.runtime.driver._obj = GPUDriver()

    # For Triton v2.3.x, there are some driver code that goes into
    # the generic code path, so we need to patch it as well.
    try:
        from triton.compiler.backends.cuda import CUDABackend

        def make_launcher_stub(*args, **kwargs):
            tmp_file = tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w")
            tmp_file.write("def launch(): pass")
            return tmp_file.name

        CUDABackend.make_launcher_stub = make_launcher_stub
    except ImportError:
        pass

    # Usually, Triton will compile the kernel and run it when we call
    # `kernel[grid](args)`. However, we want to dump the compiled kernel
    # without actually running it. So we patch the `__getitem__` method
    # of `triton.JITFunction` with `wramup=True` to avoid actual execution.
    def override_getitem(self: triton.JITFunction, grid):
        def inner(*args, **kwargs):
            return self.run(
                grid=grid,
                warmup=True,  # avoids actual kernel execution
                *args,
                **kwargs,
            )

        return inner

    triton.JITFunction.__getitem__ = override_getitem

    # For Triton v3.1.0 and below, we don't have TRITON_DUMP_DIR
    triton.runtime.cache.default_cache_dir = lambda *args, **kwargs: output_dir


def main(input_file: Path, output_file: Path, opt_pipeline_file: Path, backend: str):
    output_dir = output_file.parent.absolute()

    # Setup triton
    if opt_pipeline_file:
        os.environ["MLIR_ENABLE_DUMP"] = "1"
        os.environ["MLIR_DUMP_PATH"] = str(opt_pipeline_file)
    os.environ["TRITON_ALWAYS_COMPILE"] = "1"
    os.environ["TRITON_KERNEL_DUMP"] = "1"
    os.environ["TRITON_DUMP_DIR"] = str(output_dir)
    patch_triton(output_dir, backend)

    # Run the script
    spec = importlib.util.spec_from_file_location("example", input_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    jit_functions = {
        name: fn
        for name, fn in inspect.getmembers(module)
        if isinstance(fn, triton.JITFunction)
    }

    # Prepare output folder
    bin_files = {}
    asm_files = {}
    for file in output_dir.rglob("*/*"):
        if file.stem in jit_functions:
            shutil.copy(file, output_dir / file.name)
            if file.suffix in (".ptx", ".amdgcn"):
                asm_files[file.stem] = file
            elif file.suffix in (".hsaco", ".cubin"):
                bin_files[file.stem] = file

    # Write the output
    with open(output_file, "w") as fout:
        for name, file in asm_files.items():
            with open(file, "r") as fin:
                fout.write(fin.read())
            fout.write("\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triton wrapper")
    parser.add_argument("--output_file", type=Path)
    parser.add_argument("--input_file", type=Path)
    parser.add_argument("--opt_pipeline_file", type=Path)
    parser.add_argument("--backend", type=str, default="cuda", choices=["cuda", "hip"])
    args = parser.parse_args()
    main(
        input_file=args.input_file,
        output_file=args.output_file,
        opt_pipeline_file=args.opt_pipeline_file,
        backend=args.backend,
    )
