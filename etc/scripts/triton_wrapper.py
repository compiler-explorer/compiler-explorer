import argparse
import importlib.util
import inspect
import os
import shutil
import tempfile
from unittest.mock import MagicMock
from pathlib import Path
from typing import Any, Dict, TYPE_CHECKING, Union

import triton


def patch_triton(output_dir: Path, backend: str, arch: Union[int, str], warp_size: int):
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

    # Usually, Triton will compile the kernel and run it when we call
    # `kernel[grid](args)`. However, we want to dump the compiled kernel
    # without actually running it.
    # `CompiledKernel` represents a handle to a compiled kernel, ready to be
    # launched. We patch it to be a no-op.
    triton.compiler.compiler.CompiledKernel = MagicMock()

    # We mock a GPU driver to avoid the need to initialize CUDA/ROCm.
    # The driver is only used in runtime instead of compile time,
    # so it's safe to do this.
    def get_current_target():
        try:
            from triton.compiler.compiler import GPUTarget

            return GPUTarget(backend=backend, arch=arch, warp_size=warp_size)
        except ImportError:
            # For Triton v2.3.x, we don't have GPUTarget
            return (backend, arch)

    def get_benchmarker():
        return lambda kernel_call, quantiles: [0.0] * len(quantiles)

    mockGPUDriver = MagicMock(
        get_current_target=get_current_target,
        get_benchmarker=get_benchmarker,
         # This is needed for Triton v2.3.x, which doesn't support AMD, so we just assume it's CUDA
        binary_ext = "cubin",
    )

    # For Triton v2.3.x, there is no `triton.runtime.driver.set_active`,
    # so manually set the driver to the mocked one.
    try:
        from triton.runtime.driver import DriverConfig

        triton.runtime.driver.set_active(mockGPUDriver)
    except ImportError:
        triton.runtime.driver._obj = mockGPUDriver

    # For Triton v2.3.x, there are some driver code that goes into
    # the generic code path, so we need to patch it as well.
    try:
        from triton.compiler.backends.cuda import CUDABackend
        print("hi")

        CUDABackend.make_launcher_stub = MagicMock()
    except ImportError:
        pass

    # For Triton v3.1.0 and below, we don't have TRITON_DUMP_DIR
    triton.runtime.cache.default_cache_dir = lambda *args, **kwargs: output_dir


def main(
    input_file: Path,
    output_file: Path,
    opt_pipeline_file: Path,
    backend: str,
    arch: Union[int, str],
    warp_size: int,
):
    output_dir = output_file.parent.absolute()

    # Setup triton
    if opt_pipeline_file:
        os.environ["MLIR_ENABLE_DUMP"] = "1"
        os.environ["MLIR_DUMP_PATH"] = str(opt_pipeline_file)
    os.environ["TRITON_ALWAYS_COMPILE"] = "1"
    os.environ["TRITON_KERNEL_DUMP"] = "1"
    os.environ["TRITON_DUMP_DIR"] = str(output_dir)
    patch_triton(output_dir, backend, arch, warp_size)

    # Run the script by importing it as a module
    spec = importlib.util.spec_from_file_location("example", input_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    jit_functions = {
        name: fn
        for name, fn in inspect.getmembers(module)
        if isinstance(fn, (triton.JITFunction, triton.runtime.autotuner.Autotuner))
    }

    # Prepare output folder.
    # Since Triton dumps the compiled kernels to a folder with a random name,
    # we need to copy the files to the output folder.
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
    # Compiler Explorer expects the output to be a single file, so we need to
    # concatenate all the files into a single one.
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
    parser.add_argument("--arch", type=str, default=None)
    parser.add_argument("--warp_size", type=int, default=32)

    args = parser.parse_args()

    # Set some sane defaults for the arch
    if args.arch is None:
        if args.backend == "cuda":
            args.arch = 89
        elif args.backend == "hip":
            args.arch = "gfx942"

    # Triton expects the arch to be an int for CUDA and a string for HIP
    if args.backend == "cuda":
        args.arch = int(args.arch)

    main(**vars(args))
