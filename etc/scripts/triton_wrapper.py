import argparse
import importlib.util
import inspect
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, TYPE_CHECKING, Union

import triton

IS_TRITON_3 = triton.__version__.startswith("3.")


def override_getitem(self: triton.JITFunction, grid):
    def inner(*args, **kwargs):
        return self.run(
            grid=grid,
            warmup=True,  # avoids actual kernel execution
            *args,
            **kwargs,
        )

    return inner


def patch_triton(output_dir: Path):
    class GPUDriver:
        def get_current_device(self):
            return 0

        def get_current_stream(self, device):
            return 0

        def get_current_target(self):
            try:
                from triton.compiler.compiler import GPUTarget

                return GPUTarget(backend="cuda", arch=89, warp_size=32)
            except ImportError:
                # For Triton v2.3.1 and below, we don't have GPUTarget
                return ("cuda", 89)

    if IS_TRITON_3:
        triton.runtime.driver.set_active(GPUDriver())
    else:
        # For Triton v2.3.1 and below, we don't have set_active
        triton.runtime.driver._obj = GPUDriver()

    triton.JITFunction.__getitem__ = override_getitem
    # For Triton v3.1.0 and below, we don't have TRITON_DUMP_DIR
    triton.runtime.cache.default_cache_dir = lambda *args, **kwargs: output_dir


def main(input_file: Path, output_file: Path, opt_pipeline_file: Path):
    output_dir = output_file.parent.absolute()

    # Setup triton
    if opt_pipeline_file:
        os.environ["MLIR_ENABLE_DUMP"] = "1"
        os.environ["MLIR_DUMP_PATH"] = str(opt_pipeline_file)
    os.environ["TRITON_ALWAYS_COMPILE"] = "1"
    os.environ["TRITON_KERNEL_DUMP"] = "1"
    os.environ["TRITON_DUMP_DIR"] = str(output_dir)
    patch_triton(output_dir)

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
    args = parser.parse_args()
    main(
        input_file=args.input_file,
        output_file=args.output_file,
        opt_pipeline_file=args.opt_pipeline_file,
    )
