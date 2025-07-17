import argparse
import importlib.util
import inspect
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Union

from triton import JITFunction
from triton.compiler.compiler import ASTSource, IRSource


def override_getitem(self: JITFunction, grid):
    def inner(*args, **kwargs):
        return self.run(
            grid=grid,
            warmup=True,  # avoids actual kernel execution
            *args,
            **kwargs,
        )

    return inner


def load_input(input_file: Path) -> Dict[str, JITFunction]:
    spec = importlib.util.spec_from_file_location("example", input_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {
        name: fn
        for name, fn in inspect.getmembers(module)
        if isinstance(fn, JITFunction)
    }


def main(input_file: Path, output_file: Path, opt_pipeline_file: Path):
    output_dir = output_file.parent.absolute()

    # Setup triton
    if opt_pipeline_file:
        os.environ["MLIR_ENABLE_DUMP"] = "1"
        os.environ["MLIR_DUMP_PATH"] = str(opt_pipeline_file)
    os.environ["TRITON_ALWAYS_COMPILE"] = "1"
    os.environ["TRITON_KERNEL_DUMP"] = "1"
    os.environ["TRITON_DUMP_DIR"] = str(output_dir)
    JITFunction.__getitem__ = override_getitem

    # Run the script
    jit_functions = load_input(input_file)

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
