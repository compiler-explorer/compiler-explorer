# Copyright (c) 2025, Compiler Explorer Authors
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
import importlib.util
import json
import os
from pathlib import Path
from typing import Dict, Optional, Union
from unittest.mock import MagicMock

import torch
import triton
from torch._subclasses.fake_tensor import FakeTensorMode


class MockCacheManager(triton.runtime.cache.CacheManager):
    """
    A mock cache manager that dumps the intermediate files to a given output path.

    There are various ways to dump the intermediate files:
    1. The most obvious way is to use the `TRITON_KERNEL_DUMP` & ``TRITON_DUMP_DIR`
       environment variables. e.g.,
            os.environ["TRITON_KERNEL_DUMP"] = "1"
            os.environ["TRITON_DUMP_DIR"] = str(output_dir)
        However, `TRITON_DUMP_DIR` is introduced in Triton v3.2.0 at
        https://github.com/triton-lang/triton/commit/ca469d7b6b6def316b5f5ee6ad2bd19dcb840bd8,
        and thus not available in older versions.

    2. The second attempt is to patch the `default_cache_dir` function. e.g.,
            triton.runtime.cache.default_cache_dir = MagicMock(return_value=output_dir)
        This is a bit hacky, and less flexible in terms of controlling the file output.
        (In fact, Triton dumps the compiled kernels to a folder with a random name.)

    3. Another option is to use various hooks in Triton. e.g.,
            triton.knobs.runtime.{jit_post_compile_hook,launch_enter_hook}
            JITFunction.{compiled_hook,cache_hook}
        This approach is taken by TritonParse(https://github.com/pytorch-labs/tritonparse),
        but it does not support older versions of Triton prior to the following commits:
            https://github.com/triton-lang/triton/commit/0e9267202532ed1709dcc12c636220cf239dc377,
            https://github.com/triton-lang/triton/commit/850525276426fb9814399a8e0ee8fdf744229b02.

    4. The current apporach is to mock a `CacheManager` class. This is the most flexible
       approach, and works for all versions of Triton.
    """

    output_file: Path

    def __init__(self, key, override=False, dump=False):
        self.dump = dump
        # filename -> data
        self.files = {}
        # filename -> group dict
        self.groups = {}
        # current stage for a given kernel
        self.stage = 0

    def get_file(self, filename) -> Optional[str]:
        return self.files.get(filename, None)

    def put(self, data, filename, binary=True) -> str:
        name = Path(filename).stem
        suffix = Path(filename).suffix
        binary = isinstance(data, bytes)
        if not binary:
            data = str(data)

        # Write the final file to the output file, so that we can view it in the default assembly view.
        if suffix in (".ptx", ".amdgcn"):
            with open(MockCacheManager.output_file, "a") as f:
                f.write(data)
                f.write("\n\n")

        # Write intermediate files to the output file, so that we can see them in the Device View.
        self.stage += 1
        if suffix == ".json":
            path = MockCacheManager.output_file.parent / filename
            with open(path, "w") as fout:
                json.dump(json.loads(data), fout, indent=2)
        else:
            path = (
                MockCacheManager.output_file.parent
                / f"{name} [stage {self.stage}]{suffix}"
            )
            if not binary:
                with open(path, "w") as fout:
                    fout.write(data)
            elif suffix == ".cubin":
                try:
                    # The try-catch is needed because `disasm` was broken in Triton v3.0.0 and v3.1.0. See
                    # https://github.com/triton-lang/triton/commit/f424f656b3528c47d8c48126cdccafca29e536ae
                    from triton.tools import disasm

                    with open(path.with_suffix(".sass"), "w") as fout:
                        fout.write(disasm.get_sass(data))
                except Exception:
                    pass

        # Write the file to the "cache"
        self.files[filename] = data
        return filename

    def get_group(self, filename: str) -> Optional[Dict[str, str]]:
        self.groups.get(filename, None)

    def put_group(self, filename: str, group: Dict[str, str]):
        self.groups[filename] = group


def setup_triton(
    output_file: Path,
    opt_pipeline_file: Path,
    backend: str,
    arch: Union[int, str],
    warp_size: int,
):
    """
    Patch Triton to dump the compiled kernels to output dir without actually running them.

    This is needed because
    1. Triton does not easily support such use case. There exists an AOT compiler at
       https://github.com/triton-lang/triton/blob/main/python/triton/tools/compile.py,
       but it requires a bunch of boilerplate code and also requires additional user
       input to specify the kernel name, signature, etc.
    2. Even if Triton adds such support, older versions of Triton (e.g., v2.3.x) still
       requirs such patching to work.

    This function is a collection of hacks. It has been tested to work with Triton versions:
    2.3.0, 2.3.1, 3.0.0, 3.1.0, 3.2.0, 3.3.0, 3.3.1, 3.4.0.
    """

    os.environ["TRITON_ALWAYS_COMPILE"] = "1"
    if opt_pipeline_file:
        os.environ["MLIR_ENABLE_DUMP"] = "1"
        # Supported in Triton v3.3.0 and later since
        # https://github.com/triton-lang/triton/commit/3d7d9e33e7e4cba17dc366d207af2c657bd4fbd1
        os.environ["MLIR_DUMP_PATH"] = str(opt_pipeline_file)
    else:
        # Disable dumping other files w/ opt_pipeline_file since they race with each other
        os.environ["TRITON_CACHE_MANAGER"] = "__main__:MockCacheManager"
        MockCacheManager.output_file = output_file

    # Usually, Triton compiles and run a kernel when we call `kernel[grid](args)`.
    # However, we want to dump the compiled kernel without actually running it.
    # The class `CompiledKernel` represents a handle to a compiled kernel,
    # ready to be launched. We patch it to be a no-op.
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

    mockGPUDriver = MagicMock(
        get_current_target=get_current_target,
        get_benchmarker=lambda: MagicMock(return_value=[0.0]),
    )

    # Set the active driver to the mocked one.
    # `DriverConfig` and `triton.runtime.driver.set_active` is introduced in Triton v3.0.0 at
    # https://github.com/triton-lang/triton/commit/b844d519bc5e86edf00fe6b3c6c2d1badcd509a4
    # For older versions of Triton, we directly assign to the `_obj` field of `LazyProxy`.
    try:
        from triton.runtime.driver import DriverConfig

        triton.runtime.driver.set_active(mockGPUDriver)
    except ImportError:
        triton.runtime.driver._obj = mockGPUDriver

    # For Triton v2.3.x, there are some driver code that goes into
    # the generic code path, so we need to patch it as well.
    try:
        from triton.compiler.backends.cuda import CUDABackend

        CUDABackend.make_launcher_stub = MagicMock()
    except ImportError:
        pass

    # Triton used to depends on the ld.lld binary, and search for the following paths:
    #   - TRITON_HIP_LLD_PATH (see https://github.com/triton-lang/triton/pull/3917)
    #   - third_party/amd/backend/llvm/bin/ld.lld (see https://github.com/triton-lang/triton/pull/3662)
    #   - /opt/rocm/llvm/bin/ld.lld
    #   - /usr/bin/ld.lld (see https://github.com/triton-lang/triton/pull/3197)
    # However, none of them are available in the production environment, which causes
    #     Exception: ROCm linker /opt/rocm/llvm/bin/ld.lld not found. Set 'TRITON_HIP_LLD_PATH' to its path.
    # Since the linker is only used to generate the binary .hsaco file (from the .amdgcn assembly),
    # we don't really need it. We can just mock it to be a no-op.
    # Note: Triton no longer depends on the ld.lld binary after v3.4.0 (exclusive) with commit:
    #     https://github.com/triton-lang/triton/pull/7548
    os.environ["TRITON_HIP_LLD_PATH"] = "/bin/true"


def main(
    input_file: Path,
    output_file: Path,
    opt_pipeline_file: Path,
    backend: str,
    arch: Union[int, str],
    warp_size: int,
):
    setup_triton(output_file, opt_pipeline_file, backend, arch, warp_size)

    # Run the script by importing it as a module
    spec = importlib.util.spec_from_file_location("example", input_file)
    module = importlib.util.module_from_spec(spec)
    with FakeTensorMode():
        # Use FakeTensor (developed during Dynamo) to avoid actually creating tensors
        # See https://docs.pytorch.org/docs/stable/torch.compiler_fake_tensor.html
        # Also set the data_ptr to 0 to avoid PyTorch warning and make alignment check happy
        torch._subclasses.FakeTensor.data_ptr = MagicMock(return_value=0)
        spec.loader.exec_module(module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triton wrapper")
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to the input Python file",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        required=True,
        help="Path to the output file",
    )
    parser.add_argument(
        "--opt_pipeline_file",
        type=Path,
        help="Path to the output opt pipeline file",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="cuda",
        choices=["cuda", "hip"],
    )
    parser.add_argument(
        "--arch",
        type=str,
        default=None,  # Default value set later based on backend
    )
    parser.add_argument(
        "--warp_size",
        type=int,
        default=32,
    )

    args = parser.parse_args()

    # Set some sane defaults for the arch
    if args.arch is None:
        if args.backend == "cuda":
            args.arch = 90
        elif args.backend == "hip":
            args.arch = "gfx942"

    # Triton expects the arch to be an int for CUDA and a string for HIP
    if args.backend == "cuda":
        args.arch = int(args.arch)

    main(**vars(args))
