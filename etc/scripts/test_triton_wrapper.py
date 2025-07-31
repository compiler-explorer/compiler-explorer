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

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pytest


@dataclass(frozen=True)
class Config:
    version_tuple: Tuple[int, int, int]
    backend: str
    opt_pipeline: bool

    @property
    def python_path(self) -> Path:
        version = f"v{'.'.join(str(v) for v in self.version_tuple)}"
        return Path(f"/opt/compiler-explorer/triton/{version}/bin/python3")


CONFIGS = [
    Config(version_tuple=version_tuple, backend=backend, opt_pipeline=opt_pipeline)
    for backend in ["cuda", "hip"]
    for opt_pipeline in [True, False]
    for version_tuple in [
        (3, 4, 0),
        (3, 3, 1),
        (3, 3, 0),
        (3, 2, 0),
        (3, 1, 0),
        (3, 0, 0),
        (2, 3, 1),
        (2, 3, 0),
    ]
    if (
        not any(
            [
                # AMD support added in 3.0.0
                (backend == "hip" and version_tuple < (3, 0, 0)),
                # Opt pipeline support added in 3.3.0
                (opt_pipeline and version_tuple < (3, 3, 0)),
            ]
        )
    )
]

# Simple Triton example code for testing
KERNEL_NAME = "store_kernel"
EXAMPLE_CODE = """
import torch
import triton
import triton.language as tl

@triton.jit
def store_kernel(ptr, val):
    tl.store(ptr, val)

x = torch.empty(1)
store_kernel[(1,)](x, 1)
"""


@pytest.mark.parametrize("config", CONFIGS)
def test_triton_wrapper(config: Config):
    if not config.python_path.exists():
        pytest.skip(f"Python interpreter not found: {config.python_path}")

    # Create temporary files for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        # Create example input file
        input_file = temp_dir_path / "example.py"
        with open(input_file, "w") as f:
            f.write(EXAMPLE_CODE)

        # Output files
        output_file = temp_dir_path / "output.s"
        opt_pipeline_file = temp_dir_path / "opt_pipeline.txt"

        # Run triton_wrapper.py
        script_path = Path(__file__).parent / "triton_wrapper.py"
        cmd = [
            str(config.python_path),
            str(script_path),
            str(input_file),
            "--output_file",
            str(output_file),
            "--backend",
            config.backend,
        ]

        if config.opt_pipeline:
            cmd.append("--opt_pipeline")
            cmd.append(str(opt_pipeline_file))

        try:
            subprocess.run(cmd, check=True, timeout=60)

            if config.opt_pipeline:
                # Check that the opt pipeline file was created
                assert opt_pipeline_file.exists(), f"Opt pipeline file not created"
                assert (
                    KERNEL_NAME in opt_pipeline_file.read_text()
                ), f"Opt pipeline file does not contain kernel name: {KERNEL_NAME}"
            else:
                # Check that the output file was created
                assert output_file.exists(), f"Output file not created"

                # Check that the output file has content
                output_content = output_file.read_text()
                assert output_content, f"Output file is empty"

                # Check for auxiliary files
                files = list(temp_dir_path.iterdir())
                exts = [".ttir", ".ttgir", ".llir", ".json"]
                if config.backend == "hip":
                    exts.append(".amdgcn")
                elif config.backend == "cuda":
                    exts.append(".ptx")
                for ext in exts:
                    file = next((f for f in files if f.suffix == ext), None)
                    assert file is not None, f"Missing file with extension {ext}"
                    assert (
                        KERNEL_NAME in file.read_text()
                    ), f"File does not contain kernel name: {KERNEL_NAME}"

        except subprocess.CalledProcessError:
            pytest.fail(f"Error running triton_wrapper.py")
        except subprocess.TimeoutExpired:
            pytest.fail(f"Timeout running triton_wrapper.py")
