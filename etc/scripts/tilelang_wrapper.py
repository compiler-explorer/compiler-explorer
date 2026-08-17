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

"""Compiler Explorer wrapper for TileLang compile-only output.

CE invokes::

    python3 -I tilelang_wrapper.py --output_file out.s example.py

This forwards to the official CLI (default ``--target c``)::

    python3 -I -m tilelang.tools.compile_only --output_file out.s example.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TileLang Compiler Explorer wrapper")
    parser.add_argument("input_file", type=Path, help="Path to the input Python file")
    parser.add_argument(
        "--output_file",
        type=Path,
        required=True,
        help="Path to the output kernel source",
    )
    parser.add_argument(
        "--target",
        default="c",
        help="Explicit tilelang target (default: c). CUDA is later.",
    )
    args = parser.parse_args(argv)

    try:
        from tilelang.tools.compile_only import cli_main
    except ModuleNotFoundError as exc:
        print(
            "tilelang compile-only error: tilelang is not importable. "
            "Use a Python with tilelang installed. "
            f"Missing module: {exc.name}.",
            file=sys.stderr,
        )
        return 1

    return cli_main(
        [
            str(args.input_file),
            "--output_file",
            str(args.output_file),
            "--target",
            args.target,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
