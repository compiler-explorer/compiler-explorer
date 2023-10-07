# Copyright (c) 2023, Compiler Explorer Authors
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
import inspect
import sys

from numba.core.dispatcher import Dispatcher

# TODO(Rupt) notes before I forget:
# - Add tests
# - Add an executor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Output compiled asm from public numba-compiled functions."
    )
    parser.add_argument("--inputfile", required=True)
    parser.add_argument("--outputfile")

    args = parser.parse_args()

    # TODO(Rupt): Make these arguments to a function
    writer = (
        sys.stdout
        if args.outputfile is None
        else open(args.outputfile, "w", encoding="utf-8")
    )

    # TODO(Rupt):
    # - Put this in a function
    # - Capture outputs
    # - Capture exceptions
    spec = importlib.util.spec_from_file_location("compiler_explorer", args.inputfile)
    assert spec is not None and spec.loader is not None  # For static type checkers
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    dispatchers = [
        value
        for name, value in inspect.getmembers(module)
        # Leading underscore means private.
        # Numpy dispatches compiled functions with Dispatcher objects.
        if not name.startswith("_") and isinstance(value, Dispatcher)
    ]
    dispatchers.sort(key=_line_number)  # We prefer asm in source order

    for dispatcher in dispatchers:
        for asm in dispatcher.inspect_asm().values():
            asm = _add_line_number_comments(asm, _line_number(dispatcher))
            writer.write(asm)


def _line_number(dispatcher: Dispatcher) -> int:
    return dispatcher.py_func.__code__.co_firstlineno


def _add_line_number_comments(asm: str, lineno: int) -> str:
    return asm.replace("\n", f";{lineno}\n")


if __name__ == "__main__":
    main()
