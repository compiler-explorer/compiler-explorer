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
import contextlib
import importlib.util
import inspect
import sys
import traceback
from types import ModuleType
from typing import Iterator, TextIO

from numba.core.dispatcher import Dispatcher

# TODO(Rupt) notes before I forget:
# - Add tests


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Output compiled asm from public numba-compiled functions."
    )
    parser.add_argument("--inputfile", required=True)
    parser.add_argument("--outputfile")
    args = parser.parse_args()

    with (
        handle_exceptions(),
        open_or_stdout(args.outputfile) as writer,
    ):
        write_module_asm(path=args.inputfile, writer=writer)


def write_module_asm(*, path: str, writer: TextIO) -> None:
    """Write assembly code from compiled Numba functions in the module at `path`.

    - We only take code from public Numba Dispatchers in the module.
    - We add comments to indicate source line numbers.

    Args:
        path: Target file path containing Python code to load as a module.
        writer: Where we write the resulting code.
    """
    module = load_module(path=path)
    dispatchers = [
        value
        for name, value in inspect.getmembers(module)
        # Leading underscore means private in Python.
        # Numba manages compiled functions with Dispatcher objects.
        if not name.startswith("_") and isinstance(value, Dispatcher)
    ]
    dispatchers.sort(key=_lineno)  # We prefer source-ordered code for stable colors.
    for dispatcher in dispatchers:
        for asm in dispatcher.inspect_asm().values():
            asm = _add_lineno_comments(asm, _lineno(dispatcher))
            writer.write(asm)


@contextlib.contextmanager
def handle_exceptions() -> Iterator[None]:
    try:
        yield
    except Exception as error:
        # We prefer to hide the full traceback.
        messages = traceback.format_exception_only(type(error), error)
        sys.stderr.writelines(messages)
        sys.exit(255)


@contextlib.contextmanager
def open_or_stdout(maybe_path: str | None) -> Iterator[TextIO]:
    if maybe_path is None:
        yield sys.stdout
        return
    with open(maybe_path, "w", encoding="utf-8") as writer:
        yield writer


def load_module(*, path: str, name: str = "compiler_explorer") -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None  # For static type checkers
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _lineno(dispatcher: Dispatcher) -> int:
    return dispatcher.py_func.__code__.co_firstlineno


def _add_lineno_comments(asm: str, lineno: int) -> str:
    return asm.replace("\n", f";{lineno}\n")


if __name__ == "__main__":
    main()
