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
import sys
import typing

from numba.core.dispatcher import Dispatcher
from numba.core.types.abstract import Type

# TODO(Rupt) notes before I forget:
# - Add signature and name as comments
# - Make filter options work
# - Make translation to intel syntax work
# - Add numba-specific filter options
# - Move exec an writer to their own functions
# - Add tests
# - Allow positional arguments?
# - Inspect other states?
# - Name demangling?
# - Move to disasms?


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Output compiled asm from public numba-compiled functions."
    )
    parser.add_argument("--inputfile", required=True)
    parser.add_argument("--outputfile")

    args = parser.parse_args()

    writer = (
        sys.stdout
        if args.outputfile is None
        else open(args.outputfile, "w", encoding="utf-8")
    )

    # TODO(Rupt):
    # - Put this in a function
    # - Capture outputs
    # - Capture exceptions
    namespace = {}
    exec(open(args.inputfile).read(), namespace)

    for key, value in namespace.items():
        if key.startswith("_"):
            # A leading underscore in Python means "private".
            continue
        if not isinstance(value, Dispatcher):
            # Dispatcher is a common base class for Numba's compiled functions.
            continue
        lineno = value.py_func.__code__.co_firstlineno
        for signature, asm in value.inspect_asm().items():
            symbol = _overload(value, signature)
            writer.write(f"; CE_NUMBA_SYMBOL {symbol}\n")
            writer.write(f"; CE_NUMBA_LINENO {lineno}\n")
            writer.write(asm)


def _overload(dispatcher: Dispatcher, signature: tuple[Type, ...]) -> str:
    # NOTE: Including return type in a function symbol is not standard.
    # But in this Python world where two generated functions can have
    # otherwise identical signatures, can help to discriminate them.
    arguments = ", ".join(_templated(type_) for type_ in signature)
    return_type = dispatcher.overloads[signature].signature.return_type
    returns = _templated(return_type)
    # Numba uses __qualname__ in its mangled symbols.
    name = dispatcher.py_func.__qualname__
    return f"{name}({arguments}) -> {returns}"


def _templated(type_: Type) -> str:
    base, args_mistyped = type_.mangling_args
    # Pyright infers args_mistyped as an empty tuple.
    args = typing.cast(tuple[Type, ...], args_mistyped)
    if not args:
        return str(base)
    parameters = ", ".join(str(arg) for arg in args)
    return f"{base}<{parameters}>"


if __name__ == "__main__":
    main()
