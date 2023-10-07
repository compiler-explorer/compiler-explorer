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
import re
import sys
import typing

from numba.core.dispatcher import Dispatcher
from numba.core.types.abstract import Type

# TODO(Rupt) notes before I forget:
# - Make filter options work
# - Make translation to intel syntax work
# - Add numba-specific filter options
# - Move exec an writer to their own functions
# - Add tests
# - Inspect other states?


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
            # writer.write(f"; CE_NUMBA_SYMBOL {symbol}\n")
            # writer.write(f"; CE_NUMBA_MANGLED {mangled}\n")
            writer.write(f"; CE_NUMBA_LINENO {lineno}\n")
            demangle = True  # TODO(Rupt) configure
            if demangle:
                symbol = _overloaded_symbol(value, signature)
                asm = _demangle(asm, symbol)

            writer.write(asm)


# Naming
#
# Numba uses a custom mangling scheme that is not directly invertible.
# For example, it encodes '<locals>' as '_3clocals_3e', which is identical to
# a user-defined function or class with the valid name  '_3clocals_3e'.
# See: https://github.com/numba/numba/blob/0.58.0/numba/core/itanium_mangler.py
#
# We therefore use a non-mangled scheme that resembles their pre-mangling names.


def _overloaded_symbol(dispatcher: Dispatcher, signature: tuple[Type, ...]) -> str:
    """Return this overload's name in Numba's C++-style template syntax."""
    # NOTE: Including return type in a function symbol is not standard, but in this
    # Python world where two generated functions can have otherwise identical
    # signatures, return types can help to discriminate them.
    arguments = ", ".join(_templated_type(type_) for type_ in signature)
    return_type = dispatcher.overloads[signature].signature.return_type
    returns = _templated_type(return_type)
    # Numba uses fully qualified names in its mangled symbols.
    name = dispatcher.py_func.__qualname__.replace(".", "::")
    return f"{name}({arguments}) -> {returns}"


def _templated_type(type_: Type) -> str:
    """Return type_'s name in Numba's C++-style template syntax."""
    base, args_mistyped = type_.mangling_args
    # Pyright infers args_mistyped as an empty tuple.
    args = typing.cast(tuple[Type, ...], args_mistyped)
    if not args:
        return str(base)
    parameters = ", ".join(str(arg) for arg in args)
    return f"{base}<{parameters}>"


# Parsing


def _demangle(asm: str, symbol: str) -> str:
    """Return asm with Numba's mangled names replaced with forms of symbol."""
    mangled = _parse_mangled_symbol(asm)
    assert mangled.startswith("_ZN")
    return _multi_replace(
        {
            mangled: symbol,
            # Numba adds other symbols related to our function:
            # - An environment object
            "_ZN08NumbaEnv" + mangled[3:]: "NumbaEnv::" + symbol,
            # - A wrapper using the cpython API
            "_ZN7cpython" + mangled[3:]: "cpython::" + symbol,
            # - A wrapper using native code
            "cfunc." + mangled: "cfunc::" + symbol,
        },
        asm,
    )


def _parse_mangled_symbol(asm: str) -> str:
    """Return the mangled Numba function symbol parsed from this inspected asm code."""
    match = re.search(r"^\t\.globl\t+(_ZN[\w\d]+)", asm, flags=re.MULTILINE)
    assert match is not None
    return match[1]


def _multi_replace(key_to_value: dict[str, str], text: str) -> str:
    """Return text with substrings matching keys replaced by their values."""
    pattern = "|".join(re.escape(key) for key in key_to_value)
    return re.sub(pattern, lambda match: key_to_value[match[0]], text)


if __name__ == "__main__":
    main()
