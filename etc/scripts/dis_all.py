# -*- coding: utf-8 -*-
# Copyright (c) 2019, Sebastian Rath
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

import os
import sys
import dis
import argparse
import traceback

from dis import dis, disassemble, distb, _have_code, _disassemble_bytes, _try_compile

parser = argparse.ArgumentParser(description='Disassembles Python source code given by an input file and writes the output to a file')
parser.add_argument('-i', '--inputfile', type=str,
                    help='Input source code file (*.py)')
parser.add_argument('-o', '--outputfile', type=str,
                    help='Optional output file to write output (or error message if syntax error).',
                    default='')
parser.add_argument('-O', action='store_true', dest='optimize_1',
                    help="Enable Python's -O optimization flag (remove assert and __debug__-dependent statements)")
parser.add_argument('-OO', action='store_true', dest='optimize_2',
                    help="Enable Python's -OO optimization flag (do -O changes and also discard docstrings)")


def _disassemble_recursive(co, depth=None):
    disassemble(co)
    if depth is None or depth > 0:
        if depth is not None:
            depth = depth - 1
        for x in co.co_consts:
            if hasattr(x, 'co_code'):
                print()
                print("Disassembly of %r:" % (x,))
                _disassemble_recursive(x, depth=depth)


def _disassemble_str(source, **kwargs):
    """Compile the source string, then disassemble the code object."""
    _disassemble_recursive(_try_compile(source, '<dis>'), **kwargs)


# This function is copied from Py 3.7 and compatible with Py 3.3, 3.4, 3.5 to support recursive diassemble
def dis37(x=None, depth=None):
    """Disassemble classes, methods, functions, and other compiled objects.

    With no argument, disassemble the last traceback.

    Compiled objects currently include generator objects, async generator
    objects, and coroutine objects, all of which store their code object
    in a special attribute.
    """
    if x is None:
        distb()
        return
    # Extract functions from methods.
    if hasattr(x, '__func__'):
        x = x.__func__
    # Extract compiled code objects from...
    if hasattr(x, '__code__'):  # ...a function, or
        x = x.__code__
    elif hasattr(x, 'gi_code'):  #...a generator object, or (added in Py 3.5)
        x = x.gi_code
    elif hasattr(x, 'ag_code'):  #...an asynchronous generator object, or (added in Py 3.7)
        x = x.ag_code
    elif hasattr(x, 'cr_code'):  #...a coroutine. (added in Py 3.7)
        x = x.cr_code
    # Perform the disassembly.
    if hasattr(x, '__dict__'):  # Class or module (added in Py 3.7)
        items = sorted(x.__dict__.items())
        for name, x1 in items:
            if isinstance(x1, _have_code):
                print("Disassembly of %s:" % name)
                try:
                    dis(x1, depth=depth)
                except TypeError as msg:
                    print("Sorry:", msg)
                print()
    elif hasattr(x, 'co_code'): # Code object
        _disassemble_recursive(x, depth=depth)
    elif isinstance(x, (bytes, bytearray)): # Raw bytecode
        _disassemble_bytes(x)
    elif isinstance(x, str):    # Source code
        _disassemble_str(x, depth=depth)
    else:
        raise TypeError("don't know how to disassemble %s objects" %
                        type(x).__name__)


if __name__ == '__main__':
    args = parser.parse_args()

    if not args.inputfile:
        parser.print_help(sys.stderr)
        sys.exit(1)

    with open(args.inputfile, 'r', encoding='utf8') as fp:
        source = fp.read()

    name = os.path.basename(args.inputfile)

    optimize=0
    if args.optimize_1:
        optimize = 1
    if args.optimize_2:
        optimize = 2

    try:
        code = compile(source, name, 'exec', optimize=optimize)
    except Exception as e:
        # redirect any other by compile(..) to stderr in order to hide traceback of this script
        sys.stderr.write(''.join(traceback.format_exception_only(type(e), e)))
        sys.exit(255)

    if args.outputfile:
        sys.stdout = open(args.outputfile, 'w', encoding='utf8')

    if sys.version_info < (3, 7):
        # Any Python version older than 3.7 doesn't support recursive diassembly,
        # so we call our own function
        dis37(code)
    else:
        dis(code)
