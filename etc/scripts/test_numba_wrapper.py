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
import inspect
import io
import os
import sys
import unittest
import unittest.mock

import numba
from numba.core.caching import tempfile

from . import numba_wrapper


class TestMain(unittest.TestCase):
    def test_with_output_file(self):
        source = "<source>"
        with (
            tempfile.NamedTemporaryFile() as output_file,
            unittest.mock.patch.object(numba_wrapper, "write_module_asm") as mock,
            unittest.mock.patch.object(
                argparse.ArgumentParser,
                "parse_args",
                return_value=argparse.Namespace(
                    inputfile=source, outputfile=output_file.name
                ),
            ),
        ):
            numba_wrapper.main()
        mock.assert_called_once()
        self.assertEqual(mock.call_args.kwargs["path"], source)
        self.assertEqual(mock.call_args.kwargs["writer"].name, output_file.name)

    def test_without_output_file(self):
        with (
            unittest.mock.patch.object(numba_wrapper, "write_module_asm") as mock,
            unittest.mock.patch.object(
                argparse.ArgumentParser,
                "parse_args",
                return_value=argparse.Namespace(inputfile="test", outputfile=None),
            ),
        ):
            numba_wrapper.main()
        self.assertEqual(mock.call_args.kwargs["writer"], sys.stdout)


class TestWriteModuleAsm(unittest.TestCase):
    def test_asm(self):
        # This test is slow, (~0.2s in local testing).
        # Reducing the optimization level (NUMBA_OPT=0) made negligible difference.
        # Adding the second compiled function gave only a small increase, suggesting
        # that we suffer startup overhead.
        source = (
            "import numba\n"
            "\n"
            "\n"
            "@numba.njit(numba.int32(numba.int32))\n"
            "def square(num):\n"
            "    return num * num\n"
            "\n"
            "\n"
            "square_alias = square\n"
            "\n"
            "@numba.njit(numba.int32(numba.int32))\n"
            "def cube(num):\n"
            "    return num * num * num\n"
        )
        writer = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "test_empty.py")
            with open(path, "w") as file_:
                file_.write(source)
            numba_wrapper.write_module_asm(path=path, writer=writer)

        asm = writer.getvalue()
        self.assertIn("square", asm)
        self.assertIn("cube", asm)
        # Must be sorted by line number.
        self.assertLess(asm.index("square"), asm.index("cube"))
        # Aliasing `square` must not duplicate its code.
        self.assertEqual(asm.count("square"), asm.count("cube"))

    def test_no_error_on_no_dispatcher(self):
        writer = io.StringIO()
        numba_wrapper.write_module_asm(path=numba_wrapper.__file__, writer=writer)
        self.assertEqual(writer.getvalue(), "")


class TestLineNumber(unittest.TestCase):
    def test_encode_line_number(self):
        source = (
            " push    rbp\n"
            " mov     rbp, rsp\n"
            " mov     DWORD PTR [rbp-4], edi\n"
            " mov     eax, DWORD PTR [rbp-4]\n"
            " imul    eax, eax\n"
            " pop     rbp\n"
            " ret\n"
        )
        line_number = 5678
        source_commented = numba_wrapper._encode_line_number(line_number, source)

        source_lines = source.split("\n")
        result_lines = source_commented.split("\n")
        self.assertEqual(len(source_lines), len(result_lines))
        for before, after in zip(source_lines, result_lines, strict=True):
            if before == "":
                self.assertEqual(after, "")
                continue
            prefix, suffix = after.split(";")
            self.assertEqual(before, prefix)
            self.assertEqual(line_number, int(suffix))

    def test_line_number(self):
        def square(x):
            return x * x

        _, line_number = inspect.getsourcelines(square)
        self.assertEqual(numba_wrapper._line_number(numba.njit(square)), line_number)


class TestLoadModule(unittest.TestCase):
    def test_simple(self):
        name = "simple"
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "test_simple.py")
            with open(path, "w") as file_:
                file_.write("x = 123")
            module = numba_wrapper.load_module(path=file_.name, name=name)
        self.assertEqual(module.__name__, name)
        self.assertEqual(module.x, 123)


class TestHandleExceptions(unittest.TestCase):
    def test_no_exception(self):
        with numba_wrapper.handle_exceptions() as nil:
            self.assertIsNone(nil)

    def test_exception(self):
        secret = "dQw4w9WgXcQ"
        stderr = io.StringIO()
        with (
            self.assertRaisesRegex(SystemExit, "255"),
            unittest.mock.patch.object(sys, "stderr", stderr),
            numba_wrapper.handle_exceptions(),
        ):
            assert not secret, secret  # Just to raise an exception
        self.assertIn(secret, stderr.getvalue())


class TestOpenOrStdout(unittest.TestCase):
    def test_open(self):
        secret = "hi mom"
        with tempfile.TemporaryDirectory() as directory:
            filename = os.path.join(directory, "test_open.txt")
            with numba_wrapper.open_or_stdout(filename) as file_:
                self.assertIsNot(file_, sys.stdout)
                file_.write(secret)
            with open(filename) as file_:
                self.assertEqual(file_.read(), secret)

    def test_stdout(self):
        with numba_wrapper.open_or_stdout(None) as file_:
            self.assertIs(file_, sys.stdout)


if __name__ == "__main__":
    unittest.main()
