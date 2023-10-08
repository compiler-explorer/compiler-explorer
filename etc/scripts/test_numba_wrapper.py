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
import inspect
import io
import os
import sys
import unittest
import unittest.mock

import numba
from numba.core.caching import tempfile

from . import numba_wrapper


class TestHandleExceptions(unittest.TestCase):
    def test_no_exception(self):
        with numba_wrapper.handle_exceptions() as nil:
            self.assertIsNone(nil)

    def test_exception(self):
        secret = "dQw4w9WgXcQ"
        stderr = io.StringIO()
        with (
            unittest.mock.patch.object(sys, "exit") as exit_,
            unittest.mock.patch.object(sys, "stderr", stderr),
            numba_wrapper.handle_exceptions(),
        ):
            assert not secret, secret
        self.assertEqual(exit_.call_count, 1)
        self.assertEqual(exit_.call_args.args, (255,))
        self.assertFalse(exit_.call_args.kwargs)
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


class TestLineNumber(unittest.TestCase):
    def test_lineno(self):
        def square(x):
            return x * x

        _, line_number = inspect.getsourcelines(square)
        self.assertEqual(numba_wrapper._lineno(numba.njit(square)), line_number)

    def test_add_lineno_comments(self):
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
        source_commented = numba_wrapper._add_lineno_comments(source, line_number)
        for before, after in zip(
            source.split("\n"), source_commented.split("\n"), strict=True
        ):
            if before == "":
                self.assertEqual(after, "")
                continue
            prefix, suffix = after.split(";")
            self.assertEqual(before, prefix)
            self.assertEqual(line_number, int(suffix))


if __name__ == "__main__":
    unittest.main()
