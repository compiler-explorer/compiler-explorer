// Copyright (c) 2022, Compiler Explorer Authors
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright notice,
//       this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

export const cases = [
    {
        input: `# 0 "/app/example.cpp"
# 1 "/app//"
# 0 "<built-in>"
# 0 "<command-line>"
# 1 "/usr/include/stdc-predef.h" 1 3 4
# 0 "<command-line>" 2
# 1 "/app/example.cpp"
# 1 "/usr/include/assert.h" 1 3 4
# 35 "/usr/include/assert.h" 3 4
# 1 "/usr/include/features.h" 1 3 4
# 461 "/usr/include/features.h" 3 4
# 1 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 1 3 4
# 452 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4
# 1 "/usr/include/x86_64-linux-gnu/bits/wordsize.h" 1 3 4
# 453 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 2 3 4
# 1 "/usr/include/x86_64-linux-gnu/bits/long-double.h" 1 3 4
# 454 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 2 3 4
# 462 "/usr/include/features.h" 2 3 4
# 485 "/usr/include/features.h" 3 4
# 1 "/usr/include/x86_64-linux-gnu/gnu/stubs.h" 1 3 4
# 10 "/usr/include/x86_64-linux-gnu/gnu/stubs.h" 3 4
# 1 "/usr/include/x86_64-linux-gnu/gnu/stubs-64.h" 1 3 4
# 11 "/usr/include/x86_64-linux-gnu/gnu/stubs.h" 2 3 4
# 486 "/usr/include/features.h" 2 3 4
# 36 "/usr/include/assert.h" 2 3 4
# 66 "/usr/include/assert.h" 3 4

# 66 "/usr/include/assert.h" 3 4
extern "C" {

extern void __assert_fail (const char *__assertion, const char *__file,
        unsigned int __line, const char *__function)
        throw () __attribute__ ((__noreturn__));

extern void __assert_perror_fail (int __errnum, const char *__file,
        unsigned int __line, const char *__function)
        throw () __attribute__ ((__noreturn__));

extern void __assert (const char *__assertion, const char *__file, int __line)
        throw () __attribute__ ((__noreturn__));

}
# 2 "/app/example.cpp" 2

#line 11 "C:/WinSdk/Include/10.0.18362.0/ucrt\\assert.h"
foo
# 2 "<source>" 2
bar
# 66 "/usr/include/assert.h" 3 4
baz
# 2 "<stdin>"
biz
# 66 "/usr/include/assert.h" 3 4


# 3 "/app/example.cpp"
int main() {

# 4 "/app/example.cpp"
#pragma foo bar
# 4 "/app/example.cpp"

# 5 "/app/example.cpp" 3 4
    (static_cast <bool> (
# 5 "/app/example.cpp"
    false
# 5 "/app/example.cpp" 3 4
    ) ? void (0) : __assert_fail (
# 5 "/app/example.cpp"
    "false"
# 5 "/app/example.cpp" 3 4
    , "/app/example.cpp", 5, __extension__ __PRETTY_FUNCTION__))
# 5 "/app/example.cpp"
                ;
}`,
        output: `bar
biz
int main() {

#pragma foo bar

    (static_cast <bool> (
    false
    ) ? void (0) : __assert_fail (
    "false"
    , "/app/example.cpp", 5, __extension__ __PRETTY_FUNCTION__))
                ;
}`,
    },
];
