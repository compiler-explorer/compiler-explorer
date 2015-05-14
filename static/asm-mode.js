// Copyright (c) 2012-2015, Matt Godbold
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

CodeMirror.defineMode("asm", function () {
    function tokenString(quote) {
        return function (stream) {
            var escaped = false, next, end = false;
            while ((next = stream.next()) !== null) {
                if (next == quote && !escaped) {
                    end = true;
                    break;
                }
                escaped = !escaped && next == "\\";
            }
            return "string";
        };
    }

    return {
        token: function (stream) {
            if (stream.match(/^.+:$/)) {
                return "variable-2";
            }
            if (stream.sol() && stream.match(/^\s*\.\w+/)) {
                return "header";
            }
            if (stream.sol() && stream.match(/^\s\w+/)) {
                return "keyword";
            }
            if (stream.eatSpace()) return null;
            var ch = stream.next();
            if (ch == '"' || ch == "'") {
                return tokenString(ch)(stream);
            }
            if (/[\[\]{}\(\),;\:]/.test(ch)) return null;
            if (/[\d$]/.test(ch) || (ch == '-' && stream.peek().match(/[0-9]/))) {
                stream.eatWhile(/[\w\.]/);
                return "number";
            }
            if (ch == '%') {
                stream.eatWhile(/\w+/);
                return "variable-3";
            }
            if (ch == '#') {
                stream.eatWhile(/.*/);
                return "comment";
            }
            stream.eatWhile(/[\w\$_]/);
            return "word";
        }
    };
});

CodeMirror.defineMIME("text/x-asm", "asm");
