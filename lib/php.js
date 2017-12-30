// Copyright (c) 2017, Mike Cochrane
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

var _ = require('underscore-node');
var utils = require('./utils');

function PHPByteCodeParser(compilerProps) {
    const startOfClass = /^\tClass([a-zA-Z][\\a-zA-Z0-9$_.]*):/;
    const startOfFunction = /^\tFunction([a-zA-Z][\a-zA-Z0-9_]*):$/;
    const startOfAnnonymousFunction = /^\tFunction\0(.*?)\/(.*)(0x[a-z0-9]+):$/;
    const endOfClass = /^\tEndofclass([a-zA-Z][a-zA-Z0-9_\\]*).$/;
    const endOfFunction = /^\tEndoffunction([a-zA-Z][a-zA-Z0-9_]*)$/;
    const endOfAnnonymousFunction = /^\tEndoffunction\0(.*?)\/(.*)(0x[a-z0-9]+)$/;
    const compiledVars = /^\tcompiledvars:[^!]*(\!.*)$/;
    const byteCodeHeader = /^\tline#/;
    const byteCodeLineNumber = /^\t([ 0-9]{4})/;

    function processAsm(asm, filters) {
        var result = [];
        var asmLines = utils.splitLines(asm);

        var inByteCode = false,
            inClass = false,
            inFunction = false;

        var lineNumber = -1;
        var toReplace = [];
        asmLines.forEach(function (line) {
            line = decodeURIComponent(line);
            if (line.trim() === "") {
                // Blank line at end of bytecode for a function
                inByteCode = false;
                lineNumber = -1;
                toReplace = [];
                return;
            }

            // Is this the line before the byte code begins?
            if (line.match(byteCodeHeader)) {
                inByteCode = true;
                return;
            }

            // Indent byte code within a class
            if (line.match(startOfClass)) {
                inClass = true;
                result.push({text: "", source: null});
                result.push({text: "Class " + line.match(startOfClass)[1] + ':', source: null});
                return;
            } else if (line.match(endOfClass)) {
                inClass = false;
                result.push({text: "End of class " + line.match(endOfClass)[1], source: null});
                return;
            }

            // Indent byte code within a function
            if (line.match(startOfFunction)) {
                inFunction = true;
                let prefix = inClass ? "    " : "";
                if (!inClass) {
                    result.push({text: "", source: null});
                }
                result.push({text: prefix + "Function " + line.match(startOfFunction)[1] + ':', source: null});
                return;
            } else if (line.match(endOfFunction)) {
                inFunction = false;
                let prefix = inClass ? "    " : "";
                result.push({
                    text: prefix + "End of function " + line.match(endOfFunction)[1],
                    source: null
                });
                return;
            }

            // Indent byte code within an annonymous funtion
            if (line.match(startOfAnnonymousFunction)) {
                inFunction = true;
                let prefix = inClass ? "    " : "";
                if (!inClass) {
                    result.push({text: "", source: null});
                }
                result.push({
                    text: prefix + "Annonymous function " + line.match(startOfAnnonymousFunction)[1] + ':',
                    source: null
                });
                return;
            } else if (line.match(endOfAnnonymousFunction)) {
                inFunction = false;
                let prefix = inClass ? "    " : "";
                result.push({
                    text: prefix + "End of annonymous function " + line.match(endOfAnnonymousFunction)[1],
                    source: null
                });
                return;
            }

            // Find the compiled vars so they can be subsitiuted
            if (line.match(compiledVars)) {
                let match = line.match(compiledVars)[1];
                match.split(",\t").forEach(function(pair) {
                    toReplace = toReplace.concat([pair.split('=')]);
                });
                return;
            }

            if (!inByteCode) {
                // This line isn't interesting
                return;
            }

            // Find the line number for the bytecode
            let parts = line.split("\t");
            if (parts[1] && parseInt(parts[1])) {
                lineNumber = parseInt(parts[1]);
            }

            // Construct the text to output
            let toDisplay = (inClass ? "    " : "") + (inFunction ? "    " : "");
            toDisplay += parts[2].replace(/^[^A-Z]*/, '').replace(/^.*>/, ''); // Op Code, without prefix junk
            toDisplay += parts[4] ? ' ' + parts[4] : ''; // Return/destination
            // Operands
            for (let i = 5; i <= parts.length; i++) {
                if (parts[i] == ',') {
                    toDisplay += ',';
                } else {
                    toDisplay += parts[i] ? ' ' + parts[i] : '';
                }
            }

            // Replace any compiler variables with what they're called in php
            toReplace.forEach(function(pair) {
                toDisplay = toDisplay.replace(pair[0], pair[1]);
            });

            result.push({text: toDisplay, source: {file: null, line: lineNumber}});
            return;
        });
        return result;
    }

    this.process = function (asm, filters) {
        return processAsm(asm, filters);
    };
}

module.exports = {
    PHPByteCodeParser: PHPByteCodeParser
};
