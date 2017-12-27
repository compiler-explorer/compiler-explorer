// Copyright (c) 2012-2017, Matt Godbolt
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
    const startOfClass = /^Class [a-zA-Z][\\a-zA-Z0-9$_.]*:/;
    const startOfFunction = /^Function [a-zA-Z][a-zA-Z0-9_]*:/;
    const endOfClass = /^End of class [a-zA-Z][a-zA-Z0-9_]*./;
    const endOfFunction = /^End of function [a-zA-Z][a-zA-Z0-9_]*/;
    const compiledVars = /^compiled vars:  (\!.*)$/;
    const byteCodeHeader = /^line/;
    const byteCodeStart = /^[-]{80}/;
    const byteCodeLineNumber = /^([ 0-9]{4})/;

    function processAsm(asm, filters) {
        var result = [];
        var asmLines = utils.splitLines(asm);

        var inByteCode = false,
        	inClass = false,
        	inFunction = false;

        var lineNumber = -1;
        var toReplace = [];
        asmLines.forEach(function (line) {
            if (line.trim() === "") {
            	// Blank line at end of bytecode for a function
            	inByteCode = false;
            	lineNumber = -1;
            	toReplace = [];
                return;
            }

			// Is this the line before the byte code begins?
            if (line.match(byteCodeStart)) {
            	inByteCode = true;
				return;
            }

            // Indent byte code within a class
            if (line.match(startOfClass)) {
            	inClass = true
            	result.push({text: "", source: null});
            	result.push({text: line, source: null});
            	return;
            } else if (line.match(endOfClass)) {
            	inClass = false;
            	result.push({text: line, source: null});
            	result.push({text: "", source: null});
            	return;
            }

            // Indent byte code within a function
            if (line.match(startOfFunction)) {
            	inFunction = true;
				let prefix = inClass ? "    " : "";
            	result.push({text: prefix + line, source: null});
            	return;
            } else if (line.match(endOfFunction)) {
            	inFunction = false;
				let prefix = inClass ? "    " : "";
            	result.push({text: prefix + line, source: null});
            	return;
            }

            let prefix = (inClass ? "    " : "") + (inFunction ? "    " : "");
            if (line.match(byteCodeHeader)) {
            	// Include raw bytecode header for now
            	result.push({text: prefix + line.substr(18), source: null});
            	return;
            }

            // Find the compiled vars so they can be subsitiuted
            if (line.match(compiledVars)) {
            	let match = line.match(compiledVars)[1];
            	match.split(', ').forEach(function(pair) {
					toReplace = toReplace.concat([pair.split(' = ')]);
            	});
            	return;
            }

            if (!inByteCode) {
            	// This line isn't interesting
                return;
            }

            // Find the line number for the bytecode
			let match = line.match(byteCodeLineNumber);
			if (match && parseInt(match[1])) {
				lineNumber = parseInt(match[1]);
			}

			// Replace any compiler variables with what they're called in php
			toReplace.forEach(function(pair) {
				line = line.replace(pair[0], pair[1]);
			});

            if (lineNumber > -1) {
            	result.push({text: prefix + decodeURIComponent(line.substr(18)), source: {file: null, line: lineNumber}});
            } else {
            	result.push({text: prefix + line, source: null});
            }
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
