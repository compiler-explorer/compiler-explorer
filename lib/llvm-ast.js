// Copyright (c) 2021, Compiler Explorer Authors
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

export class LlvmAstParser {
    constructor(compilerProps) {
        this.maxAstLines = 500;
        if (compilerProps) {
            this.maxAstLines = compilerProps('maxLinesOfAst', this.maxAstLines);
        }

        // Almost every line of AST includes a span of related source lines:
        // In different forms like <line:a:b, line:c:d>
        this.spanTypes = {
            NONE: 'none', // No span specified
            EMPTY: 'empty', // span mentions no lines
            LEFT: 'left', // span mentions only the starting line
            RIGHT: 'right', // span mentions only the finishing line
            FULL: 'full', // span mentions both lines
        };
    }

    // Extract one, two, or no line numbers:
    // <line:a:b, line:c:d> -> [a, b] - FULL
    // <col:b, line:c:d> -> [-, b] - RIGHT
    // <line:a:b, col:c> -> [a, -] - LEFT
    // <col:a, col:b> -> EMPTY
    // <no source span> -> NONE
    parseSpan(line) {
        const spanRegex = /<((?:line|col)[\d ,:ceilno]+)>/;
        const twoLineRegex = /line:(\d+)[,:].*line:(\d+)[:>]/;
        const leftLineRegex = /^line:(\d+)/;
        const rightLineRegex = / line:(\d+):/;

        const m = line.match(spanRegex);
        if (m) {
            const span = m[1];
            const ll = span.match(twoLineRegex);
            if (ll) {
                return { type : this.spanTypes.FULL, left : Number(ll[1]), right: Number(ll[2])};
            }
            const left = span.match(leftLineRegex);
            if (left) {
                return { type : this.spanTypes.LEFT, left : Number(left[1])};
            }
            const right = span.match(rightLineRegex);
            if (right) {
                return { type : this.spanTypes.RIGHT, right : Number(right[1])};
            }
            return { type : this.spanTypes.EMPTY };
        }
        return { type : this.spanTypes.NONE };
    }

    // Link the AST lines with spans of source lines
    parseAndSetSourceLines(astDump) {
        var lfrom, lto;
        for (var line of astDump) {
            const span = this.parseSpan(line.text);
            switch(span.type) {
                case this.spanTypes.NONE:
                case this.spanTypes.EMPTY:
                    break;
                case this.spanTypes.LEFT:
                    lfrom = span.left;
                    lto = lfrom;
                    break;
                case this.spanTypes.RIGHT:
                    lfrom = lto;
                    lto = span.right;
                    break;
                case this.spanTypes.FULL:
                    lfrom = span.left;
                    lto = span.right;
                    break;
            }
            if (span.type !== this.spanTypes.NONE) {
                line.source = { from : lfrom, to : lto };
            }
        }
    }

    processAst(output) {
        output = output.stdout;

        // Top level decls start with |- or `-
        const topLevelRegex = /^([`|])-/;

        // Refers to the user's source file rather than a system header
        const sourceRegex = /<source>/g;

        // Refers to whatever the most recent file specified was
        const lineRegex = /<line:/;

        let mostRecentIsSource = false;

        // Remove all AST nodes which aren't directly from the user's source code
        for (let i = 0; i < output.length; ++i) {
            if (output[i].text.match(topLevelRegex)) {
                if (output[i].text.match(lineRegex) && mostRecentIsSource) {
                    // do nothing
                } else if (!output[i].text.match(sourceRegex)) {
                    // This is a system header or implicit definition,
                    // remove everything up to the next top level decl
                    // Top level decls with invalid sloc as the file don't change the most recent file
                    const slocRegex = /<<invalid sloc>>/;
                    if (!output[i].text.match(slocRegex)) {
                        mostRecentIsSource = false;
                    }

                    let spliceMax = i + 1;
                    while (output[spliceMax] && !output[spliceMax].text.match(topLevelRegex)) {
                        spliceMax++;
                    }
                    output.splice(i, spliceMax - i);
                    --i;
                } else {
                    mostRecentIsSource = true;
                }
            }
            // Filter out the symbol addresses
            const addressRegex = /^([^A-Za-z]*[A-Za-z]+) 0x[\da-z]+/gm;
            output[i].text = output[i].text.replace(addressRegex, '$1');

            // Filter out <invalid sloc> and <<invalid sloc>>
            const slocRegex = / ?<?<invalid sloc>>?/g;
            output[i].text = output[i].text.replace(slocRegex, '');

            // Unify file references
            output[i].text = output[i].text.replace(sourceRegex, 'line');
        }
        this.parseAndSetSourceLines(output);
        return output;
    }
}
