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
        this.locTypes = {
            NONE: 'none', // No location specified
            POINT: 'point', // A single location: beginning of a token
            SPAN: 'span', // Two locations: first token to last token (beginning)
        };
    }

    // Accepts "line:a:b" and "col:b"
    parsePoint(ptLine, lastLineNo) {
        const lineRegex = /line:(\d+):/;
        const colRegex = /(?:col|\d):(\d+)(?::|$)/;
        const lineMatch = ptLine.match(lineRegex);
        const colMatch = ptLine.match(colRegex);
        const line = lineMatch ? Number(lineMatch[1]) : lastLineNo;
        const col = colMatch ? Number(colMatch[1]) : null; // Does not happen for well-formed strings
        return {line, col};
    }

    // Accepts "<X, X>" and "<X>", where
    // X can be "col:a" or "line:a:b"
    // lastLineNo - the line number of the previous node,
    // reused when only a column specified.
    parseSpan(line, lastLineNo) {
        const spanRegex = /<((?:line|col)[\d ,:ceilno]+)>/;
        const m = line.match(spanRegex);
        if (m) {
            const span = m[1];
            const beginEnd = span.split(',');
            if (beginEnd.length === 2) {
                const begin = this.parsePoint(beginEnd[0], lastLineNo);
                const end = this.parsePoint(beginEnd[1], begin.line);
                return { type : this.locTypes.SPAN, begin, end };
            } else {
                return { type : this.locTypes.POINT, loc : this.parsePoint(span, lastLineNo) };
            }
        }
        return { type : this.locTypes.NONE };
    }

    // Link the AST lines with spans of source locations (lines+columns)
    parseAndSetSourceLines(astDump) {
        var lfrom = {line:null, loc:null}, lto = {line:null, loc:null};
        for (var line of astDump) {
            const span = this.parseSpan(line.text, lfrom.line);
            switch (span.type) {
                case this.locTypes.NONE:
                    break;
                case this.locTypes.POINT:
                    lfrom = span.loc;
                    lto = span.loc;
                    break;
                case this.locTypes.SPAN:
                    lfrom = span.begin;
                    lto = span.end;
                    break;
            }
            if (span.type !== this.locTypes.NONE) {
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
        const lineRegex = /<(col|line):/;

        let mostRecentIsSource = false;

        // Remove all AST nodes which aren't directly from the user's source code
        for (let i = 0; i < output.length; ++i) {
            if (topLevelRegex.test(output[i].text)) {
                if (lineRegex.test(output[i].text) && mostRecentIsSource) {
                    // do nothing
                } else if (!sourceRegex.test(output[i].text)) {
                    // This is a system header or implicit definition,
                    // remove everything up to the next top level decl
                    // Top level decls with invalid sloc as the file don't change the most recent file
                    const slocRegex = /<<invalid sloc>>/;
                    if (!slocRegex.test(output[i].text)) {
                        mostRecentIsSource = false;
                    }

                    let spliceMax = i + 1;
                    while (output[spliceMax] && !topLevelRegex.test(output[spliceMax].text)) {
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
