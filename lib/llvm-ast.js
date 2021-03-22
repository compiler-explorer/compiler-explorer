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
                    let slocRegex = /<<invalid sloc>>/;
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
            let slocRegex = / ?<?<invalid sloc>>?/g;
            output[i].text = output[i].text.replace(slocRegex, '');

            // Unify file references
            output[i].text = output[i].text.replace(sourceRegex, 'line');
        }
        const spanRegex = /<((?:line|col)[\d ,:ceilno]+)>/;
        var lfrom, lto;
        for (const element of output) {
            let m = element.text.match(spanRegex);
            if (m) {
                let span = m[1];
                const twoLineRegex = /line:(\d+)[,:].*line:(\d+)[:>]/;
                let ll = span.match(twoLineRegex);
                if (ll) {
                    lfrom = Number(ll[1]);
                    lto = Number(ll[2]);
                } else {
                    const leftLineRegex = /^line:(\d+)/;
                    let left = span.match(leftLineRegex);
                    if (left) {
                        lfrom = Number(left[1]);
                        lto = lfrom;
                    } else {
                        const rightLineRegex = / line:(\d+):/;
                        let right = span.match(rightLineRegex);
                        if (right) {
                            lfrom = lto;
                            lto = Number(right[1]);
                        }
                    }
                }
                element.source = { from : lfrom, to : lto };
            }
        }
        return output;
    }
}
