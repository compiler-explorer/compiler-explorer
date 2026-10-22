// Copyright (c) 2026, Compiler Explorer Authors
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

export function getDeclaredFunctionNames(source: string): string[] {
    const names = new Set<string>();
    const declarationPattern = /^\s*(?:fun|fundecl)\s+([A-Za-z_]\w*)\s*\(/gm;
    let match = declarationPattern.exec(source);

    while (match) {
        if (match[1]) names.add(match[1]);
        match = declarationPattern.exec(source);
    }

    return [...names];
}

export interface DucklingFunctionDefinition {
    name: string;
    lineNumber: number;
    column: number;
}

export function getFunctionDefinitions(source: string): DucklingFunctionDefinition[] {
    const definitions: DucklingFunctionDefinition[] = [];
    const definitionPattern = /^[ \t]*fun[ \t]+([A-Za-z_]\w*)[ \t]*\(/gm;
    let lineNumber = 1;
    let scannedThrough = 0;
    let match = definitionPattern.exec(source);

    while (match) {
        lineNumber += (source.slice(scannedThrough, match.index).match(/\n/g) ?? []).length;
        const name = match[1];
        if (name) {
            definitions.push({
                name,
                lineNumber,
                column: match[0].indexOf(name) + 1,
            });
        }
        scannedThrough = definitionPattern.lastIndex;
        match = definitionPattern.exec(source);
    }

    return definitions;
}
