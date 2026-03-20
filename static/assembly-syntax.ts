// Copyright (c) 2025, Compiler Explorer Authors
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

import {type AssemblyInstructionInfo} from '../types/assembly-docs.interfaces.js';

const AssemblySyntaxesList = ['att', 'intel'] as const;
export type AssemblySyntax = (typeof AssemblySyntaxesList)[number];

const ATT_SYNTAX_WARNING = 'WARNING: The information shown pertains to Intel syntax.';
const CARDINALITY_REGEX = /\b(?:first|second|third)\s+operands?\b/i;
const SOURCE_DEST_REGEX = /\b(?:source|destination)\b/i;

function needsAttWarning(text: string): boolean {
    return CARDINALITY_REGEX.test(text) && SOURCE_DEST_REGEX.test(text);
}

export function addAttSyntaxWarning(data: AssemblyInstructionInfo, syntax: AssemblySyntax): AssemblyInstructionInfo {
    if (syntax !== 'att') return data;
    if (!needsAttWarning(data.tooltip) && !needsAttWarning(data.html)) return data;
    return {
        ...data,
        tooltip: '***' + ATT_SYNTAX_WARNING + '***\n\n' + data.tooltip,
        html: '<b><em>' + ATT_SYNTAX_WARNING + '</em></b><br><br>' + data.html,
    };
}
