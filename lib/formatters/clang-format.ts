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

import {UnprocessedExecResult} from '../../types/execution/execution.interfaces';
import * as exec from '../exec';

import {BaseFormatter} from './base';
import {FormatOptions} from './base.interfaces';

export class ClangFormatFormatter extends BaseFormatter {
    static get key() {
        return 'clangformat';
    }

    override async format(source: string, options: FormatOptions): Promise<UnprocessedExecResult> {
        const tabText = options.useSpaces ? 'Never' : 'AlignWithSpaces';
        const arg = `{BasedOnStyle: ${options.baseStyle}, IndentWidth: ${options.tabWidth}, UseTab: ${tabText}}`;
        const result = await exec.execute(this.formatterInfo.exe, [`--style=${arg}`], {input: source});
        if (result.code === 0) {
            // Repair http(s) includes, i.e., remove the spaces inserted before the hier-part of the URI
            const includeFind = /^(\s*#\s*include\s*["<]https?:)\s+(\/\/[^">]+[">].*)/;
            const includeReplacement = '$1$2';
            const lines = result.stdout.split('\n');
            for (const idx in lines) lines[idx] = lines[idx].replace(includeFind, includeReplacement);
            result.stdout = lines.join('\n');
        }
        return result;
    }
}
