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

import * as utils from '../utils';

import {Win32Demangler} from './win32';

export class LLVMWin32Demangler extends Win32Demangler {
    static get key() {
        // this should be used for llvm-undname
        return 'win32-llvm';
    }

    async execDemangler() {
        const translations = {};
        const flags = ['--no-access-specifier', '--no-calling-convention'];

        const demangleFromStdin = async stdin => {
            const args = [...flags];
            const execOptions = this.compiler.getDefaultExecOptions();
            execOptions.input = stdin;
            const output = await this.compiler.exec(this.demanglerExe, args, execOptions);
            const oArray = utils.splitLines(output.stdout);
            const outputArray = oArray.filter(Boolean);

            // llvm-undname just output:
            // mangledName
            // unmangledName
            for (let i = 0; i < outputArray.length; ++i) {
                if (this.hasQuotesAroundDecoratedLabels) {
                    translations[`"${outputArray[i]}"`] = outputArray[++i];
                } else {
                    translations[outputArray[i]] = outputArray[++i];
                }
            }
        };

        this.win32RawSymbols.sort();

        let lastSymbol = null;
        let symbolArray = [];
        for (const symb of this.win32RawSymbols) {
            if (symb === lastSymbol) {
                continue;
            }
            lastSymbol = symb;
            symbolArray.push(symb);
        }

        const stdin = symbolArray.join('\n') + '\n';
        await demangleFromStdin(stdin);

        return translations;
    }
}
