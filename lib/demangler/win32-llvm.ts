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

import {unwrap} from '../assert.js';
import * as utils from '../utils.js';
import {Win32Demangler} from './win32.js';

export class LLVMWin32Demangler extends Win32Demangler {
    static override get key() {
        // this should be used for llvm-undname
        return 'win32-llvm';
    }

    override async createTranslations() {
        const translations: Record<string, string> = {};
        const flags = ['--no-access-specifier', '--no-calling-convention'];

        const symbols = new Set(unwrap(this.win32RawSymbols));
        const execOptions = this.compiler.getDefaultExecOptions();
        execOptions.input = [...symbols].join('\n') + '\n';
        const output = await this.compiler.exec(this.demanglerExe, flags, execOptions);

        // For each input line llvm-undname echoes the line, then prints the demangled name only if demangling
        // succeeded (the error goes to stderr), then an empty line: so a symbol yields one or two lines.
        for (const block of output.stdout.split(/(?:\r?\n){2,}/)) {
            const [mangled, demangled] = utils.splitLines(block);
            if (demangled && symbols.has(mangled)) {
                translations[this.hasQuotesAroundDecoratedLabels ? `"${mangled}"` : mangled] = demangled;
            }
        }

        this.translations = translations;
        return translations;
    }
}
