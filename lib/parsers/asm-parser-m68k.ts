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

import type {PropertyGetter} from '../properties.interfaces.js';
import {AsmParser} from './asm-parser.js';

export class AsmParserM68k extends AsmParser {
    constructor(compilerProps?: PropertyGetter) {
        super(compilerProps);
        // m68k GAS uses | as the comment character instead of #
        this.commentRe = /[#;|]/;
        this.commentOnly = /^\s*(((#|@|\/\/|\|).*)|(\/\*.*\*\/)|(;\s*)|(;[^;].*)|(;;\s*[^\s#].*))$/;

        // m68k GCC emits |APP / |NO_APP instead of #APP / #NO_APP for inline asm blocks
        this.startAppBlock = /^\|APP.*$/;
        this.endAppBlock = /^\|NO_APP.*$/;
        this.startAsmNesting = /^\| Begin ASM.*$/;
        this.endAsmNesting = /^\| End ASM.*$/;
    }
}
