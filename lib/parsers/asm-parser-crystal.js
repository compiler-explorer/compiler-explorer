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

import {AsmParser} from './asm-parser';

export class CrystalAsmParser extends AsmParser {
    constructor() {
        super();

        // Crystal labels are not mangled, instead they are enclosed in quotes
        // whenever necessary
        this.labelDef = /^([\w$.@]+|"(?:[^"\\]|\\.)+"):$/;
        this.indentedLabelDef = /^\s*([\w$.@]+|"(?:[^"\\]|\\.)+"):/;
        this.labelFindNonMips = /[\w.][\w$.@]*|"(?:[^"\\]|\\.)+"/g;
        this.labelFindMips = /[\w$.@]+|"(?:[^"\\]|\\.)+"/g;
        this.definesGlobal = /^\s*\.(?:globa?l|GLB|export)\s*([.A-Z_a-z][\w$.]*|"(?:[^"\\]|\\.)+")/;
        this.definesWeak = /^\s*\.(?:weakext|weak)\s*([.A-Z_a-z][\w$.]*|"(?:[^"\\]|\\.)+")/;
    }
}
