// Copyright (c) 2018, 2021, Microsoft Corporation & Compiler Explorer Authors
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

import {logger} from '../logger';
import {PropertyGetter} from '../properties.interfaces';
import * as utils from '../utils';

import {AsmParser} from './asm-parser';
import {VcAsmParser} from './asm-parser-vc';
import {AsmRegex} from './asmregex';

export class Vc6AsmParser extends VcAsmParser {
    constructor(compilerProps?: PropertyGetter) {
        super(compilerProps);
        this.miscDirective =
            /^\s*(include|INCLUDELIB|TITLE|\.|else$|endif$|if @Version|FLAT|ASSUME|THUMB|ARM64|TTL|END$)/;
        this.beginSegment =
            /^(CONST|_BSS|_DATA|_TLS|\.?[prx]?data(\$[A-Za-z]+)?|CRT(\$[A-Za-z]+)?|_TEXT|\.?text(\$[A-Za-z]+)?)\s+SEGMENT|\s*AREA/;
        this.endSegment =
            /^(CONST|_BSS|_DATA|_TLS|[prx]?data(\$[A-Za-z]+)?|CRT(\$[A-Za-z]+)?|_TEXT|text(\$[A-Za-z]+)?)\s+ENDS/;
    }

    // NB processAsm previously used `definesFunction` to "checkBeginFunction". The base class uses `beginFunction`
    // All tests pass with both using definesFunction, so I don't want to introduce complexity unnecessarily, and so
    // will leave this behaviour. If this breaks our downstream friends, we can reinstate (with tests to prevent it
    // happening again).
}
