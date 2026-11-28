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

import type {UnprocessedExecResult} from '../../types/execution/execution.interfaces.js';
import * as exec from '../exec.js';
import type {FormatOptions} from './base.interfaces.js';
import {BaseFormatter} from './base.js';

export class MachFmtFormatter extends BaseFormatter {
    static get key() {
        return 'machfmt';
    }

    /**
     * Format the provided source code
     *
     * The `-` operand reads the buffer from stdin and writes its canonical
     * layout to stdout, reading no manifest. A buffer that does not parse
     * leaves stdout empty, reports on stderr and exits non-zero, so the caller
     * never sees a partial document.
     */
    override async format(source: string, options: FormatOptions): Promise<UnprocessedExecResult> {
        return await exec.execute(this.formatterInfo.exe, ['fmt', '-'], {input: source});
    }

    /**
     * mach fmt has one canonical layout and no styling options
     */
    override isValidStyle(style: string): boolean {
        return true;
    }
}
