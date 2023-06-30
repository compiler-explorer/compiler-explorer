// Copyright (c) 2023, Compiler Explorer Authors
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

import {PropertyGetter} from '../properties.interfaces.js';
import {AsmParser} from './asm-parser.js';

export class HexagonAsmParser extends AsmParser {
    vliwPacketBegin: RegExp;
    vliwPacketEnd: RegExp;

    constructor(compilerProps?: PropertyGetter) {
        super(compilerProps);

        this.vliwPacketBegin = /^\s*{\s*$/;
        this.vliwPacketEnd = /^\s*}\s*(?::\w+)?\s*$/;
    }

    override checkVLIWpacket(line, inVLIWpacket) {
        if (this.vliwPacketBegin.test(line)) {
            return true;
        } else if (this.vliwPacketEnd.test(line)) {
            return false;
        }

        return inVLIWpacket;
    }

    override hasOpcode(line, inNvccCode, inVLIWpacket?) {
        // Remove any leading label definition...
        const match = line.match(this.labelDef);
        if (match) {
            line = line.substr(match[0].length);
        }
        // Strip any comments
        line = line.split(this.commentRe, 1)[0];
        // .inst generates an opcode, so also counts
        if (this.instOpcodeRe.test(line)) return true;
        if (inVLIWpacket) {
            return !(this.vliwPacketBegin.test(line) || this.vliwPacketEnd.test(line));
        }
        // Detect assignment, that's not an opcode...
        if (this.assignmentDef.test(line)) return false;
        return !!this.hasOpcodeRe.test(line);
    }
}
