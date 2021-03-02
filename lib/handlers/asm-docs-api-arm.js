// Copyright (c) 2017, Compiler Explorer Authors
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

import * as props from '../properties';

import { getAsmOpcode } from './asm-docs-arm';

export class AsmDocsHandler {
    constructor() {
        const asmProps = props.propsFor('asm-docs');
        this.staticMaxAgeSecs = asmProps('staticMaxAgeSecs', 10);
        this.conditionalRe = /([A-Za-z])+(EQ|NE|CS|CC|MI|PL|VS|VC|HI|LS|GE|LT|GT|LE|AL)/;
    }

    // Notes for conditionals:
    // https://developer.arm.com/documentation/dui0473/m/condition-codes/condition-code-suffixes-and-related-flags
    getOpcodeConditional(opcode) {
        if (!opcode) return;

        const matches = opcode.match(this.conditionalRe);
        if (matches) {
            // Get the base opcode
            var description = getAsmOpcode(matches[1]);
            var prependText;
            // Add the conditional data
            switch (matches[2]) {
                case "EQ":
                    prependText = "If equal, "
                    break;
                case "NE":
                    prependText = "If not equal, "
                    break;
                case "CS":
                    prependText = "If carry set, "
                    break;
                case "CC":
                    prependText = "If carry clear, "
                    break;
                case "MI":
                    prependText = "If negative, "
                    break;
                case "PL":
                    prependText = "If positive or zero, "
                    break;
                case "VS":
                    prependText = "If overflow, "
                    break;
                case "VC":
                    prependText = "If no overflow, "
                    break;
                case "HI":
                    prependText = "If unsigned higher, "
                    break;
                case "LS":
                    prependText = "If unsigned lower or same, "
                    break;
                case "GE":
                    prependText = "If signed greater than or equal, "
                    break;
                case "LT":
                    prependText = "If signed less than, "
                    break;
                case "GT":
                    prependText = "If signed greater than, "
                    break;
                case "LE":
                    prependText = "If signed less than or equal, "
                    break;
            }

            description["tooltip"] = prependText + description["tooltip"]
            description["html"] = prependText + description["html"]

            return description;
        }
    }

    handle(req, res) {
        let info = getAsmOpcode(req.params.opcode);
        if (!info) {
            info = this.getOpcodeConditional(req.params.opcode);
        }
        if (this.staticMaxAgeSecs) {
            res.setHeader('Cache-Control', 'public, max-age=' + this.staticMaxAgeSecs);
        }
        if (req.accepts(['text', 'json']) === 'json') {
            res.send({ found: !!info, result: info });
        } else {
            if (info)
                res.send(info.html);
            else
                res.send('Unknown opcode');
        }
    }
}