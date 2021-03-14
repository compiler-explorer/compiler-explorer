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

import * as props from '../properties';

import { getAsmOpcode } from './asm-docs-arm32';

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

        const conditionals = {
            EQ: 'If equal, ',
            NE: 'If not equal, ',
            CS: 'If carry set, ',
            CC: 'If carry clear, ',
            MI: 'If negative, ',
            PL: 'If positive or zero, ',
            VS: 'If overflow, ',
            VC: 'If no overflow, ',
            HI: 'If unsigned higher, ',
            LS: 'If unsigned lower or same, ',
            GE: 'If signed greater than or equal, ',
            LT: 'If signed less than, ',
            GT: 'If signed greater than, ',
            LE: 'If signed less than or equal, ',
        };

        const matches = opcode.match(this.conditionalRe);
        if (matches) {
            let opcodeDescription = getAsmOpcode(matches[1]);
            if (!opcodeDescription) return;

            let conditionalText = conditionals[matches[2]] || '';

            opcodeDescription.tooltip = conditionalText + opcodeDescription.tooltip;
            opcodeDescription.html = conditionalText + opcodeDescription.html;

            return opcodeDescription;
        }
    }

    handle(req, res) {
        const opcode = req.params.opcode.toUpperCase();
        const info = getAsmOpcode(opcode) || this.getOpcodeConditional(opcode);
        if (this.staticMaxAgeSecs) {
            res.setHeader('Cache-Control', 'public, max-age=' + this.staticMaxAgeSecs);
        }
        if (req.accepts(['text', 'json']) === 'json') {
            res.send({found: !!info, result: info});
        } else {
            if (info)
                res.send(info.html);
            else
                res.send('Unknown opcode');
        }
    }
}
