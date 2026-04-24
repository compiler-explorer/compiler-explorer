// Copyright (C) 2026 Hudson River Trading LLC <opensource@hudson-trading.com>
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

import type {McpServer} from '@modelcontextprotocol/sdk/server/mcp.js';
import {z} from 'zod';

import {getDocumentationProviderTypeByKey} from '../../asm-docs/index.js';

export function registerAsmDocsTool(server: McpServer): void {
    server.tool(
        'lookup_asm_instruction',
        'Look up documentation for an assembly instruction',
        {
            instruction_set: z
                .string()
                .describe('Instruction set architecture (e.g. "amd64", "aarch64", "arm32", "avr", "java")'),
            opcode: z.string().describe('Assembly instruction mnemonic (e.g. "MOV", "ADD", "JMP")'),
        },
        async ({instruction_set, opcode}) => {
            try {
                const Provider = getDocumentationProviderTypeByKey(instruction_set);
                const provider = new Provider();
                const information = provider.getInstructionInformation(opcode.toUpperCase());
                if (information) {
                    return {content: [{type: 'text', text: JSON.stringify(information, null, 2)}]};
                }
                return {
                    content: [{type: 'text', text: `No documentation found for ${opcode} on ${instruction_set}`}],
                    isError: true,
                };
            } catch (e) {
                return {
                    content: [
                        {type: 'text', text: `Unknown instruction set: ${instruction_set}. ${(e as Error).message}`},
                    ],
                    isError: true,
                };
            }
        },
    );
}
