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

import express from 'express';

import {getDocumentationProviderTypeByKey} from '../../asm-docs/index.js';
import {cached, cors} from '../middleware.js';

import {HttpController} from './controller.interfaces.js';

export class AssemblyDocumentationController implements HttpController {
    createRouter(): express.Router {
        const router = express.Router();
        router.get('/api/asm/:arch/:opcode', cors, cached, this.getOpcodeDocumentation.bind(this));
        return router;
    }

    /**
     * Handle request to `/asm/:arch/:opcode` endpoint
     */
    public async getOpcodeDocumentation(req: express.Request, res: express.Response) {
        try {
            const Provider = getDocumentationProviderTypeByKey(req.params.arch);
            const provider = new Provider();
            // If there is no opcode, we should just fail with 404 anyways... Assumes that no assembly language has
            // a __unknown_opcode instruction.
            const instruction = (req.params.opcode || '__UNKNOWN_OPCODE').toUpperCase();
            const information = provider.getInstructionInformation(instruction);
            if (information === null) {
                return res.status(404).send({error: `Unknown opcode '${instruction}'`});
            }

            const contentType = req.accepts(['text', 'json']);
            switch (contentType) {
                case 'text': {
                    res.send(information.html);
                    break;
                }
                case 'json': {
                    res.send(information);
                    break;
                }
                default: {
                    res.status(406).send({error: 'Not Acceptable'});
                    break;
                }
            }
        } catch {
            return res.status(404).send({error: `No documentation for '${req.params.arch}'`});
        }
    }
}
