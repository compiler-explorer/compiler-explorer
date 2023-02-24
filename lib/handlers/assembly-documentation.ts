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

import {BaseAssemblyDocumentationProvider, getDocumentationProviderTypeByKey} from '../asm-docs';
import {propsFor} from '../properties';

const MAX_STATIC_AGE = propsFor('asm-docs')('staticMaxAgeSecs', 10);

const onDocumentationProviderRequest = (
    provider: BaseAssemblyDocumentationProvider,
    request: express.Request,
    response: express.Response
) => {
    // If the request had no opcode parameter, we should fail. This assumes
    // no assembly language has a __unknown_opcode instruction.
    const instruction = (request.params.opcode || '__UNKNOWN_OPCODE').toUpperCase();
    const information = provider.getInstructionInformation(instruction);
    if (information === null) {
        return response.status(404).send({error: `Unknown opcode '${instruction}'`});
    }
    // Accept either JSON or Plaintext Content-Type
    const requestedContentType = request.accepts(['text', 'json']);
    switch (requestedContentType) {
        case 'text': {
            response.send(information.html);
            break;
        }
        case 'json': {
            response.send(information);
            break;
        }
        default: {
            response.status(406).send({error: 'Not Acceptable'});
            break;
        }
    }
};

/** Initialize API routes for assembly documentation */
export const withAssemblyDocumentationProviders = (router: express.Router) =>
    router.get('/asm/:arch/:opcode', (req, res) => {
        if (MAX_STATIC_AGE > 0) {
            res.setHeader('Cache-Control', `public, max-age=${MAX_STATIC_AGE}`);
        }
        const arch = req.params.arch;
        // makeKeyedTypeGetter throws if the key is not found. We do not wish
        // crash CE if this happens, so we catch the error and return a 404.
        try {
            const providerClass = getDocumentationProviderTypeByKey(arch);
            onDocumentationProviderRequest(new providerClass(), req, res);
        } catch {
            res.status(404)
                .json({error: `No documentation for '${arch}'`})
                .send();
        }
    });
