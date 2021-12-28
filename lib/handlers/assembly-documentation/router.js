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

import { propsFor } from '../../properties';

import { Amd64DocumentationHandler } from './amd64';
import { Arm32DocumentationHandler } from './arm32';
import { JavaDocumentationHandler } from './java';
import { Mos6502DocumentationHandler } from './mos6502';

/** @type {Record.<string, BaseAssemblyDocumentationHandler>} */
const ASSEMBLY_DOCUMENTATION_HANDLERS = {
    amd64: new Amd64DocumentationHandler(),
    arm32: new Arm32DocumentationHandler(),
    java: new JavaDocumentationHandler(),
    6502: new Mos6502DocumentationHandler(),
};

const MAX_STATIC_AGE = propsFor('asm-docs')('staticMaxAgeSecs', 10);

/**
 * Initialize all Assembly Docs routes
 *
 * @param {e.Router} router
 */
export const setup = (router) => router.get('/asm/:arch/:opcode', (req, res) => {
    if (MAX_STATIC_AGE > 0) {
        res.setHeader('Cache-Control', `public, max-age=${MAX_STATIC_AGE}`);
    }
    const architecture = req.params.arch;
    const handler = ASSEMBLY_DOCUMENTATION_HANDLERS[architecture];
    if (handler !== undefined) {
        return handler.handle(req, res);
    }
    res.status(404).json({ error: `No documentation for '${architecture}'` }).send();
});
