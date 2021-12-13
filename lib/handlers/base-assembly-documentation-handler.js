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

/**
 * @typedef {object} AssemblyInstructionInfo
 * @property {string} url - URL to more in-depth documentation
 * @property {string} html - Raw HTML to embed into the popup
 * @property {string} tooltip - Text snippet to show in the tooltip
 */

export class BaseAssemblyDocumentationHandler {
    /**
     * Gather the assembly instruction information by the instruction name.
     *
     * Implementors should return null if the instruction is not supported.
     *
     * @param {string} instruction
     * @returns {AssemblyInstructionInfo | null}
     */
    // eslint-disable-next-line no-unused-vars
    getInstructionInformation(instruction) {
        return null;
    }

    /**
     * Handle a request for assembly instruction documentation.
     *
     * Implementors should not have to override this.
     *
     * @param {e.Request} request
     * @param {e.Response} response
     */
    handle(request, response) {
        // If the request had no opcode parameter, we should fail. This assumes
        // no assembly language has a __unknown_opcode instruction.
        const instruction = (request.params.opcode || '__UNKNOWN_OPCODE').toUpperCase();
        const information = this.getInstructionInformation(instruction);
        if (information === null) {
            return response.status(404).send({ error: `Unknown opcode '${instruction}'` });
        }
        // Accept either JSON or Plaintext Content-Type
        const requestedContentType = request.accepts(['text', 'json']);
        switch (requestedContentType) {
            case 'text': response.send(information.html); break;
            case 'json': response.send(information); break;
            default: response.status(406).send({ error: 'Not Acceptable' }); break;
        }
    }
}
