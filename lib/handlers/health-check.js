// Copyright (c) 2017, Matt Godbolt
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

const
    fs = require('fs-extra'),
    logger = require('../logger').logger,
    Sentry = require('@sentry/node');

class HealthCheckHandler {
    constructor(filePath) {
        this.filePath = filePath;

        this.handle = this._handle.bind(this);
    }

    async _handle(req, res) {
        if (!this.filePath) {
            res.send('Everything is awesome');
            return;
        }

        try {
            const content = await fs.readFile(this.filePath);
            if (content.length === 0) throw Error("File is empty");
            res.send(content);
        } catch (e) {
            logger.error(`*** HEALTH CHECK FAILURE: while reading file '${this.filePath}' got ${e}`);
            Sentry.captureException(e);
            res.status(500).end();
        }
    }
}

module.exports.HealthCheckHandler = HealthCheckHandler;
