// Copyright (c) 2024, Compiler Explorer Authors
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

import fs from 'node:fs/promises';

import express from 'express';

import {CompilationQueue} from '../../compilation-queue.js';
import {logger} from '../../logger.js';
import {SentryCapture} from '../../sentry.js';
import {ICompileHandler} from '../compile.interfaces.js';

import {HttpController} from './controller.interfaces.js';

export class HealthcheckController implements HttpController {
    public constructor(
        private readonly compilationQueue: CompilationQueue,
        private readonly healthCheckFilePath: string | null,
        private readonly compileHandler: ICompileHandler,
        private readonly isExecutionWorker: boolean,
    ) {}

    createRouter(): express.Router {
        const router = express.Router();
        // TODO: Consider if this could be `GET` only. It doesn't quite make sense to do anything but GET the
        //  healthcheck
        router.use('/healthcheck', this.healthcheck.bind(this));
        return router;
    }

    /**
     * Handle request to `/healthcheck` endpoint
     * @param req
     * @param res
     */
    public async healthcheck(req: express.Request, res: express.Response) {
        // Enqueue an empty job to ensure the queue is running. This check simply tests that the queue is being
        // processed by _something_.
        await this.compilationQueue.enqueue(async () => {}, {highPriority: true});

        // If this is a worker, we don't require that the server has languages configured.
        if (!this.isExecutionWorker && !this.compileHandler.hasLanguages()) {
            logger.error(`*** HEALTH CHECK FAILURE: no languages/compilers detected`);
            return res.status(500).send();
        }

        // If we have a healthcheck file, we require that it exists and it is non-empty. The /efs/.health file contents
        // are not important, but the file acts as a health check for EFS mounts.
        if (this.healthCheckFilePath !== null) {
            try {
                const content = await fs.readFile(this.healthCheckFilePath);
                if (content.length === 0) {
                    throw new Error('File is empty');
                }
                res.set('Content-Type', 'text/html');
                return res.send(content);
            } catch (e) {
                logger.error(`*** HEALTH CHECK FAILURE: while reading file '${this.healthCheckFilePath}' got ${e}`);
                SentryCapture(e, 'Health check');
                return res.status(500).send();
            }
        }

        return res.send('Everything is awesome');
    }
}
