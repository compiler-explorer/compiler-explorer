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

import * as Sentry from '@sentry/node';
import express from 'express';
import fs from 'fs-extra';

import {CompilationQueue} from '../compilation-queue';
import {logger} from '../logger';

export class HealthCheckHandler {
    public readonly handle: (req: any, res: any) => Promise<void>;

    private lastJobPass = Date.now();

    private interval: NodeJS.Timer;

    constructor(private readonly compilationQueue: CompilationQueue, private readonly filePath: any) {
        this.handle = this._handle.bind(this);

        /* Every minute, try to pass something through the compilation
         * queue to ensure nothing gets deadlocked. Queue it at the
         * front but it may have to wait a couple minutes if it gets
         * stuck behind a super long compilation.
         *
         */
        this.interval = setInterval(() => {
            this.compilationQueue.enqueue(
                async () => {
                    this.lastJobPass = Date.now();
                },
                {jumpTheQueue: true},
            );
        }, 60 * 1000);
    }

    async _handle(req: express.Request, res: express.Response) {
        const timeSinceLast = Date.now() - this.lastJobPass;
        // We've seen requests take as long as 130/140s
        // Threshold is 140s + 60s interval + another 40s grace period
        if (timeSinceLast > 240 * 1000) {
            logger.error(
                `*** HEALTH CHECK FAILURE: Our checks haven't passed through the queue in ${Math.round(
                    timeSinceLast,
                )} seconds`,
            );
            res.status(500).end();
        }

        if (!this.filePath) {
            res.send('Everything is awesome');
            return;
        }

        try {
            const content = await fs.readFile(this.filePath);
            if (content.length === 0) throw new Error('File is empty');
            res.set('Content-Type', 'text/html');
            res.send(content);
        } catch (e) {
            logger.error(`*** HEALTH CHECK FAILURE: while reading file '${this.filePath}' got ${e}`);
            Sentry.captureException(e);
            res.status(500).end();
        }
    }

    // util for test cases / not making mocha hang
    internal_clear_interval() {
        clearInterval(this.interval);
    }
}
