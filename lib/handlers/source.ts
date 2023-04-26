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

import express from 'express';
import _ from 'underscore';

import {Source} from '../sources/index.js';

// TODO(supergrecko): Maybe provide a more elegant way to do this instead of accessing keys?
const ALLOWED_ACTIONS = new Set(['list', 'load']);

export class SourceHandler {
    public constructor(private fileSources: Source[], private addStaticHeaders: (res: express.Response) => void) {}

    private getSourceForHandler(handler: string): Source | null {
        const records = _.indexBy(this.fileSources, 'urlpart');
        return records[handler] || null;
    }

    private getActionForSource(source: Source, action: string): ((...args: unknown[]) => Promise<unknown>) | null {
        return ALLOWED_ACTIONS.has(action) ? source[action] : null;
    }

    public handle(req: express.Request, res: express.Response, next: express.NextFunction): void {
        // Split URLs with the scheme /source/browser/list into the source and the action to perform
        const [_, handler, action, ...rest] = req.url.split('/');
        const source = this.getSourceForHandler(handler);
        if (source === null) {
            next();
            return;
        }
        const callback = this.getActionForSource(source, action);
        if (callback === null) {
            next();
            return;
        }
        callback(...rest)
            .then(response => {
                this.addStaticHeaders(res);
                res.send(response);
            })
            .catch(err => {
                res.send({err: err});
            });
    }
}
