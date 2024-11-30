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

import express from 'express';
import _ from 'underscore';

import {Source} from '../../../types/source.interfaces.js';
import {cached, cors} from '../middleware.js';

import {HttpController} from './controller.interfaces.js';

export class SourceController implements HttpController {
    public constructor(private readonly sources: Source[]) {}

    createRouter(): express.Router {
        const router = express.Router();
        router.get('/source/:source/list', cors, cached, this.listEntries.bind(this));
        router.get('/source/:source/load/:language/:filename', cors, cached, this.loadEntry.bind(this));
        return router;
    }

    /**
     * Handle request to `/source/<source>/list` endpoint
     */
    public async listEntries(req: express.Request, res: express.Response) {
        const source = this.getSourceForHandler(req.params.source);
        if (source === null) {
            res.sendStatus(404);
            return;
        }
        const entries = await source.list();
        res.json(entries);
    }

    /**
     * Handle request to `/source/<source>/load/<language>/<filename>` endpoint
     */
    public async loadEntry(req: express.Request, res: express.Response) {
        const source = this.getSourceForHandler(req.params.source);
        if (source === null) {
            res.sendStatus(404);
            return;
        }
        const entry = await source.load(req.params.language, req.params.filename);
        res.json(entry);
    }

    private getSourceForHandler(handler: string): Source | null {
        const records = _.indexBy(this.sources, 'urlpart');
        return records[handler] || null;
    }
}
