import express from 'express';
import _ from 'underscore';

import {addStaticHeaders} from '../../../app.js';
import {Source} from '../../../types/source.interfaces.js';

export class SourceController {
    public constructor(private readonly sources: Source[]) {}

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
        addStaticHeaders(res);
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
        addStaticHeaders(res);
        res.json(entry);
    }

    private getSourceForHandler(handler: string): Source | null {
        const records = _.indexBy(this.sources, 'urlpart');
        return records[handler] || null;
    }
}
