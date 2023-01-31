import * as express from 'express';
import request from 'request';

import { BaseShortener } from './base.js';

export class FwdShortener extends BaseShortener {
    static override get key() { return 'fwd'; }

    override handle(req: express.Request, res: express.Response) {
        const url = `${req.protocol}://enzyme.mit.edu/explorer#${req.body.config}`;
        const options = {
            url: 'https://fwd.gymni.ch/shorten',
            json: { url: url },
            method: 'POST',
        };
        const callback = (err, resp: request.Response, body) => {
            if (!err && resp.statusCode === 200) {
                res.send({ url: body["url"] });
            } else {
                res.status(resp.statusCode).send(err);
            }
        };
        request.post(options, callback);
    }
}