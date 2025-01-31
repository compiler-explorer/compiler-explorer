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

import {FormattingService} from '../../formatting-service.js';
import {cached, cors, jsonOnly} from '../middleware.js';

import {HttpController} from './controller.interfaces.js';

export class FormattingController implements HttpController {
    public constructor(private formattingService: FormattingService) {}

    createRouter(): express.Router {
        const router = express.Router();
        router.post('/api/format/:tool', cors, cached, jsonOnly, this.format.bind(this));
        router.get('/api/formats', cors, cached, this.getFormatters.bind(this));
        return router;
    }

    /** Handle requests to /api/format/:tool */
    public async format(req: express.Request, res: express.Response) {
        const formatter = this.formattingService.getFormatterById(req.params.tool);
        // Ensure the target formatter exists
        if (formatter === null) {
            return res.status(422).json({exit: 2, answer: `Unknown format tool '${req.params.tool}'`});
        }
        // Ensure there is source code to format
        if (!req.body || !req.body.source) {
            res.status(400).json({exit: 0, answer: ''});
            return;
        }
        // Ensure that the requested style is supported by the formatter
        if (!formatter.isValidStyle(req.body.base)) {
            return res.status(422).json({exit: 3, answer: `Style '${req.body.base}' is not supported`});
        }
        // Do the formatting
        try {
            const formatted = await formatter.format(req.body.source, {
                useSpaces: req.body.useSpaces === undefined ? true : req.body.useSpaces,
                tabWidth: req.body.tabWidth === undefined ? 4 : req.body.tabWidth,
                baseStyle: req.body.base,
            });
            res.json({exit: formatted.code, answer: formatted.stdout || formatted.stderr || ''});
        } catch (err: unknown) {
            res.status(500).json({
                exit: 1,
                thrown: true,
                answer:
                    (err && Object.hasOwn(err, 'message') && (err as Record<'message', 'string'>).message) ||
                    'Internal server error',
            });
        }
    }

    /** Handle requests to /api/formats */
    public async getFormatters(_: express.Request, res: express.Response) {
        const formatters = this.formattingService.getFormatters();
        res.send(formatters.map(formatter => formatter.formatterInfo));
    }
}
