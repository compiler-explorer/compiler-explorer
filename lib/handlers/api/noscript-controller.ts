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

import {ClientOptionsHandler} from '../../options-handler.js';
import {StorageBase} from '../../storage/index.js';
import {CompileHandler} from '../compile.js';
import {cached} from '../middleware.js';

import {HttpController} from './controller.interfaces.js';

export class NoscriptController implements HttpController {
    public constructor(
        private readonly clientOptionsHandler: ClientOptionsHandler,
        private readonly renderConfig: (a: any, b: any) => any,
        private readonly storageHandler: StorageBase,
        // TODO: Noscript should not depend on the compile handler, instead, the compile handler should assign the route
        private readonly compileHandler: CompileHandler,
        private readonly bodyParserLimit: number,
    ) {}

    createRouter(): express.Router {
        const router = express.Router();
        router.get('/noscript', cached, this.viewNoscript.bind(this));
        router.get('/noscript/z/:id', cached, this.viewNoscriptShortlink.bind(this));
        router.get('/noscript/sponsors', cached, this.viewNoscriptSponsors.bind(this));
        router.get('/noscript/:language', cached, this.viewNoscriptLanguages.bind(this));
        router.post(
            '/api/noscript/compile',
            cached,
            express.urlencoded({
                type: 'application/x-www-form-urlencoded',
                limit: this.bodyParserLimit,
                extended: false,
            }),
            this.createNoscriptCompilation.bind(this),
        );
        return router;
    }

    public async viewNoscript(req: express.Request, res: express.Response) {}

    public async viewNoscriptSponsors(req: express.Request, res: express.Response) {}

    public async viewNoscriptShortlink(req: express.Request, res: express.Response) {}

    public async viewNoscriptLanguages(req: express.Request, res: express.Response) {}

    public async createNoscriptCompilation(req: express.Request, res: express.Response) {}
}
