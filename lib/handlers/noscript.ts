// Copyright (c) 2020, Compiler Explorer Authors
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

import bodyParser from 'body-parser';
import express from 'express';
import _ from 'underscore';

import {assert} from '../assert';
import {ClientState} from '../clientstate';
import {ClientStateNormalizer} from '../clientstate-normalizer';
import {isString} from '../common-utils';
import {logger} from '../logger';
import {ClientOptionsHandler} from '../options-handler';
import {StorageBase} from '../storage';

import {CompileHandler} from './compile';

function isMobileViewer(req: express.Request) {
    return req.header('CloudFront-Is-Mobile-Viewer') === 'true';
}

export class NoScriptHandler {
    readonly staticHeaders: (res: express.Response) => void;
    readonly contentPolicyHeader: (res: express.Response) => void;
    readonly clientOptionsHandler: ClientOptionsHandler;
    readonly renderConfig: (a: any, b: any) => any;
    readonly storageHandler: StorageBase;
    readonly defaultLanguage: any;
    readonly compileHandler: CompileHandler;

    formDataParser: ReturnType<typeof bodyParser.urlencoded> | undefined;

    /* the type for config makes the most sense to define in app.js or api.js */
    constructor(private readonly router: express.Router, config: any) {
        this.staticHeaders = config.staticHeaders;
        this.contentPolicyHeader = config.contentPolicyHeader;
        this.clientOptionsHandler = config.clientOptionsHandler;
        this.renderConfig = config.renderConfig;
        this.storageHandler = config.storageHandler;

        this.defaultLanguage = config.opts.wantedLanguage;
        this.compileHandler = config.compileHandler;
    }

    InitializeRoutes(options: {limit: string}) {
        this.formDataParser = bodyParser.urlencoded({
            type: 'application/x-www-form-urlencoded',
            limit: options.limit,
            extended: false,
        });

        this.router
            .get('/noscript', (req, res) => {
                this.staticHeaders(res);
                this.contentPolicyHeader(res);
                this.renderNoScriptLayout(undefined, req, res);
            })
            .get('/noscript/z/:id', this.storedStateHandlerNoScript.bind(this))
            .get('/noscript/sponsors', (req, res) => {
                this.staticHeaders(res);
                this.contentPolicyHeader(res);
                res.render(
                    'noscript/sponsors',
                    this.renderConfig(
                        {
                            embedded: false,
                            mobileViewer: isMobileViewer(req),
                        },
                        req.query,
                    ),
                );
            })
            .get('/noscript/:language', (req, res) => {
                this.staticHeaders(res);
                this.contentPolicyHeader(res);
                this.renderNoScriptLayout(undefined, req, res);
            })
            .post('/api/noscript/compile', this.formDataParser, this.compileHandler.handle.bind(this.compileHandler));
    }

    storedStateHandlerNoScript(req: express.Request, res: express.Response, next: express.NextFunction) {
        const id = req.params.id;
        this.storageHandler
            .expandId(id)
            .then(result => {
                const config = JSON.parse(result.config);

                let clientstate: ClientState;
                if (config.content) {
                    const normalizer = new ClientStateNormalizer();
                    normalizer.fromGoldenLayout(config);

                    clientstate = normalizer.normalized;
                } else {
                    clientstate = new ClientState(config);
                }

                this.renderNoScriptLayout(clientstate, req, res);

                this.storageHandler.incrementViewCount(id).catch(err => {
                    logger.error(`Error incrementing view counts for ${id} - ${err}`);
                });
            })
            .catch(err => {
                logger.warn(`Could not expand ${id}: ${err}`);
                next({
                    statusCode: 404,
                    message: `ID "${id}" could not be found`,
                });
            });
    }

    createDefaultState(wantedLanguage: string) {
        const options = this.clientOptionsHandler.get();

        const state = new ClientState();
        const session = state.findOrCreateSession(1);
        session.language = wantedLanguage;
        if (options.languages[wantedLanguage]) {
            session.source = options.languages[wantedLanguage].example;
        } else {
            session.source = '';
        }

        const compiler = session.findOrCreateCompiler(1);
        if (options.defaultCompiler[wantedLanguage]) {
            compiler.id = options.defaultCompiler[wantedLanguage];
        } else {
            compiler.id = '';
        }

        return state;
    }

    renderNoScriptLayout(state: ClientState | undefined, req: express.Request, res: express.Response) {
        this.staticHeaders(res);
        this.contentPolicyHeader(res);

        let wantedLanguage = 'c++';
        if (req.params && req.params.language) {
            wantedLanguage = req.params.language;
        } else {
            if (this.defaultLanguage) wantedLanguage = this.defaultLanguage;
            if (req.query.language) {
                const lang = req.query.language;
                assert(isString(lang));
                wantedLanguage = lang;
            }
        }

        if (!state) {
            state = this.createDefaultState(wantedLanguage);
        }

        res.render(
            'noscript/index',
            this.renderConfig(
                {
                    embedded: false,
                    mobileViewer: isMobileViewer(req),
                    wantedLanguage: wantedLanguage,
                    clientstate: state,
                    storedStateId: req.params.id || false,
                },
                req.query,
            ),
        );
    }
}
