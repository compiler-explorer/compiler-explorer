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

const
    _ = require('underscore'),
    bodyParser = require('body-parser'),
    logger = require('../logger').logger,
    clientState = require('../clientstate'),
    clientStateNormalizer = require('../clientstate-normalizer').ClientStateNormalizer;


function isMobileViewer(req) {
    return req.header('CloudFront-Is-Mobile-Viewer') === 'true';
}

class NoScriptHandler {
    constructor(router, config) {
        this.router = router;
        this.staticHeaders = config.staticHeaders;
        this.contentPolicyHeader = config.contentPolicyHeader;
        this.clientOptionsHandler = config.clientOptionsHandler;
        this.renderConfig = config.renderConfig;
        this.storageHandler = config.storageHandler;

        this.defaultLanguage = config.opts.wantedLanguage;
        this.compileHandler = config.compileHandler;
    }

    InitializeRoutes(options) {
        this.formDataParser = bodyParser.urlencoded({
            type: 'application/x-www-form-urlencoded',
            limit: options.limit,
            extended: false,
        });

        this.router
            .get('/noscript', (req, res) => {
                this.staticHeaders(res);
                this.contentPolicyHeader(res);
                this.renderNoScriptLayout(false, req, res);
            })
            .get('/noscript/z/:id', _.bind(this.storedStateHandlerNoScript, this))
            .get('/noscript/sponsors', (req, res) => {
                this.staticHeaders(res);
                this.contentPolicyHeader(res);
                res.render('noscript/sponsors', this.renderConfig({
                    embedded: false,
                    mobileViewer: isMobileViewer(req),
                }, req.query));
            })
            .get('/noscript/:language', (req, res) => {
                this.staticHeaders(res);
                this.contentPolicyHeader(res);
                this.renderNoScriptLayout(false, req, res);
            })
            .post('/api/noscript/compile', this.formDataParser, this.compileHandler.handle.bind(this.compileHandler));
    }

    storedStateHandlerNoScript(req, res, next) {
        const id = req.params.id;
        this.storageHandler.expandId(id)
            .then(result => {
                const config = JSON.parse(result.config);

                let clientstate = false;
                if (config.content) {
                    const normalizer = new clientStateNormalizer();
                    normalizer.fromGoldenLayout(config);

                    clientstate = normalizer.normalized;
                } else {
                    clientstate = new clientState.ClientState(config);
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

    createDefaultState(wantedLanguage) {
        const options = this.clientOptionsHandler.get();

        const state = new clientState.State();
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

    renderNoScriptLayout(state, req, res) {
        this.staticHeaders(res);
        this.contentPolicyHeader(res);

        let wantedLanguage = 'c++';
        if (req.params && req.params.language) {
            wantedLanguage = req.params.language;
        } else {
            if (this.defaultLanguage) wantedLanguage = this.defaultLanguage;
            if (req.query.language) wantedLanguage = req.query.language;
        }

        if (!state) {
            state = this.createDefaultState(wantedLanguage);
        }

        res.render('noscript/index', this.renderConfig({
            embedded: false,
            mobileViewer: isMobileViewer(req),
            wantedLanguage: wantedLanguage,
            clientstate: state,
            storedStateId: req.params.id ? req.params.id : false,
        }, req.query));
    }

}

module.exports = NoScriptHandler;
