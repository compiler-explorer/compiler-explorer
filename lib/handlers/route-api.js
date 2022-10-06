// Copyright (c) 2019, Compiler Explorer Authors
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

import {ClientState} from '../clientstate';
import {ClientStateGoldenifier, ClientStateNormalizer} from '../clientstate-normalizer';
import {logger} from '../logger';
import * as utils from '../utils';

import {ApiHandler} from './api';

export class RouteAPI {
    constructor(router, config) {
        this.router = router;
        this.renderGoldenLayout = config.renderGoldenLayout;

        this.storageHandler = config.storageHandler;

        // if for testing purposes
        if (config.compileHandler) {
            this.apiHandler = new ApiHandler(
                config.compileHandler,
                config.ceProps,
                config.storageHandler,
                config.clientOptionsHandler.options.urlShortenService,
            );

            this.apiHandler.setReleaseInfo(config.defArgs.gitReleaseName, config.defArgs.releaseBuildNumber);
        }
    }

    InitializeRoutes() {
        this.router
            .use('/api', this.apiHandler.handle)
            .get('/z/:id', this.storedStateHandler.bind(this))
            .get('/z/:id/code/:session', this.storedCodeHandler.bind(this))
            .get('/resetlayout/:id', this.storedStateHandlerResetLayout.bind(this))
            .get('/clientstate/:clientstatebase64', this.unstoredStateHandler.bind(this))
            .get('/fromsimplelayout', this.simpleLayoutHandler.bind(this));
    }

    storedCodeHandler(req, res, next) {
        const id = req.params.id;
        const sessionid = parseInt(req.params.session);
        this.storageHandler
            .expandId(id)
            .then(result => {
                const config = JSON.parse(result.config);

                let clientstate = false;
                if (config.content) {
                    const normalizer = new ClientStateNormalizer();
                    normalizer.fromGoldenLayout(config);

                    clientstate = normalizer.normalized;
                } else {
                    clientstate = new ClientState(config);
                }

                const session = clientstate.findSessionById(sessionid);
                if (!session) throw {msg: `Session ${sessionid} doesn't exist in this shortlink`};

                res.set('Content-Type', 'text/plain');
                res.send(session.source);
            })
            .catch(err => {
                logger.debug(`Exception thrown when expanding ${id}: `, err);
                next({
                    statusCode: 404,
                    message: `ID "${id}/${sessionid}" could not be found`,
                });
            });
    }

    storedStateHandler(req, res, next) {
        const id = req.params.id;
        this.storageHandler
            .expandId(id)
            .then(result => {
                let config = JSON.parse(result.config);
                if (config.sessions) {
                    config = this.getGoldenLayoutFromClientState(new ClientState(config));
                }
                const metadata = this.getMetaDataFromLink(req, result, config);
                this.renderGoldenLayout(config, metadata, req, res);
                // And finally, increment the view count
                // If any errors pop up, they are just logged, but the response should still be valid
                // It's really  unlikely that it happens as a result of the id not being there though,
                // but can be triggered with a missing implementation for a derived storage (s3/local...)
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

    getGoldenLayoutFromClientState(state) {
        const goldenifier = new ClientStateGoldenifier();
        goldenifier.fromClientState(state);
        return goldenifier.golden;
    }

    unstoredStateHandler(req, res) {
        const state = JSON.parse(Buffer.from(req.params.clientstatebase64, 'base64').toString());
        const config = this.getGoldenLayoutFromClientState(new ClientState(state));
        const metadata = this.getMetaDataFromLink(req, null, config);

        this.renderGoldenLayout(config, metadata, req, res);
    }

    simpleLayoutHandler(req, res) {
        const state = new ClientState();
        const session = state.findOrCreateSession(1);
        session.lang = req.query.lang;
        session.source = req.query.code;
        const compiler = session.findOrCreateCompiler(1);
        compiler.id = req.query.compiler;
        compiler.options = req.query.compiler_flags ? req.query.compiler_flags : '';

        this.renderClientState(state, undefined, req, res);
    }

    renderClientState(clientstate, metadata, req, res) {
        const config = this.getGoldenLayoutFromClientState(clientstate);

        this.renderGoldenLayout(config, metadata, req, res);
    }

    storedStateHandlerResetLayout(req, res, next) {
        const id = req.params.id;
        this.storageHandler
            .expandId(id)
            .then(result => {
                let config = JSON.parse(result.config);

                if (config.content) {
                    const normalizer = new ClientStateNormalizer();
                    normalizer.fromGoldenLayout(config);
                    config = normalizer.normalized;
                } else {
                    config = new ClientState(config);
                }

                const metadata = this.getMetaDataFromLink(req, result, config);
                this.renderClientState(config, metadata, req, res);
            })
            .catch(err => {
                logger.warn(`Exception thrown when expanding ${id}`);
                logger.debug('Exception value:', err);
                next({
                    statusCode: 404,
                    message: `ID "${id}" could not be found`,
                });
            });
    }

    escapeLine(req, line) {
        return line.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }

    filterCode(req, code, lang) {
        let lines = code.split('\n');
        if (lang.previewFilter !== null) {
            lines = lines.filter(line => !lang.previewFilter.test(line));
        }
        return lines.map(line => this.escapeLine(req, line)).join('\n');
    }

    getMetaDataFromLink(req, link, config) {
        const metadata = {
            ogDescription: null,
            ogAuthor: null,
            ogTitle: 'Compiler Explorer',
        };

        if (link) {
            metadata.ogDescription = link.specialMetadata ? link.specialMetadata.description.S : null;
            metadata.ogAuthor = link.specialMetadata ? link.specialMetadata.author.S : null;
            metadata.ogTitle = link.specialMetadata ? link.specialMetadata.title.S : 'Compiler Explorer';
        }

        if (!metadata.ogDescription) {
            let lang;
            let source = '';

            const sources = utils.glGetMainContents(config.content);
            if (sources.editors.length === 1) {
                const editor = sources.editors[0];
                lang = this.apiHandler.languages[editor.language];
                source = editor.source;
            } else {
                const normalizer = new ClientStateNormalizer();
                normalizer.fromGoldenLayout(config);
                const clientstate = normalizer.normalized;

                if (clientstate.trees && clientstate.trees.length === 1) {
                    const tree = clientstate.trees[0];
                    lang = this.apiHandler.languages[tree.compilerLanguageId];

                    if (!tree.isCMakeProject) {
                        const firstSource = tree.files.find(file => {
                            return file.filename.startsWith('example.');
                        });

                        if (firstSource) {
                            source = firstSource.content;
                        }
                    } else {
                        const firstSource = tree.files.find(file => {
                            return file.filename.startsWith('CMakeLists.txt');
                        });

                        if (firstSource) {
                            source = firstSource.content;
                        }
                    }
                }
            }

            if (lang) {
                metadata.ogDescription = this.filterCode(req, source, lang);
                metadata.ogTitle += ` - ${lang.name}`;
                if (sources && sources.compilers.length === 1) {
                    const compilerId = sources.compilers[0].compiler;
                    const compiler = this.apiHandler.compilers.find(c => c.id === compilerId);
                    if (compiler) {
                        metadata.ogTitle += ` (${compiler.name})`;
                    }
                }
            } else {
                metadata.ogDescription = source;
            }
        } else if (metadata.ogAuthor && metadata.ogAuthor !== '.') {
            metadata.ogDescription += `\nAuthor(s): ${metadata.ogAuthor}`;
        }

        return metadata;
    }
}
