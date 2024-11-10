// Copyright (c) 2023, Compiler Explorer Authors
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

import zlib from 'zlib';

import express from 'express';

import {AppDefaultArguments, CompilerExplorerOptions} from '../../app.js';
import {isString} from '../../shared/common-utils.js';
import {Language} from '../../types/languages.interfaces.js';
import {assert, unwrap} from '../assert.js';
import {ClientStateGoldenifier, ClientStateNormalizer} from '../clientstate-normalizer.js';
import {ClientState} from '../clientstate.js';
import {CompilationEnvironment} from '../compilation-env.js';
import {logger} from '../logger.js';
import {ClientOptionsHandler} from '../options-handler.js';
import {PropertyGetter} from '../properties.interfaces.js';
import {SentryCapture} from '../sentry.js';
import {ExpandedShortLink} from '../storage/base.js';
import {StorageBase} from '../storage/index.js';
import * as utils from '../utils.js';

import {ApiHandler} from './api.js';
import {CompileHandler} from './compile.js';
import {cached, csp} from './middleware.js';

export type HandlerConfig = {
    compileHandler: CompileHandler;
    clientOptionsHandler: ClientOptionsHandler;
    storageHandler: StorageBase;
    ceProps: PropertyGetter;
    opts: CompilerExplorerOptions;
    defArgs: AppDefaultArguments;
    renderConfig: any;
    renderGoldenLayout: any;
    compilationEnvironment: CompilationEnvironment;
};

export type ShortLinkMetaData = {
    ogDescription?: string;
    ogAuthor?: string;
    ogTitle?: string;
    ogCreated?: Date;
};

export class RouteAPI {
    renderGoldenLayout: any;
    storageHandler: StorageBase;
    apiHandler: ApiHandler;

    constructor(
        private readonly router: express.Router,
        config: HandlerConfig,
    ) {
        this.renderGoldenLayout = config.renderGoldenLayout;

        this.storageHandler = config.storageHandler;

        // if for testing purposes
        if (config.compileHandler) {
            this.apiHandler = new ApiHandler(
                config.compileHandler,
                config.ceProps,
                config.storageHandler,
                config.clientOptionsHandler.options.urlShortenService,
                config.compilationEnvironment,
            );

            this.apiHandler.setReleaseInfo(config.defArgs.gitReleaseName, config.defArgs.releaseBuildNumber);
        } else {
            this.apiHandler = undefined as any;
        }
    }

    InitializeRoutes() {
        this.router
            .use('/api', this.apiHandler.handle)
            .get('/z/:id', cached, csp, this.storedStateHandler.bind(this))
            .get('/z/:id/code/:session', cached, csp, this.storedCodeHandler.bind(this))
            .get('/resetlayout/:id', cached, csp, this.storedStateHandlerResetLayout.bind(this))
            .get('/clientstate/:clientstatebase64([^]*)', cached, csp, this.unstoredStateHandler.bind(this))
            .get('/fromsimplelayout', cached, csp, this.simpleLayoutHandler.bind(this));
    }

    storedCodeHandler(req: express.Request, res: express.Response, next: express.NextFunction) {
        const id = req.params.id;
        const sessionid = parseInt(req.params.session);
        this.storageHandler
            .expandId(id)
            .then(result => {
                const config = JSON.parse(result.config);

                let clientstate: ClientState | null = null;
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
                logger.warn('Exception value:', err);
                SentryCapture(err, 'storedCodeHandler');
                next({
                    statusCode: 404,
                    message: `ID "${id}/${sessionid}" could not be found`,
                });
            });
    }

    storedStateHandler(req: express.Request, res: express.Response, next: express.NextFunction) {
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

    getGoldenLayoutFromClientState(state: ClientState) {
        const goldenifier = new ClientStateGoldenifier();
        goldenifier.fromClientState(state);
        return goldenifier.golden;
    }

    unstoredStateHandler(req: express.Request, res: express.Response) {
        const state = extractJsonFromBufferAndInflateIfRequired(Buffer.from(req.params.clientstatebase64, 'base64'));
        const config = this.getGoldenLayoutFromClientState(new ClientState(state));
        const metadata = this.getMetaDataFromLink(req, null, config);

        this.renderGoldenLayout(config, metadata, req, res);
    }

    simpleLayoutHandler(req: express.Request, res: express.Response) {
        const state = new ClientState();
        const session = state.findOrCreateSession(1);
        assert(isString(req.query.lang));
        session.language = req.query.lang;
        assert(isString(req.query.code));
        session.source = req.query.code;
        const compiler = session.findOrCreateCompiler(1);
        compiler.id = req.query.compiler as string;
        compiler.options = (req.query.compiler_flags as string) || '';

        this.renderClientState(state, null, req, res);
    }

    renderClientState(
        clientstate: ClientState,
        metadata: ShortLinkMetaData | null,
        req: express.Request,
        res: express.Response,
    ) {
        const config = this.getGoldenLayoutFromClientState(clientstate);

        this.renderGoldenLayout(config, metadata, req, res);
    }

    storedStateHandlerResetLayout(req: express.Request, res: express.Response, next: express.NextFunction) {
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
                logger.warn('Exception value:', err);
                SentryCapture(err, 'storedStateHandlerResetLayout');
                next({
                    statusCode: 404,
                    message: `ID "${id}" could not be found`,
                });
            });
    }

    escapeLine(req: express.Request, line: string) {
        return line.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
    }

    filterCode(req: express.Request, code: string, lang: Language) {
        let lines = code.split('\n');
        if (lang.previewFilter !== null) {
            lines = lines.filter(line => !lang.previewFilter || !lang.previewFilter.test(line));
        }
        return lines.map(line => this.escapeLine(req, line)).join('\n');
    }

    getMetaDataFromLink(req: express.Request, link: ExpandedShortLink | null, config: any) {
        const metadata: ShortLinkMetaData = {
            ogTitle: 'Compiler Explorer',
        };

        if (link) {
            if (link.specialMetadata) {
                metadata.ogDescription = link.specialMetadata.description.S;
                metadata.ogAuthor = link.specialMetadata.author.S;
                metadata.ogTitle = link.specialMetadata.title.S;
            }

            if (link.created) metadata.ogCreated = link.created;
        }

        if (!metadata.ogDescription) {
            let lang: Language | undefined;
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

                    if (tree.isCMakeProject) {
                        const firstSource = tree.files.find(file => {
                            return unwrap(file.filename).startsWith('CMakeLists.txt');
                        });

                        if (firstSource) {
                            source = firstSource.content;
                        }
                    } else {
                        const firstSource = tree.files.find(file => {
                            return unwrap(file.filename).startsWith('example.');
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

export function extractJsonFromBufferAndInflateIfRequired(buffer: Buffer): any {
    const firstByte = buffer.at(0); // for uncompressed data this is probably '{'
    const isGzipUsed = firstByte !== undefined && (firstByte & 0x0f) === 0x8; // https://datatracker.ietf.org/doc/html/rfc1950, https://datatracker.ietf.org/doc/html/rfc1950, for '{' this yields 11
    if (isGzipUsed) {
        try {
            return JSON.parse(zlib.inflateSync(buffer).toString());
        } catch (_) {
            return JSON.parse(buffer.toString());
        }
    } else {
        return JSON.parse(buffer.toString());
    }
}
