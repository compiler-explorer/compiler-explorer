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

import * as Sentry from '@sentry/node';
import bodyParser from 'body-parser';
import express from 'express';
import _ from 'underscore';

import {CompilerInfo} from '../../types/compiler.interfaces';
import {Language, LanguageKey} from '../../types/languages.interfaces';
import {assert, unwrap} from '../assert';
import {ClientStateNormalizer} from '../clientstate-normalizer';
import {isString, unique} from '../common-utils';
import {logger} from '../logger';
import {ClientOptionsHandler} from '../options-handler';
import {PropertyGetter} from '../properties.interfaces';
import {BaseShortener, getShortenerTypeByKey} from '../shortener';
import {StorageBase} from '../storage';
import * as utils from '../utils';

import {withAssemblyDocumentationProviders} from './assembly-documentation';
import {CompileHandler} from './compile';
import {FormattingHandler} from './formatting';
import {getSiteTemplates} from './site-templates';

function methodNotAllowed(req: express.Request, res: express.Response) {
    res.send('Method Not Allowed');
    return res.status(405).end();
}

export class ApiHandler {
    public compilers: CompilerInfo[] = [];
    public languages: Partial<Record<LanguageKey, Language>> = {};
    private usedLangIds: LanguageKey[] = [];
    private options: ClientOptionsHandler | null = null;
    public readonly handle: express.Router;
    public readonly shortener: BaseShortener;
    private release = {
        gitReleaseName: '',
        releaseBuildNumber: '',
    };

    constructor(
        compileHandler: CompileHandler,
        ceProps: PropertyGetter,
        private readonly storageHandler: StorageBase,
        urlShortenService: string
    ) {
        this.handle = express.Router();
        const cacheHeader = `public, max-age=${ceProps('apiMaxAgeSecs', 24 * 60 * 60)}`;
        this.handle.use((req, res, next) => {
            res.header({
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Origin, X-Requested-With, Content-Type, Accept',
                'Cache-Control': cacheHeader,
            });
            next();
        });
        this.handle.route('/compilers').get(this.handleCompilers.bind(this)).all(methodNotAllowed);

        this.handle.route('/compilers/:language').get(this.handleCompilers.bind(this)).all(methodNotAllowed);

        this.handle.route('/languages').get(this.handleLanguages.bind(this)).all(methodNotAllowed);

        this.handle.route('/libraries/:language').get(this.handleLangLibraries.bind(this)).all(methodNotAllowed);

        this.handle.route('/libraries').get(this.handleAllLibraries.bind(this)).all(methodNotAllowed);

        // Binding for assembly documentation
        withAssemblyDocumentationProviders(this.handle);
        // Legacy binding for old clients.
        this.handle
            .route('/asm/:opcode')
            .get((req, res) => res.redirect(`amd64/${req.params.opcode}`))
            .all(methodNotAllowed);

        const maxUploadSize = ceProps('maxUploadSize', '1mb');
        const textParser = bodyParser.text({limit: ceProps('bodyParserLimit', maxUploadSize), type: () => true});

        this.handle
            .route('/compiler/:compiler/compile')
            .post(textParser, compileHandler.handle.bind(compileHandler))
            .all(methodNotAllowed);
        this.handle
            .route('/compiler/:compiler/cmake')
            .post(compileHandler.handleCmake.bind(compileHandler))
            .all(methodNotAllowed);

        this.handle
            .route('/popularArguments/:compiler')
            .post(compileHandler.handlePopularArguments.bind(compileHandler))
            .get(compileHandler.handlePopularArguments.bind(compileHandler))
            .all(methodNotAllowed);
        this.handle
            .route('/optimizationArguments/:compiler')
            .post(compileHandler.handleOptimizationArguments.bind(compileHandler))
            .get(compileHandler.handleOptimizationArguments.bind(compileHandler))
            .all(methodNotAllowed);

        const formatHandler = new FormattingHandler(ceProps);
        this.handle
            .route('/format/:tool')
            .post((req, res) => formatHandler.handle(req, res))
            .all(methodNotAllowed);
        this.handle
            .route('/formats')
            .get((req, res) => {
                const all = formatHandler.getFormatterInfo();
                res.send(all);
            })
            .all(methodNotAllowed);
        this.handle
            .route('/siteTemplates')
            .get((req, res) => {
                res.send(getSiteTemplates());
            })
            .all(methodNotAllowed);

        this.handle.route('/shortlinkinfo/:id').get(this.shortlinkInfoHandler.bind(this)).all(methodNotAllowed);

        const shortenerType = getShortenerTypeByKey(urlShortenService);
        this.shortener = new shortenerType(storageHandler);
        this.handle.route('/shortener').post(this.shortener.handle.bind(this.shortener)).all(methodNotAllowed);

        this.handle.route('/version').get(this.handleReleaseName.bind(this)).all(methodNotAllowed);
        this.handle.route('/releaseBuild').get(this.handleReleaseBuild.bind(this)).all(methodNotAllowed);
    }

    shortlinkInfoHandler(req: express.Request, res: express.Response, next: express.NextFunction) {
        const id = req.params.id;
        this.storageHandler
            .expandId(id)
            .then(result => {
                const config = JSON.parse(result.config);

                if (config.content) {
                    const normalizer = new ClientStateNormalizer();
                    normalizer.fromGoldenLayout(config);

                    res.send(normalizer.normalized);
                } else {
                    res.send(config);
                }
            })
            .catch(err => {
                logger.warn(`Exception thrown when expanding ${id}: `, err);
                logger.warn('Exception value:', err);
                Sentry.captureException(err);
                next({
                    statusCode: 404,
                    message: `ID "${id}" could not be found`,
                });
            });
    }

    handleLanguages(req: express.Request, res: express.Response) {
        const availableLanguages = this.usedLangIds.map(val => {
            const lang = this.languages[val];
            const newLangObj: Language = Object.assign({}, lang);
            if (this.options) {
                newLangObj.defaultCompiler = this.options.options.defaultCompiler[unwrap(lang).id];
            }
            return newLangObj;
        });

        this.outputList(availableLanguages, 'Id', req, res);
    }

    filterCompilerProperties(list: CompilerInfo[] | Language[], selectedFields: string[]) {
        return list.map(compiler => {
            return _.pick(compiler, selectedFields);
        });
    }

    outputList(list: CompilerInfo[] | Language[], title: string, req: express.Request, res: express.Response) {
        if (req.accepts(['text', 'json']) === 'json') {
            if (req.query.fields === 'all') {
                res.send(list);
            } else {
                const defaultfields = [
                    'id',
                    'name',
                    'lang',
                    'compilerType',
                    'semver',
                    'extensions',
                    'monaco',
                    'instructionSet',
                ];
                if (req.query.fields) {
                    assert(isString(req.query.fields));
                    const filteredList = this.filterCompilerProperties(list, req.query.fields.split(','));
                    res.send(filteredList);
                } else {
                    const filteredList = this.filterCompilerProperties(list, defaultfields);
                    res.send(filteredList);
                }
            }
            return;
        }

        const maxLength = Math.max(
            ...list
                .map(item => item.id)
                .concat([title])
                .map(item => item.length)
        );
        res.set('Content-Type', 'text/plain');
        res.send(
            utils.padRight(title, maxLength) +
                ' | Name\n' +
                list.map(lang => utils.padRight(lang.id, maxLength) + ' | ' + lang.name).join('\n')
        );
    }

    getLibrariesAsArray(languageId: LanguageKey) {
        const libsForLanguageObj = unwrap(this.options).options.libs[languageId];
        if (!libsForLanguageObj) return [];

        return Object.keys(libsForLanguageObj).map(key => {
            const language = libsForLanguageObj[key];
            const versionArr = Object.keys(language.versions).map(key => {
                const versionObj = Object.assign({}, language.versions[key]);
                versionObj.id = key;
                return versionObj;
            });

            return {
                id: key,
                name: language.name,
                description: language.description,
                url: language.url,
                versions: versionArr,
            };
        });
    }

    handleLangLibraries(req: express.Request, res: express.Response, next: express.NextFunction) {
        if (this.options) {
            if (req.params.language) {
                res.send(this.getLibrariesAsArray(req.params.language as LanguageKey));
            } else {
                next({
                    statusCode: 404,
                    message: 'Language is required',
                });
            }
        } else {
            next({
                statusCode: 500,
                message: 'Internal error',
            });
        }
    }

    handleAllLibraries(req: express.Request, res: express.Response, next: express.NextFunction) {
        if (this.options) {
            res.send(this.options.options.libs);
        } else {
            next({
                statusCode: 500,
                message: 'Internal error',
            });
        }
    }

    handleCompilers(req: express.Request, res: express.Response) {
        let filteredCompilers = this.compilers;
        if (req.params.language) {
            filteredCompilers = this.compilers.filter(compiler => compiler.lang === req.params.language);
        }

        this.outputList(filteredCompilers, 'Compiler Name', req, res);
    }

    handleReleaseName(req: express.Request, res: express.Response) {
        res.send(this.release.gitReleaseName);
    }

    handleReleaseBuild(req: express.Request, res: express.Response) {
        res.send(this.release.releaseBuildNumber);
    }

    setCompilers(compilers: CompilerInfo[]) {
        this.compilers = compilers;
        this.usedLangIds = unique(this.compilers.map(compiler => compiler.lang));
    }

    setLanguages(languages: Partial<Record<LanguageKey, Language>>) {
        this.languages = languages;
    }

    setOptions(options: ClientOptionsHandler) {
        this.options = options;
    }

    setReleaseInfo(gitReleaseName: string | undefined, releaseBuildNumber: string | undefined) {
        this.release = {
            gitReleaseName: gitReleaseName || '',
            releaseBuildNumber: releaseBuildNumber || '',
        };
    }
}
