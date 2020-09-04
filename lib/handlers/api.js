// Copyright (c) 2017, Matt Godbolt
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

const express = require('express'),
    _ = require('underscore'),
    AsmDocsApi = require('./asm-docs-api'),
    FormatterHandler = require('./formatting'),
    bodyParser = require('body-parser'),
    utils = require('../utils'),
    logger = require('../logger').logger,
    clientStateNormalizer = require('../clientstate-normalizer').ClientStateNormalizer;

class ApiHandler {
    constructor(compileHandler, ceProps, storageHandler) {
        this.compilers = [];
        this.languages = [];
        this.usedLangIds = [];
        this.options = null;
        this.storageHandler = storageHandler;
        this.handle = express.Router();
        const cacheHeader = `public, max-age=${ceProps('apiMaxAgeSecs', 24 * 60 * 60)}`;
        this.handle.use((req, res, next) => {
            res.header(
                {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'Origin, X-Requested-With, Content-Type, Accept',
                    'Cache-Control': cacheHeader,
                },
            );
            next();
        });
        this.handle.get('/compilers', this.handleCompilers.bind(this));
        this.handle.get('/compilers/:language', this.handleCompilers.bind(this));
        this.handle.get('/languages', this.handleLanguages.bind(this));
        this.handle.get('/libraries/:language', this.handleLangLibraries.bind(this));
        this.handle.get('/libraries', this.handleAllLibraries.bind(this));

        const asmDocsHandler = new AsmDocsApi.Handler();
        this.handle.get('/asm/:opcode', asmDocsHandler.handle.bind(asmDocsHandler));

        const maxUploadSize = ceProps('maxUploadSize', '1mb');
        const textParser = bodyParser.text({limit: ceProps('bodyParserLimit', maxUploadSize), type: () => true});

        this.handle.post('/compiler/:compiler/compile', textParser, compileHandler.handle.bind(compileHandler));
        this.handle.post('/popularArguments/:compiler', compileHandler.handlePopularArguments.bind(compileHandler));
        this.handle.post('/optimizationArguments/:compiler',
            compileHandler.handleOptimizationArguments.bind(compileHandler));

        this.handle.get('/popularArguments/:compiler', compileHandler.handlePopularArguments.bind(compileHandler));
        this.handle.get('/optimizationArguments/:compiler',
            compileHandler.handleOptimizationArguments.bind(compileHandler));

        const formatter = new FormatterHandler(ceProps);
        this.handle.post('/format/:tool', formatter.formatHandler.bind(formatter));
        this.handle.get('/formats', formatter.listHandler.bind(formatter));

        this.handle.get('/shortlinkinfo/:id', this.shortlinkInfoHandler.bind(this));
    }

    shortlinkInfoHandler(req, res, next) {
        const id = req.params.id;
        this.storageHandler.expandId(id)
            .then(result => {
                const config = JSON.parse(result.config);

                if (config.content) {
                    const normalizer = new clientStateNormalizer();
                    normalizer.fromGoldenLayout(config);

                    res.send(normalizer.normalized);
                } else {
                    res.send(config);
                }
            })
            .catch(err => {
                logger.warn(`Exception thrown when expanding ${id}: `, err);
                next({
                    statusCode: 404,
                    message: `ID "${id}" could not be found`,
                });
            });
    }

    handleLanguages(req, res) {
        const availableLanguages = this.usedLangIds.map(val => {
            let lang = this.languages[val];
            let newLangObj = Object.assign({}, lang);
            if (this.options) {
                newLangObj.defaultCompiler = this.options.options.defaultCompiler[lang.id];
            }
            return newLangObj;
        });

        this.outputList(availableLanguages, 'Id', req, res);
    }

    filterCompilerProperties(list, selectedFields) {
        return _.map(list, (compiler) => {
            return _.pick(compiler, selectedFields);
        });
    }

    outputList(list, title, req, res) {
        if (req.accepts(['text', 'json']) === 'json') {
            if (req.query.fields === 'all') {
                res.send(list);
            } else {
                const defaultfields = ['id', 'name', 'lang', 'compilerType', 'semver', 'extensions', 'monaco', 'exe'];
                if (req.query.fields) {
                    const filteredList = this.filterCompilerProperties(list, req.query.fields.split(','));
                    res.send(filteredList);
                } else {
                    const filteredList = this.filterCompilerProperties(list, defaultfields);
                    res.send(filteredList);
                }
            }
            return;
        }

        const maxLength = _.max(_.pluck(_.pluck(list, 'id').concat([title]), 'length'));
        res.set('Content-Type', 'text/plain');
        res.send(utils.padRight(title, maxLength) + ' | Name\n'
            + list.map(lang => utils.padRight(lang.id, maxLength) + ' | ' + lang.name).join('\n'));
    }

    getLibrariesAsArray(languageId) {
        const libsForLanguageObj = this.options.options.libs[languageId];
        if (!libsForLanguageObj) return [];

        return Object.keys(libsForLanguageObj).map((key) => {
            const language = libsForLanguageObj[key];
            const versionArr = Object.keys(language.versions).map((key) => {
                let versionObj = Object.assign({}, language.versions[key]);
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

    handleLangLibraries(req, res, next) {
        if (this.options) {
            if (req.params.language) {
                res.send(this.getLibrariesAsArray(req.params.language));
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

    handleAllLibraries(req, res, next) {
        if (this.options) {
            res.send(this.options.options.libs);
        } else {
            next({
                statusCode: 500,
                message: 'Internal error',
            });
        }
    }

    handleCompilers(req, res) {
        let filteredCompilers = this.compilers;
        if (req.params.language) {
            filteredCompilers = this.compilers.filter((val) => val.lang === req.params.language);
        }

        this.outputList(filteredCompilers, 'Compiler Name', req, res);
    }

    setCompilers(compilers) {
        this.compilers = compilers;
        this.usedLangIds = _.uniq(_.pluck(this.compilers, 'lang'));
    }

    setLanguages(languages) {
        this.languages = languages;
    }

    setOptions(options) {
        this.options = options;
    }
}

module.exports.Handler = ApiHandler;
