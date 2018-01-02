// Copyright (c) 2012-2018, Matt Godbolt
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
    _ = require('underscore-node'),
    AsmDocsApi = require('./asm-docs-api'),
    utils = require('../utils');

class ApiHandler {
    constructor(compileHandler) {
        this.compilers = [];
        this.languages = [];
        this.handle = express.Router();
        this.handle.use((req, res, next) => {
            res.header("Access-Control-Allow-Origin", "*");
            res.header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept");
            next();
        });
        this.handle.get('/compilers', this.handleCompilers.bind(this));
        this.handle.get('/compilers/:language', this.handleCompilers.bind(this));
        this.handle.get('/languages', this.handleLanguages.bind(this));
        const asmDocsHandler = new AsmDocsApi.Handler();
        this.handle.get('/asm/:opcode', asmDocsHandler.handle.bind(asmDocsHandler));
        this.handle.post('/compiler/:compiler/compile', compileHandler.handle.bind(compileHandler));
    }

    handleLanguages(req, res) {
        const availableLanguageIds = _.uniq(_.pluck(this.compilers, 'lang'));
        const availableLanguages = availableLanguageIds.map(val => this.languages[val]);

        if (req.accepts(['text', 'json']) === 'json') {
            res.set('Content-Type', 'application/json');
            res.end(JSON.stringify(availableLanguages));
        } else {
            res.set('Content-Type', 'text/plain');
            const title = 'Id';
            const maxLength = _.max(_.pluck(_.pluck(availableLanguages, 'id').concat([title]), 'length'));
            res.write(utils.padRight(title, maxLength) + ' | Name\n');
            res.end(availableLanguages.map(lang => utils.padRight(lang.id, maxLength) + ' | ' + lang.name)
                .join("\n"));
        }
    }

    handleCompilers(req, res) {
        let filteredCompilers = this.compilers;
        if (req.params.language) {
            filteredCompilers = this.compilers.filter((val) => val.lang === req.params.language);
        }

        if (req.accepts(['text', 'json']) === 'json') {
            res.set('Content-Type', 'application/json');
            res.end(JSON.stringify(filteredCompilers));
        } else {
            res.set('Content-Type', 'text/plain');
            const title = 'Compiler Name';
            const maxLength = _.max(_.pluck(_.pluck(filteredCompilers, 'id').concat([title]), 'length'));
            res.write(utils.padRight(title, maxLength) + ' | Description\n');
            res.end(
                filteredCompilers.map(compiler => utils.padRight(compiler.id, maxLength) + ' | ' + compiler.name)
                    .join("\n"));
        }
    }

    setCompilers(compilers) {
        this.compilers = compilers;
    }

    setLanguages(languages) {
        this.languages = languages;
    }
}

module.exports.Handler = ApiHandler;
