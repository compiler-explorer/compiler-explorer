// Copyright (c) 2012-2017, Matt Godbolt
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
        this.compileHandler = compileHandler;
        this.handle = express.Router();
        this.handle.use((req, res, next) => {
            res.header("Access-Control-Allow-Origin", "*");
            res.header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept");
            next();
        });
        this.handle.get('/compilers', (req, res) => {
            if (req.accepts(['text', 'json']) === 'json') {
                res.set('Content-Type', 'application/json');
                res.end(JSON.stringify(this.compilers));
            } else {
                res.set('Content-Type', 'text/plain');
                const title = 'Compiler Name';
                const maxLength = _.max(_.pluck(_.pluck(this.compilers, 'id').concat([title]), 'length'));
                res.write(utils.padRight(title, maxLength) + ' | Description\n');
                res.end(
                    this.compilers.map(compiler => utils.padRight(compiler.id, maxLength) + ' | ' + compiler.name)
                        .join("\n"));
            }
        });
        this.handle.get('/asm/:opcode', AsmDocsApi.handler);
        this.handle.param('compiler', (req, res, next, compilerName) => {
            req.compiler = compilerName;
            next();
        });
        this.handle.post('/compiler/:compiler/compile', this.compileHandler.handler);
    }

    setCompilers(compilers) {
        this.compilers = compilers;
    }
}

module.exports.ApiHandler = ApiHandler;
