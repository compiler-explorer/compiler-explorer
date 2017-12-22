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

const chai = require('chai'),
    ApiHandler = require('../../lib/handlers/api').Handler,
    express = require('express');

chai.use(require("chai-http"));
chai.should();

describe('API handling', () => {
    const app = express();
    const apiHandler = new ApiHandler({
        handle: (res, req, next) => {
            res.end("compile");
        }
    });
    app.use('/api', apiHandler.handle);
    apiHandler.setCompilers([{
        id: "gcc900",
        name: "GCC 9.0.0"
    }]);

    it('should respond to plain text compiler requests', () => {
        return chai.request(app)
            .get('/api/compilers')
            .then(res => {
                res.should.have.status(200);
                res.should.be.text;
                res.text.should.contain("Compiler Name");
                res.text.should.contain("gcc900");
                res.text.should.contain("GCC 9.0.0");
            })
            .catch(function (err) {
                throw err;
            });
    });
    it('should respond to JSON compiler requests', () => {
        return chai.request(app)
            .get('/api/compilers')
            .set('Accept', 'application/json')
            .then(res => {
                res.should.have.status(200);
                res.should.be.json;
                res.body.should.deep.equals([{
                    id: "gcc900",
                    name: "GCC 9.0.0"
                }]);
            })
            .catch(function (err) {
                throw err;
            });
    });
    it('should respond to ASM doc requests', () => {
        return chai.request(app)
            .get('/api/asm/MOVQ')
            .set('Accept', 'application/json')
            .then(res => {
                res.should.have.status(200);
                res.should.be.json;
                res.body.found.should.be.true;
            })
            .catch(function (err) {
                throw err;
            });
    });
    // TODO: more tests!
});
