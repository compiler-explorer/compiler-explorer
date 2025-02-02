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

import * as props from '../properties.js';

const ceProps = props.propsFor('compiler-explorer');

/**
 * Require the Content-Type header to be application/json
 *
 * TODO: Consider if this should return 422 instead of 400
 */
export const jsonOnly: express.Handler = (req, res, next) => {
    if (req.headers['content-type'] !== 'application/json') {
        return res.status(400).json({message: 'bad request, expected json content'});
    }
    return next();
};

/** Add static headers to the response */
export const cached: express.Handler = (_, res, next) => {
    // Cannot elide the ceProps() call here, because this file may be imported by other files such as app.ts before the
    // properties are loaded from config files.
    res.set('Cache-Control', `public, max-age=${ceProps('staticMaxAgeSecs', 31536000)}, must-revalidate`);
    return next();
};

/** Allow any client to make cross-origin requests */
export const cors: express.Handler = (_, res, next) => {
    res.set('Access-Control-Allow-Origin', '*');
    res.set('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept');
    return next();
};

/** Add Content-Security-Policy header to the response */
export const csp: express.Handler = (_, res, next) => {
    // TODO: Consider if CSP should be re-enabled
    // res.set('Content-Security-Policy', `default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self' data:; connect-src 'self'; frame-src 'self';`);
    return next();
};
