// Copyright (c) 2017, Compiler Explorer Authors
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

import _ from 'underscore';

export class SourceHandler {
    constructor(fileSources, addStaticHeaders) {
        this.allowedActions = new Set(['list', 'load', 'save']);
        this.sourceToHandler = _.indexBy(fileSources, 'urlpart');
        this.addStaticHeaders = addStaticHeaders;
    }

    handlerForAction(handler, action) {
        return this.allowedActions.has(action) ? handler[action] : null;
    }

    handle(req, res, next) {
        const bits = req.url.split('/');
        const handler = this.sourceToHandler[bits[1]];
        if (!handler) {
            next();
            return;
        }
        const handlerAction = this.handlerForAction(handler, bits[2]);
        if (!handlerAction) {
            next();
            return;
        }
        handlerAction
            .apply(handlerAction, bits.slice(3))
            .then(response => {
                this.addStaticHeaders(res);
                res.send(response);
            })
            .catch(err => {
                res.send({err: err});
            });
    }
}
