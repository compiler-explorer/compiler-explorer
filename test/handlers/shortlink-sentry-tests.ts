// Copyright (c) 2026, Compiler Explorer Authors
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
import request from 'supertest';
import {beforeEach, describe, expect, it, vi} from 'vitest';

import {CompilationEnvironment} from '../../lib/compilation-env.js';
import {ApiHandler} from '../../lib/handlers/api.js';
import {CompileHandler} from '../../lib/handlers/compile.js';
import {HandlerConfig} from '../../lib/handlers/handler.interfaces.js';
import {RouteAPI} from '../../lib/handlers/route-api.js';
import {fakeProps} from '../../lib/properties.js';
import {SentryCapture} from '../../lib/sentry.js';
import {ShortLinkNotFoundError, StorageBase} from '../../lib/storage/base.js';

vi.mock('../../lib/sentry.js', () => ({SentryCapture: vi.fn()}));

const noop = () => {};
const compileHandler = {
    handle: noop,
    handleCmake: noop,
    handleBuildProject: noop,
    handlePopularArguments: noop,
    handleOptimizationArguments: noop,
} as unknown as CompileHandler;

function storageFailingWith(error: Error): StorageBase {
    return {expandId: vi.fn().mockRejectedValue(error)} as unknown as StorageBase;
}

function makeApp(storage: StorageBase): express.Express {
    const app = express();
    new RouteAPI(app, {storageHandler: storage} as unknown as HandlerConfig).initializeRoutes();
    const apiHandler = new ApiHandler(compileHandler, fakeProps({}), storage, 'default', {
        ceProps: (_key: string, def: unknown) => def,
    } as unknown as CompilationEnvironment);
    app.use('/api', apiHandler.handle);
    return app;
}

describe('Unknown short link IDs', () => {
    beforeEach(() => {
        vi.mocked(SentryCapture).mockClear();
    });

    const routes = [
        ['/resetlayout/nosuchid', 'storedStateHandlerResetLayout'],
        ['/api/shortlinkinfo/nosuchid', 'shortlinkInfoHandler'],
    ];

    for (const [url, context] of routes) {
        it(`${url} returns 404 without reporting to Sentry`, async () => {
            const app = makeApp(storageFailingWith(new ShortLinkNotFoundError('nosuchid')));
            const response = await request(app).get(url);
            expect(response.status).toBe(404);
            expect(SentryCapture).not.toHaveBeenCalled();
        });

        it(`${url} still reports other storage failures to Sentry`, async () => {
            const failure = new Error('S3 exploded');
            const app = makeApp(storageFailingWith(failure));
            const response = await request(app).get(url);
            expect(response.status).toBe(404);
            expect(SentryCapture).toHaveBeenCalledWith(failure, context);
        });
    }
});
