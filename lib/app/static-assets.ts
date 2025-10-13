// Copyright (c) 2025, Compiler Explorer Authors
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

import fs from 'node:fs/promises';
import path from 'node:path';
import type {Router} from 'express';
import express from 'express';
import urljoin from 'url-join';

import {unwrap} from '../assert.js';
import {logger} from '../logger.js';
import {PugRequireHandler, ServerOptions} from './server.interfaces.js';

/**
 * Creates a default handler for Pug requires
 * @param staticRoot - The static assets root URL
 * @param manifest - Optional manifest mapping file paths to hashed versions
 * @returns Function to handle Pug requires
 */
export function createDefaultPugRequireHandler(
    staticRoot: string,
    manifest?: Record<string, string>,
): PugRequireHandler {
    return (path: string) => {
        if (manifest && Object.hasOwn(manifest, path)) {
            return urljoin(staticRoot, manifest[path]);
        }
        if (manifest) {
            logger.error(`Failed to locate static asset '${path}' in manifest`);
            return '';
        }
        return urljoin(staticRoot, path);
    };
}

/**
 * Sets up webpack dev middleware for development mode
 * @param options - Server options
 * @param router - Express router
 * @returns Function to handle Pug requires
 */
export async function setupWebPackDevMiddleware(options: ServerOptions, router: Router): Promise<PugRequireHandler> {
    logger.info('  using webpack dev middleware');

    /* eslint-disable n/no-unpublished-import,import/extensions, */
    const {default: webpackDevMiddleware} = await import('webpack-dev-middleware');
    const {default: webpackConfig} = await import('../../webpack.config.esm.js');
    const {default: webpack} = await import('webpack');
    /* eslint-enable */

    const webpackCompiler = unwrap(webpack(webpackConfig));
    router.use(
        webpackDevMiddleware(webpackCompiler, {
            publicPath: '/',
            stats: {
                preset: 'errors-only',
                timings: true,
            },
        }),
    );

    return path => urljoin(options.httpRoot, path);
}

/**
 * Sets up static file middleware for production mode
 * @param options - Server options
 * @param router - Express router
 * @returns Function to handle Pug requires
 */
export async function setupStaticMiddleware(options: ServerOptions, router: Router): Promise<PugRequireHandler> {
    const staticManifest = JSON.parse(await fs.readFile(path.join(options.manifestPath, 'manifest.json'), 'utf-8'));

    if (options.staticUrl) {
        logger.info(`  using static files from '${options.staticUrl}'`);
    } else {
        logger.info(`  serving static files from '${options.staticPath}'`);
        router.use(
            '/',
            express.static(options.staticPath, {
                maxAge: options.staticMaxAgeSecs * 1000,
            }),
        );
    }

    return createDefaultPugRequireHandler(options.staticRoot, staticManifest);
}

/**
 * Gets the appropriate favicon filename based on the environment
 * @param isDevMode - Whether the app is running in development mode
 * @param env - The environment names array
 * @returns The favicon filename to use
 */
export function getFaviconFilename(isDevMode: boolean, env?: string[]): string {
    if (isDevMode) {
        return 'favicon-dev.ico';
    }
    if (env?.includes('beta')) {
        return 'favicon-beta.ico';
    }
    if (env?.includes('staging')) {
        return 'favicon-staging.ico';
    }
    return 'favicon.ico';
}
