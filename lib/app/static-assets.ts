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
import {resolvePathFromAppRoot} from '../utils.js';
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

export async function loadStaticManifest(manifestPath: string): Promise<Record<string, string>> {
    return JSON.parse(await fs.readFile(path.join(manifestPath, 'manifest.json'), 'utf-8'));
}

/**
 * Sets up static file middleware for production mode
 * @param options - Server options
 * @param router - Express router
 * @param staticManifest - Webpack manifest mapping asset names to hashed filenames
 * @returns Function to handle Pug requires
 */
export function setupStaticMiddleware(
    options: ServerOptions,
    router: Router,
    staticManifest: Record<string, string>,
): PugRequireHandler {
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

// `siteLogo` and `favicon` are paths relative to public/.
export interface BrandingConfig {
    extraBodyClass: string;
    siteLogo?: string;
    favicon?: string;
}

export function getFaviconFilename(extraBodyClass: string, favicon?: string): string {
    if (extraBodyClass) return `favicon-${extraBodyClass}.ico`;
    return favicon || 'favicon.ico';
}

export function getLogoOverlayFilename(extraBodyClass: string): string | undefined {
    return extraBodyClass ? `site-logo-${extraBodyClass}.svg` : undefined;
}

/**
 * The directory branding assets are served from in dev mode: the webpack dev middleware
 * serves public/ directly (they aren't on disk in staticPath).
 */
export function getBrandingPublicDir(): string {
    return resolvePathFromAppRoot('public');
}

// The plain favicon.ico always ships, so it is only checked when something else was asked for.
function requiredBrandingAssets({extraBodyClass, siteLogo, favicon}: BrandingConfig): string[] {
    const favicons = extraBodyClass || favicon ? [getFaviconFilename(extraBodyClass, favicon)] : [];
    return [...favicons, getLogoOverlayFilename(extraBodyClass), siteLogo].filter((f): f is string => !!f);
}

function throwIfMissingBrandingAssets(where: string, missing: string[]): void {
    if (missing.length > 0) {
        throw new Error(`Missing branding assets in ${where}: ${missing.join(', ')}`);
    }
}

export async function validateBrandingAssetsOnDisk(assetDir: string, branding: BrandingConfig): Promise<void> {
    const missing: string[] = [];
    for (const filename of requiredBrandingAssets(branding)) {
        try {
            await fs.access(path.join(assetDir, filename));
        } catch (e: unknown) {
            const err = e as NodeJS.ErrnoException;
            if (err.code === 'ENOENT') missing.push(filename);
            else throw err;
        }
    }
    throwIfMissingBrandingAssets(assetDir, missing);
}

/**
 * Production deploys ship the static assets in a separate CDN bundle (see build-dist.sh), so
 * they are not on disk next to the node app and cannot be checked with the filesystem. The
 * webpack manifest *does* ship with the node app and lists every asset in the static bundle
 * (public/ is copied in wholesale), so presence of a manifest key proves the asset shipped.
 */
export function validateBrandingAssetsInManifest(manifest: Record<string, string>, branding: BrandingConfig): void {
    const missing = requiredBrandingAssets(branding).filter(filename => !Object.hasOwn(manifest, filename));
    throwIfMissingBrandingAssets('the static manifest', missing);
}
