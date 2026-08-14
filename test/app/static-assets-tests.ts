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

import path from 'node:path';

import {afterAll, beforeAll, describe, expect, it, vi} from 'vitest';

import {
    createDefaultPugRequireHandler,
    getBrandingPublicDir,
    getFaviconFilename,
    getLogoOverlayFilename,
    validateBrandingAssetsInManifest,
    validateBrandingAssetsOnDisk,
} from '../../lib/app/static-assets.js';
import {resolvePathFromAppRoot} from '../../lib/utils.js';

// Mock the logger
vi.mock('../../lib/logger.js', () => ({
    logger: {
        error: vi.fn(),
    },
}));

import {logger} from '../../lib/logger.js';

describe('Static assets', () => {
    describe('createDefaultPugRequireHandler', () => {
        it('should handle paths with manifest', () => {
            const manifest = {
                'file1.js': 'file1.hash123.js',
            };
            const handler = createDefaultPugRequireHandler('/static', manifest);

            expect(handler('file1.js')).toBe('/static/file1.hash123.js');
            expect(handler('file2.js')).toBe(''); // Not in manifest
            expect(logger.error).toHaveBeenCalledWith("Failed to locate static asset 'file2.js' in manifest");
        });

        it('should handle paths without manifest', () => {
            const handler = createDefaultPugRequireHandler('/static');

            expect(handler('file1.js')).toBe('/static/file1.js');
        });

        it('should handle staticRoot with trailing slash correctly', () => {
            const manifest = {
                'vendor.js': 'vendor.v57.0aa06caf727cfaf434aa.js',
                'app.css': 'app.v42.123456789abcdef.css',
            };

            // Test with trailing slash - this is the production scenario
            const handler = createDefaultPugRequireHandler('https://static.ce-cdn.net/', manifest);

            expect(handler('vendor.js')).toBe('https://static.ce-cdn.net/vendor.v57.0aa06caf727cfaf434aa.js');
            expect(handler('app.css')).toBe('https://static.ce-cdn.net/app.v42.123456789abcdef.css');

            // Test path not in manifest
            expect(handler('unknown.js')).toBe('');
        });

        it('should handle staticRoot with double slash correctly', () => {
            const manifest = {
                'vendor.js': 'vendor.v57.0aa06caf727cfaf434aa.js',
            };

            // Test with double slash - this is what happens with config bug
            const handler = createDefaultPugRequireHandler('https://static.ce-cdn.net//', manifest);

            // Should normalize to single slash
            expect(handler('vendor.js')).toBe('https://static.ce-cdn.net/vendor.v57.0aa06caf727cfaf434aa.js');
        });

        it('should handle staticRoot without trailing slash correctly', () => {
            const manifest = {
                'vendor.js': 'vendor.v57.0aa06caf727cfaf434aa.js',
            };

            // Test without trailing slash
            const handler = createDefaultPugRequireHandler('https://static.ce-cdn.net', manifest);

            expect(handler('vendor.js')).toBe('https://static.ce-cdn.net/vendor.v57.0aa06caf727cfaf434aa.js');
        });

        it('should handle local paths with various trailing slash scenarios', () => {
            const manifest = {
                'main.js': 'main.hash123.js',
            };

            // Test various local path formats
            const scenarios = ['/static', '/static/', '/static//'];

            for (const staticRoot of scenarios) {
                const handler = createDefaultPugRequireHandler(staticRoot, manifest);
                const result = handler('main.js');

                // All should resolve to the same normalized path
                expect(result).toBe('/static/main.hash123.js');
            }
        });
    });

    describe('getFaviconFilename', () => {
        it('uses the default favicon when no body class is set', () => {
            expect(getFaviconFilename('')).toBe('favicon.ico');
        });

        it('derives the favicon name from the body class', () => {
            expect(getFaviconFilename('dev')).toBe('favicon-dev.ico');
            expect(getFaviconFilename('beta')).toBe('favicon-beta.ico');
            expect(getFaviconFilename('staging')).toBe('favicon-staging.ico');
            expect(getFaviconFilename('anything-else')).toBe('favicon-anything-else.ico');
        });
    });

    describe('getLogoOverlayFilename', () => {
        it('returns undefined when no body class is set', () => {
            expect(getLogoOverlayFilename('')).toBeUndefined();
        });

        it('derives the overlay filename from the body class', () => {
            expect(getLogoOverlayFilename('dev')).toBe('site-logo-dev.svg');
            expect(getLogoOverlayFilename('beta')).toBe('site-logo-beta.svg');
            expect(getLogoOverlayFilename('intern')).toBe('site-logo-intern.svg');
        });
    });

    describe('validateBrandingAssetsOnDisk', () => {
        let tmpDir: string;

        beforeAll(async () => {
            const fs = await import('node:fs/promises');
            const os = await import('node:os');
            tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'ce-branding-test-'));
            await fs.writeFile(path.join(tmpDir, 'favicon-good.ico'), '');
            await fs.writeFile(path.join(tmpDir, 'site-logo-good.svg'), '');
            await fs.writeFile(path.join(tmpDir, 'favicon-partial.ico'), '');
        });

        afterAll(async () => {
            const fs = await import('node:fs/promises');
            await fs.rm(tmpDir, {recursive: true, force: true});
        });

        it('is a no-op when no body class is set', async () => {
            await expect(validateBrandingAssetsOnDisk('/nonexistent/path', '')).resolves.toBeUndefined();
        });

        it('passes when both assets exist', async () => {
            await expect(validateBrandingAssetsOnDisk(tmpDir, 'good')).resolves.toBeUndefined();
        });

        it('throws listing every missing asset', async () => {
            await expect(validateBrandingAssetsOnDisk(tmpDir, 'missing')).rejects.toThrow(/favicon-missing\.ico/);
            await expect(validateBrandingAssetsOnDisk(tmpDir, 'missing')).rejects.toThrow(/site-logo-missing\.svg/);
        });

        it('throws when only one asset is missing', async () => {
            await expect(validateBrandingAssetsOnDisk(tmpDir, 'partial')).rejects.toThrow(/site-logo-partial\.svg/);
        });
    });

    describe('validateBrandingAssetsInManifest', () => {
        // Regression for the gh-8755 revert: production nodes don't have the static files on
        // disk (they ship in the CDN bundle), so validation must go via the manifest, which
        // lists everything webpack copied from public/ into that bundle.
        const manifest = {
            'favicon-staging.ico': 'favicon-staging.ico',
            'site-logo-staging.svg': 'site-logo-staging.svg',
            'favicon-partial.ico': 'favicon-partial.ico',
        };

        it('is a no-op when no body class is set', () => {
            expect(() => validateBrandingAssetsInManifest({}, '')).not.toThrow();
        });

        it('passes when both assets are in the manifest', () => {
            expect(() => validateBrandingAssetsInManifest(manifest, 'staging')).not.toThrow();
        });

        it('throws listing every missing asset', () => {
            expect(() => validateBrandingAssetsInManifest(manifest, 'missing')).toThrow(/favicon-missing\.ico/);
            expect(() => validateBrandingAssetsInManifest(manifest, 'missing')).toThrow(/site-logo-missing\.svg/);
        });

        it('throws when only one asset is missing', () => {
            expect(() => validateBrandingAssetsInManifest(manifest, 'partial')).toThrow(/site-logo-partial\.svg/);
        });
    });

    describe('getBrandingPublicDir', () => {
        it('points at the public source dir (dev serves branding from there, not staticPath)', () => {
            // Regression: in dev the webpack middleware serves branding from public/, so validating
            // staticPath (the source dir, which has no branding assets) spuriously crashes startup.
            expect(getBrandingPublicDir()).toBe(resolvePathFromAppRoot('public'));
        });
    });

    describe('branding assets ship for every deployed environment', () => {
        // Guards the dev path end-to-end: the real files must exist in public/, so a missing
        // asset fails the build, not just a deploy. Adding an environment? Add its class here.
        const publicDir = getBrandingPublicDir();

        it.each([
            'dev',
            'beta',
            'staging',
            'winprod',
            'winstaging',
            'wintest',
        ])('has favicon + logo overlay for %s', async extraBodyClass => {
            await expect(validateBrandingAssetsOnDisk(publicDir, extraBodyClass)).resolves.toBeUndefined();
        });
    });
});
