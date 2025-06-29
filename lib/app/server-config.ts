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

import * as Sentry from '@sentry/node';
import compression from 'compression';
import express from 'express';
import type {NextFunction, Request, Response, Router} from 'express';
import morgan from 'morgan';
import sanitize from 'sanitize-filename';

import {cached, csp} from '../handlers/middleware.js';
import {logger, makeLogStream} from '../logger.js';
import {ClientOptionsSource} from '../options-handler.interfaces.js';
import {PropertyGetter} from '../properties.interfaces.js';
import * as utils from '../utils.js';
import {ServerOptions} from './server.interfaces.js';
import {RenderConfigFunction} from './server.interfaces.js';
import {LegacyGoogleUrlHandler, isMobileViewer} from './url-handlers.js';

/**
 * Setup base server configuration
 * @param options - Server options
 * @param renderConfig - Function to render configuration for templates
 * @param webServer - Express web server
 * @param router - Express router
 */
export function setupBaseServerConfig(
    options: ServerOptions,
    renderConfig: RenderConfigFunction,
    webServer: express.Express,
    router: Router,
): void {
    webServer
        .set('trust proxy', true)
        .set('view engine', 'pug')
        .on('error', err => logger.error('Caught error in web handler; continuing:', err))
        .use(
            responseTime((req, res, time) => {
                if (options.sentrySlowRequestMs > 0 && time >= options.sentrySlowRequestMs) {
                    Sentry.withScope((scope: Sentry.Scope) => {
                        scope.setExtra('duration_ms', time);
                        Sentry.captureMessage('SlowRequest', 'warning');
                    });
                }
            }),
        )
        .use(options.httpRoot, router)
        .use((req, res, next) => {
            next({status: 404, message: `page "${req.path}" could not be found`});
        });

    Sentry.setupExpressErrorHandler(webServer);

    // eslint-disable-next-line no-unused-vars
    webServer.use((err: any, req: Request, res: Response, _next: NextFunction) => {
        const status = err.status || err.statusCode || err.status_code || err.output?.statusCode || 500;
        const message = err.message || 'Internal Server Error';
        res.status(status);
        res.render('error', renderConfig({error: {code: status, message: message}}));
        if (status >= 500) {
            logger.error('Internal server error:', err);
        }
    });
}

/**
 * Creates a response time middleware
 * @param fn - Function to call with response time
 * @returns Express middleware
 */
function responseTime(fn: (req: Request, res: Response, time: number) => void): express.Handler {
    return (req: Request, res: Response, next: NextFunction) => {
        const start = process.hrtime();

        res.on('finish', () => {
            const diff = process.hrtime(start);
            const ms = diff[0] * 1000 + diff[1] / 1000000;
            fn(req, res, ms);
        });

        next();
    };
}

/**
 * Setup logging middleware
 * @param isDevMode - Whether the app is running in development mode
 * @param router - Express router
 */
export function setupLoggingMiddleware(isDevMode: boolean, router: Router): void {
    morgan.token('gdpr_ip', (req: any) => (req.ip ? utils.anonymizeIp(req.ip) : ''));

    // Based on combined format, but: GDPR compliant IP, no timestamp & no unused fields for our usecase
    const morganFormat = isDevMode ? 'dev' : ':gdpr_ip ":method :url" :status';

    router.use(
        morgan(morganFormat, {
            stream: makeLogStream('info'),
            // Skip for non errors (2xx, 3xx)
            skip: (req: Request, res: Response) => res.statusCode >= 400 || isFaviconRequest(req),
        }),
    );
    router.use(
        morgan(morganFormat, {
            stream: makeLogStream('warn'),
            // Skip for non user errors (4xx)
            skip: (req: Request, res: Response) =>
                res.statusCode < 400 || res.statusCode >= 500 || isFaviconRequest(req),
        }),
    );
    router.use(
        morgan(morganFormat, {
            stream: makeLogStream('error'),
            // Skip for non server errors (5xx)
            skip: (req: Request, res: Response) => res.statusCode < 500 || isFaviconRequest(req),
        }),
    );
}

function isFaviconRequest(req: Request): boolean {
    return [
        'favicon.ico',
        'favicon-beta.ico',
        'favicon-dev.ico',
        'favicon-staging.ico',
        'favicon-suspend.ico',
    ].includes(req.path);
}

/**
 * Setup basic routes for the web server
 * @param router - Express router
 * @param renderConfig - Function to render configuration for templates
 * @param embeddedHandler - Handler for embedded mode
 * @param ceProps - Compiler Explorer properties
 * @param awsProps - AWS properties
 * @param options - Server options
 * @param clientOptionsHandler - Client options handler
 */
export function setupBasicRoutes(
    router: Router,
    renderConfig: RenderConfigFunction,
    embeddedHandler: express.Handler,
    ceProps: PropertyGetter,
    awsProps: PropertyGetter,
    options: ServerOptions,
    clientOptionsHandler: ClientOptionsSource,
): void {
    const legacyGoogleUrlHandler = new LegacyGoogleUrlHandler(ceProps, awsProps);

    router
        .use(compression())
        .get('/', cached, csp, (req, res) => {
            res.render(
                'index',
                renderConfig(
                    {
                        embedded: false,
                        mobileViewer: isMobileViewer(req),
                    },
                    req.query,
                ),
            );
        })
        .get('/e', cached, csp, embeddedHandler)
        // legacy. not a 301 to prevent any redirect loops between old e links and embed.html
        .get('/embed.html', cached, csp, embeddedHandler)
        .get('/embed-ro', cached, csp, (req, res) => {
            res.render(
                'embed',
                renderConfig(
                    {
                        embedded: true,
                        readOnly: true,
                        mobileViewer: isMobileViewer(req),
                    },
                    req.query,
                ),
            );
        })
        .get('/robots.txt', cached, (req, res) => {
            res.end('User-agent: *\nSitemap: https://godbolt.org/sitemap.xml\nDisallow:');
        })
        .get('/sitemap.xml', cached, (req, res) => {
            res.set('Content-Type', 'application/xml');
            res.render('sitemap');
        });

    router
        .get('/client-options.js', cached, (req, res) => {
            res.set('Content-Type', 'application/javascript');
            res.end(`window.compilerExplorerOptions = ${clientOptionsHandler.getJSON()};`);
        })
        .use('/bits/:bits.html', cached, csp, (req, res) => {
            res.render(
                `bits/${sanitize(req.params.bits)}`,
                renderConfig(
                    {
                        embedded: false,
                        mobileViewer: isMobileViewer(req),
                    },
                    req.query,
                ),
            );
        })
        .use(express.json({limit: ceProps('bodyParserLimit', options.maxUploadSize)}))
        .get('/g/:id', legacyGoogleUrlHandler.handle.bind(legacyGoogleUrlHandler));
}
