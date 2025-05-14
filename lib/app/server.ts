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
import process from 'node:process';
import url from 'node:url';

import * as Sentry from '@sentry/node';
import compression from 'compression';
import express from 'express';
import type {NextFunction, Request, Response} from 'express';
// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-ignore
import morgan from 'morgan';
import PromClient from 'prom-client';
import responseTime from 'response-time';
import sanitize from 'sanitize-filename';
import sFavicon from 'serve-favicon';
// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-ignore
import systemdSocket from 'systemd-socket';
import _ from 'underscore';
import urljoin from 'url-join';

import {ElementType} from '../../shared/common-utils.js';
import type {AppArguments} from '../app.interfaces.js';
import {GoldenLayoutRootStruct} from '../clientstate-normalizer.js';
import * as normalizer from '../clientstate-normalizer.js';
import type {ShortLinkMetaData} from '../handlers/handler.interfaces.js';
import {cached, csp} from '../handlers/middleware.js';
import {logger, makeLogStream} from '../logger.js';
import {PropertyGetter} from '../properties.interfaces.js';
import {ShortLinkResolver} from '../shortener/google.js';
import * as utils from '../utils.js';
import {isDevMode} from './config.js';
import type {
    PugRequireHandler,
    RenderConfig,
    RenderConfigFunction,
    ServerDependencies,
    ServerOptions,
    WebServerResult,
} from './server.interfaces.js';

function createDefaultPugRequireHandler(staticRoot: string, manifest?: Record<string, string>) {
    return (path: string) => {
        if (manifest && Object.prototype.hasOwnProperty.call(manifest, path)) {
            return `${staticRoot}/${manifest[path]}`;
        }
        if (manifest) {
            console.error(`Failed to locate static asset '${path}' in manifest`);
            return '';
        }
        return `${staticRoot}/${path}`;
    };
}

/**
 * Gets the appropriate favicon filename based on the environment.
 * @param isDevModeValue - Whether the app is running in development mode
 * @param env - The environment names array
 */
export function getFaviconFilename(isDevModeValue: boolean, env?: string[]): string {
    if (isDevModeValue) {
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

/**
 * Detects if the request is from a mobile viewer
 */
export function isMobileViewer(req: Request): boolean {
    return req.header('CloudFront-Is-Mobile-Viewer') === 'true';
}

/**
 * Handles legacy Google URL shortener redirects
 */
class OldGoogleUrlHandler {
    private readonly googleShortUrlResolver: ShortLinkResolver;
    constructor(private readonly ceProps: PropertyGetter) {
        this.googleShortUrlResolver = new ShortLinkResolver();
    }
    async handle(req: Request, res: Response, next: NextFunction) {
        const id = req.params.id;
        const googleUrl = `https://goo.gl/${encodeURIComponent(id)}`;

        try {
            const resultObj = await this.googleShortUrlResolver.resolve(googleUrl);
            const parsed = new url.URL(resultObj.longUrl);
            const allowedRe = new RegExp(this.ceProps<string>('allowedShortUrlHostRe'));

            if (parsed.host.match(allowedRe) === null) {
                logger.warn(`Denied access to short URL ${id} - linked to ${resultObj.longUrl}`);
                return next({
                    statusCode: 404,
                    message: `ID "${id}" could not be found`,
                });
            }

            res.writeHead(301, {
                Location: resultObj.longUrl,
                'Cache-Control': 'public',
            });
            res.end();
        } catch (err: unknown) {
            logger.error(`Failed to expand ${googleUrl} - ${err}`);
            next({
                statusCode: 404,
                message: `ID "${id}" could not be found`,
            });
        }
    }
}

async function setupWebPackDevMiddleware(options: ServerOptions, router: express.Router): Promise<PugRequireHandler> {
    logger.info('  using webpack dev middleware');

    /* eslint-disable n/no-unpublished-import,import/extensions, */
    const {default: webpackDevMiddleware} = await import('webpack-dev-middleware');
    const {default: webpackConfig} = await import('../../webpack.config.esm.js');
    const {default: webpack} = await import('webpack');
    /* eslint-enable */
    type WebpackConfiguration = ElementType<Parameters<typeof webpack>[0]>;

    const webpackCompiler = webpack([webpackConfig as WebpackConfiguration]);
    router.use(
        webpackDevMiddleware(webpackCompiler, {
            publicPath: '/static',
            stats: {
                preset: 'errors-only',
                timings: true,
            },
        }),
    );

    return path => urljoin(options.httpRoot, 'static', path);
}

async function setupStaticMiddleware(options: ServerOptions, router: express.Router): Promise<PugRequireHandler> {
    const staticManifest = JSON.parse(await fs.readFile(path.join(options.distPath, 'manifest.json'), 'utf-8'));

    if (options.staticUrl) {
        logger.info(`  using static files from '${options.staticUrl}'`);
    } else {
        logger.info(`  serving static files from '${options.staticPath}'`);
        router.use(
            '/static',
            express.static(options.staticPath, {
                maxAge: options.staticMaxAgeSecs * 1000,
            }),
        );
    }

    return createDefaultPugRequireHandler(options.staticRoot, staticManifest);
}

function setupBaseServerConfig(
    options: ServerOptions,
    renderConfig: RenderConfigFunction,
    webServer: express.Express,
    router: express.Router,
) {
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

function setupLoggingMiddleware(router: express.Router) {
    morgan.token('gdpr_ip', (req: any) => (req.ip ? utils.anonymizeIp(req.ip) : ''));

    // Based on combined format, but: GDPR compliant IP, no timestamp & no unused fields for our usecase
    const morganFormat = isDevMode() ? 'dev' : ':gdpr_ip ":method :url" :status';

    router.use(
        morgan(morganFormat, {
            stream: makeLogStream('info'),
            // Skip for non errors (2xx, 3xx)
            skip: (req: Request, res: Response) => res.statusCode >= 400,
        }),
    );
    router.use(
        morgan(morganFormat, {
            stream: makeLogStream('warn'),
            // Skip for non user errors (4xx)
            skip: (req: Request, res: Response) => res.statusCode < 400 || res.statusCode >= 500,
        }),
    );
    router.use(
        morgan(morganFormat, {
            stream: makeLogStream('error'),
            // Skip for non server errors (5xx)
            skip: (req: Request, res: Response) => res.statusCode < 500,
        }),
    );
}

function setupBasicRoutes(
    router: express.Router,
    renderConfig: RenderConfigFunction,
    embeddedHandler: express.Handler,
    ceProps: PropertyGetter,
    faviconFilename: string,
    options: ServerOptions,
    clientOptionsHandler: any,
) {
    const oldGoogleUrlHandler = new OldGoogleUrlHandler(ceProps);

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

    // Try to add favicon support, but don't fail if it's not available (useful for tests)
    try {
        router.use(sFavicon(utils.resolvePathFromAppRoot('static/favicons', faviconFilename)));
    } catch (err: unknown) {
        const error = err as Error;
        logger.warn(`Could not set up favicon: ${error.message}`);
    }

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
        .get('/g/:id', oldGoogleUrlHandler.handle.bind(oldGoogleUrlHandler));
}

/**
 * Configure a web server and its routes
 */
export async function setupWebServer(
    appArgs: AppArguments,
    options: ServerOptions,
    dependencies: ServerDependencies,
): Promise<WebServerResult> {
    const {ceProps, sponsorConfig, clientOptionsHandler} = dependencies;
    const {staticRoot, httpRoot, extraBodyClass} = options;

    // Create the Express application and router
    const webServer = express();
    const router = express.Router();

    // Initialize services
    // TODO: ideally remove this (cc @junlarsen for details)
    let pugRequireHandler: PugRequireHandler = () => {
        logger.error('pug require handler not configured');
        return '';
    };

    /**
     * Renders configuration for templates
     */
    function renderConfig(extra: Record<string, any>, urlOptions?: Record<string, any>): RenderConfig {
        const urlOptionsAllowed = ['readOnly', 'hideEditorToolbars', 'language'];
        const filteredUrlOptions = _.mapObject(_.pick(urlOptions || {}, urlOptionsAllowed), val =>
            utils.toProperty(val),
        );
        const allExtraOptions = _.extend({}, filteredUrlOptions, extra);

        if (allExtraOptions.mobileViewer && allExtraOptions.config) {
            const clnormalizer = new normalizer.ClientStateNormalizer();
            clnormalizer.fromGoldenLayout(allExtraOptions.config);
            const clientstate = clnormalizer.normalized;

            const glnormalizer = new normalizer.ClientStateGoldenifier();
            allExtraOptions.slides = glnormalizer.generatePresentationModeMobileViewerSlides(clientstate);
        }

        const options = _.extend({}, allExtraOptions, clientOptionsHandler.get());
        options.optionsHash = clientOptionsHandler.getHash();
        options.compilerExplorerOptions = JSON.stringify(allExtraOptions);
        options.extraBodyClass = options.embedded ? 'embedded' : extraBodyClass;
        options.httpRoot = httpRoot;
        options.staticRoot = staticRoot;
        options.storageSolution = options.storageSolution || dependencies.storageSolution;
        options.require = pugRequireHandler;
        options.sponsors = sponsorConfig;
        return options;
    }

    /**
     * Renders GoldenLayout for a given configuration
     */
    function renderGoldenLayout(
        config: GoldenLayoutRootStruct,
        metadata: ShortLinkMetaData,
        req: Request,
        res: Response,
    ) {
        const embedded = req.query.embedded === 'true';

        res.render(
            embedded ? 'embed' : 'index',
            renderConfig(
                {
                    embedded: embedded,
                    mobileViewer: isMobileViewer(req),
                    config: config,
                    metadata: metadata,
                    storedStateId: req.params.id || false,
                },
                req.query,
            ),
        );
    }

    /**
     * Handles rendering embedded pages
     */
    const embeddedHandler = (req: Request, res: Response) => {
        res.render(
            'embed',
            renderConfig(
                {
                    embedded: true,
                    mobileViewer: isMobileViewer(req),
                },
                req.query,
            ),
        );
    };

    // Set up the base Express server configuration
    setupBaseServerConfig(options, renderConfig, webServer, router);

    // Configure static file handling based on environment
    try {
        pugRequireHandler = await (isDevMode()
            ? setupWebPackDevMiddleware(options, router)
            : setupStaticMiddleware(options, router));
    } catch (err: unknown) {
        const error = err as Error;
        logger.warn(`Error setting up static middleware: ${error.message}`);
        // Import helper for testing without circular dependencies
        pugRequireHandler = createDefaultPugRequireHandler(staticRoot);
    }

    setupLoggingMiddleware(router);
    setupBasicRoutes(
        router,
        renderConfig,
        embeddedHandler,
        ceProps,
        getFaviconFilename(isDevMode(), appArgs.env),
        options,
        clientOptionsHandler,
    );

    // Return needed server elements
    return {
        webServer,
        router,
        pugRequireHandler,
        renderConfig,
        renderGoldenLayout,
    };
}

/**
 * Starts the web server listening for connections
 */
export function startListening(webServer: express.Express, appArgs: AppArguments): void {
    const ss: {fd: number} | null = systemdSocket(); // TODO: I'm not sure this works any more
    if (ss) {
        setupSystemdSocketListening(webServer, ss);
    } else {
        setupStandardHttpListening(webServer, appArgs);
    }

    setupStartupMetrics();
}

function setupSystemdSocketListening(webServer: express.Express, ss: {fd: number}): void {
    // ms (5 min default)
    const idleTimeout = process.env.IDLE_TIMEOUT;
    const timeout = (idleTimeout === undefined ? 300 : Number.parseInt(idleTimeout)) * 1000;
    if (idleTimeout) {
        setupIdleTimeout(webServer, timeout);
        logger.info(`  IDLE_TIMEOUT: ${idleTimeout}`);
    }
    logger.info(`  Listening on systemd socket: ${JSON.stringify(ss)}`);
    webServer.listen(ss);
}

function setupIdleTimeout(webServer: express.Express, timeout: number): void {
    const exit = () => {
        logger.info('Inactivity timeout reached, exiting.');
        process.exit(0);
    };
    let idleTimer = setTimeout(exit, timeout);
    const reset = () => {
        clearTimeout(idleTimer);
        idleTimer = setTimeout(exit, timeout);
    };
    webServer.all('*', reset);
}

function setupStandardHttpListening(webServer: express.Express, appArgs: AppArguments): void {
    logger.info(`  Listening on http://${appArgs.hostname || 'localhost'}:${appArgs.port}/`);
    if (appArgs.hostname) {
        webServer.listen(appArgs.port, appArgs.hostname);
    } else {
        webServer.listen(appArgs.port);
    }
}

function setupStartupMetrics(): void {
    try {
        const startupGauge = new PromClient.Gauge({
            name: 'ce_startup_seconds',
            help: 'Time taken from process start to serving requests',
        });
        startupGauge.set(process.uptime());
    } catch (err: unknown) {
        const error = err as Error;
        logger.warn(`Error setting up startup metric: ${error.message}`);
    }
    const startupDurationMs = Math.floor(process.uptime() * 1000);
    logger.info(`  Startup duration: ${startupDurationMs}ms`);
    logger.info('=======================================');
}
