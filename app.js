#!/usr/bin/env node

// Copyright (c) 2012, Matt Godbolt
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

const startTime = new Date();

// Initialise options and properties. Don't load any handlers here; they
// may need an initialised properties library.
const nopt = require('nopt'),
    os = require('os'),
    props = require('./lib/properties'),
    child_process = require('child_process'),
    path = require('path'),
    process = require('process'),
    fs = require('fs-extra'),
    systemdSocket = require('systemd-socket'),
    url = require('url'),
    _ = require('underscore'),
    express = require('express'),
    Sentry = require('@sentry/node'),
    {logger, logToPapertrail, suppressConsoleLog} = require('./lib/logger'),
    webpackDevMiddleware = require("webpack-dev-middleware"),
    utils = require('./lib/utils'),
    initialiseWine = require('./lib/exec').initialiseWine,
    RouteAPI = require('./lib/handlers/route-api');


// Parse arguments from command line 'node ./app.js args...'
const opts = nopt({
    env: [String, Array],
    rootDir: [String],
    host: [String],
    port: [Number],
    propDebug: [Boolean],
    debug: [Boolean],
    static: [String],
    archivedVersions: [String],
    // Ignore fetch marks and assume every compiler is found locally
    noRemoteFetch: [Boolean],
    tmpDir: [String],
    wsl: [Boolean],
    // If specified, only loads the specified language, resulting in faster loadup/iteration times
    language: [String],
    // Do not use caching for compilation results (Requests might still be cached by the client's browser)
    noCache: [Boolean],
    // Don't cleanly run if two or more compilers have clashing ids
    ensureNoIdClash: [Boolean],
    logHost: [String],
    logPort: [Number],
    suppressConsoleLog: [Boolean]
});

if (opts.debug) logger.level = 'debug';

// AP: Detect if we're running under Windows Subsystem for Linux. Temporary modification
// of process.env is allowed: https://nodejs.org/api/process.html#process_process_env
if (process.platform === "linux" && child_process.execSync('uname -a').toString().indexOf('Microsoft') > -1) {
    process.env.wsl = true;
}

// AP: Allow setting of tmpDir (used in lib/base-compiler.js & lib/exec.js) through opts.
// WSL requires a directory on a Windows volume. Set that to Windows %TEMP% if no tmpDir supplied.
// If a tempDir is supplied then assume that it will work for WSL processes as well.
if (opts.tmpDir) {
    process.env.tmpDir = opts.tmpDir;
    process.env.winTmp = opts.tmpDir;
} else if (process.env.wsl) {
    // Dec 2017 preview builds of WSL include /bin/wslpath; do the parsing work for now.
    // Parsing example %TEMP% is C:\Users\apardoe\AppData\Local\Temp
    const windowsTemp = child_process.execSync('cmd.exe /c echo %TEMP%').toString().replace(/\\/g, "/");
    const driveLetter = windowsTemp.substring(0, 1).toLowerCase();
    const directoryPath = windowsTemp.substring(2).trim();
    process.env.winTmp = path.join("/mnt", driveLetter, directoryPath);
}

// Use the canned git_hash if provided
let gitReleaseName = '';
if (opts.static && fs.existsSync(opts.static + "/git_hash")) {
    gitReleaseName = fs.readFileSync(opts.static + "/git_hash").toString().trim();
} else if (fs.existsSync('.git/')) { // Just if we have been cloned and not downloaded (Thanks David!)
    gitReleaseName = child_process.execSync('git rev-parse HEAD').toString().trim();
}

// Set default values for omitted arguments
const defArgs = {
    rootDir: opts.rootDir || './etc',
    env: opts.env || ['dev'],
    hostname: opts.host,
    port: opts.port || 10240,
    staticDir: opts.static || 'static',
    gitReleaseName: gitReleaseName,
    wantedLanguage: opts.language || null,
    doCache: !opts.noCache,
    fetchCompilersFromRemote: !opts.noRemoteFetch,
    ensureNoCompilerClash: opts.ensureNoIdClash,
    suppressConsoleLog: opts.suppressConsoleLog || false
};

if (opts.logHost && opts.logPort) {
    logToPapertrail(opts.logHost, opts.logPort, defArgs.env.join("."));
}

if (defArgs.suppressConsoleLog) {
    logger.info("Disabling further console logging");
    suppressConsoleLog();
}

const webpackConfig = require('./webpack.config.js')[1],
    webpackCompiler = require('webpack')(webpackConfig),
    manifestName = 'manifest.json',
    staticManifestPath = path.join(__dirname, defArgs.staticDir, webpackConfig.output.publicPath),
    assetManifestPath = path.join(staticManifestPath, 'assets'),
    staticManifest = require(path.join(staticManifestPath, manifestName)),
    assetManifest = require(path.join(assetManifestPath, manifestName));

const isDevMode = () => process.env.NODE_ENV !== "production";

const propHierarchy = _.flatten([
    'defaults',
    defArgs.env,
    _.map(defArgs.env, e => `${e}.${process.platform}`),
    process.platform,
    os.hostname(),
    'local']);
logger.info(`properties hierarchy: ${propHierarchy.join(', ')}`);

// Propagate debug mode if need be
if (opts.propDebug) props.setDebug(true);

// *All* files in config dir are parsed
props.initialize(defArgs.rootDir + '/config', propHierarchy);

// Now load up our libraries.
const aws = require('./lib/aws'),
    google = require('./lib/google');

// Instantiate a function to access records concerning "compiler-explorer"
// in hidden object props.properties
const ceProps = props.propsFor("compiler-explorer");

let languages = require('./lib/languages').list;

if (defArgs.wantedLanguage) {
    const filteredLangs = {};
    _.each(languages, lang => {
        if (lang.id === defArgs.wantedLanguage ||
            lang.name === defArgs.wantedLanguage ||
            (lang.alias && lang.alias.indexOf(defArgs.wantedLanguage) >= 0)) {
            filteredLangs[lang.id] = lang;
        }
    });
    languages = filteredLangs;
}

if (languages.length === 0) {
    logger.error("Trying to start Compiler Explorer without a language");
}

const compilerProps = new props.CompilerProps(languages, ceProps);

const staticMaxAgeSecs = ceProps('staticMaxAgeSecs', 0);
const maxUploadSize = ceProps('maxUploadSize', '1mb');
const extraBodyClass = ceProps('extraBodyClass', isDevMode() ? 'dev' : '');
const storageSolution = compilerProps.ceProps('storageSolution', 'local');
const httpRoot = ceProps('httpRoot', '/');
const httpRootDir = httpRoot.endsWith('/') ? httpRoot : (httpRoot + '/');

function staticHeaders(res) {
    if (staticMaxAgeSecs) {
        res.setHeader('Cache-Control', 'public, max-age=' + staticMaxAgeSecs + ', must-revalidate');
    }
}

const csp = require('./lib/csp').policy;

function contentPolicyHeader(res) {
    if (csp) {
        res.setHeader('Content-Security-Policy', csp);
    }
}

function measureEventLoopLag(delayMs) {
    return new Promise((resolve) => {
        const start = process.hrtime.bigint();
        setTimeout(() => {
            const elapsed = process.hrtime.bigint() - start;
            const delta = elapsed - BigInt(delayMs * 1000000);
            return resolve(Number(delta) / 1000000.0);
        }, delayMs);
    });
}

function setupEventLoopLagLogging() {
    const lagIntervalMs = ceProps('eventLoopMeasureIntervalMs', 0);
    const thresWarn = ceProps('eventLoopLagThresholdWarn', 0);
    const thresErr = ceProps('eventLoopLagThresholdErr', 0);

    async function eventLoopLagHandler() {
        const lagMs = await measureEventLoopLag(lagIntervalMs);

        if (thresErr && lagMs >= thresErr) {
            logger.error(`Event Loop Lag: ${lagMs} ms`);
        }
        else if (thresWarn && lagMs >= thresWarn) {
            logger.warn(`Event Loop Lag: ${lagMs} ms`);
        }

        setImmediate(eventLoopLagHandler);
    }

    if (lagIntervalMs > 0) {
        setImmediate(eventLoopLagHandler);
    }
}

const awsProps = props.propsFor("aws");

aws.initConfig(awsProps)
    .then(initialiseWine)
    .then(() => {
        // function to load internal binaries (i.e. lib/source/*.js)
        function loadSources() {
            const sourcesDir = "lib/sources";
            return fs.readdirSync(sourcesDir)
                .filter(file => file.match(/.*\.js$/))
                .map(file => require("./" + path.join(sourcesDir, file)));
        }

        const fileSources = loadSources();
        const ClientOptionsHandler = require('./lib/options-handler');
        const clientOptionsHandler = new ClientOptionsHandler(fileSources, compilerProps, defArgs);
        const CompilationEnvironment = require('./lib/compilation-env');
        const compilationEnvironment = new CompilationEnvironment(compilerProps, defArgs.doCache);
        const CompileHandler = require('./lib/handlers/compile').Handler;
        const compileHandler = new CompileHandler(compilationEnvironment, awsProps);
        const StorageHandler = require('./lib/storage/storage');
        const storageHandler = StorageHandler.storageFactory(storageSolution, compilerProps, awsProps, httpRootDir);
        const SourceHandler = require('./lib/handlers/source').Handler;
        const sourceHandler = new SourceHandler(fileSources, staticHeaders);
        const CompilerFinder = require('./lib/compiler-finder');
        const compilerFinder = new CompilerFinder(compileHandler, compilerProps, awsProps, defArgs,
            clientOptionsHandler);
        const googleShortUrlResolver = new google.ShortLinkResolver();

        function oldGoogleUrlHandler(req, res, next) {
            const bits = req.url.split("/");
            if (bits.length !== 2 || req.method !== "GET") return next();
            const googleUrl = `https://goo.gl/${encodeURIComponent(bits[1])}`;
            googleShortUrlResolver.resolve(googleUrl)
                .then(resultObj => {
                    const parsed = url.parse(resultObj.longUrl);
                    const allowedRe = new RegExp(ceProps('allowedShortUrlHostRe'));
                    if (parsed.host.match(allowedRe) === null) {
                        logger.warn(`Denied access to short URL ${bits[1]} - linked to ${resultObj.longUrl}`);
                        return next();
                    }
                    res.writeHead(301, {
                        Location: resultObj.longUrl,
                        'Cache-Control': 'public'
                    });
                    res.end();
                })
                .catch(e => {
                    logger.error(`Failed to expand ${googleUrl} - ${e}`);
                    next();
                });
        }

        function startListening(server) {
            const ss = systemdSocket();
            let _port;
            if (ss) {
                // ms (5 min default)
                const idleTimeout = process.env.IDLE_TIMEOUT;
                const timeout = (typeof idleTimeout !== 'undefined' ? idleTimeout : 300) * 1000;
                if (idleTimeout) {
                    let exit = () => {
                        logger.info("Inactivity timeout reached, exiting.");
                        process.exit(0);
                    };
                    let idleTimer = setTimeout(exit, timeout);
                    let reset = () => {
                        clearTimeout(idleTimer);
                        idleTimer = setTimeout(exit, timeout);
                    };
                    server.all('*', reset);
                    logger.info(`  IDLE_TIMEOUT: ${idleTimeout}`);
                }
                _port = ss;
            } else {
                _port = defArgs.port;
            }
            logger.info(`  Listening on http://${defArgs.hostname || 'localhost'}:${_port}/`);
            logger.info(`  Startup duration: ${new Date() - startTime}ms`);
            logger.info("=======================================");
            server.listen(_port, defArgs.hostname);
        }

        compilerFinder.find()
            .then(result => {
                let compilers = result.compilers;
                let prevCompilers;
                if (defArgs.ensureNoCompilerClash) {
                    logger.warn('Ensuring no compiler ids clash');
                    if (result.foundClash) {
                        // If we are forced to have no clashes, throw an error with some explanation
                        throw new Error('Clashing compilers in the current environment found!');
                    } else {
                        logger.info('No clashing ids found, continuing normally...');
                    }
                }

                const sentryDsn = aws.getConfig('sentryDsn');
                if (sentryDsn) {
                    Sentry.init({
                        dsn: sentryDsn,
                        release: gitReleaseName,
                        environment: defArgs.env
                    });
                    logger.info(`Configured with Sentry endpoint ${sentryDsn}`);
                } else {
                    logger.info("Not configuring sentry");
                }

                const webServer = express(),
                    sFavicon = require('serve-favicon'),
                    bodyParser = require('body-parser'),
                    morgan = require('morgan'),
                    compression = require('compression'),
                    router = express.Router(),
                    healthCheck = require('./lib/handlers/health-check');

                const healthCheckFilePath = ceProps("healthCheckFilePath", false);

                const routeApi = new RouteAPI(router, compileHandler, ceProps,
                    storageHandler, renderConfig, renderGoldenLayout);

                function onCompilerChange(compilers) {
                    if (JSON.stringify(prevCompilers) === JSON.stringify(compilers)) {
                        return;
                    }
                    logger.debug("Compilers:", compilers);
                    if (compilers.length === 0) {
                        logger.error("#### No compilers found: no compilation will be done!");
                    }
                    prevCompilers = compilers;
                    clientOptionsHandler.setCompilers(compilers);
                    routeApi.apiHandler.setCompilers(compilers);
                    routeApi.apiHandler.setLanguages(languages);
                    routeApi.apiHandler.setOptions(clientOptionsHandler);
                }
    
                onCompilerChange(compilers);

                const rescanCompilerSecs = ceProps('rescanCompilerSecs', 0);
                if (rescanCompilerSecs) {
                    logger.info(`Rescanning compilers every ${rescanCompilerSecs} secs`);
                    setInterval(() => compilerFinder.find().then(result => onCompilerChange(result.compilers)),
                        rescanCompilerSecs * 1000);
                }

                webServer
                    .set('trust proxy', true)
                    .set('view engine', 'pug')
                    .on('error', err => logger.error('Caught error in web handler; continuing:', err))
                    // Handle healthchecks at the root, as they're not expected from the outside world
                    .use('/healthcheck', new healthCheck.HealthCheckHandler(healthCheckFilePath).handle)
                    .use(httpRootDir, router)
                    .use((req, res, next) => {
                        next({status: 404, message: `page "${req.path}" could not be found`});
                    })
                    .use(Sentry.Handlers.errorHandler)
                    // eslint-disable-next-line no-unused-vars
                    .use((err, req, res, next) => {
                        const status =
                            err.status ||
                            err.statusCode ||
                            err.status_code ||
                            (err.output && err.output.statusCode) ||
                            500;
                        const message = err.message || 'Internal Server Error';
                        res.status(status);
                        res.render('error', renderConfig({error: {code: status, message: message}}));
                    });

                logger.info("=======================================");
                if (gitReleaseName) logger.info(`  git release ${gitReleaseName}`);

                function renderConfig(extra, urlOptions) {
                    const urlOptionsWhitelist = [
                        'readOnly', 'hideEditorToolbars'
                    ];
                    const filteredUrlOptions = _.mapObject(
                        _.pick(urlOptions, urlOptionsWhitelist),
                        val => utils.toProperty(val));
                    const allExtraOptions = _.extend({}, filteredUrlOptions, extra);
                    const options = _.extend({}, allExtraOptions, clientOptionsHandler.get());
                    options.optionsHash = clientOptionsHandler.getHash();
                    options.compilerExplorerOptions = JSON.stringify(allExtraOptions);
                    options.extraBodyClass = options.embedded ? 'embedded' : extraBodyClass;
                    options.httpRoot = httpRoot;
                    options.httpRootDir = httpRootDir;
                    options.storageSolution = storageSolution;
                    options.require = function (path) {
                        if (isDevMode()) {
                            if (fs.existsSync('static/assets/' + path)) {
                                return '/assets/' + path;
                            } else {
                                //this will break assets in dev mode for now
                                return '/dist/' + path;
                            }
                        }
                        if (staticManifest.hasOwnProperty(path)) {
                            return `${httpRootDir}dist/${staticManifest[path]}`;
                        }
                        if (assetManifest.hasOwnProperty(path)) {
                            return `${httpRootDir}dist/assets/${assetManifest[path]}`;
                        }
                        logger.warn("Requested an asset I don't know about");
                    };
                    return options;
                }

                function renderGoldenLayout(config, metadata, req, res) {
                    staticHeaders(res);
                    contentPolicyHeader(res);
                    res.render('index', renderConfig({
                        embedded: false,
                        config: config,
                        metadata: metadata
                    }, req.query));
                }

                const embeddedHandler = function (req, res) {
                    staticHeaders(res);
                    contentPolicyHeader(res);
                    res.render('embed', renderConfig({embedded: true}, req.query));
                };
                if (isDevMode()) {
                    router.use(webpackDevMiddleware(webpackCompiler, {
                        publicPath: webpackConfig.output.publicPath,
                        logger: logger
                    }));
                    router.use(express.static(defArgs.staticDir));
                    logger.info("  using webpack dev middleware");
                } else {
                    /* Assume that anything not dev is just production.
                     * This gives sane defaults for anyone who isn't messing with this */
                    logger.info(`  serving static files from '${defArgs.staticDir}'`);
                    router.use(express.static(defArgs.staticDir, {maxAge: staticMaxAgeSecs * 1000}));
                }

                morgan.token('gdpr_ip', req => utils.anonymizeIp(req.ip));

                // Based on combined format, but: GDPR compliant IP, no timestamp & no unused fields for our usecase
                const morganFormat = isDevMode() ? 'dev' : ':gdpr_ip ":method :url" :status';

                const shortenerLib = require(`./lib/shortener-${clientOptionsHandler.options.urlShortenService}`);
                const shortener = shortenerLib({storageHandler});

                router
                    .use(Sentry.Handlers.requestHandler())
                    .use(morgan(morganFormat, {
                        stream: logger.stream,
                        // Skip for non errors (2xx, 3xx)
                        skip: (req, res) => res.statusCode >= 400
                    }))
                    .use(morgan(morganFormat, {
                        stream: logger.warnStream,
                        // Skip for non user errors (4xx)
                        skip: (req, res) => res.statusCode < 400 || res.statusCode >= 500
                    }))
                    .use(morgan(morganFormat, {
                        stream: logger.errStream,
                        // Skip for non server errors (5xx)
                        skip: (req, res) => res.statusCode < 500
                    }))
                    .use(compression())
                    .get('/', (req, res) => {
                        staticHeaders(res);
                        contentPolicyHeader(res);
                        res.render('index', renderConfig({embedded: false}, req.query));
                    })
                    .get('/e', embeddedHandler)
                    // legacy. not a 301 to prevent any redirect loops between old e links and embed.html
                    .get('/embed.html', embeddedHandler)
                    .get('/embed-ro', (req, res) => {
                        staticHeaders(res);
                        contentPolicyHeader(res);
                        res.render('embed', renderConfig({embedded: true, readOnly: true}, req.query));
                    })
                    .get('/robots.txt', (req, res) => {
                        staticHeaders(res);
                        res.end('User-agent: *\nSitemap: https://godbolt.org/sitemap.xml\nDisallow:');
                    })
                    .get('/sitemap.xml', (req, res) => {
                        staticHeaders(res);
                        res.set('Content-Type', 'application/xml');
                        res.render('sitemap');
                    })
                    .use(sFavicon(path.join(defArgs.staticDir, webpackConfig.output.publicPath, 'favicon.ico')))
                    .get('/client-options.js', (req, res) => {
                        staticHeaders(res);
                        res.set('Content-Type', 'application/javascript');
                        const options = JSON.stringify(clientOptionsHandler.get());
                        res.end(`window.compilerExplorerOptions = ${options};`);
                    })
                    .use(bodyParser.json({limit: ceProps('bodyParserLimit', maxUploadSize)}))
                    .use(bodyParser.text({limit: ceProps('bodyParserLimit', maxUploadSize), type: () => true}))
                    .use('/source', sourceHandler.handle.bind(sourceHandler))
                    .use('/g', oldGoogleUrlHandler)
                    .post('/shortener', shortener);

                routeApi.InitializeRoutes();

                if (!defArgs.doCache) {
                    logger.info("  with disabled caching");
                }
                setupEventLoopLagLogging();
                startListening(webServer);
            })
            .catch(err => {
                logger.error("Promise error (shutting down):", err);
                process.exit(1);
            });
    })
    .catch(err => {
        logger.error("AWS Init Promise error (shutting down)", err);
        process.exit(1);
    });
