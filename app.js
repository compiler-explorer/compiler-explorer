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

// Initialise options and properties. Don't load any handlers here; they
// may need an initialised properties library.
const nopt = require('nopt'),
    os = require('os'),
    props = require('./lib/properties'),
    child_process = require('child_process'),
    path = require('path'),
    fs = require('fs-extra'),
    systemdSocket = require('systemd-socket'),
    url = require('url'),
    _ = require('underscore'),
    express = require('express'),
    Raven = require('raven'),
    logger = require('./lib/logger').logger,
    webpackDevMiddleware = require("webpack-dev-middleware");


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
    noRemoteFetch: [Boolean],
    tmpDir: [String],
    wsl: [Boolean],
    language: [String],
    noCache: [Boolean]
});

if (opts.debug) logger.level = 'debug';

// AP: Detect if we're running under Windows Subsystem for Linux. Temporary modification
// of process.env is allowed: https://nodejs.org/api/process.html#process_process_env
if ((process.platform === "win32") || child_process.execSync('uname -a').toString().indexOf('Microsoft') > -1)
    process.env.wsl = true;

// AP: Allow setting of tmpDir (used in lib/base-compiler.js & lib/exec.js) through opts.
// WSL requires a directory on a Windows volume. Set that to Windows %TEMP% if no tmpDir supplied.
// If a tempDir is supplied then assume that it will work for WSL processes as well.
if (opts.tmpDir) {
    process.env.tmpDir = opts.tmpDir;
    process.env.winTmp = opts.tmpDir;
}
else if (process.env.wsl) {
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
    fetchCompilersFromRemote: !opts.noRemoteFetch
};

const webpackConfig = require('./webpack.config.js')[1],
    webpackCompiler = require('webpack')(webpackConfig),
    manifestName = 'manifest.json',
    staticManifestPath = path.join(__dirname, defArgs.staticDir, webpackConfig.output.publicPath),
    assetManifestPath = path.join(staticManifestPath, 'assets'),
    staticManifest = require(path.join(staticManifestPath, manifestName)),
    assetManifest = require(path.join(assetManifestPath, manifestName));

const isDevMode = () => process.env.NODE_ENV === "DEV";

const propHierarchy = _.flatten([
    'defaults',
    defArgs.env,
    _.map(defArgs.env, e => e + '.' + process.platform),
    process.platform,
    os.hostname(),
    'local']);
logger.info("properties hierarchy: " + propHierarchy.join(', '));

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
const extraBodyClass = ceProps('extraBodyClass', '');

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

const awsProps = props.propsFor("aws");

// function to load internal binaries (i.e. lib/source/*.js)
function loadSources() {
    const sourcesDir = "lib/sources";
    return fs.readdirSync(sourcesDir)
        .filter(file => file.match(/.*\.js$/))
        .map(file => require("./" + path.join(sourcesDir, file)));
}

const fileSources = loadSources();
const ClientOptionsHandler = require('./lib/options-handler');
const clientOptionsHandler = new ClientOptionsHandler(fileSources, languages, compilerProps, defArgs);
const CompilationEnvironment = require('./lib/compilation-env');
const compilationEnvironment = new CompilationEnvironment(compilerProps, defArgs.doCache);
const CompileHandler = require('./lib/handlers/compile').Handler;
const compileHandler = new CompileHandler(compilationEnvironment);
const ApiHandler = require('./lib/handlers/api').Handler;
const apiHandler = new ApiHandler(compileHandler, ceProps);
const SourceHandler = require('./lib/handlers/source').Handler;
const sourceHandler = new SourceHandler(fileSources, staticHeaders);
const CompilerFinder = require('./lib/compiler-finder');
const compilerFinder = new CompilerFinder(compileHandler, compilerProps, awsProps,
    languages, defArgs);

function shortUrlHandler(req, res, next) {
    const resolver = new google.ShortLinkResolver(aws.getConfig('googleApiKey'));
    const bits = req.url.split("/");
    if (bits.length !== 2 || req.method !== "GET") return next();
    const googleUrl = `http://goo.gl/${encodeURIComponent(bits[1])}`;
    resolver.resolve(googleUrl)
        .then(resultObj => {
            const parsed = url.parse(resultObj.longUrl);
            const allowedRe = new RegExp(ceProps('allowedShortUrlHostRe'));
            if (parsed.host.match(allowedRe) === null) {
                logger.warn(`Denied access to short URL ${bits[1]} - linked to ${resultObj.longUrl}`);
                return next();
            }
            res.writeHead(301, {
                Location: resultObj.id,
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
        const timeout = (typeof process.env.IDLE_TIMEOUT !== 'undefined' ? process.env.IDLE_TIMEOUT : 300) * 1000;
        if (timeout) {
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
        }
        _port = ss;
    } else {
        _port = defArgs.port;
    }
    logger.info(`  Listening on http://${defArgs.hostname || 'localhost'}:${_port}/`);
    logger.info("=======================================");
    server.listen(_port, defArgs.hostname);
}

Promise.all([compilerFinder.find(), aws.initConfig(awsProps)])
    .then(args => {
        let compilers = args[0];
        let prevCompilers;

        const ravenPrivateEndpoint = aws.getConfig('ravenPrivateEndpoint');
        if (ravenPrivateEndpoint) {
            Raven.config(ravenPrivateEndpoint, {
                release: gitReleaseName,
                environment: defArgs.env
            }).install();
            logger.info("Configured with raven endpoint", ravenPrivateEndpoint);
        } else {
            Raven.config(false).install();
        }

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
            apiHandler.setCompilers(compilers);
            apiHandler.setLanguages(languages);
        }

        onCompilerChange(compilers);

        const rescanCompilerSecs = ceProps('rescanCompilerSecs', 0);
        if (rescanCompilerSecs) {
            logger.info(`Rescanning compilers every ${rescanCompilerSecs} secs`);
            setInterval(() => compilerFinder.find().then(onCompilerChange),
                rescanCompilerSecs * 1000);
        }

        const webServer = express(),
            sFavicon = require('serve-favicon'),
            bodyParser = require('body-parser'),
            morgan = require('morgan'),
            compression = require('compression'),
            restreamer = require('./lib/restreamer');

        logger.info("=======================================");
        if (gitReleaseName) logger.info(`  git release ${gitReleaseName}`);

        function renderConfig(extra) {
            const options = _.extend(extra, clientOptionsHandler.get());
            options.compilerExplorerOptions = JSON.stringify(options);
            options.extraBodyClass = extraBodyClass;
            options.require = function (path) {
                if (isDevMode()) {
                    //this will break assets in dev mode for now
                    return '/dist/' + path;
                }
                if (staticManifest.hasOwnProperty(path)) {
                    return "dist/" + staticManifest[path];
                }
                if (assetManifest.hasOwnProperty(path)) {
                    return "dist/assets/" + assetManifest[path];
                }
                logger.warn("Requested an asset I don't know about");
            };
            return options;
        }

        const embeddedHandler = function (req, res) {
            staticHeaders(res);
            contentPolicyHeader(res);
            res.render('embed', renderConfig({embedded: true}));
        };
        const healthCheck = require('./lib/handlers/health-check');

        if (isDevMode()) {
            webServer.use(webpackDevMiddleware(webpackCompiler, {
                publicPath: webpackConfig.output.publicPath,
                logger: logger
            }));
            webServer.use(express.static(defArgs.staticDir));
        } else {
            /* Assume that anything not dev is just production.
             * This gives sane defaults for anyone who isn't messing with this */
            logger.info("  serving static files from '" + defArgs.staticDir + "'");
            webServer.use(express.static(defArgs.staticDir, {maxAge: staticMaxAgeSecs * 1000}));
        }

        // Removes last 3 octets for IPv6 and last octet for IPv4
        // There's probably a better way to do this
        morgan.token('gdpr_ip', req => {
            const ip = req.ip;
            if (ip.includes('localhost')) {
                return ip;
            }
            else if (ip.includes(':')) {
                // IPv6
                return ip.replace(/:[\da-fA-F]{0,4}:[\da-fA-F]{0,4}:[\da-fA-F]{0,4}$/, ':::');
            } else {
                // IPv4
                return ip.replace(/\.\d{1,3}$/, '.0');
            }
        });

        // Based on combined format, but: GDPR compilant IP, no timestamp & no unused fields for our usecase
        const morganFormat = isDevMode() ? 'dev' : ':gdpr_ip ":method :url" :status';

        webServer
            .use(Raven.requestHandler())
            .set('trust proxy', true)
            .set('view engine', 'pug')
            // before morgan so healthchecks aren't logged
            .use('/healthcheck', new healthCheck.HealthCheckHandler().handle)
            // Skip for non errors (2xx, 3xx)
            .use(morgan(morganFormat, {stream: logger.stream, skip: (req, res) => res.statusCode >= 400}))
            // Skip for non user errors (4xx)
            .use(morgan(morganFormat, {
                stream: logger.warnStream,
                skip: (req, res) => res.statusCode < 400 && res.statusCode >= 500
            }))
            // Skip for non server errors (5xx)
            .use(morgan(morganFormat, {stream: logger.errStream, skip: (req, res) => res.statusCode < 500}))
            .use(compression())
            .get('/', (req, res) => {
                staticHeaders(res);
                contentPolicyHeader(res);
                res.render('index', renderConfig({embedded: false}));
            })
            .get('/e', embeddedHandler)
            // legacy. not a 301 to prevent any redirect loops between old e links and embed.html
            .get('/embed.html', embeddedHandler)
            .get('/embed-ro', (req, res) => {
                staticHeaders(res);
                contentPolicyHeader(res);
                res.render('embed', renderConfig({embedded: true, readOnly: true}));
            })
            .get('/robots.txt', (req, res) => {
                staticHeaders(res);
                res.end('User-agent: *\nSitemap: https://godbolt.org/sitemap.xml');
            })
            .get('/sitemap.xml', (req, res) => {
                staticHeaders(res);
                res.set('Content-Type', 'application/xml');
                res.render('sitemap');
            })
            .use(sFavicon(path.join(defArgs.staticDir, webpackConfig.output.publicPath, 'favicon.ico')));

        webServer
            .use(bodyParser.json({limit: ceProps('bodyParserLimit', maxUploadSize)}))
            .use(bodyParser.text({limit: ceProps('bodyParserLimit', maxUploadSize), type: () => true}))
            .use(restreamer())
            .use('/source', sourceHandler.handle.bind(sourceHandler))
            .use('/api', apiHandler.handle)
            .use('/g', shortUrlHandler);
        if (!defArgs.doCache) {
            logger.info("  not caching due to --noCache parameter being present");
        }
        webServer.use(Raven.errorHandler());
        webServer.on('error', err => logger.error('Caught error:', err, "(in web error handler; continuing)"));

        startListening(webServer);
    })
    .catch(err => {
        logger.error("Promise error:", err, "(shutting down)");
        process.exit(1);
    });
