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
    _ = require('underscore-node'),
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

// Instantiate a function to access records concerning the chosen language
// in hidden object props.properties
let compilerPropsFuncsL = {};
_.each(languages, lang => compilerPropsFuncsL[lang.id] = props.propsFor(lang.id));

// Get a property from the specified langId, and if not found, use defaults from CE,
// and at last return whatever default value was set by the caller
function compilerPropsL(lang, property, defaultValue) {
    const forLanguage = compilerPropsFuncsL[lang];
    if (forLanguage) {
        const forCompiler = forLanguage(property);
        if (forCompiler !== undefined) return forCompiler;
    }
    return ceProps(property, defaultValue);
}

// For every lang passed, get its corresponding compiler property
function compilerPropsA(langs, property, defaultValue) {
    let forLanguages = {};
    _.each(langs, lang => {
        forLanguages[lang.id] = compilerPropsL(lang.id, property, defaultValue);
    });
    return forLanguages;
}

// Same as A version, but transforms each value by fn(original, lang)
function compilerPropsAT(langs, transform, property, defaultValue) {
    let forLanguages = {};
    _.each(langs, lang => {
        forLanguages[lang.id] = transform(compilerPropsL(lang.id, property, defaultValue), lang);
    });
    return forLanguages;
}

const staticMaxAgeSecs = ceProps('staticMaxAgeSecs', 0);
const maxUploadSize = ceProps('maxUploadSize', '1mb');
let extraBodyClass = ceProps('extraBodyClass', '');

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
const clientOptionsHandler = new ClientOptionsHandler(fileSources);
const CompilationEnvironment = require('./lib/compilation-env');
const compilationEnvironment = new CompilationEnvironment(ceProps, compilerPropsL, defArgs.doCache);
const CompileHandler = require('./lib/handlers/compile').Handler;
const compileHandler = new CompileHandler(compilationEnvironment);
const ApiHandler = require('./lib/handlers/api').Handler;
const apiHandler = new ApiHandler(compileHandler);
const SourceHandler = require('./lib/handlers/source').Handler;
const sourceHandler = new SourceHandler(fileSources, staticHeaders);
const CompilerFinder = require('./lib/compiler-finder');
const compilerFinder = new CompilerFinder(compileHandler, compilerPropsL, compilerPropsAT, ceProps, awsProps,
    languages, defArgs);

function ClientOptionsHandler(fileSources) {
    const sources = _.sortBy(fileSources.map(source => {
        return {name: source.name, urlpart: source.urlpart};
    }), 'name');

    const supportsBinary = compilerPropsAT(languages, res => !!res, 'supportsBinary', true);
    const supportsExecutePerLanguage = compilerPropsAT(languages, (res, lang) => {
        return supportsBinary[lang.id] && !!res;
    }, 'supportsExecute', true);
    const supportsExecute = Object.values(supportsExecutePerLanguage).some((value) => value);

    const libs = {};
    const baseLibs = compilerPropsA(languages, 'libs');
    _.each(baseLibs, (forLang, lang) => {
        if (lang && forLang) {
            libs[lang] = {};
            _.each(forLang.split(':'), lib => {
                const libBaseName = `libs.${lib}`;
                libs[lang][lib] = {
                    name: compilerPropsL(lang, libBaseName + '.name'),
                    url: compilerPropsL(lang, libBaseName + '.url'),
                    description: compilerPropsL(lang, libBaseName + '.description')
                };
                libs[lang][lib].versions = {};
                const listedVersions = `${compilerPropsL(lang, libBaseName + '.versions')}`;
                if (listedVersions) {
                    _.each(listedVersions.split(':'), version => {
                        const libVersionName = libBaseName + `.versions.${version}`;
                        libs[lang][lib].versions[version] = {};
                        libs[lang][lib].versions[version].version = compilerPropsL(lang, libVersionName + '.version');
                        libs[lang][lib].versions[version].path = [];
                        const includes = compilerPropsL(lang, libVersionName + '.path');
                        if (includes) {
                            _.each(includes.split(':'), path => libs[lang][lib].versions[version].path.push(path));
                        } else {
                            logger.warn(`No paths found for ${lib} - ${version}`);
                        }
                    });
                } else {
                    logger.warn(`No versions found for ${lib} library`);
                }
            });
        }
    });
    const options = {
        googleAnalyticsAccount: ceProps('clientGoogleAnalyticsAccount', 'UA-55180-6'),
        googleAnalyticsEnabled: ceProps('clientGoogleAnalyticsEnabled', false),
        sharingEnabled: ceProps('clientSharingEnabled', true),
        githubEnabled: ceProps('clientGitHubRibbonEnabled', true),
        gapiKey: ceProps('googleApiKey', ''),
        googleShortLinkRewrite: ceProps('googleShortLinkRewrite', '').split('|'),
        urlShortenService: ceProps('urlShortenService', 'none'),
        defaultSource: ceProps('defaultSource', ''),
        compilers: [],
        libs: libs,
        defaultCompiler: compilerPropsA(languages, 'defaultCompiler', ''),
        compileOptions: compilerPropsA(languages, 'defaultOptions', ''),
        supportsBinary: supportsBinary,
        supportsExecute: supportsExecute,
        languages: languages,
        sources: sources,
        raven: ceProps('ravenUrl', ''),
        release: gitReleaseName,
        environment: defArgs.env,
        localStoragePrefix: ceProps('localStoragePrefix'),
        cvCompilerCountMax: ceProps('cvCompilerCountMax', 6),
        defaultFontScale: ceProps('defaultFontScale', 1.0),
        doCache: defArgs.doCache
    };
    this.setCompilers = compilers => {
        const blacklistedKeys = ['exe', 'versionFlag', 'versionRe', 'compilerType', 'demangler', 'objdumper',
            'postProcess'];
        const copiedCompilers = JSON.parse(JSON.stringify(compilers));
        _.each(options.compilers, (compiler, compilersKey) => {
            _.each(compiler, (_, propKey) => {
                if (blacklistedKeys.includes(propKey)) {
                    delete copiedCompilers[compilersKey][propKey];
                }
            });
        });
        options.compilers = copiedCompilers;
    };
    this.setCompilers([]);
    this.get = () => options;
}

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

        webServer
            .use(Raven.requestHandler())
            .set('trust proxy', true)
            .set('view engine', 'pug')
            // before morgan so healthchecks aren't logged
            .use('/healthcheck', new healthCheck.HealthCheckHandler().handle)
            .use(morgan('combined', {stream: logger.stream}))
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
