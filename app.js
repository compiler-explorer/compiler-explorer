#!/usr/bin/env node

// Copyright (c) 2012-2018, Matt Godbolt
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
    http = require('http'),
    url = require('url'),
    _ = require('underscore-node'),
    express = require('express'),
    Raven = require('raven'),
    logger = require('./lib/logger').logger,
    webpackDevMiddleware = require("webpack-dev-middleware");



// Parse arguments from command line 'node ./app.js args...'
const opts = nopt({
    'env': [String, Array],
    'rootDir': [String],
    'host': [String],
    'port': [Number],
    'propDebug': [Boolean],
    'debug': [Boolean],
    'static': [String],
    'archivedVersions': [String],
    'noRemoteFetch': [Boolean],
    'tmpDir': [String],
    'wsl': [Boolean],
    'language': [String]
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

// Set default values for omitted arguments
const rootDir = opts.rootDir || './etc';
const env = opts.env || ['dev'];
const hostname = opts.host;
const port = opts.port || 10240;
const staticDir = opts.static || 'static';
const archivedVersions = opts.archivedVersions;
let gitReleaseName = "";
let versionedRootPrefix = "";
const wantedLanguage = opts.language || null;
let builtResourcesRoot = "";


const config = require('./webpack.config.js')[1],
    webpackCompiler = require('webpack')(config),
    manifestName = 'manifest.json',
    staticManifestPath = path.join(__dirname, staticDir, config.output.publicPath),
    assetManifestPath = path.join(staticManifestPath, 'assets'),
    staticManifest = require(path.join(staticManifestPath, manifestName)),
    assetManifest = require(path.join(assetManifestPath, manifestName));

// Use the canned git_hash if provided
if (opts.static && fs.existsSync(opts.static + "/git_hash")) {
    gitReleaseName = fs.readFileSync(opts.static + "/git_hash").toString().trim();
} else if (fs.existsSync('.git/')) { // Just if we have been cloned and not downloaded (Thanks David!)
    gitReleaseName = child_process.execSync('git rev-parse HEAD').toString().trim();
}
if (opts.static && fs.existsSync(opts.static + '/v/' + gitReleaseName))
    versionedRootPrefix = "v/" + gitReleaseName + "/";

if(process.env.NODE_ENV == "DEV") {
    builtResourcesRoot = "static/";
}
// Don't treat @ in paths as remote adresses
const fetchCompilersFromRemote = !opts.noRemoteFetch;

const propHierarchy = _.flatten([
    'defaults',
    env,
    _.map(env, function (e) {
        return e + '.' + process.platform;
    }),
    process.platform,
    os.hostname(),
    'local']);
logger.info("properties hierarchy: " + propHierarchy.join(', '));

// Propagate debug mode if need be
if (opts.propDebug) props.setDebug(true);

// *All* files in config dir are parsed 
props.initialize(rootDir + '/config', propHierarchy);

// Now load up our libraries.
const aws = require('./lib/aws'),
    google = require('./lib/google');

// Instantiate a function to access records concerning "compiler-explorer" 
// in hidden object props.properties
const ceProps = props.propsFor("compiler-explorer");

let languages = require('./lib/languages').list;

if (wantedLanguage) {
    const filteredLangs = {};
    _.each(languages, lang => {
        if (lang.id === wantedLanguage ||
            lang.name === wantedLanguage ||
            (lang.alias && lang.alias.indexOf(wantedLanguage) >= 0)) {
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

// Same as A version, but transfroms each value by fn(original, lang)
function compilerPropsAT(langs, transform, property, defaultValue) {
    let forLanguages = {};
    _.each(langs, lang => {
        forLanguages[lang.id] = transform(compilerPropsL(lang.id, property, defaultValue), lang);
    });
    return forLanguages;
}

const staticMaxAgeSecs = ceProps('staticMaxAgeSecs', 0);
const contentPolicy = ceProps('contentPolicy', 'default-src *');
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
        res.setHeader('Content-Security-Policy-Report-Only', csp);
    }
}

const awsProps = props.propsFor("aws");
let awsPoller = null;

function awsInstances() {
    if (!awsPoller) awsPoller = new aws.InstanceFetcher(awsProps);
    return awsPoller.getInstances();
}

// function to load internal binaries (i.e. lib/source/*.js)
function loadSources() {
    const sourcesDir = "lib/sources";
    return fs.readdirSync(sourcesDir)
        .filter(function (file) {
            return file.match(/.*\.js$/);
        })
        .map(function (file) {
            return require("./" + path.join(sourcesDir, file));
        });
}

const fileSources = loadSources();
const clientOptionsHandler = new ClientOptionsHandler(fileSources);
const CompileHandler = require('./lib/handlers/compile').Handler;
const compileHandler = new CompileHandler(ceProps, compilerPropsL);
const ApiHandler = require('./lib/handlers/api').Handler;
const apiHandler = new ApiHandler(compileHandler);
const SourceHandler = require('./lib/handlers/source').Handler;
const sourceHandler = new SourceHandler(fileSources, staticHeaders);

function ClientOptionsHandler(fileSources) {
    const sources = _.sortBy(fileSources.map(source => {
        return {name: source.name, urlpart: source.urlpart};
    }), 'name');

    const supportsBinary = compilerPropsAT(languages, res => !!res, 'supportsBinary', true);
    const supportsExecutePerLanguage = compilerPropsAT(languages, (res, lang) => supportsBinary[lang.id] && !!res, 'supportsExecute', true);
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
                    url: compilerPropsL(lang, libBaseName + '.url')
                };
                libs[lang][lib].versions = {};
                const listedVersions = compilerPropsL(lang, libBaseName + '.versions');
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
        environment: env,
        localStoragePrefix: ceProps('localStoragePrefix'),
        cvCompilerCountMax: ceProps('cvCompilerCountMax', 6),
        defaultFontScale: ceProps('defaultFontScale', 1.0)
    };
    this.setCompilers = compilers => options.compilers = compilers;
    this.setCompilers([]);
    this.handler = function getClientOptions(req, res) {
        res.set('Content-Type', 'application/json');
        staticHeaders(res);
        res.end(JSON.stringify(options));
    };
    this.get = () => options;
}

function retryPromise(promiseFunc, name, maxFails, retryMs) {
    return new Promise(function (resolve, reject) {
        let fails = 0;

        function doit() {
            const promise = promiseFunc();
            promise.then(function (arg) {
                resolve(arg);
            }, function (e) {
                fails++;
                if (fails < maxFails) {
                    logger.warn("Failed " + name + " : " + e + ", retrying");
                    setTimeout(doit, retryMs);
                } else {
                    logger.error("Too many retries for " + name + " : " + e);
                    reject(e);
                }
            });
        }

        doit();
    });
}

function findCompilers() {
    const exes = compilerPropsAT(languages, exs => _.compact(exs.split(":")), "compilers", "");

    const ndk = compilerPropsA(languages, 'androidNdk');
    _.each(ndk, (ndkPath, langId) => {
        if (ndkPath) {
            let toolchains = fs.readdirSync(ndkPath + "/toolchains");
            toolchains.forEach((version, index, a) => {
                const path = ndkPath + "/toolchains/" + version + "/prebuilt/linux-x86_64/bin/";
                if (fs.existsSync(path)) {
                    const cc = fs.readdirSync(path).filter(filename => filename.indexOf("g++") !== -1);
                    a[index] = path + cc[0];
                } else {
                    a[index] = null;
                }
            });
            toolchains = toolchains.filter(x => x !== null);
            exes[langId].push(toolchains);
        }
    });

    function fetchRemote(host, port, props) {
        logger.info("Fetching compilers from remote source " + host + ":" + port);
        return retryPromise(() => {
                return new Promise((resolve, reject) => {
                    let request = http.get({
                        hostname: host,
                        port: port,
                        path: "/api/compilers",
                        headers: {
                            'Accept': 'application/json'
                        }
                    }, res => {
                        let str = '';
                        res.on('data', chunk => {
                            str += chunk;
                        });
                        res.on('end', () => {
                            let compilers = JSON.parse(str).map(compiler => {
                                compiler.exe = null;
                                compiler.remote = "http://" + host + ":" + port;
                                return compiler;
                            });
                            resolve(compilers);
                        });
                    })
                        .on('error', reject)
                        .on('timeout', () => reject("timeout"));
                    request.setTimeout(awsProps('proxyTimeout', 1000));
                });
            },
            host + ":" + port,
            props('proxyRetries', 5),
            props('proxyRetryMs', 500))
            .catch(() => {
                logger.warn("Unable to contact " + host + ":" + port + "; skipping");
                return [];
            });
    }

    function fetchAws() {
        logger.info("Fetching instances from AWS");
        return awsInstances().then(instances => {
            return Promise.all(instances.map(instance => {
                logger.info("Checking instance " + instance.InstanceId);
                let address = instance.PrivateDnsName;
                if (awsProps("externalTestMode", false)) {
                    address = instance.PublicDnsName;
                }
                return fetchRemote(address, port, awsProps);
            }));
        });
    }

    function compilerConfigFor(langId, compilerName, parentProps) {
        const base = "compiler." + compilerName + ".";

        function props(propName, def) {
            let propsForCompiler = parentProps(langId, base + propName, undefined);
            if (propsForCompiler === undefined) {
                propsForCompiler = parentProps(langId, propName, def);
            }
            return propsForCompiler;
        }

        const supportsBinary = !!props("supportsBinary", true);
        const supportsExecute = supportsBinary && !!props("supportsExecute", true);
        const group = props("group", "");
        const compilerInfo = {
            id: compilerName,
            exe: props("exe", compilerName),
            name: props("name", compilerName),
            alias: props("alias"),
            options: props("options"),
            versionFlag: props("versionFlag"),
            versionRe: props("versionRe"),
            compilerType: props("compilerType", ""),
            demangler: props("demangler", ""),
            objdumper: props("objdumper", ""),
            intelAsm: props("intelAsm", ""),
            needsMulti: !!props("needsMulti", true),
            supportsBinary: supportsBinary,
            supportsExecute: supportsExecute,
            postProcess: props("postProcess", "").split("|"),
            lang: langId,
            group: group,
            groupName: props("groupName", "")
        };
        logger.debug("Found compiler", compilerInfo);
        return Promise.resolve(compilerInfo);
    }

    function recurseGetCompilers(langId, compilerName, parentProps) {
        if (fetchCompilersFromRemote && compilerName.indexOf("@") !== -1) {
            const bits = compilerName.split("@");
            const host = bits[0];
            const port = parseInt(bits[1]);
            return fetchRemote(host, port, ceProps);
        }
        if (compilerName.indexOf("&") === 0) {
            const groupName = compilerName.substr(1);

            const props = function (langId, propName, def) {
                if (propName === "group") {
                    return groupName;
                }
                return compilerPropsL(langId, "group." + groupName + "." + propName, parentProps(langId, propName, def));
            };

            const compilerExes = _.compact(props(langId, 'compilers', '').split(":"));
            logger.debug("Processing compilers from group " + groupName);
            return Promise.all(compilerExes.map(compiler => recurseGetCompilers(langId, compiler, props)));
        }
        if (compilerName === "AWS") return fetchAws();
        return compilerConfigFor(langId, compilerName, parentProps);
    }

    function getCompilers() {
        let compilers = [];
        _.each(exes, (exs, langId) => {
            _.each(exs, exe => compilers.push(recurseGetCompilers(langId, exe, compilerPropsL)));
        });
        return compilers;
    }

    function ensureDistinct(compilers) {
        let ids = {};
        _.each(compilers, compiler => {
            if (!ids[compiler.id]) ids[compiler.id] = [];
            ids[compiler.id].push(compiler);
        });
        _.each(ids, (list, id) => {
            if (list.length !== 1) {
                logger.error(`Compiler ID clash for '${id}' - used by ${
                    _.map(list, o => 'lang:' + o.lang + " name:" + o.name).join(', ')
                    }`);
            }
        });
        return compilers;
    }

    return Promise.all(getCompilers())
        .then(_.flatten)
        .then(compilers => compileHandler.setCompilers(compilers))
        .then(compilers => _.compact(compilers))
        .then(ensureDistinct)
        .then(compilers => _.sortBy(compilers, "name"));
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

Promise.all([findCompilers(), aws.initConfig(awsProps)])
    .then(args => {
        let compilers = args[0];
        let prevCompilers;

        const ravenPrivateEndpoint = aws.getConfig('ravenPrivateEndpoint');
        if (ravenPrivateEndpoint) {
            Raven.config(ravenPrivateEndpoint, {
                release: gitReleaseName,
                environment: env
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
            setInterval(() => findCompilers().then(onCompilerChange),
                rescanCompilerSecs * 1000);
        }

        const webServer = express(),
            sFavicon = require('serve-favicon'),
            bodyParser = require('body-parser'),
            morgan = require('morgan'),
            compression = require('compression'),
            restreamer = require('./lib/restreamer');

        logger.info("=======================================");
        logger.info("Listening on http://" + (hostname || 'localhost') + ":" + port + "/");
        if (gitReleaseName) logger.info("  git release " + gitReleaseName);

        function renderConfig(extra) {
            const options = _.extend(extra, clientOptionsHandler.get());
            options.compilerExplorerOptions = JSON.stringify(options);
            options.root = versionedRootPrefix;
            options.extraBodyClass = extraBodyClass;
            options.builtResourcesRoot = builtResourcesRoot;
            options.require = function(path) { 
                if(process.env.NODE_ENV == "DEV") {
                    //I have no idea why main => maps to styles i need to dig into this
                    if(path == 'main.css') {
                        return '/dist/styles.css';
                    }
                    //this will break assets in dev mode for now
                    return '/dist/' + path;
                }
                if(staticManifest.hasOwnProperty(path)) {
                    return versionedRootPrefix + "/dist/" + staticManifest[path];
                }
                if(assetManifest.hasOwnProperty(path)) {
                    return versionedRootPrefix + "/dist/assets/" + assetManifest[path];
                }
                console.warn("requested an asset I don't know about");
            };
            return options;
        }

        const embeddedHandler = function (req, res) {
            staticHeaders(res);
            contentPolicyHeader(res);
            res.render('embed', renderConfig({embedded: true}));
        };
        const healthCheck = require('./lib/handlers/health-check');
        
        
        if(process.env.NODE_ENV == "DEV") {
            webServer.use(webpackDevMiddleware(webpackCompiler, {
                publicPath: config.output.publicPath
            }));
            webServer.use(express.static(staticDir));
        } else {
               //assume that anything not dev is just production this gives sane defaults for anyone who isn't messing with this
            logger.info("  serving static files from '" + staticDir + "'");
            webServer.use(express.static(staticDir, {maxAge: staticMaxAgeSecs * 1000}));
            webServer.use('/v', express.static(staticDir + '/v', {maxAge: Infinity, index: false}));
        }

        
        webServer
            .use(Raven.requestHandler())
            .set('trust proxy', true)
            .set('view engine', 'pug')
            .use('/healthcheck', new healthCheck.HealthCheckHandler().handle) // before morgan so healthchecks aren't logged
            .use(morgan('combined', {stream: logger.stream}))
            .use(compression())
            .get('/', (req, res) => {
                staticHeaders(res);
                contentPolicyHeader(res);
                res.render('index', renderConfig({embedded: false}));
            })
            .get('/e', embeddedHandler)
            .get('/embed.html', embeddedHandler) // legacy. not a 301 to prevent any redirect loops between old e links and embed.html
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
            .use(sFavicon(staticDir + '/favicon.ico'));
            
        if (archivedVersions) {
            // The archived versions directory is used to serve "old" versioned data during updates. It's expected
            // to contain all the SHA-hashed directories from previous versions of Compiler Explorer.
            logger.info("  serving archived versions from", archivedVersions);
            webServer.use('/v', express.static(archivedVersions, {maxAge: Infinity, index: false}));
        }
        webServer
            .use(bodyParser.json({limit: ceProps('bodyParserLimit', maxUploadSize)}))
            .use(bodyParser.text({limit: ceProps('bodyParserLimit', maxUploadSize), type: () => true}))
            .use(restreamer())
            .get('/client-options.json', clientOptionsHandler.handler)
            .use('/source', sourceHandler.handle.bind(sourceHandler))
            .use('/api', apiHandler.handle)
            .use('/g', shortUrlHandler);
        logger.info("=======================================");

        webServer.use(Raven.errorHandler());

        webServer.on('error', err => logger.error('Caught error:', err, "(in web error handler; continuing)"));

        webServer.listen(port, hostname);
    })
    .catch(err => {
        logger.error("Promise error:", err, "(shutting down)");
        process.exit(1);
    });
