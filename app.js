#!/usr/bin/env node

// Copyright (c) 2012-2017, Matt Godbolt
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
    https = require('https'),
    url = require('url'),
    _ = require('underscore-node'),
    utils = require('./lib/utils'),
    express = require('express'),
    logger = require('./lib/logger').logger,
    Raven = require('raven');

// Parse arguments from command line 'node ./app.js args...'
var opts = nopt({
    'env': [String, Array],
    'rootDir': [String],
    'language': [String],
    'host': [String],
    'port': [Number],
    'propDebug': [Boolean],
    'debug': [Boolean],
    'static': [String],
    'archivedVersions': [String],
    'noRemoteFetch': [Boolean],
    'tmpDir': [String],
    'wsl': [Boolean]
});

if (opts.debug) logger.level = 'debug';

// AP: Detect if we're running under Windows Subsystem for Linux. Temporary modification
// of process.env is allowed: https://nodejs.org/api/process.html#process_process_env
if (child_process.execSync('uname -a').toString().indexOf('Microsoft') > -1)
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
    var windowsTemp = child_process.execSync('cmd.exe /c echo %TEMP%').toString().replace(/\\/g, "/");
    var driveLetter = windowsTemp.substring(0, 1).toLowerCase();
    var directoryPath = windowsTemp.substring(2).trim();
    process.env.winTmp = "/mnt/".concat(driveLetter).concat(directoryPath);
}

// Set default values for omitted arguments
var rootDir = opts.rootDir || './etc';
var language = opts.language || "C++";
var env = opts.env || ['dev'];
var hostname = opts.host;
var port = opts.port || 10240;
var staticDir = opts.static || 'static';
var archivedVersions = opts.archivedVersions;
var gitReleaseName = "";
var versionedRootPrefix = "";
// Use the canned git_hash if provided
if (opts.static && fs.existsSync(opts.static + "/git_hash")) {
    gitReleaseName = fs.readFileSync(opts.static + "/git_hash").toString().trim();
} else {
    gitReleaseName = child_process.execSync('git rev-parse HEAD').toString().trim();
}
if (opts.static && fs.existsSync(opts.static + '/v/' + gitReleaseName))
    versionedRootPrefix = "v/" + gitReleaseName + "/";
// Don't treat @ in paths as remote adresses
var fetchCompilersFromRemote = !opts.noRemoteFetch;

var propHierarchy = _.flatten([
    'defaults',
    env,
    language,
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
const CompileHandler = require('./lib/compile-handler').CompileHandler,
    aws = require('./lib/aws'),
    google = require('./lib/google');

// Instantiate a function to access records concerning "compiler-explorer" 
// in hidden object props.properties
var gccProps = props.propsFor("compiler-explorer");

// Instantiate a function to access records concerning the chosen language
// in hidden object props.properties
var compilerPropsFunc = props.propsFor(language.toLowerCase());

// If no option for the compiler ... use gcc's options (??)
function compilerProps(property, defaultValue) {
    // My kingdom for ccs... [see Matt's github page]
    var forCompiler = compilerPropsFunc(property, undefined);
    if (forCompiler !== undefined) return forCompiler;
    return gccProps(property, defaultValue); // gccProps comes from lib/compile-handler.js
}

var staticMaxAgeSecs = gccProps('staticMaxAgeSecs', 0);
let extraBodyClass = gccProps('extraBodyClass', '');

function staticHeaders(res) {
    if (staticMaxAgeSecs) {
        res.setHeader('Cache-Control', 'public, max-age=' + staticMaxAgeSecs + ', must-revalidate');
    }
}

var awsProps = props.propsFor("aws");
var awsPoller = null;

function awsInstances() {
    if (!awsPoller) awsPoller = new aws.InstanceFetcher(awsProps);
    return awsPoller.getInstances();
}

// function to load internal binaries (i.e. lib/source/*.js)
function loadSources() {
    var sourcesDir = "lib/sources";
    return fs.readdirSync(sourcesDir)
        .filter(function (file) {
            return file.match(/.*\.js$/);
        })
        .map(function (file) {
            return require("./" + path.join(sourcesDir, file));
        });
}

// load effectively
var fileSources = loadSources();
var sourceToHandler = {};
fileSources.forEach(function (source) {
    sourceToHandler[source.urlpart] = source;
});

var clientOptionsHandler = new ClientOptionsHandler(fileSources);
var compileHandler = new CompileHandler(gccProps, compilerProps);
const ApiHandler = require('./lib/handlers/api').ApiHandler;
const apiHandler = new ApiHandler(compileHandler);

// auxiliary function used in clientOptionsHandler
function compareOn(key) {
    return function (xObj, yObj) {
        var x = xObj[key];
        var y = yObj[key];
        if (x < y) return -1;
        if (x > y) return 1;
        return 0;
    };
}

// instantiate a function that generate javascript code,
function ClientOptionsHandler(fileSources) {
    var sources = fileSources.map(function (source) {
        return {name: source.name, urlpart: source.urlpart};
    });
    // sort source file alphabetically
    sources = sources.sort(compareOn("name"));
    var languages = _.compact(_.map(gccProps("languages", '').split(':'), function (thing) {
        if (!thing) return null;
        var splat = thing.split("=");
        return {language: splat[0], url: splat[1]};
    }));
    var supportsBinary = !!compilerProps("supportsBinary", true);
    var supportsExecute = supportsBinary && !!compilerProps("supportsExecute", true);
    var libs = {};

    var baseLibs = compilerProps("libs");

    if (baseLibs) {
        _.each(baseLibs.split(':'), function (lib) {
            libs[lib] = {name: compilerProps('libs.' + lib + '.name')};
            libs[lib].versions = {};
            var listedVersions = compilerProps("libs." + lib + '.versions');
            if (listedVersions) {
                _.each(listedVersions.split(':'), function (version) {
                    libs[lib].versions[version] = {};
                    libs[lib].versions[version].version = compilerProps("libs." + lib + '.versions.' + version + '.version');
                    libs[lib].versions[version].path = [];
                    var listedIncludes = compilerProps("libs." + lib + '.versions.' + version + '.path');
                    if (listedIncludes) {
                        _.each(listedIncludes.split(':'), function (path) {
                            libs[lib].versions[version].path.push(path);
                        });
                    } else {
                        logger.warn("No paths found for " + lib + " version " + version);
                    }
                });
            } else {
                logger.warn("No versions found for " + lib + " library");
            }
        });
    }
    var options = {
        googleAnalyticsAccount: gccProps('clientGoogleAnalyticsAccount', 'UA-55180-6'),
        googleAnalyticsEnabled: gccProps('clientGoogleAnalyticsEnabled', false),
        sharingEnabled: gccProps('clientSharingEnabled', true),
        githubEnabled: gccProps('clientGitHubRibbonEnabled', true),
        gapiKey: gccProps('googleApiKey', ''),
        googleShortLinkRewrite: gccProps('googleShortLinkRewrite', '').split('|'),
        defaultSource: gccProps('defaultSource', ''),
        language: language,
        compilers: [],
        libs: libs,
        sourceExtension: compilerProps('compileFilename').split('.', 2)[1],
        defaultCompiler: compilerProps('defaultCompiler', ''),
        compileOptions: compilerProps('defaultOptions', ''),
        supportsBinary: supportsBinary,
        supportsExecute: supportsExecute,
        languages: languages,
        sources: sources,
        raven: gccProps('ravenUrl', ''),
        release: gitReleaseName,
        environment: env,
        localStoragePrefix: gccProps('localStoragePrefix'),
        cvCompilerCountMax: gccProps('cvCompilerCountMax', 6),
        defaultFontScale: gccProps('defaultFontScale', 1.0)
    };
    this.setCompilers = function (compilers) {
        options.compilers = compilers;
    };
    this.setCompilers([]);
    this.handler = function getClientOptions(req, res) {
        res.set('Content-Type', 'application/json');
        staticHeaders(res);
        res.end(JSON.stringify(options));
    };
    this.get = function () {
        return options;
    };
}

// function used to enable loading and saving source code from web interface
function getSource(req, res, next) {
    var bits = req.url.split("/");
    var handler = sourceToHandler[bits[1]];
    if (!handler) {
        next();
        return;
    }
    var action = bits[2];
    if (action === "list") action = handler.list;
    else if (action === "load") action = handler.load;
    else if (action === "save") action = handler.save;
    else action = null;
    if (action === null) {
        next();
        return;
    }
    action.apply(handler, bits.slice(3).concat(function (err, response) {
        staticHeaders(res);
        if (err) {
            res.end(JSON.stringify({err: err}));
        } else {
            res.end(JSON.stringify(response));
        }
    }));
}

function retryPromise(promiseFunc, name, maxFails, retryMs) {
    return new Promise(function (resolve, reject) {
        var fails = 0;

        function doit() {
            var promise = promiseFunc();
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
    var exes = compilerProps("compilers", "/usr/bin/g++").split(":");
    var ndk = compilerProps('androidNdk');
    if (ndk) {
        var toolchains = fs.readdirSync(ndk + "/toolchains");
        toolchains.forEach(function (v, i, a) {
            var path = ndk + "/toolchains/" + v + "/prebuilt/linux-x86_64/bin/";
            if (fs.existsSync(path)) {
                var cc = fs.readdirSync(path).filter(function (filename) {
                    return filename.indexOf("g++") !== -1;
                });
                a[i] = path + cc[0];
            } else {
                a[i] = null;
            }
        });
        toolchains = toolchains.filter(function (x) {
            return x !== null;
        });
        exes.push.apply(exes, toolchains);
    }

    function fetchRemote(host, port, props) {
        logger.info("Fetching compilers from remote source " + host + ":" + port);
        return retryPromise(function () {
                return new Promise(function (resolve, reject) {
                    var request = http.get({
                        hostname: host,
                        port: port,
                        path: "/api/compilers",
                        headers: {
                            'Accept': 'application/json'
                        }
                    }, function (res) {
                        var str = '';
                        res.on('data', function (chunk) {
                            str += chunk;
                        });
                        res.on('end', function () {
                            var compilers = JSON.parse(str).map(function (compiler) {
                                compiler.exe = null;
                                compiler.remote = "http://" + host + ":" + port;
                                return compiler;
                            });
                            resolve(compilers);
                        });
                    }).on('error', function (e) {
                        reject(e);
                    }).on('timeout', function () {
                        reject("timeout");
                    });
                    request.setTimeout(awsProps('proxyTimeout', 1000));
                });
            },
            host + ":" + port,
            props('proxyRetries', 5),
            props('proxyRetryMs', 500))
            .catch(function () {
                logger.warn("Unable to contact " + host + ":" + port + "; skipping");
                return [];
            });
    }

    function fetchAws() {
        logger.info("Fetching instances from AWS");
        return awsInstances().then(function (instances) {
            return Promise.all(instances.map(function (instance) {
                logger.info("Checking instance " + instance.InstanceId);
                var address = instance.PrivateDnsName;
                if (awsProps("externalTestMode", false)) {
                    address = instance.PublicDnsName;
                }
                return fetchRemote(address, port, awsProps);
            }));
        });
    }

    function compilerConfigFor(name, parentProps) {
        const base = "compiler." + name,
            exe = compilerProps(base + ".exe", name);

        function props(name, def) {
            return parentProps(base + "." + name, parentProps(name, def));
        }

        var supportsBinary = !!props("supportsBinary", true);
        var supportsExecute = supportsBinary && !!props("supportsExecute", true);
        var compilerInfo = {
            id: name,
            exe: exe,
            name: props("name", name),
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
            postProcess: props("postProcess", "").split("|")
        };
        logger.info("Found compiler", compilerInfo);
        return Promise.resolve(compilerInfo);
    }

    function recurseGetCompilers(name, parentProps) {
        if (fetchCompilersFromRemote && name.indexOf("@") !== -1) {
            var bits = name.split("@");
            var host = bits[0];
            var port = parseInt(bits[1]);
            return fetchRemote(host, port, gccProps);
        }
        if (name.indexOf("&") === 0) {
            var groupName = name.substr(1);

            var props = function (name, def) {
                if (name === "group") {
                    return groupName;
                }
                return compilerProps("group." + groupName + "." + name, parentProps(name, def));
            };

            var exes = props('compilers', '').split(":");
            logger.info("Processing compilers from group " + groupName);
            return Promise.all(exes.map(function (compiler) {
                return recurseGetCompilers(compiler, props);
            }));
        }
        if (name === "AWS") return fetchAws();
        return compilerConfigFor(name, parentProps);
    }

    return Promise.all(
        exes.map(function (compiler) {
            return recurseGetCompilers(compiler, compilerProps);
        }))
        .then(_.flatten)
        .then(function (compilers) {
            return compileHandler.setCompilers(compilers);
        })
        .then(function (compilers) {
            return _.filter(compilers, function (x) {
                return x;
            });
        })
        .then(function (compilers) {
            compilers = compilers.sort(compareOn("name"));
            return compilers;
        });
}

function shortUrlHandler(req, res, next) {
    const resolver = new google.ShortLinkResolver(aws.getConfig('googleApiKey'));
    const bits = req.url.split("/");
    if (bits.length !== 2 || req.method !== "GET") return next();
    const googleUrl = `http://goo.gl/${encodeURIComponent(bits[1])}`;
    resolver.resolve(googleUrl)
        .then(resultObj => {
            var parsed = url.parse(resultObj.longUrl);
            var allowedRe = new RegExp(gccProps('allowedShortUrlHostRe'));
            if (parsed.host.match(allowedRe) === null) {
                logger.warn("Denied access to short URL " + bits[1] + " - linked to " + resultObj.longUrl);
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
    .then(function (args) {
        let compilers = args[0];
        var prevCompilers;

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

        const newRelicLicense = aws.getConfig('newRelicLicense');
        if (newRelicLicense) {
            process.env.NEW_RELIC_NO_CONFIG_FILE = true;
            process.env.NEW_RELIC_APP_NAME = 'Compiler Explorer';
            process.env.NEW_RELIC_LICENSE_KEY = newRelicLicense;
            process.env.NEW_RELIC_LABELS = 'Language:' + language;
            require('newrelic');
            logger.info('New relic configured with license', newRelicLicense);
        }

        function onCompilerChange(compilers) {
            if (JSON.stringify(prevCompilers) === JSON.stringify(compilers)) {
                return;
            }
            logger.info("Compilers:", compilers);
            if (compilers.length === 0) {
                logger.error("#### No compilers found: no compilation will be done!");
            }
            prevCompilers = compilers;
            clientOptionsHandler.setCompilers(compilers);
            apiHandler.setCompilers(compilers);
        }

        onCompilerChange(compilers);

        var rescanCompilerSecs = gccProps('rescanCompilerSecs', 0);
        if (rescanCompilerSecs) {
            logger.info("Rescanning compilers every " + rescanCompilerSecs + "secs");
            setInterval(function () {
                findCompilers().then(onCompilerChange);
            }, rescanCompilerSecs * 1000);
        }

        var webServer = express(),
            sFavicon = require('serve-favicon'),
            bodyParser = require('body-parser'),
            morgan = require('morgan'),
            compression = require('compression'),
            restreamer = require('./lib/restreamer');

        logger.info("=======================================");
        logger.info("Listening on http://" + (hostname || 'localhost') + ":" + port + "/");
        logger.info("  serving static files from '" + staticDir + "'");
        logger.info("  git release " + gitReleaseName);

        function renderConfig(extra) {
            var options = _.extend(extra, clientOptionsHandler.get());
            options.compilerExplorerOptions = JSON.stringify(options);
            options.root = versionedRootPrefix;
            options.extraBodyClass = extraBodyClass;
            return options;
        }

        var embeddedHandler = function (req, res) {
            staticHeaders(res);
            res.render('embed', renderConfig({embedded: true}));
        };
        const healthCheck = require('./lib/handlers/health-check');
        webServer
            .use(Raven.requestHandler())
            .set('trust proxy', true)
            .set('view engine', 'pug')
            .use('/healthcheck', new healthCheck.HealthCheckHandler().handle) // before morgan so healthchecks aren't logged
            .use(morgan('combined', {stream: logger.stream}))
            .use(compression())
            .get('/', function (req, res) {
                staticHeaders(res);
                res.render('index', renderConfig({embedded: false}));
            })
            .get('/e', embeddedHandler)
            .get('/embed.html', embeddedHandler) // legacy. not a 301 to prevent any redirect loops between old e links and embed.html
            .get('/embed-ro', function (req, res) {
                staticHeaders(res);
                res.render('embed', renderConfig({embedded: true, readOnly: true}));
            })
            .get('/robots.txt', function (req, res) {
                staticHeaders(res);
                res.end('User-agent: *\nSitemap: https://godbolt.org/sitemap.xml');
            })
            .get('/sitemap.xml', function (req, res) {
                staticHeaders(res);
                res.set('Content-Type', 'application/xml');
                res.render('sitemap');
            })
            .use(sFavicon(staticDir + '/favicon.ico'))
            .use('/v', express.static(staticDir + '/v', {maxAge: Infinity, index: false}))
            .use(express.static(staticDir, {maxAge: staticMaxAgeSecs * 1000}));
        if (archivedVersions) {
            // The archived versions directory is used to serve "old" versioned data during updates. It's expected
            // to contain all the SHA-hashed directories from previous versions of Compiler Explorer.
            logger.info("  serving archived versions from", archivedVersions);
            webServer.use('/v', express.static(archivedVersions, {maxAge: Infinity, index: false}));
        }
        webServer
            .use(bodyParser.json({limit: gccProps('bodyParserLimit', '1mb')}))
            .use(bodyParser.text({
                limit: gccProps('bodyParserLimit', '1mb'), type: function () {
                    return true;
                }
            }))
            .use(restreamer())
            .get('/client-options.json', clientOptionsHandler.handler)
            .use('/source', getSource)
            .use('/api', apiHandler.handle)
            .use('/g', shortUrlHandler)
            .post('/compile', compileHandler.handler);
        logger.info("=======================================");

        webServer.use(Raven.errorHandler());

        webServer.on('error', function (err) {
            logger.error('Caught error:', err, "(in web error handler; continuing)");
        });

        webServer.listen(port, hostname);
    })
    .catch(function (err) {
        logger.error("Promise error:", err, "(shutting down)");
        process.exit(1);
    });
