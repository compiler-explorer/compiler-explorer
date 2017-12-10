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
if ((child_process.execSync('uname -a').toString().indexOf('Microsoft') > -1) && (opts.wsl))
    process.env.wsl = true;

// AP: Allow setting of tmpDir (used in lib/base-compiler.js & lib/exec.js) through
// opts. WSL requires a tmpDir as it can't see Linux volumes so set default to c:\tmp.
if (opts.tmpDir)
    process.env.tmpDir = opts.tmpDir;
else if (process.env.wsl)
    process.env.tmpDir = "/mnt/c/tmp";


// Set default values for omitted arguments
var rootDir = opts.rootDir || './etc';
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
    asm_doc_api = require('./lib/asm-docs-api'),
    google = require('./lib/google');

// Instantiate a function to access records concerning "compiler-explorer" 
// in hidden object props.properties
var ceProps = props.propsFor("compiler-explorer");

const languages = require('./lib/languages').list;

// Instantiate a function to access records concerning the chosen language
// in hidden object props.properties
var compilerPropsFuncsL = {};
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

var staticMaxAgeSecs = ceProps('staticMaxAgeSecs', 0);
let extraBodyClass = ceProps('extraBodyClass', '');

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
var compileHandler = new CompileHandler(ceProps, compilerPropsL);
var apiHandler = new ApiHandler(compileHandler);

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
    const sources = fileSources.map(function (source) {
        return {name: source.name, urlpart: source.urlpart};
    }).sort(compareOn("name"));
    // sort source file alphabetically

    var supportsBinary = compilerPropsAT(languages, res => !!res, "supportsBinary", true);
    var supportsExecute = supportsBinary && !!compilerPropsAT(languages, (res, lang) => supportsBinary[lang.id] && !!res, "supportsExecute", true);
    var libs = {};

    var baseLibs = compilerPropsA(languages, "libs");
    _.each(baseLibs, function (forLang, lang) {
        if (lang && forLang) {
            libs[lang] = {};
            _.each(forLang.split(':'), function (lib) {
                libs[lang][lib] = {name: compilerPropsL(lang, 'libs.' + lib + '.name')};
                libs[lang][lib].versions = {};
                var listedVersions = compilerPropsL(lang, "libs." + lib + '.versions');
                if (listedVersions) {
                    _.each(listedVersions.split(':'), function (version) {
                        libs[lang][lib].versions[version] = {};
                        libs[lang][lib].versions[version].version = compilerPropsL(lang, "libs." + lib + '.versions.' + version + '.version');
                        libs[lang][lib].versions[version].path = [];
                        var listedIncludes = compilerPropsL(lang, "libs." + lib + '.versions.' + version + '.path');
                        if (listedIncludes) {
                            _.each(listedIncludes.split(':'), function (path) {
                                libs[lang][lib].versions[version].path.push(path);
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
    });
    var options = {
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
    let exes = compilerPropsAT(languages, exs => {
        return exs.split(":").filter(_.identity);
    }, "compilers", "");

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
            lang: langId
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

            const compilerExes = props(langId, 'compilers', '').split(":").filter(_.identity);
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
        .then(compileHandler.setCompilers)
        .then(compilers => _.filter(compilers, compiler => !!compiler))
        .then(ensureDistinct)
        .then(compilers => compilers.sort(compareOn("name")));
}

function ApiHandler(compileHandler) {
    this.compilers = [];
    this.compileHandler = compileHandler;
    this.setCompilers = function (compilers) {
        this.compilers = compilers;
    };
    this.handler = express.Router();
    this.handler.use((req, res, next) => {
        res.header("Access-Control-Allow-Origin", "*");
        res.header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept");
        next();
    });
    this.handler.get('/compilers', _.bind((req, res) => {
        if (req.accepts(['text', 'json']) === 'json') {
            res.set('Content-Type', 'application/json');
            res.end(JSON.stringify(this.compilers));
        } else {
            res.set('Content-Type', 'text/plain');
            var title = 'Compiler Name';
            var maxLength = _.max(_.pluck(_.pluck(this.compilers, 'id').concat([title]), 'length'));
            res.write(utils.padRight(title, maxLength) + ' | Description\n');
            res.end(_.map(this.compilers, compiler => {
                return utils.padRight(compiler.id, maxLength) + ' | ' + compiler.name;
            }).join("\n"));
        }
    }, this));
    this.handler.get('/asm/:opcode', asm_doc_api.asmDocsHandler);
    this.handler.param('compiler', _.bind((req, res, next, compilerName) => {
        req.compiler = compilerName;
        next();
    }, this));
    this.handler.post('/compiler/:compiler/compile', this.compileHandler.handler);
}

function healthcheckHandler(req, res, next) {
    res.end("Everything is awesome");
}

function shortUrlHandler(req, res, next) {
    const resolver = new google.ShortLinkResolver(aws.getConfig('googleApiKey'));
    const bits = req.url.split("/");
    if (bits.length !== 2 || req.method !== "GET") return next();
    const googleUrl = `http://goo.gl/${encodeURIComponent(bits[1])}`;
    resolver.resolve(googleUrl)
        .then(resultObj => {
            var parsed = url.parse(resultObj.longUrl);
            var allowedRe = new RegExp(ceProps('allowedShortUrlHostRe'));
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
            process.env.NEW_RELIC_LABELS = 'Languages:' + _.map(languages, languages => languages.name);
            require('newrelic');
            logger.info('New relic configured with license', newRelicLicense);
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
        }

        onCompilerChange(compilers);

        var rescanCompilerSecs = ceProps('rescanCompilerSecs', 0);
        if (rescanCompilerSecs) {
            logger.info(`Rescanning compilers every ${rescanCompilerSecs} secs`);
            setInterval(() => findCompilers().then(onCompilerChange),
                rescanCompilerSecs * 1000);
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
        webServer
            .use(Raven.requestHandler())
            .set('trust proxy', true)
            .set('view engine', 'pug')
            .use('/healthcheck', healthcheckHandler) // before morgan so healthchecks aren't logged
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
            .use(bodyParser.json({limit: ceProps('bodyParserLimit', '1mb')}))
            .use(bodyParser.text({
                limit: ceProps('bodyParserLimit', '1mb'), type: function () {
                    return true;
                }
            }))
            .use(restreamer())
            .get('/client-options.json', clientOptionsHandler.handler)
            .use('/source', getSource)
            .use('/api', apiHandler.handler)
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
