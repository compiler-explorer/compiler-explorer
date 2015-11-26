#!/usr/bin/env node

// Copyright (c) 2012-2015, Matt Godbolt
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

var nopt = require('nopt'),
    os = require('os'),
    props = require('./lib/properties'),
    compileHandler = require('./lib/compile').compileHandler,
    express = require('express'),
    child_process = require('child_process'),
    path = require('path'),
    fs = require('fs-extra'),
    http = require('http'),
    Promise = require('promise');

var opts = nopt({
    'env': [String],
    'rootDir': [String]
});

var propHierarchy = [
    'defaults',
    opts.env || 'dev',
    os.hostname()];

var rootDir = opts.rootDir || './etc';

props.initialize(rootDir + '/config', propHierarchy);

var port = props.get('gcc-explorer', 'port', 10240);
var staticMaxAgeMs = props.get('gcc-explorer', 'staticMaxAgeMs', 0);

function initializeMemwatch() {
    var memwatch = require('memwatch-next');
    console.log("Initial GC");
    memwatch.gc();
    // Everything else happens a little later to let the initial GC finish.
    setTimeout(function () {
        var lastDiff = new memwatch.HeapDiff();
        memwatch.on('leak', function (info) {
            console.log("Memwatch leak: " + JSON.stringify(info));
        });

        memwatch.on('stats', function (stats) {
            console.log("Memwatch stats: " + JSON.stringify(stats));
        });

        var heapDiffEverySecs = props.get('gcc-explorer', 'gcHeapDiffEverySecs', 0);
        if (heapDiffEverySecs) {
            console.log("Diffing heap every " + heapDiffEverySecs + "s");
            setInterval(function () {
                var diff = lastDiff.end();
                lastDiff = new memwatch.HeapDiff();
                console.log("Memwatch diff from last stats: " + JSON.stringify(diff));
            }, 1000 * heapDiffEverySecs);
        }
        var gcIntervalSecs = props.get("gcc-explorer", "gcIntervalSecs", 0);
        if (gcIntervalSecs) {
            console.log("Forcing a GC every " + gcIntervalSecs + "s");
            setInterval(function () {
                memwatch.gc();
            }, 1000 * gcIntervalSecs);
        }
    }, 1 * 1000);
}

function loadSources() {
    var sourcesDir = "lib/sources";
    var sources = fs.readdirSync(sourcesDir)
        .filter(function (file) {
            return file.match(/.*\.js$/);
        })
        .map(function (file) {
            return require("./" + path.join(sourcesDir, file));
        });
    return sources;
}

var fileSources = loadSources();
var sourceToHandler = {};
fileSources.forEach(function (source) {
    sourceToHandler[source.urlpart] = source;
});

function compareOn(key) {
    return function (xObj, yObj) {
        var x = xObj[key];
        var y = yObj[key];
        if (x < y) return -1;
        if (x > y) return 1;
        return 0;
    };
}

function clientOptionsHandler(compilers, fileSources) {
    var sources = fileSources.map(function (source) {
        return {name: source.name, urlpart: source.urlpart};
    });
    sources = sources.sort(compareOn("name"));
    var options = {
        google_analytics_account: props.get('gcc-explorer', 'clientGoogleAnalyticsAccount', 'UA-55180-6'),
        google_analytics_enabled: props.get('gcc-explorer', 'clientGoogleAnalyticsEnabled', false),
        sharing_enabled: props.get('gcc-explorer', 'clientSharingEnabled', true),
        github_ribbon_enabled: props.get('gcc-explorer', 'clientGitHubRibbonEnabled', true),
        urlshortener: props.get('gcc-explorer', 'clientURLShortener', 'google'),
        gapiKey: props.get('gcc-explorer', 'google-api-key', 'AIzaSyAaz35KJv8DA0ABoime0fEIh32NmbyYbcQ'),
        defaultCompiler: props.get('gcc-explorer', 'defaultCompiler', ''),
        defaultSource: props.get('gcc-explorer', 'defaultSource', ''),
        compilers: compilers,
        language: props.get("gcc-explorer", "language"),
        compileOptions: props.get("gcc-explorer", "options"),
        sources: sources
    };
    var text = "var OPTIONS = " + JSON.stringify(options) + ";";
    return function getClientOptions(req, res) {
        res.set('Content-Type', 'application/javascript');
        res.set('Cache-Control', 'public, max-age=' + staticMaxAgeMs);
        res.end(text);
    };
}

function getSource(req, res, next) {
    var bits = req.url.split("/");
    var handler = sourceToHandler[bits[1]];
    if (!handler) {
        next();
        return;
    }
    var action = bits[2];
    if (action == "list") action = handler.list;
    else if (action == "load") action = handler.load;
    else if (action == "save") action = handler.save;
    else action = null;
    if (action === null) {
        next();
        return;
    }
    action.apply(handler, bits.slice(3).concat(function (err, response) {
        res.set('Cache-Control', 'public, max-age=' + staticMaxAgeMs);
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
                    console.log("Failed " + name + " : " + e + ", retrying");
                    setTimeout(doit, retryMs);
                } else {
                    console.log("Too many retries for " + name + " : " + e);
                    reject(e);
                }
            });
        }

        doit();
    });
}

function configuredCompilers() {
    var exes = props.get("gcc-explorer", "compilers", "/usr/bin/g++").split(":");
    var ndk = props.get('gcc-explorer', 'androidNdk');
    if (ndk) {
        var toolchains = fs.readdirSync(ndk + "/toolchains");
        toolchains.forEach(function (v, i, a) {
            var path = ndk + "/toolchains/" + v + "/prebuilt/linux-x86_64/bin/";
            if (fs.existsSync(path)) {
                var cc = fs.readdirSync(path).filter(function (filename) {
                    return filename.indexOf("g++") != -1;
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
    // Map any named compilers to their executable
    return Promise.all(exes.map(function (name) {
        if (name.indexOf("@") !== -1) {
            var bits = name.split("@");
            var host = bits[0];
            var port = parseInt(bits[1]);
            console.log("Fetching compilers from remote source " + host + ":" + port);
            return retryPromise(function () {
                    return new Promise(function (resolve, reject) {
                        http.get({
                            hostname: host,
                            port: port,
                            path: "/api/compilers"
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
                        });
                    });
                },
                host + ":" + port,
                props.get('gcc-explorer', 'proxyRetries', 20),
                props.get('gcc-explorer', 'proxyRetryMs', 500));
        }
        var base = "compiler." + name;
        var exe = props.get("gcc-explorer", base + ".exe", "");
        if (!exe) {
            return Promise.resolve({id: name, exe: name, name: name});
        }
        return Promise.resolve({
            id: name,
            exe: exe,
            name: props.get("gcc-explorer", base + ".name", name),
            alias: props.get("gcc-explorer", base + ".alias"),
            versionFlag: props.get("gcc-explorer", base + ".versionFlag"),
            is6g: !!props.get("gcc-explorer", base + ".is6g", false),
            intelAsm: props.get("gcc-explorer", base + ".intelAsm", "")
        });
    }));
}

function getCompilerInfo(compilerInfo) {
    if (Array.isArray(compilerInfo)) {
        return Promise.resolve(compilerInfo);
    }
    return new Promise(function (resolve) {
        var compiler = compilerInfo.exe;
        var versionFlag = compilerInfo.versionFlag || '--version';
        child_process.exec(compiler + ' ' + versionFlag, function (err, output) {
            if (err) return resolve(null);
            compilerInfo.version = output.split('\n')[0];
            if (compilerInfo.intelAsm) {
                return resolve(compilerInfo);
            }
            child_process.exec(compiler + ' --target-help', function (err, output) {
                var options = {};
                if (!err) {
                    var splitness = /--?[-a-zA-Z]+( ?[-a-zA-Z]+)/;
                    output.split('\n').forEach(function (line) {
                        var match = line.match(splitness);
                        if (!match) return;
                        options[match[0]] = true;
                    });
                }
                if (options['-masm']) {
                    compilerInfo.intelAsm = "-masm=intel";
                }
                resolve(compilerInfo);
            });
        });
    });
}

function findCompilers() {
    return configuredCompilers()
        .then(function (compilers) {
            return Promise.all(compilers.map(getCompilerInfo));
        })
        .then(function (compilers) {
            compilers = Array.prototype.concat.apply([], compilers);
            compilers = compilers.filter(function (x) {
                return x !== null;
            });
            compilers = compilers.sort(function (x, y) {
                return x.name < y.name ? -1 : x.name > y.name ? 1 : 0;
            });
            console.log("Compilers:");
            compilers.forEach(function (c) {
                console.log(c.id + " : " + c.name + " : " + (c.exe || c.remote));
            });
            return compilers;
        });
}

function apiHandler(compilers) {
    var reply = JSON.stringify(compilers);
    return function apiHandler(req, res, next) {
        var bits = req.url.split("/");
        if (bits.length !== 2 || req.method !== "GET") return next();
        switch (bits[1]) {
            default:
                next();
                break;

            case "compilers":
                res.set('Content-Type', 'application/json');
                res.end(reply);
                break;
        }
    };
}

findCompilers().then(function (compilers) {
    var webServer = express(),
        sFavicon = require('serve-favicon'),
        sStatic = require('serve-static'),
        bodyParser = require('body-parser'),
        logger = require('morgan'),
        compression = require('compression'),
        restreamer = require('connect-restreamer');

    webServer
        .use(logger('combined'))
        .use(compression())
        .use(sFavicon('static/favicon.ico'))
        .use(sStatic('static', {maxAge: staticMaxAgeMs}))
        .use(bodyParser.json())
        .use(restreamer())
        .get('/client-options.js', clientOptionsHandler(compilers, fileSources))
        .use('/source', getSource)
        .use('/api', apiHandler(compilers))
        .post('/compile', compileHandler(compilers));

    // GO!
    console.log("=======================================");
    console.log("Listening on http://" + os.hostname() + ":" + port + "/");
    console.log("=======================================");
    initializeMemwatch();
    webServer.listen(port);
}).catch(function (err) {
    console.log("Error: " + err.stack);
});
