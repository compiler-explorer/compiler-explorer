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
        if (err) {
            res.end(JSON.stringify({err: err}));
        } else {
            res.end(JSON.stringify(response));
        }
    }));
}

function getCompilerExecutables() {
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
    return exes;
}

function clientOptionsHandler(compilers, fileSources) {
    var sources = fileSources.map(function (source) {
        return {name: source.name, urlpart: source.urlpart};
    });
    sources = sources.sort(compareOn("name"));
    var options = {
        google_analytics_account: props.get('gcc-explorer', 'clientGoogleAnalyticsAccount', 'UA-55180-6'),
        google_analytics_enabled: props.get('gcc-explorer', 'clientGoogleAnalyticsEnabled', true),
        sharing_enabled: props.get('gcc-explorer', 'clientSharingEnabled', true),
        github_ribbon_enabled: props.get('gcc-explorer', 'clientGitHubRibbonEnabled', true),
        urlshortener: props.get('gcc-explorer', 'clientURLShortener', 'google'),
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
        res.end(text);
    };
}

function getCompilerInfo(compiler) {
    return new Promise(function (resolve, reject) {
        child_process.exec(compiler + ' --version', function (err, output) {
            if (err) return resolve(null);
            var version = output.split('\n')[0];
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
                resolve({exe: compiler, version: version, supportedOpts: options});
            });
        });
    });
}

function findCompilers() {
    var compilers = getCompilerExecutables().map(getCompilerInfo);
    return Promise.all(compilers).then(function (compilers) {
        compilers = compilers.filter(function (x) {
            return x !== null
        });
        compilers = compilers.sort(function (x, y) {
            return x.version < y.version ? -1 : x.version > y.version ? 1 : 0;
        });
        return compilers;
    });
}

findCompilers().then(function (compilers) {
    var webServer = express(),
        sFavicon = require('serve-favicon'),
        sStatic = require('serve-static'),
        bodyParser = require('body-parser'),
        logger = require('morgan');

    webServer
        .use(logger('combined'))
        .use(sFavicon('static/favicon.ico'))
        .use(sStatic('static'))
        .use(bodyParser.urlencoded({extended: true}))
        .get('/client-options.js', clientOptionsHandler(compilers, fileSources))
        .use('/source', getSource)
        .post('/compile', compileHandler(compilers));

    // GO!
    console.log("=======================================");
    console.log("Listening on http://" + os.hostname() + ":" + port + "/");
    console.log("=======================================");
    webServer.listen(port);
}).catch(function (err) {
    console.log("Error: " + err.stack);
});
