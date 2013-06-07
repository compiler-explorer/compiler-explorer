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

var nopt = require('nopt'),
    os = require('os'),
    props = require('./lib/properties'),
    querystring = require('querystring'),
    connect = require('connect'),
    child_process = require('child_process'),
    temp = require('temp'),
    path = require('path'),
    async = require('async'),
    LRU = require('lru-cache'),
    fs = require('fs');

var opts = nopt({
        'env': [String],
        'rootDir': [String]});

var propHierarchy = [
    'defaults', 
    opts.env || 'dev',
    os.hostname()];

var rootDir = opts.rootDir || './etc';

props.initialize(rootDir + '/config', propHierarchy);

var port = props.get('gcc-explorer', 'port', 10240);
var compilers = [];
var compilersByExe = {};

var cache = LRU({ 
    max: props.get('gcc-explorer', 'cacheMb') * 1024 * 1024, 
    length: function(n) { return n.length; }
});
var cacheHits = 0;
var cacheMisses = 0;

var compileQueue = async.queue(function(task, callback) {
    console.log("Compiling, queue size " + compileQueue.length());
    task.task(callback);
}, props.get("gcc-explorer", "maxConcurrentCompiles", 1));
compileQueue.drain = function() {
    console.log("Compile queue empty");
};
compileQueue.saturated = function() {
    console.log("Compile queue full");
};


function checkOptions(options) {
    var okOptions = new RegExp(props.get('gcc-options', 'whitelistRe', '.*'));
    var badOptions = new RegExp(props.get('gcc-options', 'blacklistRe'));
    var error = [];
    options.forEach(function(option) {
        if (!option.match(okOptions) || option.match(badOptions)) {
            error.push(option);
        }
    });
    if (error.length > 0) return "Bad options: " + error.join(", ");
    return null;
}

function checkSource(source) {
    var re = /^\s*#include(_next)?\s+["<"](\/|.*\.\.)/;
    var failed = [];
    source.split('\n').forEach(function(line, index) {
        if (line.match(re)) {
            failed.push("<stdin>:" + (index + 1) + ":1: no absolute or relative includes please");
        }
    });
    if (failed.length > 0) return failed.join("\n");
    return null;
}

function cacheStats() {
    console.log("Cache stats: " + cacheHits + " hits, " + cacheMisses + " misses");
}

function compile(req, res) {
    var source = req.body.source;
    var compiler = req.body.compiler;
    if (getCompilerExecutables().indexOf(compiler) < 0) {
        return res.end(JSON.stringify({code: -1, stderr: "bad compiler " + compiler}));
    }
    var compilerInfo = compilersByExe[compiler];
    if (!compilerInfo) {
        return res.end(JSON.stringify({code: -1, stderr: "bad compiler " + compiler}));
    }
    var options = req.body.options.split(' ').filter(function(x){return x!=""});
    var filters = req.body.filters;
    var optionsErr = checkOptions(options);
    if (optionsErr) {
        return res.end(JSON.stringify({code: -1, stderr: optionsErr}));
    }
    var sourceErr = checkSource(source);
    if (sourceErr) {
        return res.end(JSON.stringify({code: -1, stderr: sourceErr}));
    }

    var key = compiler + " | " + source + " | " + options + " | " + filters["intel"];
    var cached = cache.get(key);
    if (cached) {
        cacheHits++;
        cacheStats();
        res.end(cached);
        return;
    }
    cacheMisses++;

    var compileTask = function(taskFinished) {
        temp.mkdir('gcc-explorer-compiler', function(err, dirPath) {
            if (err) {
                return res.end(JSON.stringify({code: -1, stderr: "Unable to open temp file: " + err}));
            }
            var postProcess = props.get("gcc-explorer", "postProcess");
            var inputFilename = path.join(dirPath, props.get("gcc-explorer", "compileFilename"));
            var outputFilename = path.join(dirPath, 'output.S');
            if (compilerInfo.supportedOpts['-masm']) {
                var syntax = '-masm=att'; // default at&t
                if (filters["intel"] == "true") syntax = '-masm=intel';
                options = options.concat([syntax]);
            }
            options = options.concat(['-g', '-o', outputFilename, '-S', inputFilename]);
            var file = fs.openSync(inputFilename, "w");
            fs.writeSync(file, source);
            fs.closeSync(file);
            var okToCache = true;
            var compilerWrapper = props.get("gcc-explorer", "compiler-wrapper");
            if (compilerWrapper) {
                options = [compiler].concat(options);
                compiler = compilerWrapper;
            }
            var child = child_process.spawn(
                compiler,
                options,
                {detached: true}
                );
            var stdout = "";
            var stderr = "";
            var timeout = setTimeout(function() {
                okToCache = false;
                child.kill();
                stderr += "\nKilled - processing time exceeded";
            }, props.get("gcc-explorer", "compileTimeoutMs", 100));
            child.stdout.on('data', function (data) { stdout += data; });
            child.stderr.on('data', function (data) { stderr += data; });
            child.on('exit', function (code) {
                clearTimeout(timeout);
                child_process.exec('cat "' + outputFilename + '" | ' + postProcess,
                    { maxBuffer: props.get("gcc-explorer", "max-asm-size", 8 * 1024 * 1024) },
                    function(err, filt_stdout, filt_stderr) {
                        var data = filt_stdout;
                        if (err) {
                            data = '<No output: ' + err + '>';
                        }

                        var result = JSON.stringify({
                            stdout: stdout,
                            stderr: stderr,
                            asm: data,
                            code: code });
                        if (okToCache) {
                            cache.set(key, result);
                            cacheStats();
                        }
                        res.end(result);
                        fs.unlink(outputFilename, function() { 
                            fs.unlink(inputFilename, 
                                function() { fs.rmdir(dirPath); });
                        });
                        taskFinished();
                    });
            });
            child.stdin.end();
        });
    };

    compileQueue.push({task: compileTask});
}

function loadSources() {
    var sourcesDir = "lib/sources";
    var sources = fs.readdirSync(sourcesDir)
        .filter(function(file) { return file.match(/.*\.js$/); })
        .map(function(file) { return require("./" + path.join(sourcesDir, file)); });
    return sources;
}

var fileSources = loadSources();
var sourceToHandler = {};
fileSources.forEach(function(source) { sourceToHandler[source.urlpart] = source; });

function compareOn(key) {
    return function(xObj, yObj) {
        var x = xObj[key];
        var y = yObj[key];
        if (x < y) return -1;
        if (x > y) return 1;
        return 0;
    };
}

function getSources(req, res) {
    var sources = fileSources.map(function(source) { 
        return {name: source.name, urlpart: source.urlpart};
    });
    res.end(JSON.stringify(sources.sort(compareOn("name"))));
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
    if (action == null) {
        next();
        return;
    }
    action.apply(handler, bits.slice(3).concat(function(err, response) {
        if (err) {
            res.end(JSON.stringify({err: err}));
        } else {
            res.end(JSON.stringify(response));
        }}));
}

function getCompilerExecutables() {
    return props.get("gcc-explorer", "compilers", "/usr/bin/g++").split(":");
}

function getCompilers(req, res) {
    res.end(JSON.stringify(compilers));
}

function findCompilers() {
    async.map(getCompilerExecutables(),
        function (compiler, callback) {
            fs.stat(compiler, function(err, result) {
                if (err) return callback(null, null);
                child_process.exec(compiler + ' --version', function(err, output) {
                    if (err) return callback(null, null);
                    var version = output.split('\n')[0];
                    child_process.exec(compiler + ' --target-help', function(err, output) {
                        var options = {};
                        if (!err) {
                            var splitness = /--?[-a-zA-Z]+( ?[-a-zA-Z]+)/;
                            output.split('\n').forEach(function(line) {
                                var match = line.match(splitness);
                                if (!match) return;
                                options[match[0]] = true;
                            });
                        }
                        callback(null, {exe: compiler, version: version, supportedOpts: options});
                    });
                });
            });
        },
        function (err, all) {
            all = all.filter(function(x){return x!=null;});
            compilers = all.sort(function(x,y){return x.version < y.version ? -1 : x.version > y.version ? 1 : 0;});
            compilersByExe = {};
            compilers.forEach(function(compiler) {
                compilersByExe[compiler.exe] = compiler;
            });
        }
    );
}

findCompilers();

// WebServer.
var webServer = connect();
webServer
    .use(connect.logger())
    .use(connect.favicon('static/favicon.ico'))
    .use(connect.static('static'))
    .use(connect.bodyParser())
    .use('/sources', getSources)
    .use('/source', getSource)
    .use('/compilers', getCompilers)
    .use('/compile', compile);


// GO!
console.log("=======================================");
console.log("Listening on http://" + os.hostname() + ":" + port + "/");
console.log("=======================================");
webServer.listen(port);
