// Copyright (c) 2012-2015, Matt Godbold
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

var props = require('./properties'),
    child_process = require('child_process'),
    temp = require('temp'),
    path = require('path'),
    httpProxy = require('http-proxy'),
    LRU = require('lru-cache'),
    fs = require('fs-extra'),
    Promise = require('promise'),
    Queue = require('promise-queue');

Queue.configure(Promise);
temp.track();

function periodicCleanup() {
    temp.cleanup(function(err, stats) {
        if (err) console.log("Error cleaning directories: ", err);
        if (stats) console.log("Directory cleanup stats:", stats);
    });
}
setTimeout(function(){
    // property library isn't initialized until after the first tick
    var tempDirCleanupSecs = props.get("gcc-explorer", "tempDirCleanupSecs", 600);
    console.log("Cleaning temp dirs every " + tempDirCleanupSecs + " secs");
    setInterval(periodicCleanup, tempDirCleanupSecs * 1000);
}, 0);

function Compile(compilers) {
    this.compilersById = {};
    var self = this;
    compilers.forEach(function (compiler) {
        self.compilersById[compiler.id] = compiler;
    });
    this.okOptions = new RegExp(props.get('gcc-options', 'whitelistRe', '.*'));
    this.badOptions = new RegExp(props.get('gcc-options', 'blacklistRe'));
    this.cache = LRU({
        max: props.get('gcc-explorer', 'cacheMb') * 1024 * 1024,
        length: function (n) {
            return n.asm.length;
        }
    });
    this.cacheHits = 0;
    this.cacheMisses = 0;
    this.compileQueue = new Queue(props.get("gcc-explorer", "maxConcurrentCompiles", 1), Infinity);
}

Compile.prototype.newTempDir = function () {
    return new Promise(function (resolve, reject) {
        temp.mkdir('gcc-explorer-compiler', function (err, dirPath) {
            if (err)
                reject("Unable to open temp file: " + err);
            else
                resolve(dirPath);
        });
    });
};

Compile.prototype.writeFile = Promise.denodeify(fs.writeFile);
Compile.prototype.stat = Promise.denodeify(fs.stat);

Compile.prototype.convert6g = function (code) {
    var re = /^[0-9]+\s*\(([^:]+):([0-9]+)\)\s*([A-Z]+)(.*)/;
    var prevLine = null;
    var file = null
    return code.split('\n').map(function (line) {
        var match = line.match(re);
        if (match) {
            var res = "";
            if (file == null) {
                res += "\t.file 1 \"" + match[1] + "\"\n";
                file = match[1];
            }
            if (prevLine != match[2]) {
                res += "\t.loc 1 " + match[2] + "\n";
                prevLine = match[2];
            }
            return res + "\t" + match[3].toLowerCase() + match[4];
        } else
            return null;
    }).filter(function (x) {
        return x;
    }).join("\n");
};

Compile.prototype.getRemote = function (compiler) {
    var compilerInfo = this.compilersById[compiler];
    if (!compilerInfo) return false;
    if (compilerInfo.exe === null && compilerInfo.remote)
        return compilerInfo.remote;
    return false;
};

Compile.prototype.runCompiler = function (compiler, options) {
    var okToCache = true;
    var child = child_process.spawn(
        compiler,
        options,
        {detached: true}
    );
    var stdout = "";
    var stderr = "";
    var timeout = setTimeout(function () {
        okToCache = false;
        child.kill();
        stderr += "\nKilled - processing time exceeded";
    }, props.get("gcc-explorer", "compileTimeoutMs", 100));
    child.stdout.on('data', function (data) {
        stdout += data;
    });
    child.stderr.on('data', function (data) {
        stderr += data;
    });
    return new Promise(function (resolve, reject) {
        child.on('error', function (e) {
            reject(e);
        });
        child.on('exit', function (code) {
            clearTimeout(timeout);
            resolve({code: code, stdout: stdout, stderr: stderr, okToCache: okToCache});
        });
        child.stdin.end();
    });
};

Compile.prototype.compile = function (source, compiler, options, filters) {
    var self = this;
    var optionsError = self.checkOptions(options);
    if (optionsError) return Promise.reject(optionsError);
    var sourceError = self.checkSource(source);
    if (sourceError) return Promise.reject(sourceError);

    var compilerInfo = self.compilersById[compiler];
    if (!compilerInfo) {
        return Promise.reject("Bad compiler " + compiler);
    }

    var key = compiler + " | " + source + " | " + options + " | " + filters.intel;
    var cached = self.cache.get(key);
    if (cached) {
        self.cacheHits++;
        self.cacheStats();
        return Promise.resolve(cached);
    }
    self.cacheMisses++;

    var tempFileAndDirPromise = Promise.resolve().then(function () {
        return self.newTempDir().then(function (dirPath) {
            var inputFilename = path.join(dirPath, props.get("gcc-explorer", "compileFilename"));
            return self.writeFile(inputFilename, source).then(function () {
                return {inputFilename: inputFilename, dirPath: dirPath};
            });
        });
    });
    var compileToAsmPromise = tempFileAndDirPromise.then(function (info) {
        var inputFilename = info.inputFilename;
        var dirPath = info.dirPath;
        var postProcess = props.get("gcc-explorer", "postProcess");
        var outputFilename = path.join(dirPath, 'output.S');
        if (compilerInfo.intelAsm && filters.intel) {
            options = options.concat(compilerInfo.intelAsm.split(" "));
        }
        var compileToAsm = props.get("gcc-explorer", "compileToAsm", "-S").split(" ");
        options = options.concat(['-g', '-o', outputFilename]).concat(compileToAsm).concat([inputFilename]);

        var compilerExe = compilerInfo.exe;
        var compilerWrapper = props.get("gcc-explorer", "compiler-wrapper");
        if (compilerWrapper) {
            options = [compilerExe].concat(options);
            compilerExe = compilerWrapper;
        }
        var maxSize = props.get("gcc-explorer", "max-asm-size", 8 * 1024 * 1024);
        return self.runCompiler(compilerExe, options).then(function (result) {
            result.dirPath = dirPath;
            if (result.code !== 0) {
                result.asm = "<Compilation failed>";
                return result;
            }
            if (compilerInfo.is6g) {
                result.asm = self.convert6g(result.stdout);
                result.stdout = "";
                return Promise.resolve(result);
            }
            return self.stat(outputFilename).then(function (stat) {
                if (stat.size >= maxSize) {
                    result.asm = "<No output: generated assembly was too large (" + stat.size + " > " + maxSize + " bytes)>";
                    return result;
                }
                return new Promise(function (resolve) {
                    child_process.exec('cat "' + outputFilename + '" | ' + postProcess,
                        {maxBuffer: maxSize},
                        function (err, data) {
                            if (err)
                                data = '<No output: ' + err + '>';
                            result.asm = data;
                            resolve(result);
                        });
                });
            }, function () {
                result.asm = "<No output file>";
                return result;
            });
        });
    });

    return self.compileQueue.add(function () {
        return compileToAsmPromise.then(function (result) {
            if (result.dirPath) {
                fs.remove(result.dirPath);
                result.dirPath = undefined;
            }
            if (result.okToCache) {
                self.cache.set(key, result);
                self.cacheStats();
            }
            return result;
        });
    });
};

Compile.prototype.checkOptions = function (options) {
    var error = [];
    var self = this;
    options.forEach(function (option) {
        if (!option.match(self.okOptions) || option.match(self.badOptions)) {
            error.push(option);
        }
    });
    if (error.length > 0) return "Bad options: " + error.join(", ");
    return null;
};

Compile.prototype.checkSource = function (source) {
    var re = /^\s*#include(_next)?\s+["<"](\/|.*\.\.)/;
    var failed = [];
    source.split('\n').forEach(function (line, index) {
        if (line.match(re)) {
            failed.push("<stdin>:" + (index + 1) + ":1: no absolute or relative includes please");
        }
    });
    if (failed.length > 0) return failed.join("\n");
    return null;
};

Compile.prototype.cacheStats = function () {
    console.log("Cache stats: " + this.cacheHits + " hits, " + this.cacheMisses + " misses");
};

function compileHandler(compilers) {
    var compileObj = new Compile(compilers);
    var proxy = httpProxy.createProxyServer({});

    return function compile(req, res, next) {
        var compiler = req.body.compiler;
        var remote = compileObj.getRemote(compiler);
        if (remote) {
            proxy.web(req, res, {target: remote}, function (e) {
                console.log("Proxy error: ", e);
                next(e);
            });
            return;
        }
        var source = req.body.source;
        if (!source || typeof(req.body.options) !== "string") {
            return next(new Error("Bad request"));
        }
        var options = req.body.options.split(' ').filter(function (x) {
            return x !== "";
        });
        var filters = req.body.filters;
        compileObj.compile(source, compiler, options, filters).then(
            function (result) {
                res.set('Content-Type', 'application/json');
                res.end(JSON.stringify(result));
            },
            function (error) {
                console.log("Error: " + error.stack);
                if (typeof(error) !== "string") {
                    error = "Internal GCC explorer error: " + error.toString();
                }
                res.end(JSON.stringify({code: -1, stderr: error}));
            }
        );
    };
}

module.exports = {
    compileHandler: compileHandler
};
