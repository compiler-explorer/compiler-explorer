// Copyright (c) 2012-2016, Matt Godbolt
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

var child_process = require('child_process'),
    temp = require('temp'),
    path = require('path'),
    httpProxy = require('http-proxy'),
    LRU = require('lru-cache'),
    fs = require('fs-extra'),
    Promise = require('promise'), // jshint ignore:line
    Queue = require('promise-queue'),
    asm = require('./asm');

Queue.configure(Promise);
temp.track();

function periodicCleanup() {
    temp.cleanup(function (err, stats) {
        if (err) console.log("Error cleaning directories: ", err);
        // if (stats) console.log("Directory cleanup stats:", stats);
    });
}
var gccProps = null;
var compilerProps = null;
var stubRe = null;
var stubText = null;

function identity(x) {
    return x;
}

function initialise(gccProps_, compilerProps_) {
    gccProps = gccProps_;
    compilerProps = compilerProps_;
    var tempDirCleanupSecs = gccProps("tempDirCleanupSecs", 600);
    console.log("Cleaning temp dirs every " + tempDirCleanupSecs + " secs");
    setInterval(periodicCleanup, tempDirCleanupSecs * 1000);
    asm.initialise(compilerProps);
    stubRe = compilerProps("stubRe");
    stubText = compilerProps("stubText");
}

function Compile(compilers) {
    this.compilersById = {};
    var self = this;
    compilers.forEach(function (compiler) {
        self.compilersById[compiler.id] = compiler;
    });
    this.okOptions = new RegExp(gccProps('optionsWhitelistRe', '.*'));
    this.badOptions = new RegExp(gccProps('optionsBlacklistRe', '(?!)'));
    this.cache = LRU({
        max: gccProps('cacheMb') * 1024 * 1024,
        length: function (n) {
            return JSON.stringify(n).length;
        }
    });
    this.cacheHits = 0;
    this.cacheMisses = 0;
    this.compileQueue = new Queue(gccProps("maxConcurrentCompiles", 1), Infinity);
    this.multiarch = null;
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
Compile.prototype.readFile = Promise.denodeify(fs.readFile);
Compile.prototype.stat = Promise.denodeify(fs.stat);

Compile.prototype.convert6g = function (code) {
    var re = /^[0-9]+\s*\(([^:]+):([0-9]+)\)\s*([A-Z]+)(.*)/;
    var prevLine = null;
    var file = null;
    return code.split('\n').map(function (line) {
        var match = line.match(re);
        if (match) {
            var res = "";
            if (file === null) {
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
    }).filter(identity).join("\n");
};

Compile.prototype.getRemote = function (compiler) {
    var compilerInfo = this.compilersById[compiler];
    if (!compilerInfo) return false;
    if (compilerInfo.exe === null && compilerInfo.remote)
        return compilerInfo.remote;
    return false;
};

Compile.prototype.runCompiler = function (compiler, options, needsMulti) {
    var okToCache = true;
    var env = {
        PATH: process.env.PATH
    };
    if (needsMulti && this.multiarch) {
        env.LIBRARY_PATH = '/usr/lib/' + this.multiarch;
        env.C_INCLUDE_PATH = '/usr/include/' + this.multiarch;
        env.CPLUS_INCLUDE_PATH = '/usr/include/' + this.multiarch;
    }
    var child = child_process.spawn(
        compiler,
        options,
        {detached: process.platform == 'linux', env: env}
    );
    var stdout = "";
    var stderr = "";
    var timeout = setTimeout(function () {
        okToCache = false;
        child.kill();
        stderr += "\nKilled - processing time exceeded";
    }, gccProps("compileTimeoutMs", 100));
    var truncated = false;
    var maxOutput = gccProps("max-error-output", 5000);
    child.stdout.on('data', function (data) {
        if (truncated) return;
        if (stdout.length > maxOutput) {
            stdout += "\n[Truncated]";
            truncated = true;
            child.kill();
            return;
        }
        stdout += data;
    });
    child.stderr.on('data', function (data) {
        if (truncated) return;
        if (stderr.length > maxOutput) {
            stderr += "\n[Truncated]";
            truncated = true;
            child.kill();
            return;
        }
        stderr += data;
    });
    return new Promise(function (resolve, reject) {
        child.on('error', function (e) {
            reject(e);
        });
        child.on('exit', function (code) {
            clearTimeout(timeout);
            // Why is this apparently needed in some cases (e.g. when I used to use this to do getMultiarch)?
            // Without it, I apparently get stdout/stderr callbacks *after* the exit...
            setTimeout(function () {
                resolve({code: code, stdout: stdout, stderr: stderr, okToCache: okToCache});
            }, 0);
        });
        child.stdin.end();
    });
};

Compile.prototype.getMultiarch = function () {
    try {
        var multi = child_process.execSync("gcc -print-multiarch").toString().trim();
        if (multi) {
            console.log("Multiarch: " + multi);
            this.multiarch = multi;
        } else {
            console.log("No multiarch");
        }
    } catch (err) {
        console.log("Unable to get multiarch: " + err);
    }
};

Compile.prototype.objdump = function (outputFilename, result, maxSize, intelAsm) {
    var objDumpCommand = 'objdump -d -C "' + outputFilename + '" -l --insn-width=16';
    if (intelAsm) objDumpCommand += " -M intel";
    return new Promise(function (resolve) {
        child_process.exec(objDumpCommand,
            {maxBuffer: maxSize},
            function (err, data) {
                if (err)
                    data = '<No output: ' + err + '>';
                result.asm = data;
                resolve(result);
            });
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

    // Don't run binary for unsupported compiles, even if we're asked.
    if (filters.binary && !compilerInfo.supportsBinary) {
        delete filters.binary;
    }

    var key = compiler + " | " + source + " | " + options + " | " + JSON.stringify(filters);
    var cached = self.cache.get(key);
    if (cached) {
        self.cacheHits++;
        self.cacheStats();
        return Promise.resolve(cached);
    }
    self.cacheMisses++;

    if (filters.binary && !source.match(stubRe)) {
        source += "\n" + stubText + "\n";
    }

    var tempFileAndDirPromise = Promise.resolve().then(function () {
        return self.newTempDir().then(function (dirPath) {
            var inputFilename = path.join(dirPath, compilerProps("compileFilename"));
            return self.writeFile(inputFilename, source).then(function () {
                return {inputFilename: inputFilename, dirPath: dirPath};
            });
        });
    });
    var compileToAsmPromise = tempFileAndDirPromise.then(function (info) {
        var inputFilename = info.inputFilename;
        var dirPath = info.dirPath;
        var postProcess = [compilerProps("postProcess"), compilerInfo.postProcess];
        postProcess = postProcess.filter(function (x) {
            return x;
        });
        var outputFilename = path.join(dirPath, 'output.s');
        if (compilerInfo.options) {
            options = options.concat(compilerInfo.options.split(" "));
        }
        if (compilerInfo.intelAsm && filters.intel && !filters.binary) {
            options = options.concat(compilerInfo.intelAsm.split(" "));
        }
        var compileToAsm;
        if (!filters.binary) {
            compileToAsm = compilerProps("compileToAsm", compilerInfo.asmFlag).split(" ");
        } else {
            compileToAsm = compilerProps("compileToBinary", "").split(" ");
        }
        if (compilerInfo.isCl) {
            options = options.concat(['/c', '/Fa' + outputFilename, '/Fo' + outputFilename + ".obj"]);
        } else {
            options = options.concat(['-g', compilerInfo.outputFlag, outputFilename]);
        }
        options = options.concat(compileToAsm).concat([inputFilename]);

        var compilerExe = compilerInfo.exe;
        var compilerWrapper = compilerProps("compiler-wrapper");
        if (compilerWrapper) {
            options = [compilerExe].concat(options);
            compilerExe = compilerWrapper;
        }
        var maxSize = gccProps("max-asm-size", 8 * 1024 * 1024);
        options = options.filter(identity);
        return self.runCompiler(compilerExe, options, compilerInfo.needsMulti).then(function (result) {
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
            if (filters.binary && !compilerInfo.isCl) {
                return self.objdump(outputFilename, result, maxSize, filters.intel);
            }
            return self.stat(outputFilename).then(function (stat) {
                if (stat.size >= maxSize) {
                    result.asm = "<No output: generated assembly was too large (" + stat.size + " > " + maxSize + " bytes)>";
                    return result;
                }
                if (postProcess.length) {
                    return new Promise(function (resolve) {
                        child_process.exec('cat "' + outputFilename + '" | ' + postProcess.join(" | "),
                            {maxBuffer: maxSize},
                            function (err, data) {
                                if (err)
                                    data = '<No output: ' + err + '>';
                                result.asm = data;
                                resolve(result);
                            });
                    });
                } else {
                    return self.readFile(outputFilename).then(function (contents) {
                        result.asm = contents.toString();
                        return Promise.resolve(result);
                    });
                }
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
                result.asm = asm.processAsm(result.asm, filters);
                self.cache.set(key, result);
                self.cacheStats();
            } else {
                result.asm = {text: result.asm};
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
    var re = /^\s*#\s*i(nclude|mport)(_next)?\s+["<"](\/|.*\.\.)/;
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
    var pc = (100 * this.cacheHits) / (this.cacheMisses + this.cacheHits);
    console.log("Cache stats: " + this.cacheHits + " hits, " + this.cacheMisses + " misses (" + pc.toFixed(2) +
        "%), LRU has " + this.cache.itemCount + " item(s) totalling " + this.cache.length + " bytes");
};

function CompileHandler() {
    var compileObj = null;

    this.setCompilers = function (compilers) {
        compileObj = new Compile(compilers);
        compileObj.getMultiarch();
    };
    var proxy = httpProxy.createProxyServer({});

    this.handler = function compile(req, res, next) {
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
        var options = req.body.options || '';
        if (source === undefined) {
            return next(new Error("Bad request"));
        }
        options = options.split(' ').filter(identity);
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
    CompileHandler: CompileHandler,
    initialise: initialise
};
