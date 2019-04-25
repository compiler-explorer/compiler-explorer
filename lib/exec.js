// Copyright (c) 2017, Matt Godbolt
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

const child_process = require('child_process'),
    path = require('path'),
    temp = require('temp'),
    fs = require('fs-extra'),
    logger = require('./logger').logger,
    treeKill = require('tree-kill'),
    execProps = require('./properties').propsFor('execution'),
    {splitLines} = require('./utils');

function execute(command, args, options) {
    options = options || {};
    const maxOutput = options.maxOutput || 1024 * 1024;
    const timeoutMs = options.timeoutMs || 0;
    const env = options.env;

    if (options.wrapper) {
        args = args.slice(0); // prevent mutating the caller's arguments
        args.unshift(command);
        command = options.wrapper;

        if (command.startsWith('./')) command = path.join(process.cwd(), command);
    }

    let okToCache = true;
    const cwd = options.customCwd ? options.customCwd : (
        (command.startsWith("/mnt") && process.env.wsl) ? process.env.winTmp : process.env.tmpDir
    );
    logger.debug({type: "executing", command: command, args: args, env: env, cwd: cwd});
    // AP: Run Windows-volume executables in winTmp. Otherwise, run in tmpDir (which may be undefined).
    // https://nodejs.org/api/child_process.html#child_process_child_process_spawn_command_args_options
    const child = child_process.spawn(command, args, {
        cwd: cwd,
        env: env,
        detached: process.platform === 'linux'
    });
    let running = true;

    const kill = options.killChild || function () {
        if (running) treeKill(child.pid);
    };

    const streams = {
        stderr: "",
        stdout: "",
        truncated: false
    };
    let timeout;
    if (timeoutMs) timeout = setTimeout(() => {
        logger.warn("Timeout for", command, args, "after", timeoutMs, "ms");
        okToCache = false;
        kill();
        streams.stderr += "\nKilled - processing time exceeded";
    }, timeoutMs);

    function setupOnError(stream, name) {
        if (stream === undefined) return;
        stream.on('error', err => {
            logger.error('Error with ' + name + ' stream:', err);
        });
    }

    function setupStream(stream, name) {
        if (stream === undefined) return;
        stream.on('data', data => {
            if (streams.truncated) return;
            const newLength = (streams[name].length + data.length);
            if ((maxOutput > 0) && (newLength > maxOutput)) {
                streams[name] = streams[name] + data.slice(0, maxOutput - streams[name].length);
                streams[name] += "\n[Truncated]";
                streams.truncated = true;
                kill();
                return;
            }
            streams[name] += data;
        });
        setupOnError(stream, name);
    }

    setupOnError(child.stdin, 'stdin');
    setupStream(child.stdout, 'stdout');
    setupStream(child.stderr, 'stderr');
    child.on('exit', function (code) {
        logger.debug({type: 'exited', code: code});
        if (timeout !== undefined) clearTimeout(timeout);
        running = false;
    });
    return new Promise(function (resolve, reject) {
        child.on('error', function (e) {
            logger.debug("Error with " + command + " args", args, ":", e);
            reject(e);
        });
        child.on('close', function (code) {
            // Being killed externally gives a NULL error code. Synthesize something different here.
            if (code === null) code = -1;
            if (timeout !== undefined) clearTimeout(timeout);
            const result = {
                code: code,
                stdout: streams.stdout,
                stderr: streams.stderr,
                okToCache: okToCache
            };
            logger.debug({type: "executed", command: command, args: args, result: result});
            resolve(result);
        });
        if (options.input) child.stdin.write(options.input);
        child.stdin.end();
    });
}

function findFile(filename, searchPaths) {
    for (let searchPath of searchPaths) {
        const maybeFile = path.join(searchPath, filename);
        logger.debug(`Looking for ${filename} at ${maybeFile}...`);
        if (fs.existsSync(maybeFile)) {
            logger.debug(`Found ${filename} at ${maybeFile}`);
            return maybeFile;
        }
    }
    throw Error(`Unable to find path for ${filename}`);
}

function findDependentFiles(command) {
    // TODO handle configuration of objdumper
    // TODO handle configuration of search path
    const searchPaths = [
        '/usr/lib/gcc/x86_64-linux-gnu/8/',
        '/usr/lib/gcc/x86_64-linux-gnu/8/../../../../x86_64-linux-gnu/lib/x86_64-linux-gnu/8/',
        '/usr/lib/gcc/x86_64-linux-gnu/8/../../../../x86_64-linux-gnu/lib/x86_64-linux-gnu/',
        '/usr/lib/gcc/x86_64-linux-gnu/8/../../../../x86_64-linux-gnu/lib/../lib/',
        '/usr/lib/gcc/x86_64-linux-gnu/8/../../../x86_64-linux-gnu/8/',
        '/usr/lib/gcc/x86_64-linux-gnu/8/../../../x86_64-linux-gnu/',
        '/usr/lib/gcc/x86_64-linux-gnu/8/../../../../lib/',
        '/lib/x86_64-linux-gnu/8/', '/lib/x86_64-linux-gnu/',
        '/lib/../lib/',
        '/usr/lib/x86_64-linux-gnu/8/',
        '/usr/lib/x86_64-linux-gnu/',
        '/usr/lib/../lib/',
        '/usr/lib/gcc/x86_64-linux-gnu/8/../../../../x86_64-linux-gnu/lib/',
        '/usr/lib/gcc/x86_64-linux-gnu/8/../../../',
        '/lib/',
        '/usr/lib/'];
    return execute("objdump", ["-p", command])
        .then(result => {
            if (result.code !== 0) {
                return result;
            }
            const NEEDED = /^\s+NEEDED\s+(.*)$/;
            return [command].concat(
                splitLines(result.stdout)
                    .map(x => x.match(NEEDED))
                    .filter(x => x)
                    .map(x => findFile(x[1], searchPaths)));
        });
}

function tarFiles(files) {
    // TODO (maybe not here); generate cache filename, check s3 first, only do this if needed
    const tarBall = temp.path(); // TODO remove after
    return execute("tar", [
        "zcf", tarBall, // Create the file
        "--dereference", // deref symlinks
        "-P", // Allow leading /
        "--xform", "s:^.*/::" // "flatten" the hierarchy
    ].concat(files))
        .then(result => {
            if (result.code !== 0) {
                throw Error(`Unable to tar files: ${result.code}`);
            }
            return tarBall;
        });
}

function newTempDir() {
    return new Promise((resolve, reject) => {
        temp.mkdir({prefix: 'compiler-explorer-execution', dir: process.env.tmpDir}, (err, dirPath) => {
            if (err)
                reject(`Unable to open temp file: ${err}`);
            else
                resolve(dirPath);
        });
    });
}

function execScript(script) {
    const handlers = {
        untar: (obj, config) => {
            return execute(
                "tar",
                [
                    "zxf", obj.path,
                    "-C", config.tmpDir
                ]
            );
        },
        exec: (obj, config) => {
            if (!obj.options) obj.options = {};
            // Ensure any env vars we might have don't leak
            const userEnv = obj.options.env || {};
            obj.options.env = {};
            // TODO maybe? set userenv and pass as `--env=<>` options (BUT sanitize so no script attack)
            return execute(
                "/usr/local/bin/firejail", // TODO config of exe (assume not on PATH)
                [
                    // TODO: firejail config
                    "--quiet",
                    // TODO: blacklist lots of directories?
                    "--blacklist=/opt",
                    "--blacklist=/compiler-explorer-image",
                    `--private=${config.tmpDir}`,
                    "--net=none",
                    "--noroot",
                    `--env=LD_LIBRARY_PATH=/home/${process.env.USER}`,
                    "--private-dev",
                    "--private-tmp",
                    "--rlimit-cpu=1",
                    "--hostname=compiler-explorer",
                    "--shell=none",
                    "--", obj.path
                ].concat(obj.args || []),
                {} // NB we don't pass the user options here
            );
        }
    };
    return newTempDir()
        .then(dirPath => {
            let promise = Promise.resolve([]);
            script.map(command => {
                promise = promise.then(results => {
                    return handlers[command.command](command, {tmpDir: dirPath})
                        .then(result => {
                            results.push({command, result});
                            return results;
                        });
                });
            });
            promise = promise.then((results) => {
                fs.remove(dirPath);
                return results;
            });
            return promise;
        });
}

function sandboxFirejailS3(command, args, options) {
    logger.info("Sandbox execution via execution service", command, args);
    return findDependentFiles(command)
        .then(tarFiles)
        .then((tarBall) => execScript([
            {command: "untar", path: tarBall},
            {command: "exec", path: "./" + path.basename(command), arguments: args, options: options}
        ]))
        .then(results => {
            // TODO STILL NOT WORKING HERE
            logger.info("yay", results[1].result);
            if (results.ok) {
                return results[1].result; // the "exec" step
            } else {
                return {
                    code: -1,
                    stdout: "",
                    stderr: results.error,
                    okToCache: false
                };
            }
        })
        .catch(e => {
            logger.error(e);
            return e;
        });
}

function sandbox(command, args, options) {
    const type = execProps("sandboxType");
    if (type === "none") {
        logger.info("Sandbox execution (sandbox disabled)", command, args);
        return execute(command, args, options);
    }
    if (type === "firejail-s3") {
        return sandboxFirejailS3(command, args, options);
    }
    throw Error(`bad sandbox type ${type}`);
}

module.exports = {
    execute: execute,
    sandbox: sandbox,
    execScript: execScript
};
