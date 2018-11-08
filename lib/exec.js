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
    logger = require('./logger').logger,
    treeKill = require('tree-kill'),
    execProps = require('./properties').propsFor('execution');

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
    logger.debug({type: "executing", command: command, args: args, env: env});
    // AP: Run Windows-volume executables in winTmp. Otherwise, run in tmpDir (which may be undefined).
    // https://nodejs.org/api/child_process.html#child_process_child_process_spawn_command_args_options
    const child = child_process.spawn(command, args, {
        cwd: options.customCwd ? options.customCwd : (
            (command.startsWith("/mnt") && process.env.wsl) ? process.env.winTmp : process.env.tmpDir
        ),
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

function sandbox(command, args, options) {
    const type = execProps("sandboxType", "docker");
    logger.info(type);
    if (type === "none") {
        logger.info("Sandbox execution (sandbox disabled)", command, args);
        return execute(command, args, options);
    }
    logger.info("Sandbox execution via docker", command, args);
    const execPath = path.dirname(command);
    const execName = path.basename(command);
    return new Promise(function (resolve, reject) {
        logger.debug("Starting sandbox docker container for", command, args);
        let containerId = null;
        let killed = false;
        const timeoutMs = options.timeoutMs || 0;

        function removeContainer() {
            if (containerId) {
                logger.debug("Removing container", containerId);
                execute("docker", ["rm", containerId]);
            } else {
                logger.debug("No container to remove");
            }
        }

        // Start the docker container and detach...
        execute(
            "docker",
            [
                "run",
                "--detach",
                "--cpu-shares=128",
                "--cpu-quota=25000",
                "--ulimit", "nofile=20", // needs at least this to function normally it seems
                "--ulimit", "cpu=3", // hopefully 3 seconds' CPU time
                "--ulimit", "rss=" + (128 * 1024), // hopefully RSS size limit
                "--network=none",
                "--memory=128M",
                "--memory-swap=0",
                "-v" + execPath + ":/home/ce-user:ro",
                "mattgodbolt/compiler-explorer:exec",
                "./" + execName
            ].concat(args),
            {})
            .then(function (result) {
                containerId = result.stdout.trim();
                logger.debug("Docker container id is", containerId);
                if (result.code !== 0) {
                    logger.error("Failed to start docker", result);
                    result.stdout = [];
                    result.stderr = [];
                    if (containerId !== "") {
                        // If we didn't get a container ID, reject...
                        reject(result);
                    }
                }
            })
            .then(function () {
                return execute(
                    "docker",
                    [
                        "wait",
                        containerId
                    ],
                    {
                        timeoutMs: timeoutMs,
                        killChild: function () {
                            logger.debug("Killing docker container", containerId);
                            execute("docker", ["kill", containerId]);
                            killed = true;
                        }
                    });
            })
            .then(function (result) {
                if (result.code !== 0) {
                    logger.error("Failed to wait for", containerId);
                    removeContainer();
                    reject(result);
                    return;
                }
                const returnValue = parseInt(result.stdout);
                return execute(
                    "docker",
                    [
                        "logs",
                        containerId
                    ], options)
                    .then(function (logResult) {
                        if (logResult.code !== 0) {
                            logger.error("Failed to get logs for", containerId);
                            removeContainer();
                            reject(logResult);
                            return;
                        }
                        if (killed)
                            logResult.stdout += "\n### Killed after " + timeoutMs + "ms";
                        logResult.code = returnValue;
                        return logResult;
                    });
            })
            .then(function (result) {
                removeContainer();
                resolve(result);
            })
            .catch(function (err) {
                removeContainer();
                reject(err);
            });
    });
}

module.exports = {
    execute: execute,
    sandbox: sandbox
};
