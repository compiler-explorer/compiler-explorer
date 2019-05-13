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
    fs = require('fs'),
    logger = require('./logger').logger,
    treeKill = require('tree-kill'),
    execProps = require('./properties').propsFor('execution'),
    Graceful = require('node-graceful');

function setupOnError(stream, name) {
    if (stream === undefined) return;
    stream.on('error', err => {
        logger.error('Error with ' + name + ' stream:', err);
    });
}

function executeDirect(command, args, options) {
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
    logger.debug("Execution", {type: "executing", command: command, args: args, env: env, cwd: cwd});
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
        logger.debug("Execution", {type: 'exited', code: code});
        if (timeout !== undefined) clearTimeout(timeout);
        running = false;
    });
    return new Promise(function (resolve, reject) {
        child.on('error', function (e) {
            logger.debug("Execution", "Error with " + command + " args", args, ":", e);
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
            logger.debug("Execution", {type: "executed", command: command, args: args, result: result});
            resolve(result);
        });
        if (options.input) child.stdin.write(options.input);
        child.stdin.end();
    });
}


function sandboxDocker(command, args, options) {
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

function sandboxFirejail(command, args, options) {
    logger.info("Sandbox execution via firejail", command, args);
    const execPath = path.dirname(command);
    const execName = path.basename(command);
    const jailingOptions = [
        '--quiet',
        '--profile=etc/firejail/sandbox.profile',
        // '--debug',
        // TODO: If no cwd, just `--private` '--private=/tmp/private-home', else infer...
        // hack for now: which relies on knowing the execPath will be in the private temp directory.
        `--private=${execPath}`,
        `./${execName}`
    ].concat(args);
    command = 'firejail';
    args = jailingOptions;
    return executeDirect(command, args, options);
}

const sandboxDispatchTable = {
    none: (command, args, options) => {
        logger.info("Sandbox execution (sandbox disabled)", command, args);
        return executeDirect(command, args, options);
    },
    firejail: sandboxFirejail,
    docker: sandboxDocker
};

function sandbox(command, args, options) {
    const type = execProps("sandboxType", "docker");
    const dispatchEntry = sandboxDispatchTable[type];
    if (!dispatchEntry)
        return Promise.reject(`Bad sandbox type ${type}`);
    return dispatchEntry(command, args, options);
}

function initialiseWine() {
    const wine = execProps("wine");
    if (!wine) {
        logger.info("WINE not configured");
        return Promise.resolve();
    }

    const server = execProps("wineServer");
    const firejail = execProps("executionType", "none") === "firejail" ? execProps("firejail") : null;
    const env = applyWineEnv({PATH: process.env.PATH});
    const prefix = env.WINEPREFIX;

    logger.info(`Initialising WINE in ${prefix}`);
    if (!fs.existsSync(prefix) || !fs.statSync(prefix).isDirectory()) {
        logger.info(`Creating directory ${prefix}`);
        fs.mkdirSync(prefix);
    }

    logger.info(`Killing any pre-existing wine-server`);
    let result = child_process.execSync(`${server} -k || true`, {env: env});
    logger.info(`Result: ${result}`);
    logger.info(`Waiting for any pre-existing server to stop...`);
    result = child_process.execSync(`${server} -w`, {env: env});
    logger.info(`Result: ${result}`);

    // Once the server is running, we fire up a single process through to:
    // * test that WINE works
    // * be the "babysitter". WINE loves to make lots of long-lived processes
    //   on the first execution, and this upsets things like firejail, which won't
    //   exit until the last child process dies. That makes compilations hang.
    // We wait until the process has printed out some known good text, but don't wait
    // for it to exit (it won't).

    let waitForOk, wineServer;
    if (firejail) {
        logger.info(`Starting a new, firejailed, long-lived wineserver ${server}`);
        wineServer = child_process.spawn(
            firejail,
            [
                "--quiet",
                "--profile=etc/firejail/wine.profile",
                "--private",
                server,
                "-p"
            ],
            {
                env: env,
                detached: true,
                stdio: [process.stdin, process.stdout, process.stderr]
            });
        logger.info(`firejailed wineserver pid=${wineServer.pid}`);
        logger.info(`Initialising WINE babysitter with ${wine}...`);
        waitForOk = child_process.spawn(
            firejail,
            [
                "--quiet",
                "--profile=etc/firejail/wine.profile",
                wine,
                "cmd"],
            {env: env, detached: true});
    } else {
        logger.info(`Starting a new, long-lived wineserver ${server}`);
        wineServer = child_process.spawn(server, ["-p"], {
            env: env,
            detached: true,
            stdio: [process.stdin, process.stdout, process.stderr]
        });
        logger.info(`wineserver pid=${wineServer.pid}`);
        logger.info(`Initialising WINE babysitter with ${wine}...`);
        waitForOk = child_process.spawn(
            wine,
            ["cmd"],
            {env: env, detached: true});
    }

    Graceful.on('exit', () => {
        const waitingPromises = [];

        function waitForExit(process, name) {
            return new Promise((resolve) => {
                process.on('close', () => {
                    logger.info(`Process '${name}' closed`);
                    resolve();
                });
            });
        }

        if (waitForOk && !waitForOk.killed) {
            logger.info('Shutting down WINE babysitter');
            waitForOk.kill();
            if (waitForOk.killed) {
                waitingPromises.push(waitForExit(waitForOk, "WINE babysitter"));
            }
            waitForOk = null;
        }
        if (wineServer && !wineServer.killed) {
            logger.info('Shutting down WINE server');
            wineServer.kill();
            if (wineServer.killed) {
                waitingPromises.push(waitForExit(wineServer, "WINE server"));
            }
            wineServer = null;
        }
        return Promise.all(waitingPromises);
    });
    return new Promise((resolve, reject) => {
        setupOnError(waitForOk.stdin, "stdin");
        setupOnError(waitForOk.stdout, "stdout");
        setupOnError(waitForOk.stderr, "stderr");
        const magicString = "!!EVERYTHING IS WORKING!!";
        waitForOk.stdin.write(`echo ${magicString}`);
        waitForOk.stdin.end();

        let output = "";
        waitForOk.stdout.on('data', data => {
            logger.info(`Output from wine process: ${data.toString().trim()}`);
            output += data;
            if (output.includes(magicString)) {
                resolve();
            }
        });
        waitForOk.stderr.on('data',
            data => logger.info(`stderr output from wine process: ${data.toString().trim()}`));
        waitForOk.on('error', function (e) {
            logger.error(`WINE babysitting process exited with ${e}`);
            reject(e);
        });
        waitForOk.on('close', function (code) {
            logger.info(`WINE test exited with code ${code}`);
            reject();
        });
    });
}

function applyWineEnv(env) {
    const prefix = execProps("winePrefix");
    env = env || {};
    // Force use of wine vcruntime (See change 45106c382)
    env.WINEDLLOVERRIDES = "vcruntime140=b";
    env.WINEDEBUG = "-all";
    env.WINEPREFIX = prefix;
    return env;
}

function needsWine(command) {
    return command.match(/\.exe$/i) && process.platform === 'linux';
}

function executeWineDirect(command, args, options) {
    options = options || {};
    options.env = applyWineEnv(options.env);
    args.unshift(command);
    return executeDirect(execProps("wine"), args, options);
}

function executeFirejail(command, args, options) {
    options = options || {};
    const firejail = execProps("firejail");
    const baseOptions = ['--debug'];
    if (needsWine(command)) {
        logger.debug("WINE execution via firejail", command, args);
        options.env = applyWineEnv(options.env);
        args.unshift(command);
        command = execProps("wine");
        baseOptions.push('--profile=etc/firejail/wine.profile');
        delete options.customCwd; // TODO!
        baseOptions.push(command);
        return executeDirect(firejail, baseOptions.concat(args), options);
    }

    logger.debug("Regular execution via firejail", command, args);
    baseOptions.push('--profile=etc/firejail/execute.profile');
    if (options.customCwd) {
        baseOptions.push(`--private=${options.customCwd}`);
        args = args.map(opt => opt.replace(options.customCwd, "."));
        delete options.customCwd;
    } else {
        baseOptions.push('--private');
    }
    baseOptions.push(command);
    return executeDirect(firejail, baseOptions.concat(args), options);
}

function executeNone(command, args, options) {
    if (needsWine(command)) {
        return executeWineDirect(command, args, options);
    }
    return executeDirect(command, args, options);
}

const executeDispatchTable = {
    none: executeNone,
    firejail: executeFirejail
};

function execute(command, args, options) {
    const type = execProps("executionType", "none");
    const dispatchEntry = executeDispatchTable[type];
    if (!dispatchEntry)
        return Promise.reject(`Bad sandbox type ${type}`);
    return dispatchEntry(command, args, options);
}

module.exports = {
    execute,
    sandbox,
    initialiseWine
};
