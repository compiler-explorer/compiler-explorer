// Copyright (c) 2017, Compiler Explorer Authors
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

import child_process from 'child_process';
import fs from 'fs';
import path from 'path';

import Graceful from 'node-graceful';
import treeKill from 'tree-kill';
import _ from 'underscore';

import { logger } from './logger';
import { propsFor } from './properties';

const execProps = propsFor('execution');

function setupOnError(stream, name) {
    if (stream === undefined) return;
    stream.on('error', err => {
        logger.error(`Error with ${name} stream:`, err);
    });
}

function executeDirect(command, args, options, filenameTransform) {
    // filename transform is expected to have been pre-applied by the caller.
    // it is passed through here only so clients can see it in the result.
    filenameTransform = filenameTransform || (x => x);
    options = options || {};
    const maxOutput = options.maxOutput || 1024 * 1024;
    const timeoutMs = options.timeoutMs || 0;
    const env = options.env ? options.env : _.clone(process.env);

    if (options.ldPath) {
        env.LD_LIBRARY_PATH = options.ldPath;
    }

    if (options.wrapper) {
        args = args.slice(0); // prevent mutating the caller's arguments
        args.unshift(command);
        command = options.wrapper;

        if (command.startsWith('./')) command = path.join(process.cwd(), command);
    }

    let okToCache = true;
    const cwd = options.customCwd ? options.customCwd : (
        (command.startsWith('/mnt') && process.env.wsl) ? process.env.winTmp : process.env.tmpDir
    );
    logger.debug('Execution', {type: 'executing', command: command, args: args, env: env, cwd: cwd});
    // AP: Run Windows-volume executables in winTmp. Otherwise, run in tmpDir (which may be undefined).
    // https://nodejs.org/api/child_process.html#child_process_child_process_spawn_command_args_options
    const child = child_process.spawn(command, args, {
        cwd: cwd,
        env: env,
        detached: process.platform === 'linux',
    });
    let running = true;

    const kill = options.killChild || (() => {
        if (running) treeKill(child.pid);
    });

    const streams = {
        stderr: '',
        stdout: '',
        truncated: false,
    };
    let timeout;
    if (timeoutMs) timeout = setTimeout(() => {
        logger.warn(`Timeout for ${command} ${args} after ${timeoutMs}ms`);
        okToCache = false;
        kill();
        streams.stderr += '\nKilled - processing time exceeded';
    }, timeoutMs);

    function setupStream(stream, name) {
        if (stream === undefined) return;
        stream.on('data', data => {
            if (streams.truncated) return;
            const newLength = (streams[name].length + data.length);
            if ((maxOutput > 0) && (newLength > maxOutput)) {
                streams[name] = streams[name] + data.slice(0, maxOutput - streams[name].length);
                streams[name] += '\n[Truncated]';
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
    child.on('exit', code => {
        logger.debug('Execution', {type: 'exited', code: code});
        if (timeout !== undefined) clearTimeout(timeout);
        running = false;
    });
    return new Promise((resolve, reject) => {
        child.on('error', e => {
            logger.debug(`Execution error with ${command} args: ${args}:`, e);
            reject(e);
        });
        child.on('close', code => {
            // Being killed externally gives a NULL error code. Synthesize something different here.
            if (code === null) code = -1;
            if (timeout !== undefined) clearTimeout(timeout);
            const result = {
                code,
                okToCache,
                filenameTransform,
                stdout: streams.stdout,
                stderr: streams.stderr,
            };
            logger.debug('Execution', {type: 'executed', command: command, args: args, result: result});
            resolve(result);
        });
        if (child.stdin) {
            if (options.input) child.stdin.write(options.input);
            child.stdin.end();
        }
    });
}

function withNsjailTimeout(args, options) {
    if (options && options.timeoutMs) {
        const ExtraWallClockLeewayMs = 1000;
        return args.concat([
            '--time_limit',
            `${Math.round((options.timeoutMs + ExtraWallClockLeewayMs) / 1000)}`,
        ]);
    }
    return args;
}

function sandboxNsjail(command, args, options) {
    logger.info('Sandbox execution via nsjail', {command, args});
    const execPath = path.dirname(command);
    const execName = path.basename(command);

    const jailingOptions = withNsjailTimeout([
        '--config', 'etc/nsjail/sandbox.cfg',
        '--cwd', '/app',
        '--bindmount', `${execPath}:/app`,
    ]);

    if (options.ldPath) {
        jailingOptions.push(`--env=LD_LIBRARY_PATH=${options.ldPath}`);
        delete options.ldPath;
    }

    return executeDirect(
        execProps('nsjail'),
        jailingOptions
            .concat(['--', `./${execName}`])
            .concat(args),
        options);
}

// function msToFjTimeout(ms) {
//     const totalSecs = Math.round(ms / 1000);
//     return `${Math.floor(totalSecs / (60 * 60))}:${Math.floor(totalSecs / 60)}:${totalSecs % 60}`;
// }

function withFirejailTimeout(args, options) {
    if (options && options.timeoutMs) {
        // const ExtraWallClockLeewayMs = 1000;
        const ExtraCpuLeewayMs = 1500;
        return args.concat([
            // TODO: reinstate once we work out why this causes a 1s+ delay on every execution!
            // `--timeout=${msToFjTimeout(options.timeoutMs + ExtraWallClockLeewayMs)}`,
            `--rlimit-cpu=${Math.round((options.timeoutMs + ExtraCpuLeewayMs) / 1000)}`,
        ]);
    }
    return args;
}

function sandboxFirejail(command, args, options) {
    logger.info('Sandbox execution via firejail', {command, args});
    const execPath = path.dirname(command);
    const execName = path.basename(command);
    const jailingOptions = withFirejailTimeout([
        '--quiet',
        '--deterministic-exit-code',
        '--terminate-orphans',
        '--profile=etc/firejail/sandbox.profile',
        `--private=${execPath}`,
        '--private-cwd']);

    if (options.ldPath) {
        jailingOptions.push(`--env=LD_LIBRARY_PATH=${options.ldPath}`);
        delete options.ldPath;
    }

    return executeDirect(
        execProps('firejail'),
        jailingOptions
            .concat([`./${execName}`])
            .concat(args),
        options);
}

const sandboxDispatchTable = {
    none: (command, args, options) => {
        logger.info('Sandbox execution (sandbox disabled)', {command, args});
        return executeDirect(command, args, options);
    },
    nsjail: sandboxNsjail,
    firejail: sandboxFirejail,
};

export async function sandbox(command, args, options) {
    const type = execProps('sandboxType', 'firejail');
    const dispatchEntry = sandboxDispatchTable[type];
    if (!dispatchEntry)
        throw new Error(`Bad sandbox type ${type}`);
    return dispatchEntry(command, args, options);
}

const wineSandboxName = 'ce-wineserver';

export function initialiseWine() {
    const wine = execProps('wine');
    if (!wine) {
        logger.info('WINE not configured');
        return Promise.resolve();
    }

    const server = execProps('wineServer');
    const firejail = execProps('executionType', 'none') === 'firejail' ? execProps('firejail') : null;
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

    // We run a long-lived cmd process, to:
    // * test that WINE works
    // * be something which holds open a working firejail sandbox
    // All future WINE compiles go through the same sandbox.
    // We wait until the process has printed out some known good text, but don't wait
    // for it to exit (it won't, on purpose).

    let wineServer;
    if (firejail) {
        logger.info(`Starting a new, firejailed, long-lived wineserver complex`);
        wineServer = child_process.spawn(
            firejail,
            [
                '--quiet',
                '--profile=etc/firejail/wine.profile',
                '--private',
                `--name=${wineSandboxName}`,
                wine,
                'cmd',
            ],
            {env: env, detached: true});
        logger.info(`firejailed pid=${wineServer.pid}`);
    } else {
        logger.info(`Starting a new, long-lived wineserver complex ${server}`);
        wineServer = child_process.spawn(
            wine,
            ['cmd'],
            {env: env, detached: true});
        logger.info(`wineserver pid=${wineServer.pid}`);
    }

    wineServer.on('close', code => {
        logger.info(`WINE server complex exited with code ${code}`);
    });

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

        if (wineServer && !wineServer.killed) {
            logger.info('Shutting down WINE server complex');
            wineServer.kill();
            if (wineServer.killed) {
                waitingPromises.push(waitForExit(wineServer, 'WINE server'));
            }
            wineServer = null;
        }
        return Promise.all(waitingPromises);
    });

    return new Promise((resolve, reject) => {
        setupOnError(wineServer.stdin, 'stdin');
        setupOnError(wineServer.stdout, 'stdout');
        setupOnError(wineServer.stderr, 'stderr');
        const magicString = '!!EVERYTHING IS WORKING!!';
        wineServer.stdin.write(`echo ${magicString}`);

        let output = '';
        wineServer.stdout.on('data', data => {
            logger.info(`Output from wine server complex: ${data.toString().trim()}`);
            output += data;
            if (output.includes(magicString)) {
                resolve();
            }
        });
        wineServer.stderr.on('data',
            data => logger.info(`stderr output from wine server complex: ${data.toString().trim()}`));
        wineServer.on('error', e => {
            logger.error(`WINE server complex exited with error ${e}`);
            reject(e);
        });
        wineServer.on('close', code => {
            logger.info(`WINE server complex exited with code ${code}`);
            reject();
        });
    });
}

function applyWineEnv(env) {
    const prefix = execProps('winePrefix');
    env = _.clone(env) || {};
    // Force use of wine vcruntime (See change 45106c382)
    env.WINEDLLOVERRIDES = 'vcruntime140=b';
    env.WINEDEBUG = '-all';
    env.WINEPREFIX = prefix;
    return env;
}

function needsWine(command) {
    return command.match(/\.exe$/i) && process.platform === 'linux' && !process.env.wsl;
}

function executeWineDirect(command, args, options) {
    options = _.clone(options) || {};
    options.env = applyWineEnv(options.env);
    args = [command, ...args];
    return executeDirect(execProps('wine'), args, options);
}

function executeFirejail(command, args, options) {
    options = _.clone(options) || {};
    const firejail = execProps('firejail');
    const baseOptions = withFirejailTimeout(['--quiet', '--deterministic-exit-code', '--terminate-orphans'], options);
    if (needsWine(command)) {
        logger.debug('WINE execution via firejail', {command, args});
        options.env = applyWineEnv(options.env);
        args = [command, ...args];
        command = execProps('wine');
        baseOptions.push('--profile=etc/firejail/wine.profile');
        baseOptions.push(`--join=${wineSandboxName}`);
        delete options.customCwd;
        baseOptions.push(command);
        return executeDirect(firejail, baseOptions.concat(args), options);
    }

    logger.debug('Regular execution via firejail', {command, args});
    baseOptions.push('--profile=etc/firejail/execute.profile');
    let filenameTransform;
    if (options.customCwd) {
        baseOptions.push(`--private=${options.customCwd}`);
        const replacement = options.customCwd;
        filenameTransform = opt => opt.replace(replacement, '.');
        args = args.map(filenameTransform);
        delete options.customCwd;
        // TODO: once it's supported properly in our patched firejail, make this option common to both customCwd and
        // non-customCwd code paths.
        baseOptions.push('--private-cwd');
    } else {
        baseOptions.push('--private');
    }
    baseOptions.push(command);
    return executeDirect(firejail, baseOptions.concat(args), options, filenameTransform);
}

function executeNone(command, args, options) {
    if (needsWine(command)) {
        return executeWineDirect(command, args, options);
    }
    return executeDirect(command, args, options);
}

const executeDispatchTable = {
    none: executeNone,
    firejail: executeFirejail,
};

export async function execute(command, args, options) {
    const type = execProps('executionType', 'none');
    const dispatchEntry = executeDispatchTable[type];
    if (!dispatchEntry)
        throw new Error(`Bad sandbox type ${type}`);
    return dispatchEntry(command, args, options);
}
