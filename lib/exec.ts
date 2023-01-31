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
import path from 'path';

import fs from 'fs-extra';
import Graceful from 'node-graceful';
import treeKill from 'tree-kill';
import _ from 'underscore';

import {ExecutionOptions} from '../types/compilation/compilation.interfaces';
import {FilenameTransformFunc, UnprocessedExecResult} from '../types/execution/execution.interfaces';

import {logger} from './logger';
import {propsFor} from './properties';

type NsJailOptions = {
    args: string[];
    options: ExecutionOptions;
    filenameTransform: FilenameTransformFunc;
};

const execProps = propsFor('execution');

function setupOnError(stream, name) {
    if (stream === undefined) return;
    stream.on('error', err => {
        logger.error(`Error with ${name} stream:`, err);
    });
}

export function executeDirect(
    command: string,
    args: string[],
    options: ExecutionOptions,
    filenameTransform?: FilenameTransformFunc,
): Promise<UnprocessedExecResult> {
    options = options || {};
    const maxOutput = options.maxOutput || 1024 * 1024;
    const timeoutMs = options.timeoutMs || 0;
    const env = {...process.env, ...options.env};

    if (options.ldPath) {
        env.LD_LIBRARY_PATH = options.ldPath.join(path.delimiter);
    }

    if (options.wrapper) {
        args = args.slice(0); // prevent mutating the caller's arguments
        args.unshift(command);
        command = options.wrapper;

        if (command.startsWith('./')) command = path.join(process.cwd(), command);
    }

    let okToCache = true;
    let timedOut = false;
    const cwd =
        options.customCwd || (command.startsWith('/mnt') && process.env.wsl ? process.env.winTmp : process.env.tmpDir);
    logger.debug('Execution', {type: 'executing', command: command, args: args, env: env, cwd: cwd});
    const startTime = process.hrtime.bigint();

    // AP: Run Windows-volume executables in winTmp. Otherwise, run in tmpDir (which may be undefined).
    // https://nodejs.org/api/child_process.html#child_process_child_process_spawn_command_args_options
    const child = child_process.spawn(command, args, {
        cwd: cwd,
        env: env,
        detached: process.platform === 'linux',
    });
    let running = true;

    const kill =
        options.killChild ||
        (() => {
            if (running && child && child.pid) treeKill(child.pid);
        });

    const streams = {
        stderr: '',
        stdout: '',
        truncated: false,
    };
    let timeout;
    if (timeoutMs)
        timeout = setTimeout(() => {
            logger.warn(`Timeout for ${command} ${args} after ${timeoutMs}ms`);
            okToCache = false;
            timedOut = true;
            kill();
            streams.stderr += '\nKilled - processing time exceeded';
        }, timeoutMs);

    function setupStream(stream, name) {
        if (stream === undefined) return;
        stream.on('data', data => {
            if (streams.truncated) return;
            const newLength = streams[name].length + data.length;
            if (maxOutput > 0 && newLength > maxOutput) {
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
            const endTime = process.hrtime.bigint();
            const result: UnprocessedExecResult = {
                code,
                okToCache,
                timedOut,
                filenameTransform: filenameTransform || (x => x),
                stdout: streams.stdout,
                stderr: streams.stderr,
                execTime: ((endTime - startTime) / BigInt(1000000)).toString(),
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

export function getNsJailCfgFilePath(configName: string): string {
    const propKey = `nsjail.config.${configName}`;
    const configPath = execProps<string>(propKey);
    if (configPath === undefined) {
        logger.error(`Could not find ${propKey}. Are you missing a definition?`);
        throw new Error(`Missing nsjail execution config property key ${propKey}`);
    }
    return configPath;
}

export function getFirejailProfileFilePath(profileName: string): string {
    const propKey = `firejail.profile.${profileName}`;
    const profilePath = execProps<string>(propKey);
    if (profilePath === undefined) {
        logger.error(`Could not find ${propKey}. Are you missing a definition?`);
        throw new Error(`Missing firejail execution profile property key ${propKey}`);
    }
    return profilePath;
}

export function getNsJailOptions(
    configName: string,
    command: string,
    args: string[],
    options: ExecutionOptions,
): NsJailOptions {
    options = {...options};
    const jailingOptions = ['--config', getNsJailCfgFilePath(configName)];

    if (options.timeoutMs) {
        const ExtraWallClockLeewayMs = 1000;
        jailingOptions.push(`--time_limit=${Math.round((options.timeoutMs + ExtraWallClockLeewayMs) / 1000)}`);
    }

    const homeDir = '/app';
    let filenameTransform;
    if (options.customCwd) {
        let replacement = options.customCwd;
        if (options.appHome) {
            replacement = options.appHome;
            const relativeCwd = path.join(homeDir, path.relative(options.appHome, options.customCwd));
            jailingOptions.push('--cwd', relativeCwd, '--bindmount', `${options.appHome}:${homeDir}`);
        } else {
            jailingOptions.push('--cwd', homeDir, '--bindmount', `${options.customCwd}:${homeDir}`);
        }

        filenameTransform = opt => opt.replace(replacement, '/app');
        args = args.map(filenameTransform);
        delete options.customCwd;
    }

    const env = {...options.env, HOME: homeDir};
    if (options.ldPath) {
        jailingOptions.push(`--env=LD_LIBRARY_PATH=${options.ldPath.join(path.delimiter)}`);
        delete options.ldPath;
        delete env.LD_LIBRARY_PATH;
    }

    for (const [key, value] of Object.entries(env)) {
        if (value !== undefined) jailingOptions.push(`--env=${key}=${value}`);
    }
    delete options.env;

    return {
        args: jailingOptions.concat(['--', command]).concat(args),
        options,
        filenameTransform,
    };
}

export function getSandboxNsjailOptions(command: string, args: string[], options: ExecutionOptions): NsJailOptions {
    // If we already had a custom cwd, use that.
    if (options.customCwd) {
        let relativeCommand = command;
        if (command.startsWith(options.customCwd)) {
            relativeCommand = path.relative(options.customCwd, command);
            if (path.dirname(relativeCommand) === '.') {
                relativeCommand = `./${relativeCommand}`;
            }
        }
        return getNsJailOptions('sandbox', relativeCommand, args, options);
    }

    // Else, assume the executable should be run as `./exec` and run it from its directory.
    options = {...options, customCwd: path.dirname(command)};
    return getNsJailOptions('sandbox', `./${path.basename(command)}`, args, options);
}

function sandboxNsjail(command, args, options) {
    logger.info('Sandbox execution via nsjail', {command, args});
    const nsOpts = getSandboxNsjailOptions(command, args, options);
    return executeDirect(execProps<string>('nsjail'), nsOpts.args, nsOpts.options, nsOpts.filenameTransform);
}

function executeNsjail(command, args, options) {
    const nsOpts = getNsJailOptions('execute', command, args, options);
    return executeDirect(execProps<string>('nsjail'), nsOpts.args, nsOpts.options, nsOpts.filenameTransform);
}

function withFirejailTimeout(args: string[], options?) {
    if (options && options.timeoutMs) {
        // const ExtraWallClockLeewayMs = 1000;
        const ExtraCpuLeewayMs = 1500;
        return args.concat([`--rlimit-cpu=${Math.round((options.timeoutMs + ExtraCpuLeewayMs) / 1000)}`]);
    }
    return args;
}

function sandboxFirejail(command: string, args: string[], options) {
    logger.info('Sandbox execution via firejail', {command, args});
    const execPath = path.dirname(command);
    const execName = path.basename(command);
    const jailingOptions = withFirejailTimeout([
        '--quiet',
        '--deterministic-exit-code',
        '--deterministic-shutdown',
        '--profile=' + getFirejailProfileFilePath('sandbox'),
        `--private=${execPath}`,
        '--private-cwd',
    ]);

    if (options.ldPath) {
        jailingOptions.push(`--env=LD_LIBRARY_PATH=${options.ldPath.join(path.delimiter)}`);
        delete options.ldPath;
    }

    for (const key of Object.keys(options.env || {})) {
        jailingOptions.push(`--env=${key}=${options.env[key]}`);
    }
    delete options.env;

    return executeDirect(execProps<string>('firejail'), jailingOptions.concat([`./${execName}`]).concat(args), options);
}

const sandboxDispatchTable = {
    none: (command, args, options) => {
        logger.info('Sandbox execution (sandbox disabled)', {command, args});
        if (needsWine(command)) {
            return executeWineDirect(command, args, options);
        }
        return executeDirect(command, args, options);
    },
    nsjail: sandboxNsjail,
    firejail: sandboxFirejail,
};

export async function sandbox(
    command: string,
    args: string[],
    options: ExecutionOptions,
): Promise<UnprocessedExecResult> {
    const type = execProps('sandboxType', 'firejail');
    const dispatchEntry = sandboxDispatchTable[type];
    if (!dispatchEntry) throw new Error(`Bad sandbox type ${type}`);
    return await dispatchEntry(command, args, options);
}

const wineSandboxName = 'ce-wineserver';
// WINE takes a while to initialise and very often we don't need to run it at
// all during startup. So, we do just the bare minimum at startup and then make
// a promise that all subsequent WINE calls wait on.
let wineInitPromise: Promise<void> | null;

export function startWineInit() {
    const wine = execProps<string | undefined>('wine');
    if (!wine) {
        logger.info('WINE not configured');
        return;
    }

    const server = execProps('wineServer');
    const executionType = execProps('executionType', 'none');
    // We need to fire up a firejail wine server even in nsjail world (for now).
    const firejail = executionType === 'firejail' || executionType === 'nsjail' ? execProps<string>('firejail') : null;
    const env = applyWineEnv({PATH: process.env.PATH});
    const prefix = env.WINEPREFIX;

    logger.info(`Initialising WINE in ${prefix}`);

    const asyncSetup = async (): Promise<void> => {
        if (!(await fs.pathExists(prefix))) {
            logger.info(`Creating directory ${prefix}`);
            await fs.mkdir(prefix);
        }

        logger.info(`Killing any pre-existing wine-server`);
        let result = await child_process.exec(`${server} -k || true`, {env: env});
        logger.info(`Result: ${result}`);
        logger.info(`Waiting for any pre-existing server to stop...`);
        result = await child_process.exec(`${server} -w`, {env: env});
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
                    '--profile=' + getFirejailProfileFilePath('wine'),
                    '--private',
                    `--name=${wineSandboxName}`,
                    wine,
                    'cmd',
                ],
                {env: env, detached: true},
            );
            logger.info(`firejailed pid=${wineServer.pid}`);
        } else {
            logger.info(`Starting a new, long-lived wineserver complex ${server}`);
            wineServer = child_process.spawn(wine, ['cmd'], {env: env, detached: true});
            logger.info(`wineserver pid=${wineServer.pid}`);
        }

        wineServer.on('close', code => {
            logger.info(`WINE server complex exited with code ${code}`);
        });

        Graceful.on('exit', () => {
            const waitingPromises: Promise<void>[] = [];

            function waitForExit(process, name): Promise<void> {
                return new Promise(resolve => {
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
            wineServer.stderr.on('data', data =>
                logger.info(`stderr output from wine server complex: ${data.toString().trim()}`),
            );
            wineServer.on('error', e => {
                logger.error(`WINE server complex exited with error ${e}`);
                reject(e);
            });
            wineServer.on('close', code => {
                logger.info(`WINE server complex exited with code ${code}`);
                reject();
            });
        });
    };
    wineInitPromise = asyncSetup();
}

function applyWineEnv(env) {
    return {
        ...env,
        // Force use of wine vcruntime (See change 45106c382)
        WINEDLLOVERRIDES: 'vcruntime140=b',
        WINEDEBUG: '-all',
        WINEPREFIX: execProps('winePrefix'),
    };
}

function needsWine(command) {
    return command.match(/\.exe$/i) && process.platform === 'linux' && !process.env.wsl;
}

async function executeWineDirect(command, args, options) {
    options = _.clone(options) || {};
    options.env = applyWineEnv(options.env);
    args = [command, ...args];
    await wineInitPromise;
    return await executeDirect(execProps<string>('wine'), args, options);
}

async function executeFirejail(command, args, options) {
    options = _.clone(options) || {};
    const firejail = execProps<string>('firejail');
    const baseOptions = withFirejailTimeout(
        ['--quiet', '--deterministic-exit-code', '--deterministic-shutdown'],
        options,
    );
    if (needsWine(command)) {
        logger.debug('WINE execution via firejail', {command, args});
        options.env = applyWineEnv(options.env);
        args = [command, ...args];
        command = execProps('wine');
        baseOptions.push('--profile=' + getFirejailProfileFilePath('wine'), `--join=${wineSandboxName}`);
        delete options.customCwd;
        baseOptions.push(command);
        await wineInitPromise;
        return await executeDirect(firejail, baseOptions.concat(args), options);
    }

    logger.debug('Regular execution via firejail', {command, args});
    baseOptions.push('--profile=' + getFirejailProfileFilePath('execute'));

    if (options.ldPath) {
        baseOptions.push(`--env=LD_LIBRARY_PATH=${options.ldPath.join(path.delimiter)}`);
        delete options.ldPath;
    }

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
    return await executeDirect(firejail, baseOptions.concat(args), options, filenameTransform);
}

async function executeNone(command, args, options) {
    if (needsWine(command)) {
        return await executeWineDirect(command, args, options);
    }
    return await executeDirect(command, args, options);
}

const executeDispatchTable = {
    none: executeNone,
    firejail: executeFirejail,
    nsjail: (command, args, options) =>
        needsWine(command) ? executeFirejail(command, args, options) : executeNsjail(command, args, options),
};

export async function execute(
    command: string,
    args: string[],
    options: ExecutionOptions,
): Promise<UnprocessedExecResult> {
    const type = execProps('executionType', 'none');
    const dispatchEntry = executeDispatchTable[type];
    if (!dispatchEntry) throw new Error(`Bad sandbox type ${type}`);
    return await dispatchEntry(command, args, options);
}
