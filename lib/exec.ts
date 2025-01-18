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

import buffer from 'buffer';
import child_process from 'child_process';
import os from 'os';
import path from 'path';
import {Stream} from 'stream';

import fs from 'fs-extra';
import treeKill from 'tree-kill';
import _ from 'underscore';

import type {ExecutionOptions} from '../types/compilation/compilation.interfaces.js';
import type {FilenameTransformFunc, UnprocessedExecResult} from '../types/execution/execution.interfaces.js';

import {assert, unwrap, unwrapString} from './assert.js';
import {logger} from './logger.js';
import {Graceful} from './node-graceful.js';
import {propsFor} from './properties.js';
import * as utils from './utils.js';

type NsJailOptions = {
    args: string[];
    options: ExecutionOptions;
    filenameTransform: FilenameTransformFunc | undefined;
};

const execProps = propsFor('execution');

function checkExecOptions(options: ExecutionOptions) {
    if (options.env) {
        for (const key of Object.keys(options.env)) {
            const value: any = options.env[key];
            if (value !== undefined && typeof value !== 'string') {
                logger.warn(`Found non-string in environment: ${key} of ${typeof value} : '${value}'`);
                options.env[key] = value.toString();
            }
        }
    }
}

function setupOnError(stream: Stream, name: string) {
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
    const maxOutput = Math.min(options.maxOutput || 1024 * 1024, buffer.constants.MAX_STRING_LENGTH);
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
    // In WSL; run Windows-volume executables in a temp directory.
    const cwd = options.customCwd || (command.startsWith('/mnt') && process.env.wsl ? os.tmpdir() : undefined);
    logger.debug('Execution', {type: 'executing', command: command, args: args, env: env, cwd: cwd});
    const startTime = process.hrtime.bigint();

    const child = child_process.spawn(command, args, {
        cwd: cwd,
        env: env,
        detached: process.platform === 'linux',
    });
    let running = true;

    const kill =
        options.killChild ||
        (() => {
            if (running && child && child.pid) {
                // Close the stdin pipe on our end, otherwise we'll get an EPIPE
                child.stdin.destroy();
                treeKill(child.pid);
            }
        });

    const streams = {
        stderr: '',
        stdout: '',
        truncated: false,
    };
    let timeout: NodeJS.Timeout | undefined;
    if (timeoutMs)
        timeout = setTimeout(() => {
            logger.warn(`Timeout for ${command} ${args} after ${timeoutMs}ms`);
            okToCache = false;
            timedOut = true;
            kill();
            streams.stderr += '\nKilled - processing time exceeded\n';
        }, timeoutMs);

    function setupStream(stream: Stream, name: 'stdout' | 'stderr') {
        if (stream === undefined) return;
        stream.on('data', data => {
            if (streams.truncated) return;
            const newLength = streams[name].length + data.length;
            if (maxOutput > 0 && newLength > maxOutput) {
                const truncatedMsg = '\n[Truncated]';
                const spaceLeft = Math.max(maxOutput - streams[name].length - truncatedMsg.length, 0);
                streams[name] = streams[name] + data.slice(0, spaceLeft);
                streams[name] += truncatedMsg.slice(0, maxOutput - streams[name].length);
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
                truncated: streams.truncated,
                execTime: utils.deltaTimeNanoToMili(startTime, endTime),
            };
            // Check debug level explicitly as result may be a very large string
            // which we'd prefer to avoid preparing if it won't be used
            if (logger.isDebugEnabled()) {
                logger.debug('Execution', {type: 'executed', command: command, args: args, result: result});
            }
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

export function getCeWrapperCfgFilePath(configName: string): string {
    const propKey = `cewrapper.config.${configName}`;
    const configPath = execProps<string>(propKey);
    if (configPath === undefined) {
        logger.error(`Could not find '${propKey}'. Are you missing a definition?`);
        throw new Error(`Missing cewrapper execution config property key '${propKey}'`);
    }
    return path.resolve(configPath);
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
    let filenameTransform: FilenameTransformFunc | undefined;
    if (options.customCwd) {
        let replacement = options.customCwd;
        if (options.appHome) {
            replacement = options.appHome;
            const relativeCwd = path.join(homeDir, path.relative(options.appHome, options.customCwd));
            jailingOptions.push('--cwd', relativeCwd, '--bindmount', `${options.appHome}:${homeDir}`);
        } else {
            jailingOptions.push('--cwd', homeDir, '--bindmount', `${options.customCwd}:${homeDir}`);
        }

        filenameTransform = opt => opt.replaceAll(replacement, '/app');
        args = args.map(filenameTransform);
        delete options.customCwd;
    }

    const transform = filenameTransform || (x => x);

    const env: Record<string, string> = {...options.env, HOME: homeDir};
    if (options.ldPath) {
        const ldPaths = options.ldPath.filter(Boolean).map(path => transform(path));
        jailingOptions.push(`--env=LD_LIBRARY_PATH=${ldPaths.join(path.delimiter)}`);
        delete options.ldPath;
        delete env.LD_LIBRARY_PATH;
    }

    for (const [key, value] of Object.entries(env)) {
        if (value !== undefined) jailingOptions.push(`--env=${key}=${transform(value)}`);
    }
    delete options.env;

    return {
        args: jailingOptions.concat(['--', command]).concat(args),
        options,
        filenameTransform,
    };
}

export function getCeWrapperOptions(
    configName: string,
    command: string,
    args: string[],
    options: ExecutionOptions,
): NsJailOptions {
    options = {...options};
    const jailingOptions = [`--config=${getCeWrapperCfgFilePath(configName)}`];

    if (options.customCwd) {
        if (options.appHome) {
            jailingOptions.push(`--home=${options.appHome}`);
        } else {
            jailingOptions.push(`--home=${options.customCwd}`);
        }

        // note: keep the customCwd in options, dont delete
    }

    if (options.timeoutMs) {
        const ExtraWallClockLeewayMs = 1000;
        jailingOptions.push(`--time_limit=${Math.round((options.timeoutMs + ExtraWallClockLeewayMs) / 1000)}`);
    }

    return {
        args: jailingOptions.concat([command]).concat(args),
        options,
        filenameTransform: x => x,
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

export function getSandboxCEWrapperOptions(command: string, args: string[], options: ExecutionOptions): NsJailOptions {
    return getCeWrapperOptions('sandbox', command, args, options);
}

export function getExecuteCEWrapperOptions(command: string, args: string[], options: ExecutionOptions): NsJailOptions {
    return getCeWrapperOptions('execute', command, args, options);
}

function sandboxNsjail(command: string, args: string[], options: ExecutionOptions) {
    logger.info('Sandbox execution via nsjail', {command, args});
    const nsOpts = getSandboxNsjailOptions(command, args, options);
    return executeDirect(execProps<string>('nsjail'), nsOpts.args, nsOpts.options, nsOpts.filenameTransform);
}

function executeNsjail(command: string, args: string[], options: ExecutionOptions) {
    const nsOpts = getNsJailOptions('execute', command, args, options);
    return executeDirect(execProps<string>('nsjail'), nsOpts.args, nsOpts.options, nsOpts.filenameTransform);
}

function sandboxCEWrapper(command: string, args: string[], options: ExecutionOptions) {
    const nsOpts = getSandboxCEWrapperOptions(command, args, options);
    return executeDirect(execProps<string>('cewrapper'), nsOpts.args, nsOpts.options, nsOpts.filenameTransform);
}

function executeCEWrapper(command: string, args: string[], options: ExecutionOptions) {
    const nsOpts = getExecuteCEWrapperOptions(command, args, options);
    return executeDirect(execProps<string>('cewrapper'), nsOpts.args, nsOpts.options, nsOpts.filenameTransform);
}

function withFirejailTimeout(args: string[], options?: ExecutionOptions) {
    if (options && options.timeoutMs) {
        // const ExtraWallClockLeewayMs = 1000;
        const ExtraCpuLeewayMs = 1500;
        return args.concat([`--rlimit-cpu=${Math.round((options.timeoutMs + ExtraCpuLeewayMs) / 1000)}`]);
    }
    return args;
}

function sandboxFirejail(command: string, args: string[], options: ExecutionOptions) {
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

    const env = options.env || {};
    for (const key of Object.keys(env)) {
        jailingOptions.push(`--env=${key}=${env[key]}`);
    }
    delete options.env;

    return executeDirect(execProps<string>('firejail'), jailingOptions.concat([`./${execName}`]).concat(args), options);
}

const sandboxDispatchTable = {
    none: (command: string, args: string[], options: ExecutionOptions) => {
        logger.info('Sandbox execution (sandbox disabled)', {command, args});
        if (needsWine(command)) {
            return executeWineDirect(command, args, options);
        }
        return executeDirect(command, args, options);
    },
    nsjail: sandboxNsjail,
    firejail: sandboxFirejail,
    cewrapper: sandboxCEWrapper,
};

export async function sandbox(
    command: string,
    args: string[],
    options: ExecutionOptions,
): Promise<UnprocessedExecResult> {
    checkExecOptions(options);
    const type = execProps('sandboxType', 'firejail');
    const dispatchEntry = sandboxDispatchTable[type as 'none' | 'nsjail' | 'firejail' | 'cewrapper'];
    if (!dispatchEntry) throw new Error(`Bad sandbox type ${type}`);
    if (!command) throw new Error(`No executable provided`);
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
    const env = applyWineEnv({PATH: unwrapString(process.env.PATH)});
    const prefix = env.WINEPREFIX;

    logger.info(`Initialising WINE in ${prefix}`);

    const asyncSetup = async (): Promise<void> => {
        if (!(await fs.pathExists(prefix))) {
            logger.info(`Creating directory ${prefix}`);
            await fs.mkdir(prefix);
        }

        logger.info(`Killing any pre-existing wine-server`);
        child_process.exec(`${server} -k || true`, {env: env});

        // We run a long-lived cmd process, to:
        // * test that WINE works
        // * be something which holds open a working firejail sandbox
        // All future WINE compiles go through the same sandbox.
        // We wait until the process has printed out some known good text, but don't wait
        // for it to exit (it won't, on purpose).

        let wineServer: child_process.ChildProcess | undefined;
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

            function waitForExit(process: child_process.ChildProcess, name: string): Promise<void> {
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
                wineServer = undefined;
            }
            return Promise.all(waitingPromises);
        });

        return new Promise((resolve, reject) => {
            assert(wineServer);
            const [stdin, stdout, stderr] = [
                unwrap(wineServer.stdin),
                unwrap(wineServer.stdout),
                unwrap(wineServer.stderr),
            ];
            setupOnError(stdin, 'stdin');
            setupOnError(stdout, 'stdout');
            setupOnError(stderr, 'stderr');
            const magicString = '!!EVERYTHING IS WORKING!!';
            stdin.write(`echo ${magicString}`);

            let output = '';
            stdout.on('data', data => {
                logger.info(`Output from wine server complex: ${data.toString().trim()}`);
                output += data;
                if (output.includes(magicString)) {
                    resolve();
                }
            });
            stderr.on('data', data => logger.info(`stderr output from wine server complex: ${data.toString().trim()}`));
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

function applyWineEnv(env: Record<string, string>): Record<string, string> {
    return {
        ...env,
        // Force use of wine vcruntime (See change 45106c382)
        WINEDLLOVERRIDES: 'vcruntime140=b',
        WINEDEBUG: '-all',
        WINEPREFIX: execProps<string>('winePrefix'),
    };
}

function needsWine(command: string) {
    return command.match(/\.exe$/i) && process.platform === 'linux' && !process.env.wsl;
}

async function executeWineDirect(command: string, args: string[], options: ExecutionOptions) {
    options = _.clone(options) || {};
    options.env = applyWineEnv(options.env || {});
    args = [command, ...args];
    await wineInitPromise;
    return await executeDirect(unwrapString(execProps<string>('wine')), args, options);
}

async function executeFirejail(command: string, args: string[], options: ExecutionOptions) {
    options = _.clone(options) || {};
    const firejail = execProps<string>('firejail');
    const baseOptions = withFirejailTimeout(
        ['--quiet', '--deterministic-exit-code', '--deterministic-shutdown'],
        options,
    );
    if (needsWine(command)) {
        logger.debug('WINE execution via firejail', {command, args});
        options.env = applyWineEnv(options.env || {});
        args = [command, ...args];
        command = execProps<string>('wine');
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

    let filenameTransform: FilenameTransformFunc | undefined;
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

async function executeNone(command: string, args: string[], options: ExecutionOptions) {
    if (needsWine(command)) {
        return await executeWineDirect(command, args, options);
    }
    return await executeDirect(command, args, options);
}

type DispatchFunction = (command: string, args: string[], options: ExecutionOptions) => Promise<UnprocessedExecResult>;

const executeDispatchTable: Record<string, DispatchFunction> = {
    none: executeNone,
    firejail: executeFirejail,
    nsjail: (command: string, args: string[], options: ExecutionOptions) =>
        needsWine(command) ? executeFirejail(command, args, options) : executeNsjail(command, args, options),
    cewrapper: executeCEWrapper,
};

export async function execute(
    command: string,
    args: string[],
    options: ExecutionOptions,
): Promise<UnprocessedExecResult> {
    checkExecOptions(options);
    const type = execProps('executionType', 'none');
    const dispatchEntry = executeDispatchTable[type];
    if (!dispatchEntry) throw new Error(`Bad sandbox type ${type}`);
    if (!command) throw new Error(`No executable provided`);
    return await dispatchEntry(command, args, options);
}
