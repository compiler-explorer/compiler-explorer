// Copyright (c) 2025, Compiler Explorer Authors
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

import child_process from 'node:child_process';
import * as fs from 'node:fs';
import path from 'node:path';

import {Command} from 'commander';

import {AppArguments} from '../app.interfaces.js';
import {logger} from '../logger.js';
import * as utils from '../utils.js';

/**
 * Parses a command line option into a number.
 */
export function parsePortNumberForOptions(value: string): number {
    // Ensure string contains only digits
    if (!/^\d+$/.test(value)) {
        throw new Error(`Invalid port number: "${value}"`);
    }

    const parsedValue = Number.parseInt(value, 10);
    if (Number.isNaN(parsedValue) || parsedValue > 65535) {
        throw new Error(`Invalid port number: "${value}"`);
    }
    return parsedValue;
}

/**
 * Options parsed from command-line arguments
 */
export interface CompilerExplorerOptions {
    env: string[];
    rootDir: string;
    host?: string;
    port: number;
    propDebug?: boolean;
    debug?: boolean;
    dist?: boolean;
    remoteFetch: boolean;
    tmpDir?: string;
    wsl?: boolean;
    language?: string[];
    cache: boolean;
    ensureNoIdClash?: boolean;
    exitOnCompilerFailure?: boolean;
    logHost?: string;
    logPort?: number;
    hostnameForLogging?: string;
    suppressConsoleLog: boolean;
    metricsPort?: number;
    loki?: string;
    discoveryOnly?: string;
    prediscovered?: string;
    static?: string;
    local: boolean;
    version: boolean;
    devMode: boolean;
    instanceColor?: string;
}

/**
 * Parse command-line arguments and return parsed options
 * @param argv The command-line arguments to parse
 */
export function parseCommandLine(argv: string[]): CompilerExplorerOptions {
    const program = new Command();
    program
        .name('compiler-explorer')
        .description('Interactively investigate compiler output')
        .option('--env <environments...>', 'Environment(s) to use', ['dev'])
        .option('--root-dir <dir>', 'Root directory for config files', './etc')
        .option('--host <hostname>', 'Hostname to listen on')
        .option('--port <port>', 'Port to listen on', parsePortNumberForOptions, 10240)
        .option('--prop-debug', 'Debug properties')
        .option('--debug', 'Enable debug output')
        .option('--dist', 'Running in dist mode')
        .option('--no-remote-fetch', 'Ignore fetch marks and assume every compiler is found locally')
        .option('--tmpDir, --tmp-dir <dir>', 'Directory to use for temporary files')
        .option('--wsl', 'Running under Windows Subsystem for Linux')
        .option('--language <languages...>', 'Only load specified languages for faster startup')
        .option('--no-cache', 'Do not use caching for compilation results')
        .option('--ensure-no-id-clash', "Don't run if compilers have clashing ids")
        .option('--exit-on-compiler-failure', 'Exit with error code if any compilers fail to initialize')
        .option('--logHost, --log-host <hostname>', 'Hostname for remote logging')
        .option('--logPort, --log-port <port>', 'Port for remote logging', parsePortNumberForOptions)
        .option('--hostnameForLogging, --hostname-for-logging <hostname>', 'Hostname to use in logs')
        .option('--suppressConsoleLog, --suppress-console-log', 'Disable console logging')
        .option('--metricsPort, --metrics-port <port>', 'Port to serve metrics on', parsePortNumberForOptions)
        .option('--loki <url>', 'URL for Loki logging')
        .option('--discoveryonly, --discovery-only <file>', 'Output discovery info to file and exit')
        .option('--prediscovered <file>', 'Input discovery info from file')
        .option('--static <dir>', 'Path to static content')
        .option('--no-local', 'Disable local config')
        .option('--version', 'Show version information')
        .option(
            '--dev-mode',
            'Run in dev mode (default if NODE_ENV is not production)',
            process.env.NODE_ENV !== 'production',
        )
        .option('--instance-color <color>', 'Instance color (blue or green) for deployment differentiation');

    program.parse(argv);
    return program.opts() as CompilerExplorerOptions;
}

/**
 * Extract git release information from repository or file
 */
export function getGitReleaseName(distPath: string, isDist: boolean): string {
    // Use the canned git_hash if provided
    const gitHashFilePath = path.join(distPath, 'git_hash');
    if (isDist && fs.existsSync(gitHashFilePath)) {
        return fs.readFileSync(gitHashFilePath).toString().trim();
    }

    // Check if we have been cloned and not downloaded
    if (fs.existsSync('.git')) {
        return child_process.execSync('git rev-parse HEAD').toString().trim();
    }

    // unknown case
    return '<no git hash found>';
}

/**
 * Extract release build number from file
 */
export function getReleaseBuildNumber(distPath: string, isDist: boolean): string {
    // Use the canned build only if provided
    const releaseBuildPath = path.join(distPath, 'release_build');
    if (isDist && fs.existsSync(releaseBuildPath)) {
        return fs.readFileSync(releaseBuildPath).toString().trim();
    }
    return '<no build number found>';
}

/**
 * Detect if running under Windows Subsystem for Linux
 */
export function detectWsl(): boolean {
    if (process.platform === 'linux') {
        try {
            return child_process.execSync('uname -a').toString().toLowerCase().includes('microsoft');
        } catch (e) {
            logger.warn('Unable to detect WSL environment', e);
        }
    }
    return false;
}

/**
 * Convert command-line options to AppArguments for the application
 */
export function convertOptionsToAppArguments(
    options: CompilerExplorerOptions,
    gitReleaseName: string,
    releaseBuildNumber: string,
    isWsl: boolean,
): AppArguments {
    return {
        rootDir: options.rootDir,
        env: options.env,
        hostname: options.host,
        port: options.port,
        gitReleaseName: gitReleaseName,
        releaseBuildNumber: releaseBuildNumber,
        wantedLanguages: options.language,
        doCache: options.cache,
        fetchCompilersFromRemote: options.remoteFetch,
        ensureNoCompilerClash: options.ensureNoIdClash,
        exitOnCompilerFailure: options.exitOnCompilerFailure,
        prediscovered: options.prediscovered,
        discoveryOnly: options.discoveryOnly,
        staticPath: options.static,
        metricsPort: options.metricsPort,
        useLocalProps: options.local,
        propDebug: options.propDebug || false,
        tmpDir: options.tmpDir,
        isWsl: isWsl,
        devMode: options.devMode,
        instanceColor: options.instanceColor,
        loggingOptions: {
            debug: options.debug || false,
            logHost: options.logHost,
            logPort: options.logPort,
            hostnameForLogging: options.hostnameForLogging,
            loki: options.loki,
            suppressConsoleLog: options.suppressConsoleLog,
            paperTrailIdentifier: options.env.join('.'),
        },
    };
}

/**
 * Parse command-line arguments into an AppArguments object
 * @param argv The command-line arguments to parse
 * @returns Application arguments
 */
export function parseArgsToAppArguments(argv: string[]): AppArguments {
    const options = parseCommandLine(argv);
    const isWsl = detectWsl();

    const distPath = utils.resolvePathFromAppRoot('.');
    logger.debug(`Distpath=${distPath}`);

    const gitReleaseName = getGitReleaseName(distPath, options.dist === true);
    const releaseBuildNumber = getReleaseBuildNumber(distPath, options.dist === true);

    const appArgs = convertOptionsToAppArguments(options, gitReleaseName, releaseBuildNumber, isWsl);

    if (options.version) {
        // We can't use the `--version` support in Commander, as we need to parse the args
        // to find the directory for the git release and whatnot.
        logger.info('Compiler Explorer version info:');
        logger.info(`  git release ${appArgs.gitReleaseName}`);
        logger.info(`  release build ${appArgs.releaseBuildNumber}`);
        logger.info('Exiting');
        process.exit(0);
    }

    return appArgs;
}
