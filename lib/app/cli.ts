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
export function parseNumberForOptions(value: string): number {
    // Ensure string contains only digits (and optional leading minus sign)
    if (!/^-?\d+$/.test(value)) {
        throw new Error(`Invalid number: "${value}"`);
    }

    const parsedValue = Number.parseInt(value, 10);
    if (Number.isNaN(parsedValue)) {
        throw new Error(`Invalid number: "${value}"`);
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
}

/**
 * Parse command-line arguments and return parsed options
 */
export function parseCommandLine(): CompilerExplorerOptions {
    const program = new Command();
    program
        .name('compiler-explorer')
        .description('Interactively investigate compiler output')
        .option('--env <environments...>', 'Environment(s) to use', ['dev'])
        .option('--root-dir <dir>', 'Root directory for config files', './etc')
        .option('--host <hostname>', 'Hostname to listen on')
        .option('--port <port>', 'Port to listen on', parseNumberForOptions, 10240)
        .option('--prop-debug', 'Debug properties')
        .option('--debug', 'Enable debug output')
        .option('--dist', 'Running in dist mode')
        .option('--no-remote-fetch', 'Ignore fetch marks and assume every compiler is found locally')
        .option('--tmpDir, --tmp-dir <dir>', 'Directory to use for temporary files')
        .option('--wsl', 'Running under Windows Subsystem for Linux')
        .option('--language <languages...>', 'Only load specified languages for faster startup')
        .option('--no-cache', 'Do not use caching for compilation results')
        .option('--ensure-no-id-clash', "Don't run if compilers have clashing ids")
        .option('--logHost, --log-host <hostname>', 'Hostname for remote logging')
        .option('--logPort, --log-port <port>', 'Port for remote logging', parseNumberForOptions)
        .option('--hostnameForLogging, --hostname-for-logging <hostname>', 'Hostname to use in logs')
        .option('--suppressConsoleLog, --suppress-console-log', 'Disable console logging')
        .option('--metricsPort, --metrics-port <port>', 'Port to serve metrics on', parseNumberForOptions)
        .option('--loki <url>', 'URL for Loki logging')
        .option('--discoveryonly, --discovery-only <file>', 'Output discovery info to file and exit')
        .option('--prediscovered <file>', 'Input discovery info from file')
        .option('--static <dir>', 'Path to static content')
        .option('--no-local', 'Disable local config')
        .option('--version', 'Show version information');

    program.parse();
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
    if (fs.existsSync('.git/')) {
        return child_process.execSync('git rev-parse HEAD').toString().trim();
    }

    // unknown case
    return '';
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
    return '';
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
 * Set up temporary directory, especially for WSL environments
 */
export function setupTempDir(options: CompilerExplorerOptions, isWsl: boolean): void {
    // If a tempDir is supplied, use it
    if (options.tmpDir) {
        if (isWsl) {
            process.env.TEMP = options.tmpDir; // for Windows
        } else {
            process.env.TMP = options.tmpDir; // for Linux
        }
    }
    // If running under WSL without explicit tmpDir, try to use Windows %TEMP%
    else if (isWsl) {
        try {
            const windowsTemp = child_process.execSync('cmd.exe /c echo %TEMP%').toString().replaceAll('\\', '/');
            const driveLetter = windowsTemp.substring(0, 1).toLowerCase();
            const directoryPath = windowsTemp.substring(2).trim();
            process.env.TEMP = path.join('/mnt', driveLetter, directoryPath);
        } catch (e) {
            logger.warn('Unable to invoke cmd.exe to get windows %TEMP% path.');
        }
    }
}

/**
 * Convert command-line options to AppArguments for the application
 */
export function convertOptionsToAppArguments(
    options: CompilerExplorerOptions,
    gitReleaseName: string,
    releaseBuildNumber: string,
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
        suppressConsoleLog: options.suppressConsoleLog,
    };
}

/**
 * Initialize configuration from command-line arguments
 * @returns Application arguments and raw options
 */
export function initializeOptionsFromCommandLine(): {appArgs: AppArguments; options: CompilerExplorerOptions} {
    // Parse command-line arguments
    const options = parseCommandLine();

    // Set debug level if requested
    if (options.debug) {
        logger.level = 'debug';
    }

    // Detect WSL environment
    const isWsl = detectWsl();
    if (isWsl) {
        process.env.wsl = 'true';
    }

    // Set up temporary directory
    setupTempDir(options, isWsl);
    logger.info(`Using temporary dir: ${process.env.TEMP || process.env.TMP}`);

    // Get distribution path and version information
    const distPath = utils.resolvePathFromAppRoot('.');
    logger.debug(`Distpath=${distPath}`);

    const gitReleaseName = getGitReleaseName(distPath, options.dist === true);
    const releaseBuildNumber = getReleaseBuildNumber(distPath, options.dist === true);

    // Convert to AppArguments
    const appArgs = convertOptionsToAppArguments(options, gitReleaseName, releaseBuildNumber);

    return {appArgs, options};
}
