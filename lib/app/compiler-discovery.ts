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

import fs from 'node:fs/promises';
import process from 'node:process';

import {CompilerInfo} from '../../types/compiler.interfaces.js';
import {AppArguments} from '../app.interfaces.js';
import {CompilerFinder} from '../compiler-finder.js';
import {logger} from '../logger.js';

/**
 * Discover and prepare compilers for use in the application
 * @param appArgs - Application arguments
 * @param compilerFinder - Compiler finder instance
 * @param isExecutionWorker - Whether the server is running as an execution worker
 * @returns Array of discovered compilers
 */
export async function discoverCompilers(
    appArgs: AppArguments,
    compilerFinder: CompilerFinder,
    isExecutionWorker: boolean,
): Promise<CompilerInfo[]> {
    let compilers: CompilerInfo[];
    if (appArgs.prediscovered) {
        compilers = await loadPrediscoveredCompilers(appArgs.prediscovered, compilerFinder);
    } else {
        const result = await findAndValidateCompilers(appArgs, compilerFinder, isExecutionWorker);
        compilers = result.compilers;
    }

    if (appArgs.discoveryOnly) {
        await handleDiscoveryOnlyMode(appArgs.discoveryOnly, compilers, compilerFinder);
    }

    return compilers;
}

/**
 * Load compilers from a prediscovered JSON file
 * @param filename - Path to prediscovered compilers JSON file
 * @param compilerFinder - Compiler finder instance
 * @returns Array of loaded compilers
 */
export async function loadPrediscoveredCompilers(
    filename: string,
    compilerFinder: CompilerFinder,
): Promise<CompilerInfo[]> {
    const prediscoveredCompilersJson = await fs.readFile(filename, 'utf8');
    const initialCompilers = JSON.parse(prediscoveredCompilersJson) as CompilerInfo[];
    const prediscResult = await compilerFinder.loadPrediscovered(initialCompilers);
    if (prediscResult.length === 0) {
        throw new Error('Unexpected failure, no compilers found!');
    }
    return initialCompilers;
}

/**
 * Find and validate compilers for the application
 * @param appArgs - Application arguments
 * @param compilerFinder - Compiler finder instance
 * @param isExecutionWorker - Whether the server is running as an execution worker
 * @returns Object containing compilers and clash status
 */
export async function findAndValidateCompilers(
    appArgs: AppArguments,
    compilerFinder: CompilerFinder,
    isExecutionWorker: boolean,
) {
    const initialFindResults = await compilerFinder.find();
    const initialCompilers = initialFindResults.compilers;
    if (!isExecutionWorker && initialCompilers.length === 0) {
        throw new Error('Unexpected failure, no compilers found!');
    }
    if (appArgs.ensureNoCompilerClash) {
        logger.warn('Ensuring no compiler ids clash');
        if (initialFindResults.foundClash) {
            // If we are forced to have no clashes, throw an error with some explanation
            throw new Error('Clashing compilers in the current environment found!');
        }
        logger.info('No clashing ids found, continuing normally...');
    }
    return initialFindResults;
}

/**
 * Handle discovery-only mode by saving compilers to file and exiting
 * @param savePath - Path to save discovered compilers
 * @param initialCompilers - Array of discovered compilers
 * @param compilerFinder - Compiler finder instance
 */
export async function handleDiscoveryOnlyMode(
    savePath: string,
    initialCompilers: Partial<CompilerInfo>[],
    compilerFinder: CompilerFinder,
) {
    for (const compiler of initialCompilers) {
        if (compiler.buildenvsetup && compiler.buildenvsetup.id === '') delete compiler.buildenvsetup;

        if (compiler.externalparser && compiler.externalparser.id === '') delete compiler.externalparser;

        const compilerInstance =
            compiler.lang && compiler.id
                ? compilerFinder.compileHandler.findCompiler(compiler.lang, compiler.id)
                : undefined;
        if (compilerInstance) {
            compiler.cachedPossibleArguments = compilerInstance.possibleArguments.possibleArguments;
        }
    }
    await fs.writeFile(savePath, JSON.stringify(initialCompilers));
    logger.info(`Discovered compilers saved to ${savePath}`);
    process.exit(0);
}
