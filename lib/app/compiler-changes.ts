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

import {CompilerInfo} from '../../types/compiler.interfaces.js';
import type {Language, LanguageKey} from '../../types/languages.interfaces.js';
import type {AppArguments} from '../app.interfaces.js';
import {unwrap} from '../assert.js';
import {CompilerFinder} from '../compiler-finder.js';
import {RouteAPI} from '../handlers/route-api.js';
import {logger} from '../logger.js';
import {ClientOptionsHandler} from '../options-handler.js';
import {PropertyGetter} from '../properties.interfaces.js';

/**
 * Setup handling of compiler changes and periodic rescanning
 * @param initialCompilers - The initial set of compilers
 * @param clientOptionsHandler - Client options handler
 * @param routeApi - Route API instance
 * @param languages - Available languages
 * @param ceProps - CE properties
 * @param compilerFinder - Compiler finder instance
 * @param appArgs - Application arguments
 */
export async function setupCompilerChangeHandling(
    initialCompilers: CompilerInfo[],
    clientOptionsHandler: ClientOptionsHandler,
    routeApi: RouteAPI,
    languages: Record<LanguageKey, Language>,
    ceProps: PropertyGetter,
    compilerFinder: CompilerFinder,
    appArgs: AppArguments,
): Promise<void> {
    let prevCompilers = '';

    /**
     * Handle compiler change events
     * @param compilers - New set of compilers
     */
    async function onCompilerChange(compilers: CompilerInfo[]) {
        const compilersAsJson = JSON.stringify(compilers);
        if (prevCompilers === compilersAsJson) {
            return;
        }
        logger.info(`Compiler scan count: ${compilers.length}`);
        logger.debug('Compilers:', compilers);
        prevCompilers = compilersAsJson;
        await clientOptionsHandler.setCompilers(compilers);
        const apiHandler = unwrap(routeApi.apiHandler);
        apiHandler.setCompilers(compilers);
        apiHandler.setLanguages(languages);
        apiHandler.setOptions(clientOptionsHandler);
    }

    // Set initial compilers
    await onCompilerChange(initialCompilers);

    // Set up compiler rescanning if configured
    const rescanCompilerSecs = ceProps('rescanCompilerSecs', 0);
    if (rescanCompilerSecs && !appArgs.prediscovered) {
        logger.info(`Rescanning compilers every ${rescanCompilerSecs} secs`);
        setInterval(
            () => compilerFinder.find().then(result => onCompilerChange(result.compilers)),
            rescanCompilerSecs * 1000,
        );
    }
}
