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
import {getHash} from '../utils.js';

export async function setupCompilerChangeHandling(
    initialCompilers: CompilerInfo[],
    clientOptionsHandler: ClientOptionsHandler,
    routeApi: RouteAPI,
    languages: Record<LanguageKey, Language>,
    ceProps: PropertyGetter,
    compilerFinder: CompilerFinder,
    appArgs: AppArguments,
): Promise<void> {
    const rescanCompilerSecs = ceProps('rescanCompilerSecs', 0);
    const rescanEnabled = !!rescanCompilerSecs && !appArgs.prediscovered;
    let prevCompilersHash: string | undefined;

    async function onCompilerChange(compilers: CompilerInfo[]) {
        if (rescanEnabled) {
            // Compare by content hash so no-op rescans are skipped without
            // retaining a full serialised copy of every compiler, which for a
            // production-sized set runs to tens of megabytes.
            const compilersHash = getHash(compilers);
            if (prevCompilersHash === compilersHash) {
                return;
            }
            prevCompilersHash = compilersHash;
        }
        logger.info(`Compiler scan count: ${compilers.length}`);
        logger.debug('Compilers:', compilers);
        await clientOptionsHandler.setCompilers(compilers);
        const apiHandler = unwrap(routeApi.apiHandler);
        apiHandler.setCompilers(compilers);
        apiHandler.setLanguages(languages);
        apiHandler.setOptions(clientOptionsHandler);
    }

    await onCompilerChange(initialCompilers);

    if (rescanEnabled) {
        logger.info(`Rescanning compilers every ${rescanCompilerSecs} secs`);
        setInterval(
            () => compilerFinder.find().then(result => onCompilerChange(result.compilers)),
            rescanCompilerSecs * 1000,
        );
    }
}
