// Copyright (c) 2022, Compiler Explorer Authors
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

import {EventEmitter} from 'golden-layout';
import {LRUCache} from 'lru-cache';
import _ from 'underscore';

import type {BuildSystemId} from '../shared/build-systems.js';
import {CompilationResult, FiledataPair} from '../types/compilation/compilation.interfaces.js';
import {CompilerInfo} from '../types/compiler.interfaces.js';
import {ResultLine} from '../types/resultline/resultline.interfaces.js';
import {getBackendApi} from './api/backend-api.js';
import {CompilationStatus, CompilationStatusCode} from './compiler-service.interfaces.js';
import {IncludeDownloads, SourceAndFiles} from './download-service.js';
import {options} from './options.js';
import {SentryCapture} from './sentry.js';
import {compilersService} from './services/compilers.service.js';
import {languagesService} from './services/languages.service.js';
import {SiteSettings} from './settings.js';

const ASCII_COLORS_RE = new RegExp(/\x1B\[[\d;]*m(.\[K)?/g);

type HasCompilationStatus = Partial<Pick<CompilationResult, 'code' | 'stdout' | 'stderr' | 'inputFilename'>>;

export class CompilerService {
    private allowStoreCodeDebug: boolean;
    private cache: LRUCache<string, CompilationResult>;

    constructor(eventHub: EventEmitter) {
        this.allowStoreCodeDebug = true;
        this.cache = new LRUCache({
            maxSize: 200 * 1024,
            sizeCalculation: n => JSON.stringify(n).length,
        });

        eventHub.on(
            'settingsChange',
            (newSettings: SiteSettings) => (this.allowStoreCodeDebug = newSettings.allowStoreCodeDebug),
        );
    }

    private async getDefaultCompilerForLang(langId: string): Promise<string> {
        const languages = await languagesService.getLanguages();
        return languages[langId]?.defaultCompiler ?? '';
    }

    public async processFromLangAndCompiler(
        langId: string | null,
        compilerId: string,
    ): Promise<{langId: string | null; compiler: CompilerInfo | null} | null> {
        try {
            if (langId) {
                if (!compilerId) {
                    compilerId = await this.getDefaultCompilerForLang(langId);
                }

                const foundCompiler = await this.findCompiler(langId, compilerId);
                if (!foundCompiler) {
                    const compilers = Object.values((await this.getCompilersForLang(langId)) ?? {});
                    if (compilers.length > 0) {
                        return {
                            compiler: compilers[0],
                            langId: langId,
                        };
                    }
                    return {
                        compiler: null,
                        langId: langId,
                    };
                }
                return {
                    compiler: foundCompiler,
                    langId: langId,
                };
            }
            if (compilerId) {
                for (const lang of Object.values(languagesService.getLanguagesOrFail())) {
                    const compiler = await this.findCompiler(lang.id, compilerId);
                    if (compiler) {
                        return {
                            langId: lang.id,
                            compiler: compiler,
                        };
                    }
                }
                return null;
            }
            const languages = Object.values(languagesService.getLanguagesOrFail());
            if (languages.length > 0) {
                const firstLang = languages[0];
                return this.processFromLangAndCompiler(firstLang.id, compilerId);
            }
            return null;
        } catch (e) {
            SentryCapture(e, 'processFromLangAndCompiler');
        }
        return {
            langId: langId,
            compiler: null,
        };
    }

    public async getGroupsInUse(langId: string): Promise<{value: string; label: string}[]> {
        const compilers = (await this.getCompilersForLang(langId)) ?? {};
        return _.uniq(Object.values(compilers), false, compiler => compiler.group)
            .map(compiler => {
                return {value: compiler.group, label: compiler.groupName || compiler.group};
            })
            .sort((a, b) =>
                a.label.localeCompare(b.label, undefined /* Ignore language */, {
                    sensitivity: 'base',
                }),
            );
    }

    async getCompilersForLang(langId: string): Promise<Record<string, CompilerInfo>> {
        return compilersService.getCompilersForLang(langId);
    }

    private findCompilerInList(compilers: Record<string, CompilerInfo>, compilerId: string) {
        if (compilerId in compilers) {
            return compilers[compilerId];
        }
        for (const id in compilers) {
            if (compilers[id].alias.includes(compilerId)) {
                return compilers[id];
            }
        }
        return null;
    }

    async findCompiler(langId: string, compilerId: string): Promise<CompilerInfo | null> {
        if (!compilerId) return null;
        const compilers = await this.getCompilersForLang(langId);
        return this.findCompilerInList(compilers, compilerId);
    }

    private async submitToCompiler(
        request: Record<string, any>,
        compileFn: (compilerId: string, req: Record<string, unknown>) => Promise<CompilationResult>,
    ) {
        request.allowStoreCodeDebug = this.allowStoreCodeDebug;
        const jsonRequest = JSON.stringify(request);
        if (options.doCache && !request.bypassCache) {
            const cachedResult = this.cache.get(jsonRequest);
            if (cachedResult) return {request, result: cachedResult, localCacheHit: true};
        }
        const result = await compileFn(request.compiler as string, request);
        if (result?.okToCache && options.doCache) this.cache.set(jsonRequest, result);
        return {request, result, localCacheHit: false};
    }

    public submit(request: Record<string, any>) {
        return this.submitToCompiler(request, (id, req) => getBackendApi().compile(id, req));
    }

    public submitBuild(buildSystem: BuildSystemId, request: Record<string, any>) {
        return this.submitToCompiler(request, (id, req) => getBackendApi().compileBuild(id, buildSystem, req));
    }

    public async requestPopularArguments(compilerId: string, usedOptions: string) {
        const result = await getBackendApi().popularArguments(compilerId, usedOptions);
        return {request: compilerId, result, localCacheHit: false};
    }

    public async expandToFiles(source: string): Promise<SourceAndFiles> {
        const includesOrEmbeds = new IncludeDownloads();

        const includeOrEmbedFind = /^\s*#\s*(include|embed)\s*["<](https?:\/\/[^">]+)[">]/;
        const lines = source.split('\n');
        for (const idx in lines) {
            const line = lines[idx];
            const match = line.match(includeOrEmbedFind);
            if (match) {
                const download = includesOrEmbeds.include(match[2]);
                lines[idx] = line.replace(includeOrEmbedFind, `#${match[1]} "${download.filename}"`);
            }
        }

        const files: FiledataPair[] = await includesOrEmbeds.allDownloadsAsFileDataPairs();

        return {
            source: lines.join('\n'),
            files: files,
        };
    }

    public static getSelectizerOrder() {
        return [{field: '$order'}, {field: '$score'}, {field: 'name'}];
    }

    public static doesCompilationResultHaveWarnings(result: HasCompilationStatus) {
        const stdout = result.stdout ?? [];
        const stderr = result.stderr ?? [];

        // TODO: Pass what compiler did this and check if it it's actually skippable
        // Right now we're ignoring outputs that match the input filename
        // Compiler & Executor are capable of giving us the info, but conformance view is not
        if (stdout.length === 1 && stderr.length === 0 && result.inputFilename) {
            // This code is a special case for MSVC which writes the filename to stdout
            // MSVC will use back-slashes, Wine will use forward slashes
            // We could also move this calculation to the server at some point
            const lastSlashPos = _.findLastIndex(result.inputFilename, ch => ch === '\\' || ch === '/');
            return result.inputFilename.substring(lastSlashPos + 1) !== stdout[0].text;
        }
        return stdout.length > 0 || stderr.length > 0;
    }

    public static calculateStatusIcon(result: HasCompilationStatus): CompilationStatus {
        let code = 1;
        if (result.code === undefined) {
            code = 5;
        } else if (result.code !== 0) {
            code = 3;
        } else if (CompilerService.doesCompilationResultHaveWarnings(result)) {
            code = 2;
        }
        return {code: code, compilerOut: result.code};
    }

    private static getAriaLabel(status: CompilationStatus) {
        const compilationResult = (() => {
            if (status.code === CompilationStatusCode.UNKNOWN) return 'Unknown';
            // Compiling...
            if (status.code === CompilationStatusCode.COMPILING) return 'Compiling';
            if (status.compilerOut === 0) {
                // StdErr.length > 0
                if (status.code === CompilationStatusCode.WITH_ERRORS) return 'Compilation succeeded with errors';
                // StdOut.length > 0
                if (status.code === CompilationStatusCode.WITH_WARNINGS) return 'Compilation succeeded with warnings';
                return 'Compilation succeeded';
            }
            // StdErr.length > 0
            if (status.code === CompilationStatusCode.WITH_ERRORS) return 'Compilation failed with errors';
            // StdOut.length > 0
            if (status.code === CompilationStatusCode.WITH_WARNINGS) return 'Compilation failed with warnings';
            return 'Compilation failed';
        })();
        const executionResult = (() => {
            switch (status.didExecute) {
                case undefined:
                    return '';
                case true:
                    return '; execution successful';
                case false:
                    return '; execution failed';
            }
        })();
        return `${compilationResult}${executionResult}`;
    }

    private static getColor(status: CompilationStatus) {
        // Compiling...
        if (status.code === CompilationStatusCode.COMPILING) return '#888888';
        if (status.didExecute !== undefined) {
            return status.didExecute ? '#12BB12' : '#FF1212';
        }
        if (status.compilerOut === 0) {
            // StdErr.length > 0
            if (status.code === CompilationStatusCode.WITH_ERRORS) return '#FF6645';
            // StdOut.length > 0
            if (status.code === CompilationStatusCode.WITH_WARNINGS) return '#FF6500';
            return '#12BB12';
        }
        // StdErr.length > 0
        if (status.code === CompilationStatusCode.WITH_ERRORS) return '#FF1212';
        // StdOut.length > 0
        if (status.code === CompilationStatusCode.WITH_WARNINGS) return '#BB8700';
        return '#FF6645';
    }

    public static handleCompilationStatus(statusIcon: JQuery | null, status: CompilationStatus) {
        if (statusIcon != null) {
            statusIcon
                .removeClass()
                .addClass('status-icon fas')
                .css('color', CompilerService.getColor(status))
                .toggle(status.code !== CompilationStatusCode.NONE)
                .attr('aria-label', CompilerService.getAriaLabel(status))
                .toggleClass('fa-spinner fa-spin', status.code === CompilationStatusCode.COMPILING);
            let error, unknown, okay;
            if (status.didExecute === undefined) {
                error = status.code === CompilationStatusCode.WITH_ERRORS;
                unknown = status.code === CompilationStatusCode.UNKNOWN;
                okay = status.code === CompilationStatusCode.OK || status.code === CompilationStatusCode.WITH_WARNINGS;
            } else {
                error = status.code !== CompilationStatusCode.COMPILING && !status.didExecute;
                unknown = false;
                okay = status.code !== CompilationStatusCode.COMPILING && status.didExecute;
            }
            statusIcon
                .toggleClass('fa-times-circle', error)
                .toggleClass('fa-circle-question', unknown)
                .toggleClass('fa-check-circle', okay);
        }
    }

    public static handleOutputButtonTitle(element: JQuery, result: CompilationResult) {
        const stdout = result.stdout ?? [];

        const stderr = result.stderr ?? [];

        function filterAsciiColors(line: ResultLine) {
            return line.text.replace(ASCII_COLORS_RE, '');
        }

        const output = stdout.map(filterAsciiColors).concat(stderr.map(filterAsciiColors)).join('\n');

        element.prop('title', output);
    }
}
