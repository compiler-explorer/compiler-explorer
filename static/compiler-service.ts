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

import $ from 'jquery';
import _ from 'underscore';
import {LRUCache} from 'lru-cache';
import {EventEmitter} from 'golden-layout';

import {options} from './options.js';

import {ResultLine} from '../types/resultline/resultline.interfaces.js';

import jqXHR = JQuery.jqXHR;
import ErrorTextStatus = JQuery.Ajax.ErrorTextStatus;
import {CompilerInfo} from '../types/compiler.interfaces.js';
import {CompilationResult, FiledataPair} from '../types/compilation/compilation.interfaces.js';
import {CompilationStatus} from './compiler-service.interfaces.js';
import {IncludeDownloads, SourceAndFiles} from './download-service.js';
import {SentryCapture} from './sentry.js';

const ASCII_COLORS_RE = new RegExp(/\x1B\[[\d;]*m(.\[K)?/g);

export class CompilerService {
    private readonly base = window.httpRoot;
    private allowStoreCodeDebug: boolean;
    private cache: LRUCache<string, CompilationResult>;
    private readonly compilersByLang: Record<string, Record<string, CompilerInfo>>;

    constructor(eventHub: EventEmitter) {
        this.allowStoreCodeDebug = true;
        this.cache = new LRUCache({
            maxSize: 200 * 1024,
            sizeCalculation: n => JSON.stringify(n).length,
        });

        this.compilersByLang = {};

        for (const compiler of options.compilers) {
            if (!(compiler.lang in this.compilersByLang)) this.compilersByLang[compiler.lang] = {};
            this.compilersByLang[compiler.lang][compiler.id] = compiler;
        }

        eventHub.on('settingsChange', newSettings => (this.allowStoreCodeDebug = newSettings.allowStoreCodeDebug));
    }

    private getDefaultCompilerForLang(langId: string) {
        return options.defaultCompiler[langId];
    }

    public processFromLangAndCompiler(
        langId: string | null,
        compilerId: string,
    ): {langId: string | null; compiler: CompilerInfo | null} | null {
        try {
            if (langId) {
                if (!compilerId) {
                    compilerId = this.getDefaultCompilerForLang(langId);
                }

                const foundCompiler = this.findCompiler(langId, compilerId);
                if (!foundCompiler) {
                    const compilers = Object.values(this.getCompilersForLang(langId) ?? {});
                    if (compilers.length > 0) {
                        return {
                            compiler: compilers[0],
                            langId: langId,
                        };
                    } else {
                        return {
                            // There were no compilers, so return null, the selection will show up empty
                            compiler: null,
                            langId: langId,
                        };
                    }
                } else {
                    return {
                        compiler: foundCompiler,
                        langId: langId,
                    };
                }
            } else if (compilerId) {
                const matchingCompilers = Object.values(options.languages).map(lang => {
                    const compiler = this.findCompiler(lang.id, compilerId);
                    if (compiler) {
                        return {
                            langId: lang.id,
                            compiler: compiler,
                        };
                    }
                    return null;
                });
                // Ensure that if no compiler is present, we return null instead of undefined
                return matchingCompilers.find(compiler => compiler !== null) ?? null;
            } else {
                const languages = Object.values(options.languages);
                if (languages.length > 0) {
                    const firstLang = languages[0];
                    return this.processFromLangAndCompiler(firstLang.id, compilerId);
                } else {
                    // TODO: What now? No languages loaded
                    return null;
                }
            }
        } catch (e) {
            SentryCapture(e, 'processFromLangAndCompiler');
        }
        // TODO: What now? Found no compilers!
        return {
            langId: langId,
            compiler: null,
        };
    }

    public getGroupsInUse(langId: string): {value: string; label: string}[] {
        return _.chain(this.getCompilersForLang(langId))
            .map((compiler: CompilerInfo) => compiler)
            .uniq(false, compiler => compiler.group)
            .map(compiler => {
                return {value: compiler.group, label: compiler.groupName || compiler.group};
            })
            .sort((a, b) => {
                return a.label.localeCompare(b.label, undefined /* Ignore language */, {sensitivity: 'base'}) === 0;
            })
            .value();
    }

    getCompilersForLang(langId: string): Record<string, CompilerInfo> | undefined {
        return langId in this.compilersByLang ? this.compilersByLang[langId] : undefined;
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

    findCompiler(langId: string, compilerId: string): CompilerInfo | null {
        if (!compilerId) return null;
        const compilers = this.getCompilersForLang(langId) ?? {};
        return this.findCompilerInList(compilers, compilerId);
    }

    private static handleRequestError(
        request: any,
        reject: (reason?: any) => void,
        xhr: jqXHR,
        textStatus: ErrorTextStatus,
        errorThrown: string,
    ) {
        let error = errorThrown;
        if (!error) {
            switch (textStatus) {
                case 'timeout':
                    error = 'Request timed out';
                    break;
                case 'abort':
                    error = 'Request was aborted';
                    break;
                case 'error':
                    switch (xhr.status) {
                        case 500:
                            error = 'Request failed: internal server error';
                            break;
                        case 504:
                            error = 'Request failed: gateway timeout';
                            break;
                        default:
                            error = 'Request failed: HTTP error code ' + xhr.status;
                            break;
                    }
                    break;
                default:
                    error = 'Error sending request';
                    break;
            }
        }
        reject({
            request: request,
            error: error,
        });
    }

    private getBaseUrl() {
        return window.location.origin + this.base;
    }

    public async submit(request: Record<string, any>) {
        request.allowStoreCodeDebug = this.allowStoreCodeDebug;
        const jsonRequest = JSON.stringify(request);
        if (options.doCache && !request.bypassCache) {
            const cachedResult = this.cache.get(jsonRequest);
            if (cachedResult) {
                return {
                    request: request,
                    result: cachedResult,
                    localCacheHit: true,
                };
            }
        }
        return new Promise((resolve, reject) => {
            const compilerId = encodeURIComponent(request.compiler);
            $.ajax({
                type: 'POST',
                url: `${this.getBaseUrl()}api/compiler/${compilerId}/compile`,
                dataType: 'json',
                contentType: 'application/json',
                data: jsonRequest,
                success: result => {
                    if (result && result.okToCache && options.doCache) {
                        this.cache.set(jsonRequest, result);
                    }
                    resolve({
                        request: request,
                        result: result,
                        localCacheHit: false,
                    });
                },
                error: (jqXHR, textStatus, errorThrown) => {
                    CompilerService.handleRequestError(request, reject, jqXHR, textStatus, errorThrown);
                },
            });
        });
    }

    public submitCMake(request: Record<string, any>) {
        request.allowStoreCodeDebug = this.allowStoreCodeDebug;
        const jsonRequest = JSON.stringify(request);
        if (options.doCache && !request.bypassCache) {
            const cachedResult = this.cache.get(jsonRequest);
            if (cachedResult) {
                return Promise.resolve({
                    request: request,
                    result: cachedResult,
                    localCacheHit: true,
                });
            }
        }
        return new Promise((resolve, reject) => {
            const compilerId = encodeURIComponent(request.compiler);
            $.ajax({
                type: 'POST',
                url: `${this.getBaseUrl()}api/compiler/${compilerId}/cmake`,
                dataType: 'json',
                contentType: 'application/json',
                data: jsonRequest,
                success: result => {
                    if (result && result.okToCache && options.doCache) {
                        this.cache.set(jsonRequest, result);
                    }
                    resolve({
                        request: request,
                        result: result,
                        localCacheHit: false,
                    });
                },
                error: (jqXHR, textStatus, errorThrown) => {
                    CompilerService.handleRequestError(request, reject, jqXHR, textStatus, errorThrown);
                },
            });
        });
    }

    public requestPopularArguments(compilerId: string, usedOptions: string) {
        return new Promise((resolve, reject) => {
            $.ajax({
                type: 'POST',
                url: `${this.getBaseUrl()}api/popularArguments/${compilerId}`,
                dataType: 'json',
                data: JSON.stringify({
                    usedOptions: usedOptions,
                    presplit: false,
                }),
                success: result => {
                    resolve({
                        request: compilerId,
                        result: result,
                        localCacheHit: false,
                    });
                },
                error: (jqXHR, textStatus, errorThrown) => {
                    CompilerService.handleRequestError(compilerId, reject, jqXHR, textStatus, errorThrown);
                },
            });
        });
    }

    private getFilenameFromUrl(url: string): string {
        const jsurl = new URL(url);
        const urlpath = jsurl.pathname;
        return urlpath.substring(urlpath.lastIndexOf('/') + 1);
    }

    public async expandToFiles(source: string): Promise<SourceAndFiles> {
        const includes = new IncludeDownloads();

        const includeFind = /^\s*#\s*include\s*["<](https?:\/\/[^">]+)[">]/;
        const lines = source.split('\n');
        for (const idx in lines) {
            const line = lines[idx];
            const match = line.match(includeFind);
            if (match) {
                const download = includes.include(match[1]);
                lines[idx] = `#include "${download.filename}"`;
            }
        }

        const files: FiledataPair[] = await includes.allDownloadsAsFileDataPairs();

        return {
            source: lines.join('\n'),
            files: files,
        };
    }

    public static getSelectizerOrder() {
        return [{field: '$order'}, {field: '$score'}, {field: 'name'}];
    }

    public static doesCompilationResultHaveWarnings(result: CompilationResult) {
        // TODO: Types probably need to be updated here
        // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
        const stdout = result.stdout ?? [];
        // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
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

    public static calculateStatusIcon(result: CompilationResult) {
        let code = 1;
        if (result.code !== 0) {
            code = 3;
        } else if (this.doesCompilationResultHaveWarnings(result)) {
            code = 2;
        }
        return {code: code, compilerOut: result.code};
    }

    private static getAriaLabel(status: CompilationStatus) {
        // Compiling...
        if (status.code === 4) return 'Compiling';
        if (status.compilerOut === 0) {
            // StdErr.length > 0
            if (status.code === 3) return 'Compilation succeeded with errors';
            // StdOut.length > 0
            if (status.code === 2) return 'Compilation succeeded with warnings';
            return 'Compilation succeeded';
        } else {
            // StdErr.length > 0
            if (status.code === 3) return 'Compilation failed with errors';
            // StdOut.length > 0
            if (status.code === 2) return 'Compilation failed with warnings';
            return 'Compilation failed';
        }
    }

    private static getColor(status: CompilationStatus) {
        // Compiling...
        if (status.code === 4) return '#888888';
        if (status.compilerOut === 0) {
            // StdErr.length > 0
            if (status.code === 3) return '#FF6645';
            // StdOut.length > 0
            if (status.code === 2) return '#FF6500';
            return '#12BB12';
        } else {
            // StdErr.length > 0
            if (status.code === 3) return '#FF1212';
            // StdOut.length > 0
            if (status.code === 2) return '#BB8700';
            return '#FF6645';
        }
    }

    public static handleCompilationStatus(
        statusLabel: JQuery | null,
        statusIcon: JQuery | null,
        status: CompilationStatus,
    ) {
        if (statusLabel != null) {
            statusLabel.toggleClass('error', status.code === 3).toggleClass('warning', status.code === 2);
        }

        if (statusIcon != null) {
            statusIcon
                .removeClass()
                .addClass('status-icon fas')
                .css('color', this.getColor(status))
                .toggle(status.code !== 0)
                .attr('aria-label', this.getAriaLabel(status))
                .toggleClass('fa-spinner fa-spin', status.code === 4)
                .toggleClass('fa-times-circle', status.code === 3)
                .toggleClass('fa-check-circle', status.code === 1 || status.code === 2);
        }
    }

    public static handleOutputButtonTitle(element: JQuery, result: CompilationResult) {
        // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
        const stdout = result.stdout ?? [];
        // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
        const stderr = result.stderr ?? [];

        function filterAsciiColors(line: ResultLine) {
            return line.text.replace(ASCII_COLORS_RE, '');
        }

        const output = stdout.map(filterAsciiColors).concat(stderr.map(filterAsciiColors)).join('\n');

        element.prop('title', output);
    }
}
