// Copyright (c) 2026, Compiler Explorer Authors
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

import {CompilerInfo, DEDUPABLE_COMPILER_FIELDS, DedupedCompilerList} from '../../types/compiler.interfaces.js';
import {optionsHash} from '../options.js';
import {SentryCapture} from '../sentry.js';

export class CompilersService {
    private readonly loadPromises = new Map<string, Promise<Record<string, CompilerInfo>>>();

    async getCompilersForLang(langId: string): Promise<Record<string, CompilerInfo>> {
        let promise = this.loadPromises.get(langId);
        if (!promise) {
            promise = this.fetchCompilersForLang(langId);
            this.loadPromises.set(langId, promise);
            promise.catch(e => {
                SentryCapture(e, `fetchCompilersForLang(${langId})`);
                this.loadPromises.delete(langId);
            });
        }
        return promise;
    }

    private static readonly compilerFields: (keyof CompilerInfo)[] = [
        'id',
        'name',
        'version',
        'fullVersion',
        'baseName',
        'alias',
        'lang',
        'group',
        'groupName',
        'options',
        'tools',
        'libsArr',
        'license',
        'remote',
        'optPipeline',
        'irArg',
        'minIrArgs',
        'hidden',
        'intelAsm',
        'notification',
        'instructionSet',
        'unwiseOptions',
        'possibleOverrides',
        'possibleRuntimeTools',
        'compilerCategories',
        'semver',
        'isNightly',
        'disabledFilters',
        '$order',
        'supportsExecute',
        'supportsAsmDocs',
        'supportsIntel',
        'supportsBinary',
        'supportsBinaryObject',
        'supportsOptOutput',
        'supportsStackUsageOutput',
        'supportsPpView',
        'supportsGccDump',
        'supportsIrView',
        'supportsAstView',
        'supportsRustMirView',
        'supportsRustMacroExpView',
        'supportsRustHirView',
        'supportsClangirView',
        'supportsHaskellCoreView',
        'supportsHaskellStgView',
        'supportsHaskellCmmView',
        'supportsLeanCView',
        'supportsClojureMacroExpView',
        'supportsYulView',
        'supportsCfg',
        'supportsGnatDebugViews',
        'supportsLibraryCodeFilter',
        'supportsDeviceAsmView',
        'supportsIrViewOptToggleOption',
        'supportsDemangle',
        'supportsVerboseDemangling',
        'supportsFiltersInBinary',
    ];

    /**
     * Expand a `?dedupe=` envelope back into plain `CompilerInfo`s. The table entries are shared between every
     * compiler that references them, so nothing downstream may mutate a deduplicated field in place.
     */
    private static rehydrate(body: DedupedCompilerList): CompilerInfo[] {
        return body.compilers.map(entry => {
            const compiler = entry as unknown as CompilerInfo;
            for (const field of DEDUPABLE_COMPILER_FIELDS) {
                const indices = entry[field];
                const table: unknown[] | undefined = body.refs[field];
                if (indices && table) {
                    (compiler as Record<string, unknown>)[field] = indices.map(index => table[index]);
                }
            }
            return compiler;
        });
    }

    private async fetchCompilersForLang(langId: string): Promise<Record<string, CompilerInfo>> {
        // `possibleOverrides` stays in `fields` so that an instance predating `dedupe` still answers usefully: it
        // ignores the unknown parameter and sends today's inline shape, which `Array.isArray` picks out below.
        const response = await fetch(
            `${window.httpRoot}api/compilers/${encodeURIComponent(langId)}?fields=${CompilersService.compilerFields.join(',')}&dedupe=${DEDUPABLE_COMPILER_FIELDS.join(',')}&hash=${optionsHash}`,
            {headers: {Accept: 'application/json'}},
        );
        const body: CompilerInfo[] | DedupedCompilerList = await response.json();
        const compilers = Array.isArray(body) ? body : CompilersService.rehydrate(body);
        const result: Record<string, CompilerInfo> = {};
        for (const compiler of compilers) {
            result[compiler.id] = compiler;
        }
        return result;
    }
}

export const compilersService = new CompilersService();
