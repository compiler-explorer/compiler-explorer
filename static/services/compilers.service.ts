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

import {CompilerInfo} from '../../types/compiler.interfaces.js';
import {optionsHash} from '../options.js';

export class CompilersService {
    private readonly loadPromises = new Map<string, Promise<Record<string, CompilerInfo>>>();

    async getCompilersForLang(langId: string): Promise<Record<string, CompilerInfo>> {
        let promise = this.loadPromises.get(langId);
        if (!promise) {
            promise = this.fetchCompilersForLang(langId);
            this.loadPromises.set(langId, promise);
        }
        return promise;
    }

    private static readonly compilerFields = [
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
    ].join(',');

    private async fetchCompilersForLang(langId: string): Promise<Record<string, CompilerInfo>> {
        const response = await fetch(
            `${window.httpRoot}api/compilers/${encodeURIComponent(langId)}?fields=${CompilersService.compilerFields}&hash=${optionsHash}`,
            {headers: {Accept: 'application/json'}},
        );
        const compilers: CompilerInfo[] = await response.json();
        const result: Record<string, CompilerInfo> = {};
        for (const compiler of compilers) {
            result[compiler.id] = compiler;
        }
        return result;
    }
}

export const compilersService = new CompilersService();
