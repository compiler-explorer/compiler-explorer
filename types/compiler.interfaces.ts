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

import {ICompilerArguments} from './compiler-arguments.interfaces';
import {Language, LanguageKey} from './languages.interfaces';
import {Library} from './libraries/libraries.interfaces';
import {Tool, ToolInfo} from './tool.interfaces';

export type CompilerInfo = {
    id: string;
    exe: string;
    name: string;
    version: string;
    fullVersion: string;
    baseName: string;
    alias: string[];
    options: string;
    versionFlag?: string;
    versionRe?: string;
    explicitVersion?: string;
    compilerType: string;
    debugPatched: boolean;
    demangler: string;
    demanglerType: string;
    objdumper: string;
    objdumperType: string;
    intelAsm: string;
    supportsAsmDocs: boolean;
    instructionSet: string;
    needsMulti: boolean;
    adarts: string;
    supportsDeviceAsmView?: boolean;
    supportsDemangle?: boolean;
    supportsBinary?: boolean;
    supportsBinaryObject?: boolean;
    supportsIntel?: boolean;
    interpreted?: boolean;
    // (interpreted || supportsBinary) && supportsExecute
    supportsExecute?: boolean;
    supportsGccDump?: boolean;
    supportsFiltersInBinary?: boolean;
    supportsOptOutput?: boolean;
    supportsPpView?: boolean;
    supportsAstView?: boolean;
    supportsIrView?: boolean;
    supportsLLVMOptPipelineView?: boolean;
    supportsRustMirView?: boolean;
    supportsRustMacroExpView?: boolean;
    supportsRustHirView?: boolean;
    supportsHaskellCoreView?: boolean;
    supportsHaskellStgView?: boolean;
    supportsHaskellCmmView?: boolean;
    supportsCfg?: boolean;
    supportsGnatDebugViews?: boolean;
    supportsLibraryCodeFilter?: boolean;
    executionWrapper: string;
    postProcess: string[];
    lang: LanguageKey;
    group: string;
    groupName: string;
    $groups: string[];
    includeFlag: string;
    includePath: string;
    linkFlag: string;
    rpathFlag: string;
    libpathFlag: string;
    libPath: string[];
    ldPath: string[];
    // [env, setting][]
    envVars: [string, string][];
    notification: string;
    isSemVer: boolean;
    semver: string;
    libsArr: Library['id'][];
    tools: Record<ToolInfo['id'], Tool>;
    unwiseOptions: string[];
    hidden: boolean;
    buildenvsetup: {
        id: string;
        props: (name: string, def: string) => string;
    };
    license?: {
        link?: string;
        name?: string;
        preamble?: string;
    };
    remote?: {
        target: string;
        path: string;
    };
    disabledFilters: string[];
    optArg?: string;
    externalparser: any;
    removeEmptyGccDump?: boolean;
    irArg?: string[];
    llvmOptArg?: string[];
    llvmOptModuleScopeArg?: string[];
    llvmOptNoDiscardValueNamesArg?: string[];
    cachedPossibleArguments?: any;
    nvdisasm?: string;
    mtime?: any;
};

// Compiler information collected by the compiler-finder
export type PreliminaryCompilerInfo = Omit<
    CompilerInfo,
    'version' | 'fullVersion' | 'baseName' | '$groups' | 'disabledFilters'
> & {version?: string};

export interface ICompiler {
    possibleArguments: ICompilerArguments;
    lang: Language;
    compile(source, options, backendOptions, filters, bypassCache, tools, executionParameters, libraries, files);
    cmake(files, key);
    initialise(mtime: Date, clientOptions, isPrediscovered: boolean);
    getInfo(): CompilerInfo;
}
