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

import {Language} from '../types/languages.interfaces';
import {CompilerFilters} from '../types/features/filters.interfaces';
import {MessageWithLocation} from '../types/resultline/resultline.interfaces';
import {SiteSettings} from './settings';
import {Theme} from './themes';
import {PPOptions} from './panes/pp-view.interfaces';
import {GccSelectedPass} from './panes/gccdump-view.interfaces';
import {Motd} from './motd.interfaces';
import {CompilerInfo} from '../types/compiler.interfaces';
import {CompilationResult} from '../types/compilation/compilation.interfaces';

// This list comes from executing
// grep -rPo "eventHub\.(on|emit)\('.*'," static/ | cut -d "'" -f2 | sort | uniq
export type EventMap = {
    astViewClosed: (compilerId: number) => void;
    astViewOpened: (compilerId: number) => void;
    broadcastFontScale: (scale: number) => void;
    cfgViewClosed: (compilerId: number) => void;
    cfgViewOpened: (compilerId: number) => void;
    colours: (editorId: number, colours: Record<number, number>, scheme: string) => void;
    coloursForCompiler: (compilerId: number, colours: Record<number, number>, scheme: string) => void;
    coloursForEditor: (editorId: number, colours: Record<number, number>, scheme: string) => void;
    compiler: (
        compilerId: number,
        compiler: CompilerInfo | null,
        options: string,
        editorId: number,
        treeId: number
    ) => void;
    compilerClose: (compilerId: number, treeId: boolean | number) => void;
    compileResult: (
        compilerId: number,
        compiler: CompilerInfo,
        result: CompilationResult,
        language: Language | undefined
    ) => void;
    compilerFavoriteChange: (compilerId: number) => void;
    compilerFlagsChange: (compilerId: number, options: string) => void;
    compilerOpen: (compilerId: number, editorId: number, treeId: number | boolean) => void;
    // Right now nothing emits this event, but it might be useful at some point so we keep it
    compilerSetDecorations: (compilerId: number, lineNums: number[], revealLine: boolean, column?: number) => void;
    compiling: (compilerId: number, compiler: CompilerInfo) => void;
    conformanceViewClose: (editorId: number) => void;
    conformanceViewOpen: (editorId: number) => void;
    copyShortLinkToClip: () => void;
    deviceViewClosed: (compilerId: number) => void;
    deviceViewOpened: (compilerId: number) => void;
    displaySharingPopover: () => void;
    editorChange: (editorId: number, source: string, langId: string, compilerId?: number) => void;
    editorClose: (editorId: number) => void;
    editorDisplayFlow: (editorId: number, flow: MessageWithLocation[]) => void;
    editorLinkLine: (editorId: number, lineNumber: number, colBegin: number, colEnd: number, reveal: boolean) => void;
    editorOpen: (editorId: number) => void;
    editorSetDecoration: (editorId: number, lineNumber: number, reveal: boolean) => void;
    executeResult: (executorId: number, compiler: any, result: any, language: Language) => void;
    executor: (
        executorId: number,
        compiler: any,
        options: string,
        editorId: boolean | number,
        treeId: boolean | number
    ) => void;
    executorClose: (executorId: number) => void;
    executorOpen: (executorId: number, editorId: boolean | number) => void;
    filtersChange: (compilerId: number, filters: CompilerFilters) => void;
    findCompilers: () => void;
    findEditors: () => void;
    findExecutors: () => void;
    flagsViewClosed: (compilerId: number, options: string) => void;
    flagsViewOpened: (compilerId: number) => void;
    gccDumpFiltersChanged: (compilerId: number, filters: CompilerFilters, recompile: boolean) => void;
    gccDumpPassSelected: (compilerId: number, pass: GccSelectedPass, recompile: boolean) => void;
    gccDumpUIInit: (compilerId: number) => void;
    gccDumpViewClosed: (compilerId: number) => void;
    gccDumpViewOpened: (compilerId: number) => void;
    gnatDebugTreeViewClosed: (compilerId: number) => void;
    gnatDebugTreeViewOpened: (compilerId: number) => void;
    gnatDebugViewClosed: (compilerId: number) => void;
    gnatDebugViewOpened: (compilerId: number) => void;
    haskellCmmViewClosed: (compilerId: number) => void;
    haskellCmmViewOpened: (compilerId: number) => void;
    haskellCoreViewClosed: (compilerId: number) => void;
    haskellCoreViewOpened: (compilerId: number) => void;
    haskellStgViewClosed: (compilerId: number) => void;
    haskellStgViewOpened: (compilerId: number) => void;
    initialised: () => void;
    irViewClosed: (compilerId: number) => void;
    irViewOpened: (compilerId: number) => void;
    llvmOptPipelineViewClosed: (compilerId: number) => void;
    llvmOptPipelineViewOpened: (compilerId: number) => void;
    llvmOptPipelineViewOptionsUpdated: (compilerId: number, options: any, recompile: boolean) => void;
    languageChange: (editorId: number | boolean, newLangId: string, treeId?: boolean | number) => void;
    modifySettings: (modifiedSettings: Partial<SiteSettings>) => void;
    motd: (data: Motd) => void;
    newSource: (editorId: number, newSource: string) => void;
    optViewClosed: (compilerId: number) => void;
    optViewOpened: (compilerId: number) => void;
    outputClosed: (compilerId: number) => void;
    outputOpened: (compilerId: number) => void;
    panesLinkLine: (
        compilerId: number,
        lineNumber: number,
        colBegin: number,
        colEnd: number,
        reveal: boolean,
        sender: string,
        editorId?: number
    ) => void;
    ppViewClosed: (compilerId: number) => void;
    ppViewOpened: (compilerId: number) => void;
    ppViewOptionsUpdated: (compilerId: number, options: PPOptions, recompile: boolean) => void;
    requestCompilation: (editorId: number | boolean, treeId: boolean | number) => void;
    requestMotd: () => void;
    requestSettings: () => void;
    requestTheme: () => void;
    resendCompilation: (compilerId: number) => void;
    requestCompiler: (compilerId: number) => void;
    requestFilters: (compilerId: number) => void;
    resendExecution: (executorId: number) => void;
    resize: () => void;
    rustHirViewClosed: (compilerId: number) => void;
    rustHirViewOpened: (compilerId: number) => void;
    rustMacroExpViewClosed: (compilerId: number) => void;
    rustMacroExpViewOpened: (compilerId: number) => void;
    rustMirViewClosed: (compilerId: number) => void;
    rustMirViewOpened: (compilerId: number) => void;
    // TODO: There are no emitters for this event
    selectLine: (editorId: number, lineNumber: number) => void;
    settingsChange: (newSettings: SiteSettings) => void;
    setToolInput: (compilerId: number, toolId: number, string: string) => void;
    shown: () => void;
    themeChange: (newTheme: Theme | null) => void;
    toolClosed: (compilerId: number, toolState: unknown) => void;
    toolInputChange: (compilerId: number, toolId: number, input: string) => void;
    toolInputViewClosed: (compilerId: number, toolId: number, input: string) => void;
    toolInputViewCloseRequest: (compilerId: number, toolId: number) => void;
    toolOpened: (compilerId: number, toolState: unknown) => void;
    toolSettingsChange: (compilerId: number) => void;
    treeClose: (treeId: number) => void;
    treeCompilerEditorExcludeChange: (treeId: number, compilerId: number, editorId: number) => void;
    treeCompilerEditorIncludeChange: (treeId: number, compilerId: number, editorId: number) => void;
    treeOpen: (treeId: number) => void;
};
