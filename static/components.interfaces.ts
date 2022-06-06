// Copyright (c) 2021, Compiler Explorer Authors
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

export interface ComponentConfig<S> {
    type: string;
    componentName: string;
    componentState: S;
}

type StateWithLanguage = {lang: string};
type StateWithEditor = {source: number};
type StateWithTree = {tree: number};
type StateWithId = {id: number};
type EmptyState = Record<never, never>;

export type EmptyCompilerState = StateWithLanguage & StateWithEditor;
export type PopulatedCompilerState = StateWithEditor & {
    filters: Record<string, boolean>; // TODO: compilation filters
    options: unknown;
    compiler: string;
    libs?: unknown;
    lang?: string;
};
export type CompilerForTreeState = StateWithLanguage & StateWithTree;

export type EmptyExecutorState = StateWithLanguage & StateWithEditor;
export type PopulatedExecutorState = StateWithLanguage & StateWithEditor & StateWithTree & {
    compiler: string;
    libs: unknown;
    options: unknown;
};
export type ExecutorForTreeState = StateWithLanguage & StateWithTree;

export type EmptyEditorState = Partial<StateWithId & StateWithLanguage>;
export type PopulatedEditorState = StateWithId & {
    source: string;
    options: unknown;
};

export type EmptyTreeState = Partial<StateWithId>;

export type OutputState = StateWithTree & {
    compiler: string;
    editor: number; // EditorId
}

export type ToolViewState = StateWithTree & {
    compiler: string;
    editor: number; // EditorId
    toolId: string;
    args: unknown;
    monacoStdin: boolean;
}

export type EmptyToolInputViewState = EmptyState;
export type PopulatedToolInputViewState = {
    compilerId: string;
    toolId: string;
    toolName: string;
}

export type EmptyDiffViewState = EmptyState;
export type PopulatedDiffViewState = {
    lhs: unknown;
    rhs: unknown;
}

export type EmptyOptViewState = EmptyState;
export type PopulatedOptViewState = StateWithId & StateWithEditor & {
    optOutput: unknown;
    compilerName: string;
    editorid: number;
    treeid: number;
}

export type EmptyFlagsViewState = EmptyState;
export type PopulatedFlagsViewState = StateWithId & {
    compilerName: string;
    compilerFlags: unknown;
}

export type EmptyPpViewState = EmptyState;
export type PopulatedPpViewState = StateWithId & StateWithEditor & {
    ppOutput: unknown;
    compilerName: string;
    editorid: number;
    treeid: number;
}

export type EmptyAstViewState = EmptyState;
export type PopulatedAstViewState = StateWithId & StateWithEditor & {
    astOutput: unknown;
    compilerName: string;
    editorid: number;
    treeid: number;
}

export type EmptyGccDumpViewState = EmptyState;
export type GccDumpOptions =
    | 'treeDump'
    | 'rtlDump'
    | 'ipaDump'
    | 'addressOption'
    | 'slimOption'
    | 'rawOption'
    | 'detailsOption'
    | 'statsOption'
    | 'blocksOption'
    | 'vopsOption'
    | 'linenoOption'
    | 'uidOption'
    | 'allOption'
    | 'selectedPass';
export type PopulatedGccDumpViewState = {
    _compilerid: string;
    _compilerName: string;
    _editorid: number;
    _treeid: number;
} & (Record<GccDumpOptions, unknown> | EmptyState)

export type EmptyCfgViewState = EmptyState;
export type PopulatedCfgViewState = StateWithId & {
    editorid: number;
    treeid: number;
}

export type EmptyConformanceViewState = EmptyState; // TODO: unusued?
export type PopulatedConformanceViewState =  {
    editorid: number;
    treeid: number;
    langId: string;
    source: string;
}

export type EmptyIrViewState = EmptyState;
export type PopulatedIrViewState = StateWithId & {
    editorid: number;
    treeid: number;
    source: string;
    irOutput: unknown;
    compilerName: string;
}

export type EmptyRustMirViewState = EmptyState;
export type PopulatedRustMirViewState = StateWithId & {
    source: string,
    rustMirOutput: unknown,
    compilerName: string,
    editorid: number,
    treeid: number,
}

export type EmptyHaskellCoreViewState = EmptyState;
export type PopulatedHaskellCoreViewState = StateWithId & {
    source: string,
    haskellCoreOutput: unknown,
    compilerName: string,
    editorid: number,
    treeid: number,
}

export type EmptyHaskellStgViewState = EmptyState;
export type PopulatedHaskellStgViewState = StateWithId & {
    source: string,
    haskellStgOutput: unknown,
    compilerName: string,
    editorid: number,
    treeid: number,
}

export type EmptyHaskellCmmViewState = EmptyState;
export type PopulatedHaskellCmmViewState = StateWithId & {
    source: string,
    haskellCmmOutput: unknown,
    compilerName: string,
    editorid: number,
    treeid: number,
}

export type EmptyGnatDebugTreeViewState = EmptyState;
export type PopulatedGnatDebugTreeViewState = StateWithId & {
    source: string,
    gnatDebugTreeOutput: unknown,
    compilerName: string,
    editorid: number,
    treeid: number,
}

export type EmptyGnatDebugViewState = EmptyState;
export type PopulatedGnatDebugViewState = StateWithId & {
    source: string,
    gnatDebugOutput: unknown,
    compilerName: string,
    editorid: number,
    treeid: number,
}

export type EmptyRustMacroExpViewState = EmptyState;
export type PopulatedRustMacroExpViewState = StateWithId & {
    source: string,
    rustMacroExpOutput: unknown,
    compilerName: string,
    editorid: number,
    treeid: number,
}

export type EmptyRustHirViewState = EmptyState;
export type PopulatedRustHirViewState = StateWithId & {
    source: string,
    rustHirOutput: unknown,
    compilerName: string,
    editorid: number,
    treeid: number,
}

export type EmptyDeviceViewState = EmptyState;
export type PopulatedDeviceViewState = StateWithId & {
    source: string,
    deviceOutput: unknown,
    compilerName: string,
    editorid: number,
    treeid: number,
}
