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

import {CompilerOutputOptions} from '../types/features/filters.interfaces';
import {CfgState} from './panes/cfg-view.interfaces';
import {LLVMOptPipelineViewState} from './panes/llvm-opt-pipeline.interfaces';
import {GccDumpViewState} from './panes/gccdump-view.interfaces';
export const COMPILER_COMPONENT_NAME = 'compiler';
export const EXECUTOR_COMPONENT_NAME = 'executor';
export const EDITOR_COMPONENT_NAME = 'codeEditor';
export const TREE_COMPONENT_NAME = 'tree';
export const OUTPUT_COMPONENT_NAME = 'output';
export const TOOL_COMPONENT_NAME = 'tool';

export const TOOL_INPUT_VIEW_COMPONENT_NAME = 'toolInputView';
export const DIFF_VIEW_COMPONENT_NAME = 'diff';
export const OPT_VIEW_COMPONENT_NAME = 'opt';
export const FLAGS_VIEW_COMPONENT_NAME = 'flags';
export const PP_VIEW_COMPONENT_NAME = 'pp';
export const AST_VIEW_COMPONENT_NAME = 'ast';
export const GCC_DUMP_VIEW_COMPONENT_NAME = 'gccdump';
export const CFG_VIEW_COMPONENT_NAME = 'cfg';
export const CONFORMANCE_VIEW_COMPONENT_NAME = 'conformance';
export const IR_VIEW_COMPONENT_NAME = 'ir';
export const LLVM_OPT_PIPELINE_VIEW_COMPONENT_NAME = 'llvmOptPipelineView';
export const RUST_MIR_VIEW_COMPONENT_NAME = 'rustmir';
export const HASKELL_CORE_VIEW_COMPONENT_NAME = 'haskellCore';
export const HASKELL_STG_VIEW_COMPONENT_NAME = 'haskellStg';
export const HASKELL_CMM_VIEW_COMPONENT_NAME = 'haskellCmm';
export const GNAT_DEBUG_TREE_VIEW_COMPONENT_NAME = 'gnatdebugtree';
export const GNAT_DEBUG_VIEW_COMPONENT_NAME = 'gnatdebug';
export const RUST_MACRO_EXP_VIEW_COMPONENT_NAME = 'rustmacroexp';
export const RUST_HIR_VIEW_COMPONENT_NAME = 'rusthir';
export const DEVICE_VIEW_COMPONENT_NAME = 'device';

export interface ComponentConfig<S> {
    type: string;
    componentName: string;
    componentState: S;
}

export type StateWithLanguage = {lang: string};
// TODO(#4490 The War of The Types) We should normalize state types
export type StateWithEditor = {source: string | number};
export type StateWithTree = {tree: number};
export type StateWithId = {id: number};
export type EmptyState = Record<never, never>;

export type EmptyCompilerState = StateWithLanguage & StateWithEditor;
export type PopulatedCompilerState = StateWithEditor & {
    filters: CompilerOutputOptions | undefined;
    options: unknown;
    compiler: string;
    libs?: unknown;
    lang?: string;
};
export type CompilerForTreeState = StateWithLanguage & StateWithTree;

export type EmptyExecutorState = StateWithLanguage &
    StateWithEditor & {
        compilationPanelShown: boolean;
        compilerOutShown: boolean;
    };
export type PopulatedExecutorState = StateWithLanguage &
    StateWithEditor &
    StateWithTree & {
        compiler: string;
        libs: unknown;
        options: unknown;
        compilationPanelShown: boolean;
        compilerOutShown: boolean;
    };
export type ExecutorForTreeState = StateWithLanguage &
    StateWithTree & {
        compilationPanelShown: boolean;
        compilerOutShown: boolean;
    };

export type EmptyEditorState = Partial<StateWithId & StateWithLanguage>;
export type PopulatedEditorState = StateWithId & {
    source: string;
    options: unknown;
};

export type EmptyTreeState = Partial<StateWithId>;

export type OutputState = StateWithTree & {
    compiler: number; // CompilerID
    editor: number; // EditorId
};

export type ToolViewState = StateWithTree & {
    compiler: number; // CompilerId
    editor: number; // EditorId
    toolId: string;
    args: unknown;
    monacoStdin: boolean;
};

export type EmptyToolInputViewState = EmptyState;
export type PopulatedToolInputViewState = {
    compilerId: string;
    toolId: string;
    toolName: string;
};

export type EmptyDiffViewState = EmptyState;
export type PopulatedDiffViewState = {
    lhs: unknown;
    rhs: unknown;
};

export type EmptyOptViewState = EmptyState;
export type PopulatedOptViewState = StateWithId &
    StateWithEditor & {
        optOutput: unknown;
        compilerName: string;
        editorid: number;
        treeid: number;
    };

export type EmptyFlagsViewState = EmptyState;
export type PopulatedFlagsViewState = StateWithId & {
    compilerName: string;
    compilerFlags: unknown;
};

export type EmptyPpViewState = EmptyState;
export type PopulatedPpViewState = StateWithId &
    StateWithEditor & {
        ppOutput: unknown;
        compilerName: string;
        editorid: number;
        treeid: number;
    };

export type EmptyAstViewState = EmptyState;
export type PopulatedAstViewState = StateWithId &
    StateWithEditor & {
        astOutput: unknown;
        compilerName: string;
        editorid: number;
        treeid: number;
    };

export type EmptyGccDumpViewState = EmptyState;
export type PopulatedGccDumpViewState = StateWithId &
    GccDumpViewState & {
        compilerName: string;
        editorid: number;
        treeid: number;
    };

export type EmptyCfgViewState = EmptyState;
export type PopulatedCfgViewState = StateWithId &
    CfgState & {
        editorid: number;
        treeid: number;
    };

export type EmptyConformanceViewState = EmptyState; // TODO: unusued?
export type PopulatedConformanceViewState = {
    editorid: number;
    treeid: number;
    langId: string;
    source: string;
};

export type EmptyIrViewState = EmptyState;
export type PopulatedIrViewState = StateWithId & {
    editorid: number;
    treeid: number;
    source: string;
    irOutput: unknown;
    compilerName: string;
};

export type EmptyLLVMOptPipelineViewState = EmptyState;
export type PopulatedLLVMOptPipelineViewState = StateWithId &
    LLVMOptPipelineViewState & {
        compilerName: string;
        editorid: number;
        treeid: number;
    };

export type EmptyRustMirViewState = EmptyState;
export type PopulatedRustMirViewState = StateWithId & {
    source: string;
    rustMirOutput: unknown;
    compilerName: string;
    editorid: number;
    treeid: number;
};

export type EmptyHaskellCoreViewState = EmptyState;
export type PopulatedHaskellCoreViewState = StateWithId & {
    source: string;
    haskellCoreOutput: unknown;
    compilerName: string;
    editorid: number;
    treeid: number;
};

export type EmptyHaskellStgViewState = EmptyState;
export type PopulatedHaskellStgViewState = StateWithId & {
    source: string;
    haskellStgOutput: unknown;
    compilerName: string;
    editorid: number;
    treeid: number;
};

export type EmptyHaskellCmmViewState = EmptyState;
export type PopulatedHaskellCmmViewState = StateWithId & {
    source: string;
    haskellCmmOutput: unknown;
    compilerName: string;
    editorid: number;
    treeid: number;
};

export type EmptyGnatDebugTreeViewState = EmptyState;
export type PopulatedGnatDebugTreeViewState = StateWithId & {
    source: string;
    gnatDebugTreeOutput: unknown;
    compilerName: string;
    editorid: number;
    treeid: number;
};

export type EmptyGnatDebugViewState = EmptyState;
export type PopulatedGnatDebugViewState = StateWithId & {
    source: string;
    gnatDebugOutput: unknown;
    compilerName: string;
    editorid: number;
    treeid: number;
};

export type EmptyRustMacroExpViewState = EmptyState;
export type PopulatedRustMacroExpViewState = StateWithId & {
    source: string;
    rustMacroExpOutput: unknown;
    compilerName: string;
    editorid: number;
    treeid: number;
};

export type EmptyRustHirViewState = EmptyState;
export type PopulatedRustHirViewState = StateWithId & {
    source: string;
    rustHirOutput: unknown;
    compilerName: string;
    editorid: number;
    treeid: number;
};

export type EmptyDeviceViewState = EmptyState;
export type PopulatedDeviceViewState = StateWithId & {
    source: string;
    devices: unknown;
    compilerName: string;
    editorid: number;
    treeid: number;
};
