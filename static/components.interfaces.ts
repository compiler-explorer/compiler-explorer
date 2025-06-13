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

import GoldenLayout from 'golden-layout';

import {CompilerOutputOptions} from '../types/features/filters.interfaces.js';
import {ConfiguredOverrides} from './compilation/compiler-overrides.interfaces.js';
import {ConfiguredRuntimeTools} from './execution/execution.interfaces.js';
import {CfgState} from './panes/cfg-view.interfaces.js';
import {ClangirState} from './panes/clangir-view.interfaces.js';
import {GccDumpViewState} from './panes/gccdump-view.interfaces.js';
import {IrState} from './panes/ir-view.interfaces.js';
import {OptPipelineViewState} from './panes/opt-pipeline.interfaces.js';
import {MonacoPaneState} from './panes/pane.interfaces.js';
export const COMPILER_COMPONENT_NAME = 'compiler';
export const EXECUTOR_COMPONENT_NAME = 'executor';
export const EDITOR_COMPONENT_NAME = 'codeEditor';
export const TREE_COMPONENT_NAME = 'tree';
export const OUTPUT_COMPONENT_NAME = 'output';
export const TOOL_COMPONENT_NAME = 'tool';

export const TOOL_INPUT_VIEW_COMPONENT_NAME = 'toolInputView';
export const DIFF_VIEW_COMPONENT_NAME = 'diff';
export const OPT_VIEW_COMPONENT_NAME = 'opt';
export const STACK_USAGE_VIEW_COMPONENT_NAME = 'stackusage';
export const FLAGS_VIEW_COMPONENT_NAME = 'flags';
export const PP_VIEW_COMPONENT_NAME = 'pp';
export const AST_VIEW_COMPONENT_NAME = 'ast';
export const GCC_DUMP_VIEW_COMPONENT_NAME = 'gccdump';
export const CFG_VIEW_COMPONENT_NAME = 'cfg';
export const CONFORMANCE_VIEW_COMPONENT_NAME = 'conformance';
export const IR_VIEW_COMPONENT_NAME = 'ir';
export const CLANGIR_VIEW_COMPONENT_NAME = 'clangir';
export const OPT_PIPELINE_VIEW_COMPONENT_NAME = 'optPipelineView';
// Historical LLVM-specific name preserved to keep old links working
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
        overrides?: ConfiguredOverrides;
        runtimeTools?: ConfiguredRuntimeTools;
    };
export type ExecutorForTreeState = StateWithLanguage &
    StateWithTree & {
        compilationPanelShown: boolean;
        compilerOutShown: boolean;
    };

export type EmptyEditorState = Partial<StateWithId & StateWithLanguage>;
export type PopulatedEditorState = StateWithId &
    StateWithLanguage & {
        source: string;
        options: unknown;
    };

type CmakeArgsState = {cmakeArgs: string};
export type EmptyTreeState = Partial<StateWithId & CmakeArgsState>;

export type OutputState = StateWithTree & {
    compiler: number; // CompilerID
    editor: number; // EditorId
};

export type ToolState = {
    toolId: string;
    monacoStdin?: boolean;
    monacoEditorOpen?: boolean;
    monacoEditorHasBeenAutoOpened?: boolean;
    argsPanelShown?: boolean;
    stdinPanelShown?: boolean;
    args?: string;
    stdin?: string;
    wrap?: boolean;
};

export type NewToolSettings = MonacoPaneState & ToolState;

export type ToolViewState = StateWithTree &
    ToolState & {
        id: number; // CompilerID (TODO(#4703): Why is this not part of StateWithTree)
        compilerName: string; // Compiler Name (TODO(#4703): Why is this not part of StateWithTree)
        editorid: number; // EditorId
        toolId: string;
    };

export type EmptyToolInputViewState = EmptyState;
export type PopulatedToolInputViewState = {
    compilerId: number;
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

export type EmptyStackUsageViewState = EmptyState;
export type PopulatedStackUsageViewState = StateWithId &
    StateWithEditor & {
        suOutput: unknown;
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
export type PopulatedIrViewState = StateWithId &
    IrState & {
        editorid: number;
        treeid: number;
        source: string;
        compilerName: string;
    };

export type EmptyClangirViewState = EmptyState;
export type PopulatedClangirViewState = StateWithId &
    ClangirState & {
        editorid: number;
        treeid: number;
        source: string;
        compilerName: string;
    };

export type EmptyOptPipelineViewState = EmptyState;
export type PopulatedOptPipelineViewState = StateWithId &
    OptPipelineViewState & {
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

/**
 * Mapping of component names to their expected state types.
 * This provides compile-time type safety for component states.
 * Components can have either empty (default) or populated states.
 */
export interface ComponentStateMap {
    [COMPILER_COMPONENT_NAME]: EmptyCompilerState | PopulatedCompilerState | CompilerForTreeState;
    [EXECUTOR_COMPONENT_NAME]: EmptyExecutorState | PopulatedExecutorState | ExecutorForTreeState;
    [EDITOR_COMPONENT_NAME]: EmptyEditorState | PopulatedEditorState;
    [TREE_COMPONENT_NAME]: EmptyTreeState;
    [OUTPUT_COMPONENT_NAME]: OutputState;
    [TOOL_COMPONENT_NAME]: ToolViewState;
    [TOOL_INPUT_VIEW_COMPONENT_NAME]: EmptyToolInputViewState | PopulatedToolInputViewState;
    [DIFF_VIEW_COMPONENT_NAME]: EmptyDiffViewState | PopulatedDiffViewState;
    [OPT_VIEW_COMPONENT_NAME]: EmptyOptViewState | PopulatedOptViewState;
    [STACK_USAGE_VIEW_COMPONENT_NAME]: EmptyStackUsageViewState | PopulatedStackUsageViewState;
    [FLAGS_VIEW_COMPONENT_NAME]: EmptyFlagsViewState | PopulatedFlagsViewState;
    [PP_VIEW_COMPONENT_NAME]: EmptyPpViewState | PopulatedPpViewState;
    [AST_VIEW_COMPONENT_NAME]: EmptyAstViewState | PopulatedAstViewState;
    [GCC_DUMP_VIEW_COMPONENT_NAME]: EmptyGccDumpViewState | PopulatedGccDumpViewState;
    [CFG_VIEW_COMPONENT_NAME]: EmptyCfgViewState | PopulatedCfgViewState;
    [CONFORMANCE_VIEW_COMPONENT_NAME]: PopulatedConformanceViewState;
    [IR_VIEW_COMPONENT_NAME]: EmptyIrViewState | PopulatedIrViewState;
    [CLANGIR_VIEW_COMPONENT_NAME]: EmptyClangirViewState | PopulatedClangirViewState;
    [OPT_PIPELINE_VIEW_COMPONENT_NAME]: EmptyOptPipelineViewState | PopulatedOptPipelineViewState;
    [LLVM_OPT_PIPELINE_VIEW_COMPONENT_NAME]: EmptyOptPipelineViewState | PopulatedOptPipelineViewState;
    [RUST_MIR_VIEW_COMPONENT_NAME]: EmptyRustMirViewState | PopulatedRustMirViewState;
    [HASKELL_CORE_VIEW_COMPONENT_NAME]: EmptyHaskellCoreViewState | PopulatedHaskellCoreViewState;
    [HASKELL_STG_VIEW_COMPONENT_NAME]: EmptyHaskellStgViewState | PopulatedHaskellStgViewState;
    [HASKELL_CMM_VIEW_COMPONENT_NAME]: EmptyHaskellCmmViewState | PopulatedHaskellCmmViewState;
    [GNAT_DEBUG_TREE_VIEW_COMPONENT_NAME]: EmptyGnatDebugTreeViewState | PopulatedGnatDebugTreeViewState;
    [GNAT_DEBUG_VIEW_COMPONENT_NAME]: EmptyGnatDebugViewState | PopulatedGnatDebugViewState;
    [RUST_MACRO_EXP_VIEW_COMPONENT_NAME]: EmptyRustMacroExpViewState | PopulatedRustMacroExpViewState;
    [RUST_HIR_VIEW_COMPONENT_NAME]: EmptyRustHirViewState | PopulatedRustHirViewState;
    [DEVICE_VIEW_COMPONENT_NAME]: EmptyDeviceViewState | PopulatedDeviceViewState;
}

/**
 * Type-safe component configuration that enforces:
 * - type must be the literal string 'component' (not just any string)
 * - componentName must be a valid component name from ComponentStateMap
 * - componentState must match the expected type for that component
 */
export interface ComponentConfig<K extends keyof ComponentStateMap = keyof ComponentStateMap> {
    type: 'component';
    componentName: K;
    componentState: ComponentStateMap[K];
    title?: string;
    isClosable?: boolean;
    reorderEnabled?: boolean;
    width?: number;
    height?: number;
}

/**
 * Layout item types (row, column, stack) with typed content
 */
export interface LayoutItem {
    type: 'row' | 'column' | 'stack';
    content: ItemConfig[];
    isClosable?: boolean;
    reorderEnabled?: boolean;
    width?: number;
    height?: number;
    activeItemIndex?: number;
}

/**
 * Union type for all valid item configurations
 */
export type ItemConfig = ComponentConfig | LayoutItem;

/**
 * Type-safe GoldenLayout configuration
 */
export interface GoldenLayoutConfig extends Omit<GoldenLayout.Config, 'content'> {
    content?: ItemConfig[];
}

/**
 * Type guard to check if an item is a component configuration
 */
export function isComponentConfig(item: ItemConfig): item is ComponentConfig {
    return item.type === 'component';
}

/**
 * Type guard to check if an item is a layout item (row, column, stack)
 */
export function isLayoutItem(item: ItemConfig): item is LayoutItem {
    return item.type === 'row' || item.type === 'column' || item.type === 'stack';
}

/**
 * Helper type for partial component states during initialization
 */
export type PartialComponentState<K extends keyof ComponentStateMap> = Partial<ComponentStateMap[K]>;

/**
 * Type for serialized GoldenLayout state (for URL/storage).
 * This interface is designed for Phase 2/3 of the migration when we'll
 * implement type-safe serialization/deserialization for:
 * - URL sharing (when users share layout links)
 * - localStorage persistence
 * - Import/export functionality
 *
 * Currently unused but defines the target structure for serialization.
 */
export interface SerializedLayoutState {
    version: number;
    content: ItemConfig[];
    settings?: GoldenLayout.Settings;
    dimensions?: GoldenLayout.Dimensions;
    labels?: GoldenLayout.Labels;
    maximisedItemId?: string | null;
}

/**
 * Type for drag source factory functions
 */
export type DragSourceFactory<K extends keyof ComponentStateMap> = () => ComponentConfig<K>;
