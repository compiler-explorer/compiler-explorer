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

// =============================================================================
// GoldenLayout Type Safety Infrastructure
// =============================================================================
// This section provides type-safe wrappers and interfaces for GoldenLayout
// configurations. It addresses issue #4490 "The War of The Types" by creating
// a migration path from loosely-typed to strongly-typed layout configurations.

import GoldenLayout from 'golden-layout';

/**
 * Mapping of component names to their expected state types.
 * This provides compile-time type safety for component states.
 * Components can have either empty (default) or populated states.
 */
export interface ComponentStateMap {
    [COMPILER_COMPONENT_NAME]: EmptyCompilerState | PopulatedCompilerState;
    [EXECUTOR_COMPONENT_NAME]: EmptyExecutorState | PopulatedExecutorState;
    [EDITOR_COMPONENT_NAME]: EmptyEditorState | PopulatedEditorState;
    [TREE_COMPONENT_NAME]: EmptyTreeState;
    [OUTPUT_COMPONENT_NAME]: OutputState;
    [TOOL_COMPONENT_NAME]: ToolViewState;
    [TOOL_INPUT_VIEW_COMPONENT_NAME]: PopulatedToolInputViewState;
    [DIFF_VIEW_COMPONENT_NAME]: PopulatedDiffViewState;
    [OPT_VIEW_COMPONENT_NAME]: PopulatedOptViewState;
    [STACK_USAGE_VIEW_COMPONENT_NAME]: PopulatedStackUsageViewState;
    [FLAGS_VIEW_COMPONENT_NAME]: PopulatedFlagsViewState;
    [PP_VIEW_COMPONENT_NAME]: PopulatedPpViewState;
    [AST_VIEW_COMPONENT_NAME]: PopulatedAstViewState;
    [GCC_DUMP_VIEW_COMPONENT_NAME]: PopulatedGccDumpViewState;
    [CFG_VIEW_COMPONENT_NAME]: PopulatedCfgViewState;
    [CONFORMANCE_VIEW_COMPONENT_NAME]: PopulatedConformanceViewState;
    [IR_VIEW_COMPONENT_NAME]: PopulatedIrViewState;
    [CLANGIR_VIEW_COMPONENT_NAME]: PopulatedClangirViewState;
    [OPT_PIPELINE_VIEW_COMPONENT_NAME]: PopulatedOptPipelineViewState;
    [LLVM_OPT_PIPELINE_VIEW_COMPONENT_NAME]: PopulatedOptPipelineViewState;
    [RUST_MIR_VIEW_COMPONENT_NAME]: PopulatedRustMirViewState;
    [HASKELL_CORE_VIEW_COMPONENT_NAME]: PopulatedHaskellCoreViewState;
    [HASKELL_STG_VIEW_COMPONENT_NAME]: PopulatedHaskellStgViewState;
    [HASKELL_CMM_VIEW_COMPONENT_NAME]: PopulatedHaskellCmmViewState;
    [GNAT_DEBUG_TREE_VIEW_COMPONENT_NAME]: PopulatedGnatDebugTreeViewState;
    [GNAT_DEBUG_VIEW_COMPONENT_NAME]: PopulatedGnatDebugViewState;
    [RUST_MACRO_EXP_VIEW_COMPONENT_NAME]: PopulatedRustMacroExpViewState;
    [RUST_HIR_VIEW_COMPONENT_NAME]: PopulatedRustHirViewState;
    [DEVICE_VIEW_COMPONENT_NAME]: PopulatedDeviceViewState;
}

/**
 * Type-safe component configuration.
 * This is the new, improved version of ComponentConfig that enforces:
 * - type must be the literal string 'component' (not just any string)
 * - componentName must be a valid component name from ComponentStateMap
 * - componentState must match the expected type for that component
 */
export interface TypedComponentConfig<K extends keyof ComponentStateMap = keyof ComponentStateMap> {
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
export interface TypedLayoutItem {
    type: 'row' | 'column' | 'stack';
    content: TypedItemConfig[];
    isClosable?: boolean;
    reorderEnabled?: boolean;
    width?: number;
    height?: number;
    activeItemIndex?: number;
}

/**
 * Union type for all valid item configurations
 */
export type TypedItemConfig = TypedComponentConfig | TypedLayoutItem;

/**
 * Type-safe GoldenLayout configuration
 */
export interface TypedGoldenLayoutConfig extends Omit<GoldenLayout.Config, 'content'> {
    content?: TypedItemConfig[];
}

/**
 * Type guard to check if an item is a component configuration
 */
export function isComponentConfig(item: TypedItemConfig): item is TypedComponentConfig {
    return item.type === 'component';
}

/**
 * Type guard to check if an item is a layout item (row, column, stack)
 */
export function isLayoutItem(item: TypedItemConfig): item is TypedLayoutItem {
    return item.type === 'row' || item.type === 'column' || item.type === 'stack';
}

/**
 * Helper type for partial component states during initialization
 */
export type PartialComponentState<K extends keyof ComponentStateMap> = Partial<ComponentStateMap[K]>;

/**
 * Helper function to create a typed component configuration
 */
export function createTypedComponentConfig<K extends keyof ComponentStateMap>(
    componentName: K,
    componentState: ComponentStateMap[K],
    options?: {
        title?: string;
        isClosable?: boolean;
        reorderEnabled?: boolean;
        width?: number;
        height?: number;
    },
): TypedComponentConfig<K> {
    return {
        type: 'component',
        componentName,
        componentState,
        ...options,
    };
}

/**
 * Helper function to create a typed layout item
 */
export function createTypedLayoutItem(
    type: 'row' | 'column' | 'stack',
    content: TypedItemConfig[],
    options?: {
        isClosable?: boolean;
        reorderEnabled?: boolean;
        width?: number;
        height?: number;
        activeItemIndex?: number;
    },
): TypedLayoutItem {
    return {
        type,
        content,
        ...options,
    };
}

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
    content: TypedItemConfig[];
    settings?: GoldenLayout.Settings;
    dimensions?: GoldenLayout.Dimensions;
    labels?: GoldenLayout.Labels;
    maximisedItemId?: string | null;
}

/**
 * Helper to convert from GoldenLayout's internal config to our typed config.
 *
 * WARNING: This is currently just a type cast with no runtime validation!
 * Phase 2 MUST implement proper validation to ensure component states match
 * their expected types before this can be safely used with untrusted data
 * (e.g., from URLs, localStorage, or user imports).
 *
 * The proper implementation will:
 * 1. Validate each component's state matches its expected type
 * 2. Provide helpful error messages for invalid configurations
 * 3. Handle version migrations if needed
 *
 * @param config - Untyped config from GoldenLayout
 * @returns Typed config (currently just a cast)
 */
export function fromGoldenLayoutConfig(config: GoldenLayout.Config): TypedGoldenLayoutConfig {
    // TODO(Phase 2): Implement proper validation here
    // This should validate component states match their expected types
    // and throw meaningful errors for invalid configurations
    return config as TypedGoldenLayoutConfig;
}

/**
 * Helper to convert to GoldenLayout's expected config format.
 * This direction is safe since we're going from typed to untyped.
 */
export function toGoldenLayoutConfig(config: TypedGoldenLayoutConfig): GoldenLayout.Config {
    return config as GoldenLayout.Config;
}

/**
 * Type for drag source factory functions
 */
export type DragSourceFactory<K extends keyof ComponentStateMap> = () => TypedComponentConfig<K>;

/**
 * Typed wrapper for createDragSource that avoids the need for 'as any'.
 * Returns the result with _dragListener property for event handling.
 *
 * Note: We still need to cast internally because GoldenLayout's TypeScript
 * definitions don't properly type the second parameter as accepting a function.
 */
export function createTypedDragSource<K extends keyof ComponentStateMap>(
    layout: GoldenLayout,
    element: HTMLElement | JQuery,
    factory: DragSourceFactory<K>,
): any {
    return layout.createDragSource(element, factory as any);
}

/**
 * Helper to convert legacy ComponentConfig to TypedComponentConfig.
 * This function bridges between the old and new type systems during migration.
 *
 * It's called "legacy" because it converts from the existing ComponentConfig
 * (which uses type: string) to the new TypedComponentConfig (type: 'component').
 *
 * This is a temporary bridge function that will be removed in Phase 3 once
 * all code has been migrated to use TypedComponentConfig directly.
 */
export function legacyComponentConfigToTyped<T>(config: ComponentConfig<T>): any {
    return {
        ...config,
        type: 'component' as const,
    };
}

/**
 * Helper to convert TypedComponentConfig to legacy ComponentConfig.
 * This is the reverse bridge for cases where we need to pass typed configs
 * to code that still expects the legacy format.
 *
 * Also temporary and will be removed in Phase 3.
 */
export function typedComponentConfigToLegacy<K extends keyof ComponentStateMap>(
    config: TypedComponentConfig<K>,
): ComponentConfig<ComponentStateMap[K]> {
    return {
        ...config,
        type: 'component',
    };
}
