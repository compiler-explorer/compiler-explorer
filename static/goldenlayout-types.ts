// Copyright (c) 2025, Compiler Explorer Authors
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

import {
    AST_VIEW_COMPONENT_NAME,
    CFG_VIEW_COMPONENT_NAME,
    CLANGIR_VIEW_COMPONENT_NAME,
    COMPILER_COMPONENT_NAME,
    CONFORMANCE_VIEW_COMPONENT_NAME,
    DEVICE_VIEW_COMPONENT_NAME,
    DIFF_VIEW_COMPONENT_NAME,
    EDITOR_COMPONENT_NAME,
    EXECUTOR_COMPONENT_NAME,
    EmptyCompilerState,
    EmptyEditorState,
    EmptyExecutorState,
    EmptyTreeState,
    FLAGS_VIEW_COMPONENT_NAME,
    GCC_DUMP_VIEW_COMPONENT_NAME,
    GNAT_DEBUG_TREE_VIEW_COMPONENT_NAME,
    GNAT_DEBUG_VIEW_COMPONENT_NAME,
    HASKELL_CMM_VIEW_COMPONENT_NAME,
    HASKELL_CORE_VIEW_COMPONENT_NAME,
    HASKELL_STG_VIEW_COMPONENT_NAME,
    IR_VIEW_COMPONENT_NAME,
    LLVM_OPT_PIPELINE_VIEW_COMPONENT_NAME,
    OPT_PIPELINE_VIEW_COMPONENT_NAME,
    OPT_VIEW_COMPONENT_NAME,
    OUTPUT_COMPONENT_NAME,
    OutputState,
    PP_VIEW_COMPONENT_NAME,
    PopulatedAstViewState,
    PopulatedCfgViewState,
    PopulatedClangirViewState,
    PopulatedCompilerState,
    PopulatedConformanceViewState,
    PopulatedDeviceViewState,
    PopulatedDiffViewState,
    PopulatedEditorState,
    PopulatedExecutorState,
    PopulatedFlagsViewState,
    PopulatedGccDumpViewState,
    PopulatedGnatDebugTreeViewState,
    PopulatedGnatDebugViewState,
    PopulatedHaskellCmmViewState,
    PopulatedHaskellCoreViewState,
    PopulatedHaskellStgViewState,
    PopulatedIrViewState,
    PopulatedOptPipelineViewState,
    PopulatedOptViewState,
    PopulatedPpViewState,
    PopulatedRustHirViewState,
    PopulatedRustMacroExpViewState,
    PopulatedRustMirViewState,
    PopulatedStackUsageViewState,
    PopulatedToolInputViewState,
    RUST_HIR_VIEW_COMPONENT_NAME,
    RUST_MACRO_EXP_VIEW_COMPONENT_NAME,
    RUST_MIR_VIEW_COMPONENT_NAME,
    STACK_USAGE_VIEW_COMPONENT_NAME,
    TOOL_COMPONENT_NAME,
    TOOL_INPUT_VIEW_COMPONENT_NAME,
    TREE_COMPONENT_NAME,
    ToolViewState,
} from './components.interfaces.js';

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
 * Type-safe component configuration
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
 * Type for serialized GoldenLayout state (for URL/storage)
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
 * Helper to convert from GoldenLayout's internal config to our typed config
 */
export function fromGoldenLayoutConfig(config: GoldenLayout.Config): TypedGoldenLayoutConfig {
    // This is a shallow conversion - in practice, we'd need to validate
    // the component states match their expected types
    return config as TypedGoldenLayoutConfig;
}

/**
 * Helper to convert to GoldenLayout's expected config format
 */
export function toGoldenLayoutConfig(config: TypedGoldenLayoutConfig): GoldenLayout.Config {
    return config as GoldenLayout.Config;
}

/**
 * Type for drag source factory functions
 */
export type DragSourceFactory<K extends keyof ComponentStateMap> = () => TypedComponentConfig<K>;

/**
 * Typed wrapper for createDragSource that avoids the need for 'as any'
 * Returns the result with _dragListener property for event handling
 */
export function createTypedDragSource<K extends keyof ComponentStateMap>(
    layout: GoldenLayout,
    element: HTMLElement | JQuery,
    factory: DragSourceFactory<K>,
): any {
    return layout.createDragSource(element, factory as any);
}

/**
 * Helper to convert legacy ComponentConfig to TypedComponentConfig
 * This is a flexible conversion that works with any component state
 */
export function legacyComponentConfigToTyped<T>(config: import('./components.interfaces.js').ComponentConfig<T>): any {
    return {
        ...config,
        type: 'component' as const,
    };
}

/**
 * Helper to convert TypedComponentConfig to legacy ComponentConfig
 */
export function typedComponentConfigToLegacy<K extends keyof ComponentStateMap>(
    config: TypedComponentConfig<K>,
): import('./components.interfaces.js').ComponentConfig<ComponentStateMap[K]> {
    return {
        ...config,
        type: 'component',
    };
}
