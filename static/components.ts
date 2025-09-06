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
import {ConfiguredOverrides} from '../types/compilation/compiler-overrides.interfaces.js';
import {ConfiguredRuntimeTools} from '../types/execution/execution.interfaces.js';
import {ParseFiltersAndOutputOptions} from '../types/features/filters.interfaces.js';
import {LanguageKey} from '../types/languages.interfaces.js';
import {
    AnyComponentConfig,
    AST_VIEW_COMPONENT_NAME,
    CFG_VIEW_COMPONENT_NAME,
    CLANGIR_VIEW_COMPONENT_NAME,
    COMPILER_COMPONENT_NAME,
    CONFORMANCE_VIEW_COMPONENT_NAME,
    ComponentConfig,
    ComponentStateMap,
    DEVICE_VIEW_COMPONENT_NAME,
    DIFF_VIEW_COMPONENT_NAME,
    DragSourceFactory,
    EDITOR_COMPONENT_NAME,
    EXECUTOR_COMPONENT_NAME,
    EXPLAIN_VIEW_COMPONENT_NAME,
    FLAGS_VIEW_COMPONENT_NAME,
    GCC_DUMP_VIEW_COMPONENT_NAME,
    GNAT_DEBUG_TREE_VIEW_COMPONENT_NAME,
    GNAT_DEBUG_VIEW_COMPONENT_NAME,
    GoldenLayoutConfig,
    HASKELL_CMM_VIEW_COMPONENT_NAME,
    HASKELL_CORE_VIEW_COMPONENT_NAME,
    HASKELL_STG_VIEW_COMPONENT_NAME,
    IR_VIEW_COMPONENT_NAME,
    ItemConfig,
    LayoutItem,
    LLVM_OPT_PIPELINE_VIEW_COMPONENT_NAME,
    OPT_PIPELINE_VIEW_COMPONENT_NAME,
    OPT_VIEW_COMPONENT_NAME,
    OUTPUT_COMPONENT_NAME,
    PP_VIEW_COMPONENT_NAME,
    RUST_HIR_VIEW_COMPONENT_NAME,
    RUST_MACRO_EXP_VIEW_COMPONENT_NAME,
    RUST_MIR_VIEW_COMPONENT_NAME,
    STACK_USAGE_VIEW_COMPONENT_NAME,
    TOOL_COMPONENT_NAME,
    TOOL_INPUT_VIEW_COMPONENT_NAME,
    TREE_COMPONENT_NAME,
} from './components.interfaces.js';
import {GccDumpViewState} from './panes/gccdump-view.interfaces.js';
import {SentryCapture} from './sentry.js';

/** Get an empty compiler component. */
export function getCompiler(editorId: number, lang: string): ComponentConfig<typeof COMPILER_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: COMPILER_COMPONENT_NAME,
        componentState: {
            source: editorId,
            lang,
        },
    };
}

/**
 * Get a compiler component with the given configuration.
 *
 * We have legacy calls in the codebase (to keep shortlinks permanent) that use this function without
 * passing langId and libs. To keep that support, the parameters are optional.
 *
 * TODO: Find the right type for options
 */
export function getCompilerWith(
    editorId: number,
    filters: ParseFiltersAndOutputOptions | undefined,
    options: unknown,
    compilerId: string,
    langId?: string,
    libs?: unknown,
): ComponentConfig<typeof COMPILER_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: COMPILER_COMPONENT_NAME,
        componentState: {
            source: editorId,
            compiler: compilerId,
            lang: langId,
            filters,
            options,
            libs,
        },
    };
}

/** Get a compiler for a tree mode component. */
export function getCompilerForTree(treeId: number, lang: string): ComponentConfig<typeof COMPILER_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: COMPILER_COMPONENT_NAME,
        componentState: {
            tree: treeId,
            lang,
        },
    };
}

/** Get an empty executor component. */
export function getExecutor(editorId: number, lang: string): ComponentConfig<typeof EXECUTOR_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: EXECUTOR_COMPONENT_NAME,
        componentState: {
            source: editorId,
            lang,
            compilationPanelShown: true,
            compilerOutShown: true,
        },
    };
}

/** Get a compiler component with the given configuration. */
export function getExecutorWith(
    editorId: number,
    lang: string,
    compilerId: string,
    libraries: {name: string; ver: string}[],
    compilerArgs: string | undefined,
    treeId: number,
    overrides?: ConfiguredOverrides,
    runtimeTools?: ConfiguredRuntimeTools,
): ComponentConfig<typeof EXECUTOR_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: EXECUTOR_COMPONENT_NAME,
        componentState: {
            source: editorId,
            tree: treeId,
            compiler: compilerId,
            libs: libraries,
            options: compilerArgs,
            lang,
            compilationPanelShown: true,
            compilerOutShown: true,
            overrides: overrides,
            runtimeTools: runtimeTools,
        },
    };
}

/** Get an executor for a tree mode component. */
export function getExecutorForTree(treeId: number, lang: string): ComponentConfig<typeof EXECUTOR_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: EXECUTOR_COMPONENT_NAME,
        componentState: {
            tree: treeId,
            lang,
            compilationPanelShown: true,
            compilerOutShown: true,
        },
    };
}

/**
 * Get an empty editor component.
 *
 * TODO: main.js calls this with no arguments.
 */
export function getEditor(langId: LanguageKey, id?: number): ComponentConfig<typeof EDITOR_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: EDITOR_COMPONENT_NAME,
        componentState: {
            id,
            lang: langId,
        },
    };
}

/** Get an editor component with the given configuration. */
export function getEditorWith(
    id: number,
    source: string,
    options: ParseFiltersAndOutputOptions,
    langId: string,
): ComponentConfig<typeof EDITOR_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: EDITOR_COMPONENT_NAME,
        componentState: {
            id,
            source,
            options,
            lang: langId,
        },
    };
}

/**
 * Get an empty tree component.
 *
 * TODO: main.js calls this with no arguments.
 */
export function getTree(id?: number): ComponentConfig<typeof TREE_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: TREE_COMPONENT_NAME,
        componentState: {
            id,
            cmakeArgs: '-DCMAKE_BUILD_TYPE=Debug',
        },
    };
}

/** Get an output component with the given configuration. */
export function getOutput(
    compiler: number,
    editor: number,
    tree: number,
): ComponentConfig<typeof OUTPUT_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: OUTPUT_COMPONENT_NAME,
        componentState: {
            compiler,
            editor,
            tree,
        },
    };
}

/** Get a tool view component with the given configuration. */
export function getToolViewWith(
    compilerId: number,
    compilerName: string,
    editorid: number,
    toolId: string,
    args: string,
    monacoStdin: boolean,
    tree: number,
): ComponentConfig<typeof TOOL_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: TOOL_COMPONENT_NAME,
        componentState: {
            id: compilerId,
            compilerName,
            editorid,
            toolId,
            args,
            tree,
            monacoStdin,
        },
    };
}

/** Get an empty tool input view component. */
export function getToolInputView(): ComponentConfig<typeof TOOL_INPUT_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: TOOL_INPUT_VIEW_COMPONENT_NAME,
        componentState: {},
    };
}

/** Get a tool input view component with the given configuration. */
export function getToolInputViewWith(
    compilerId: number,
    toolId: string,
    toolName: string,
): ComponentConfig<typeof TOOL_INPUT_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: TOOL_INPUT_VIEW_COMPONENT_NAME,
        componentState: {
            compilerId,
            toolId,
            toolName,
        },
    };
}

/** Get an empty diff component. */
export function getDiffView(): ComponentConfig<typeof DIFF_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: DIFF_VIEW_COMPONENT_NAME,
        componentState: {},
    };
}

/**
 * Get a diff component with the given configuration.
 *
 * TODO: possibly unused?
 */
export function getDiffViewWith(lhs: unknown, rhs: unknown): ComponentConfig<typeof DIFF_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: DIFF_VIEW_COMPONENT_NAME,
        componentState: {
            lhs,
            rhs,
        },
    };
}

/** Get an empty opt view component. */
export function getOptView(): ComponentConfig<typeof OPT_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: OPT_VIEW_COMPONENT_NAME,
        componentState: {},
    };
}

/** Get an opt view with the given configuration. */
export function getOptViewWith(
    id: number,
    source: string,
    optOutput: unknown,
    compilerName: string,
    editorid: number,
    treeid: number,
): ComponentConfig<typeof OPT_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: OPT_VIEW_COMPONENT_NAME,
        componentState: {
            id,
            source,
            optOutput,
            compilerName,
            editorid,
            treeid,
        },
    };
}

export function getStackUsageView(): ComponentConfig<typeof STACK_USAGE_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: STACK_USAGE_VIEW_COMPONENT_NAME,
        componentState: {},
    };
}
export function getStackUsageViewWith(
    id: number,
    source: string,
    suOutput: unknown,
    compilerName: string,
    editorid: number,
    treeid: number,
): ComponentConfig<typeof STACK_USAGE_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: STACK_USAGE_VIEW_COMPONENT_NAME,
        componentState: {
            id,
            source,
            suOutput,
            compilerName,
            editorid,
            treeid,
        },
    };
}

/** Get an empty flags view component. */
export function getFlagsView(): ComponentConfig<typeof FLAGS_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: FLAGS_VIEW_COMPONENT_NAME,
        componentState: {},
    };
}

/** Get a flags view with the given configuration. */
export function getFlagsViewWith(
    id: number,
    compilerName: string,
    compilerFlags: unknown,
): ComponentConfig<typeof FLAGS_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: FLAGS_VIEW_COMPONENT_NAME,
        componentState: {
            id,
            compilerName,
            compilerFlags,
        },
    };
}

/** Get an empty preprocessor view component. */
export function getPpView(): ComponentConfig<typeof PP_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: PP_VIEW_COMPONENT_NAME,
        componentState: {},
    };
}

/** Get a preprocessor view with the given configuration. */
export function getPpViewWith(
    id: number,
    source: string,
    ppOutput: unknown,
    compilerName: string,
    editorid: number,
    treeid: number,
): ComponentConfig<typeof PP_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: PP_VIEW_COMPONENT_NAME,
        componentState: {
            id,
            source,
            ppOutput,
            compilerName,
            editorid,
            treeid,
        },
    };
}

/** Get an empty ast view component. */
export function getAstView(): ComponentConfig<typeof AST_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: AST_VIEW_COMPONENT_NAME,
        componentState: {},
    };
}

/** Get an ast view with the given configuration. */
export function getAstViewWith(
    id: number,
    source: string,
    astOutput: unknown,
    compilerName: string,
    editorid: number,
    treeid: number,
): ComponentConfig<typeof AST_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: AST_VIEW_COMPONENT_NAME,
        componentState: {
            id,
            source,
            astOutput,
            compilerName,
            editorid,
            treeid,
        },
    };
}

/** Get an empty gcc dump view component. */
export function getGccDumpView(): ComponentConfig<typeof GCC_DUMP_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: GCC_DUMP_VIEW_COMPONENT_NAME,
        componentState: {},
    };
}

/** Get a gcc dump view with the given configuration. */
export function getGccDumpViewWith(
    id: number,
    compilerName: string,
    editorid: number,
    treeid: number,
    gccDumpOutput: GccDumpViewState,
): ComponentConfig<typeof GCC_DUMP_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: GCC_DUMP_VIEW_COMPONENT_NAME,
        componentState: {
            // PopulatedGccDumpViewState
            id,
            compilerName,
            editorid,
            treeid,

            // & GccDumpFiltersState
            ...gccDumpOutput,
        },
    };
}

/** Get an empty cfg view component. */
export function getCfgView(): ComponentConfig<typeof CFG_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: CFG_VIEW_COMPONENT_NAME,
        componentState: {},
    };
}

/** Get a cfg view with the given configuration. */
export function getCfgViewWith(
    id: number,
    editorid: number,
    treeid: number,
    isircfg?: boolean,
): ComponentConfig<typeof CFG_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: CFG_VIEW_COMPONENT_NAME,
        componentState: {
            selectedFunction: null,
            id,
            editorid,
            treeid,
            isircfg,
            narrowtreelayout: true,
        },
    };
}

/** Get a conformance view with the given configuration. */
export function getConformanceView(
    editorid: number,
    treeid: number,
    source: string,
    langId: string,
): ComponentConfig<typeof CONFORMANCE_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: CONFORMANCE_VIEW_COMPONENT_NAME,
        componentState: {
            editorid,
            treeid,
            source,
            langId,
        },
    };
}

/** Get an empty ir view component. */
export function getIrView(): ComponentConfig<typeof IR_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: IR_VIEW_COMPONENT_NAME,
        componentState: {},
    };
}

/** Get a ir view with the given configuration. */
export function getIrViewWith(
    id: number,
    source: string,
    irOutput: unknown,
    compilerName: string,
    editorid: number,
    treeid: number,
): ComponentConfig<typeof IR_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: IR_VIEW_COMPONENT_NAME,
        componentState: {
            id,
            source,
            irOutput,
            compilerName,
            editorid,
            treeid,
        },
    };
}

export function getClangirView(): ComponentConfig<typeof CLANGIR_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: CLANGIR_VIEW_COMPONENT_NAME,
        componentState: {},
    };
}

export function getClangirViewWith(
    id: number,
    source: string,
    clangirOutput: unknown,
    compilerName: string,
    editorid: number,
    treeid: number,
): ComponentConfig<typeof CLANGIR_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: CLANGIR_VIEW_COMPONENT_NAME,
        componentState: {
            id,
            source,
            clangirOutput,
            compilerName,
            editorid,
            treeid,
        },
    };
}

/** Get an empty opt pipeline view component. */
export function getOptPipelineView(): ComponentConfig<typeof OPT_PIPELINE_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: OPT_PIPELINE_VIEW_COMPONENT_NAME,
        componentState: {},
    };
}

/** Get a opt pipeline view with the given configuration. */
export function getOptPipelineViewWith(
    id: number,
    lang: string,
    compilerId: string,
    compilerName: string,
    editorid: number,
    treeid: number,
): ComponentConfig<typeof OPT_PIPELINE_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: OPT_PIPELINE_VIEW_COMPONENT_NAME,
        componentState: {
            id,
            lang,
            compiler: compilerId,
            compilerName,
            editorid,
            treeid,
            selectedGroup: '',
            selectedIndex: 0,
            sidebarWidth: 0,
        },
    };
}

/** Get an empty rust mir view component. */
export function getRustMirView(): ComponentConfig<typeof RUST_MIR_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: RUST_MIR_VIEW_COMPONENT_NAME,
        componentState: {},
    };
}

/** Get a rust mir view with the given configuration. */
export function getRustMirViewWith(
    id: number,
    source: string,
    rustMirOutput: unknown,
    compilerName: string,
    editorid: number,
    treeid: number,
): ComponentConfig<typeof RUST_MIR_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: RUST_MIR_VIEW_COMPONENT_NAME,
        componentState: {
            id,
            source,
            rustMirOutput,
            compilerName,
            editorid,
            treeid,
        },
    };
}

/** Get an empty haskell core view component. */
export function getHaskellCoreView(): ComponentConfig<typeof HASKELL_CORE_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: HASKELL_CORE_VIEW_COMPONENT_NAME,
        componentState: {},
    };
}

/** Get a haskell core view with the given configuration. */
export function getHaskellCoreViewWith(
    id: number,
    source: string,
    haskellCoreOutput: unknown,
    compilerName: string,
    editorid: number,
    treeid: number,
): ComponentConfig<typeof HASKELL_CORE_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: HASKELL_CORE_VIEW_COMPONENT_NAME,
        componentState: {
            id,
            source,
            haskellCoreOutput,
            compilerName,
            editorid,
            treeid,
        },
    };
}

/** Get an empty haskell stg view component. */
export function getHaskellStgView(): ComponentConfig<typeof HASKELL_STG_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: HASKELL_STG_VIEW_COMPONENT_NAME,
        componentState: {},
    };
}

/** Get a haskell stg view with the given configuration. */
export function getHaskellStgViewWith(
    id: number,
    source: string,
    haskellStgOutput: unknown,
    compilerName: string,
    editorid: number,
    treeid: number,
): ComponentConfig<typeof HASKELL_STG_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: HASKELL_STG_VIEW_COMPONENT_NAME,
        componentState: {
            id,
            source,
            haskellStgOutput,
            compilerName,
            editorid,
            treeid,
        },
    };
}

/** Get an empty haskell cmm view component. */
export function getHaskellCmmView(): ComponentConfig<typeof HASKELL_CMM_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: HASKELL_CMM_VIEW_COMPONENT_NAME,
        componentState: {},
    };
}

export function getHaskellCmmViewWith(
    id: number,
    source: string,
    haskellCmmOutput: unknown,
    compilerName: string,
    editorid: number,
    treeid: number,
): ComponentConfig<typeof HASKELL_CMM_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: HASKELL_CMM_VIEW_COMPONENT_NAME,
        componentState: {
            id,
            source,
            haskellCmmOutput,
            compilerName,
            editorid,
            treeid,
        },
    };
}

/** Get an empty gnat debug tree view component. */
export function getGnatDebugTreeView(): ComponentConfig<typeof GNAT_DEBUG_TREE_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: GNAT_DEBUG_TREE_VIEW_COMPONENT_NAME,
        componentState: {},
    };
}

/** Get a gnat debug tree view with the given configuration. */
export function getGnatDebugTreeViewWith(
    id: number,
    source: string,
    gnatDebugTreeOutput: unknown,
    compilerName: string,
    editorid: number,
    treeid: number,
): ComponentConfig<typeof GNAT_DEBUG_TREE_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: GNAT_DEBUG_TREE_VIEW_COMPONENT_NAME,
        componentState: {
            id,
            source,
            gnatDebugTreeOutput,
            compilerName,
            editorid,
            treeid,
        },
    };
}

/** Get an empty gnat debug info view component. */
export function getGnatDebugView(): ComponentConfig<typeof GNAT_DEBUG_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: GNAT_DEBUG_VIEW_COMPONENT_NAME,
        componentState: {},
    };
}

/** Get a gnat debug info view with the given configuration. */
export function getGnatDebugViewWith(
    id: number,
    source: string,
    gnatDebugOutput: unknown,
    compilerName: string,
    editorid: number,
    treeid: number,
): ComponentConfig<typeof GNAT_DEBUG_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: GNAT_DEBUG_VIEW_COMPONENT_NAME,
        componentState: {
            id,
            source,
            gnatDebugOutput,
            compilerName,
            editorid,
            treeid,
        },
    };
}

/** Get an empty rust macro exp view component. */
export function getRustMacroExpView(): ComponentConfig<typeof RUST_MACRO_EXP_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: RUST_MACRO_EXP_VIEW_COMPONENT_NAME,
        componentState: {},
    };
}

/** Get a rust macro exp view with the given configuration. */
export function getRustMacroExpViewWith(
    id: number,
    source: string,
    rustMacroExpOutput: unknown,
    compilerName: string,
    editorid: number,
    treeid: number,
): ComponentConfig<typeof RUST_MACRO_EXP_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: RUST_MACRO_EXP_VIEW_COMPONENT_NAME,
        componentState: {
            id,
            source,
            rustMacroExpOutput,
            compilerName,
            editorid,
            treeid,
        },
    };
}

/** Get an empty rust hir view component. */
export function getRustHirView(): ComponentConfig<typeof RUST_HIR_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: RUST_HIR_VIEW_COMPONENT_NAME,
        componentState: {},
    };
}

/** Get a rust hir view with the given configuration. */
export function getRustHirViewWith(
    id: number,
    source: string,
    rustHirOutput: unknown,
    compilerName: string,
    editorid: number,
    treeid: number,
): ComponentConfig<typeof RUST_HIR_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: RUST_HIR_VIEW_COMPONENT_NAME,
        componentState: {
            id,
            source,
            rustHirOutput,
            compilerName,
            editorid,
            treeid,
        },
    };
}

/** Get an empty device view component. */
export function getDeviceView(): ComponentConfig<typeof DEVICE_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: DEVICE_VIEW_COMPONENT_NAME,
        componentState: {},
    };
}

/** Get a device view with the given configuration. */
export function getDeviceViewWith(
    id: number,
    source: string,
    devices: unknown,
    compilerName: string,
    editorid: number,
    treeid: number,
): ComponentConfig<typeof DEVICE_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: DEVICE_VIEW_COMPONENT_NAME,
        componentState: {
            id: id,
            source: source,
            devices: devices,
            compilerName: compilerName,
            editorid: editorid,
            treeid: treeid,
        },
    };
}

/** Get an empty explain view component. */
export function getExplainView(): ComponentConfig<typeof EXPLAIN_VIEW_COMPONENT_NAME> {
    return createComponentConfig(EXPLAIN_VIEW_COMPONENT_NAME, {});
}

/** Get an explain view with the given configuration. */
export function getExplainViewWith(
    id: number,
    compilerName: string,
    editorid: number,
    treeid: number,
): ComponentConfig<typeof EXPLAIN_VIEW_COMPONENT_NAME> {
    return createComponentConfig(EXPLAIN_VIEW_COMPONENT_NAME, {
        id,
        compilerName,
        editorid,
        treeid,
    });
}

/**
 * Helper function to create a typed component configuration
 */
export function createComponentConfig<K extends keyof ComponentStateMap>(
    componentName: K,
    componentState: ComponentStateMap[K],
    options?: Pick<AnyComponentConfig, 'title' | 'isClosable' | 'reorderEnabled' | 'width' | 'height'>,
): ComponentConfig<K> {
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
export function createLayoutItem(
    type: 'row' | 'column' | 'stack',
    content: ItemConfig[],
    options?: Pick<LayoutItem, 'isClosable' | 'reorderEnabled' | 'width' | 'height' | 'activeItemIndex'>,
): LayoutItem {
    return {
        type,
        content,
        ...options,
    };
}

/**
 * Helper to convert from GoldenLayout's internal config to our typed config.
 *
 * This function validates that the configuration is valid and all component
 * states match their expected types. It provides helpful error messages
 * for invalid configurations.
 *
 * TODO(#7808): Enable this function for configuration validation
 * Currently unused but ready for implementation - see issue for details.
 *
 * @param config - Untyped config from GoldenLayout, localStorage, or URLs
 * @returns Typed config with validated component states
 * @throws Error if the configuration is invalid (should be caught and handled)
 */
export function fromGoldenLayoutConfig(config: GoldenLayout.Config): GoldenLayoutConfig {
    if (!config || typeof config !== 'object') {
        throw new Error('Invalid configuration: must be an object');
    }

    // Validate the root structure
    return {
        ...config,
        content: config.content ? validateItemConfigs(config.content) : undefined,
    };
}

/**
 * Validates an array of item configurations (recursive)
 */
function validateItemConfigs(items: any[]): ItemConfig[] {
    if (!Array.isArray(items)) {
        throw new Error('Configuration content must be an array');
    }

    return items.map((item, index) => validateItemConfig(item, index));
}

/**
 * Validates a single item configuration (component or layout item)
 */
function validateItemConfig(item: any, index?: number): ItemConfig {
    const location = index !== undefined ? `item ${index}` : 'item';

    if (!item || typeof item !== 'object') {
        throw new Error(`Invalid ${location}: must be an object`);
    }

    if (!item.type) {
        throw new Error(`Invalid ${location}: missing 'type' property`);
    }

    if (item.type === 'component') {
        return validateComponentConfig(item, location);
    }
    if (item.type === 'row' || item.type === 'column' || item.type === 'stack') {
        return validateLayoutItem(item, location);
    }
    throw new Error(`Invalid ${location}: unknown type '${item.type}'`);
}

/**
 * Validates a component configuration
 */
function validateComponentConfig(config: any, location: string): AnyComponentConfig {
    if (!config.componentName) {
        throw new Error(`Invalid ${location}: missing 'componentName' property`);
    }

    if (typeof config.componentName !== 'string') {
        throw new Error(`Invalid ${location}: 'componentName' must be a string`);
    }

    // Validate that the component state matches the expected type for this component
    if (!validateComponentState(config.componentName, config.componentState)) {
        throw new Error(
            `Invalid ${location}: invalid component state for component '${config.componentName}'. ` +
                `State: ${JSON.stringify(config.componentState, null, 2)}`,
        );
    }

    return {
        type: 'component',
        componentName: config.componentName,
        componentState: config.componentState,
        title: config.title,
        isClosable: config.isClosable,
        reorderEnabled: config.reorderEnabled,
        width: config.width,
        height: config.height,
    };
}

/**
 * Validates a layout item (row, column, stack)
 */
function validateLayoutItem(item: any, location: string): LayoutItem {
    if (!item.content || !Array.isArray(item.content)) {
        throw new Error(`Invalid ${location}: layout items must have a 'content' array`);
    }

    return {
        type: item.type as 'row' | 'column' | 'stack',
        content: validateItemConfigs(item.content),
        isClosable: item.isClosable,
        reorderEnabled: item.reorderEnabled,
        width: item.width,
        height: item.height,
        activeItemIndex: item.activeItemIndex,
    };
}

/**
 * Helper to convert to GoldenLayout's expected config format.
 * This direction is safe since we're going from typed to untyped.
 */
export function toGoldenLayoutConfig(config: GoldenLayoutConfig): GoldenLayout.Config {
    return config as GoldenLayout.Config;
}

/**
 * Typed wrapper for createDragSource that returns the drag event emitter directly.
 * This simplifies the API by hiding the internal _dragListener implementation detail.
 *
 * Note: We still need to cast internally because GoldenLayout's TypeScript
 * definitions don't properly type the second parameter as accepting a function.
 */
export function createDragSource<K extends keyof ComponentStateMap>(
    layout: GoldenLayout,
    element: HTMLElement | JQuery,
    factory: DragSourceFactory<K>,
): GoldenLayout.EventEmitter {
    // TODO(#7808): Fix GoldenLayout TypeScript definitions to eliminate 'as any' cast
    // Both the factory parameter type and return type are incorrectly defined
    const result = layout.createDragSource(element, factory as any) as any;
    return result._dragListener;
}

/**
 * Validation function for component states.
 * This ensures that component states match their expected types.
 */
function validateComponentState(componentName: string, state: any): boolean {
    // Basic validation - state must be an object
    if (typeof state !== 'object' || state === null) {
        return false;
    }

    switch (componentName) {
        case COMPILER_COMPONENT_NAME:
            // Compiler states can have various combinations of properties
            return (
                (state.lang && state.source !== undefined) ||
                (state.source !== undefined && state.compiler) ||
                (state.lang && state.tree !== undefined)
            );

        case EXECUTOR_COMPONENT_NAME:
            // Executor states require compilation panel booleans
            return typeof state.compilationPanelShown === 'boolean' && typeof state.compilerOutShown === 'boolean';

        case EDITOR_COMPONENT_NAME:
            // Editor states are flexible but must have valid properties
            return true;

        case TREE_COMPONENT_NAME:
            // Tree states are flexible but must have valid properties
            return true;

        case OUTPUT_COMPONENT_NAME:
            // Output state needs specific numeric properties
            return (
                typeof state.tree === 'number' && typeof state.compiler === 'number' && typeof state.editor === 'number'
            );

        case TOOL_COMPONENT_NAME:
            // Tool state needs specific properties
            return (
                typeof state.tree === 'number' &&
                typeof state.toolId === 'string' &&
                typeof state.id === 'number' &&
                typeof state.editorid === 'number'
            );

        // View components have diverse state requirements but must be valid objects
        case TOOL_INPUT_VIEW_COMPONENT_NAME:
        case DIFF_VIEW_COMPONENT_NAME:
        case OPT_VIEW_COMPONENT_NAME:
        case STACK_USAGE_VIEW_COMPONENT_NAME:
        case FLAGS_VIEW_COMPONENT_NAME:
        case PP_VIEW_COMPONENT_NAME:
        case AST_VIEW_COMPONENT_NAME:
        case GCC_DUMP_VIEW_COMPONENT_NAME:
        case CFG_VIEW_COMPONENT_NAME:
        case CONFORMANCE_VIEW_COMPONENT_NAME:
        case IR_VIEW_COMPONENT_NAME:
        case CLANGIR_VIEW_COMPONENT_NAME:
        case OPT_PIPELINE_VIEW_COMPONENT_NAME:
        case LLVM_OPT_PIPELINE_VIEW_COMPONENT_NAME:
        case RUST_MIR_VIEW_COMPONENT_NAME:
        case HASKELL_CORE_VIEW_COMPONENT_NAME:
        case HASKELL_STG_VIEW_COMPONENT_NAME:
        case HASKELL_CMM_VIEW_COMPONENT_NAME:
        case GNAT_DEBUG_TREE_VIEW_COMPONENT_NAME:
        case GNAT_DEBUG_VIEW_COMPONENT_NAME:
        case RUST_MACRO_EXP_VIEW_COMPONENT_NAME:
        case RUST_HIR_VIEW_COMPONENT_NAME:
        case DEVICE_VIEW_COMPONENT_NAME:
            return true;

        default:
            // Unknown component name - this should not happen with proper typing
            SentryCapture(componentName, `Unknown component name in validateComponentState: ${componentName}`);
            return false;
    }
}
