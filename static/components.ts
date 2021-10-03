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

// here instead of in the editor.js and compiler.js etc to prevent circular dependencies.
import {
    AstOutputComponentState,
    CompilerComponentState,
    CompilerFilterComponentState, CompilerFlagsComponentState, CompilerNameComponentState,
    ComponentConfig, DeviceOutputComponentState,
    EditorComponentState, EditorIdComponentState, GccDumpComponentConfig,
    IdComponentState, IrOutputComponentState, LangIdComponentState,
    LibraryComponentState,
    LibraryItem, OptimizationComponentState,
    OptionsComponentState, RustMacroComponentState, RustMirComponentState,
    SourceComponentState, ToolInputViewComponentState, ToolViewComponentState,
    TreeComponentState,
} from './components.interfaces';
import {Dictionary} from 'underscore';

export function getCompiler(editorId: number, lang: string): ComponentConfig<SourceComponentState> {
    return {
        type: 'component',
        componentName: 'compiler',
        componentState: {
            source: editorId,
            lang: lang,
        },
    };
}

// TODO: Add options type
export function getCompilerWith(editorId: number, filters: Dictionary<boolean>, options, compilerId: string, langId: string, libs: LibraryItem[]): ComponentConfig<CompilerFilterComponentState> {
    return {
        type: 'component',
        componentName: 'compiler',
        componentState: {
            source: editorId,
            filters: filters,
            options: options,
            compiler: compilerId,
            lang: langId,
            libs: libs,
        },
    };
}

export function getCompilerForTree(treeId: number, lang: string): ComponentConfig<TreeComponentState> {
    return {
        type: 'component',
        componentName: 'compiler',
        componentState: {
            tree: treeId,
            lang: lang,
        },
    };
}

export function getExecutor(editorId: number, lang: string): ComponentConfig<SourceComponentState> {
    return {
        type: 'component',
        componentName: 'executor',
        componentState: {
            source: editorId,
            lang: lang,
        },
    };
}

type GetExecutorWithConfig = ComponentConfig<CompilerComponentState & TreeComponentState & LibraryComponentState>;

// TODO: Add compilerArgs type
export function getExecutorWith(editorId: number, lang: string, compilerId: string, libraries: LibraryItem[], compilerArgs, treeId: number): GetExecutorWithConfig {
    return {
        type: 'component',
        componentName: 'executor',
        componentState: {
            source: editorId,
            tree: treeId,
            lang: lang,
            compiler: compilerId,
            libs: libraries,
            options: compilerArgs,
        },
    };
}

export function getExecutorForTree(treeId: number, lang: string): ComponentConfig<TreeComponentState> {
    return {
        type: 'component',
        componentName: 'executor',
        componentState: {
            tree: treeId,
            lang: lang,
        },
    };
}

export function getEditor(id: number, langId: string): ComponentConfig<IdComponentState> {
    return {
        type: 'component',
        componentName: 'codeEditor',
        componentState: {
            id: id,
            lang: langId,
        }
    };
}

type GetEditorWithConfig = ComponentConfig<IdComponentState & OptionsComponentState & SourceComponentState>;

// TODO: Add options type
export function getEditorWith(id: number, source: number, options): GetEditorWithConfig {
    return {
        type: 'component',
        componentName: 'codeEditor',
        componentState: {
            id: id,
            source: source,
            options: options,
        }
    };
}

export function getTree(id: number): ComponentConfig<IdComponentState> {
    return {
        type: 'component',
        componentName: 'tree',
        componentState: {
            id: id
        },
    };
}

export function getOutput(compiler: string, editor: number, tree: number): ComponentConfig<EditorComponentState> {
    return {
        type: 'component',
        componentName: 'output',
        componentState: {
            compiler: compiler,
            editor: editor,
            tree: tree,
        },
    };
}

// TODO: Add toolId type
export function getToolViewWith(compiler: string, editor: number, toolId, args: string, monacoStdin: boolean, tree: number): ComponentConfig<ToolViewComponentState> {
    return {
        type: 'component',
        componentName: 'tool',
        componentState: {
            compiler: compiler,
            editor: editor,
            toolId: toolId,
            args: args,
            tree: tree,
            monacoStdin: monacoStdin,
        },
    };
}

export function getToolInputView(): ComponentConfig {
    return {
        type: 'component',
        componentName: 'toolInputView',
        componentState: {},
    };
}

// TODO: Add toolId type
export function getToolInputViewWith(compilerId: string, toolId, toolName: string): ComponentConfig<ToolInputViewComponentState> {
    return {
        type: 'component',
        componentName: 'toolInputView',
        componentState: {
            compilerId: compilerId,
            toolId: toolId,
            toolName: toolName,
        },
    };
}

export function getDiff(): ComponentConfig {
    return {
        type: 'component',
        componentName: 'diff',
        componentState: {},
    };
}

export function getOptView() {
    return {
        type: 'component',
        componentName: 'opt',
        componentState: {},
    };
}

type GetOptViewWithConfig = ComponentConfig<CompilerNameComponentState & SourceComponentState & EditorIdComponentState & OptimizationComponentState>;

// TODO: Add optimization type
export function getOptViewWith(id: number, source: number, optimization, compilerName: string, editorid: boolean): GetOptViewWithConfig {
    return {
        type: 'component',
        componentName: 'opt',
        componentState: {
            id: id,
            source: source,
            optOutput: optimization,
            compilerName: compilerName,
            editorid: editorid,
        },
    };
}

export function getFlagsView(): ComponentConfig {
    return {
        type: 'component',
        componentName: 'flags',
        componentState: {},
    };
}

// TODO: Add compilerFlags type
export function getFlagsViewWith(id: number, compilerName: string, compilerFlags): ComponentConfig<CompilerFlagsComponentState> {
    return {
        type: 'component',
        componentName: 'flags',
        componentState: {
            id: id,
            compilerName: compilerName,
            compilerFlags: compilerFlags,
        },
    };
}

export function getAstView(): ComponentConfig {
    return {
        type: 'component',
        componentName: 'ast',
        componentState: {},
    };
}

// TODO: Add astOutput type
export function getAstViewWith(id: number, source: number, astOutput, compilerName: string, editorid: boolean): ComponentConfig<AstOutputComponentState> {
    return {
        type: 'component',
        componentName: 'ast',
        componentState: {
            id: id,
            source: source,
            astOutput: astOutput,
            compilerName: compilerName,
            editorid: editorid,
        },
    };
}

export function getGccDumpView(): ComponentConfig {
    return {
        type: 'component',
        componentName: 'gccdump',
        componentState: {},
    };
}

// TODO: Add gccDumpOutput type
export function getGccDumpViewWith(id: number, compilerName: string, editorid: boolean, gccDumpOutput): GccDumpComponentConfig {
    var ret: GccDumpComponentConfig = {
        type: 'component',
        componentName: 'gccdump',
        componentState: {
            _compilerid: id,
            _compilerName: compilerName,
            _editorid: editorid,
        },
    };

    if (gccDumpOutput) {
        ret.treeDump = gccDumpOutput.treeDump;
        ret.rtlDump = gccDumpOutput.rtlDump;
        ret.ipaDump = gccDumpOutput.ipaDump;
        ret.addressOption = gccDumpOutput.addressOption;
        ret.slimOption = gccDumpOutput.slimOption;
        ret.rawOption = gccDumpOutput.rawOption;
        ret.detailsOption = gccDumpOutput.detailsOption;
        ret.statsOption = gccDumpOutput.statsOption;
        ret.blocksOption = gccDumpOutput.blocksOption;
        ret.vopsOption = gccDumpOutput.vopsOption;
        ret.linenoOption = gccDumpOutput.linenoOption;
        ret.uidOption = gccDumpOutput.uidOption;
        ret.allOption = gccDumpOutput.allOption;
        ret.selectedPass = gccDumpOutput.selectedPass;
    }

    return ret;
}

export function getCfgView(): ComponentConfig {
    return {
        type: 'component',
        componentName: 'cfg',
        componentState: {},
    };
}

export function getCfgViewWith(id: number, editorid: boolean): ComponentConfig<EditorIdComponentState & IdComponentState> {
    return {
        type: 'component',
        componentName: 'cfg',
        componentState: {
            id: id,
            editorid: editorid,
        },
    };
}

// langId may be string...
// TODO: Add langId type
export function getConformanceView(editorid: boolean, source: number, langId): ComponentConfig<LangIdComponentState> {
    return {
        type: 'component',
        componentName: 'conformance',
        componentState: {
            editorid: editorid,
            source: source,
            langId: langId,
        },
    };
}

export function getIrView(): ComponentConfig {
    return {
        type: 'component',
        componentName: 'ir',
        componentState: {},
    };
}

// TODO: Add irOutput type
export function getIrViewWith(id: number, source: number, irOutput, compilerName: string, editorid: boolean): ComponentConfig<IrOutputComponentState> {
    return {
        type: 'component',
        componentName: 'ir',
        componentState: {
            id: id,
            source: source,
            irOutput: irOutput,
            compilerName: compilerName,
            editorid: editorid,
        },
    };
}

export function getRustMirView(): ComponentConfig {
    return {
        type: 'component',
        componentName: 'rustmir',
        componentState: {},
    };
}

// TODO: Add rustMirOutput type
export function getRustMirViewWith(id: number, source: number, rustMirOutput, compilerName: string, editorid: boolean): ComponentConfig<RustMirComponentState> {
    return {
        type: 'component',
        componentName: 'rustmir',
        componentState: {
            id: id,
            source: source,
            rustMirOutput: rustMirOutput,
            compilerName: compilerName,
            editorid: editorid,
        },
    };
}

export function getRustMacroExpView(): ComponentConfig {
    return {
        type: 'component',
        componentName: 'rustmacroexp',
        componentState: {},
    };
}

// TODO: Add rustMacroExpOutput type
export function getRustMacroExpViewWith(id: number, source: number, rustMacroExpOutput, compilerName: string, editorid: boolean): ComponentConfig<RustMacroComponentState> {
    return {
        type: 'component',
        componentName: 'rustmacroexp',
        componentState: {
            id: id,
            source: source,
            rustMacroExpOutput: rustMacroExpOutput,
            compilerName: compilerName,
            editorid: editorid,
        },
    };
}

export function getDeviceView(): ComponentConfig {
    return {
        type: 'component',
        componentName: 'device',
        componentState: {},
    };
}

// TODO: Add deviceOutput type
export function getDeviceViewWith(id: number, source: number, deviceOutput, compilerName: string, editorid: boolean): ComponentConfig<DeviceOutputComponentState> {
    return {
        type: 'component',
        componentName: 'device',
        componentState: {
            id: id,
            source: source,
            deviceOutput: deviceOutput,
            compilerName: compilerName,
            editorid: editorid,
        },
    };
}
