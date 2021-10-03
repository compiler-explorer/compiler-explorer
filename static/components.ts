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

'use strict';

// here instead of in the editor.js and compiler.js etc to prevent circular dependencies.
import {
    CompilerComponentState,
    CompilerFilterComponentState,
    ComponentConfig, IdComponentState, LangComponentState, LibraryItem, OptionsComponentState, SourceComponentState,
    TreeComponentState,
} from './components.interfaces';

export function getCompiler(editorId: number, lang: string): ComponentConfig {
    return {
        type: 'component',
        componentName: 'compiler',
        componentState: {
            source: editorId,
            lang: lang
        }
    };
}

export function getCompilerWith(editorId: number, filters, options, compilerId: string, langId: string, libs: LibraryItem[]): ComponentConfig<CompilerFilterComponentState> {
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
            lang: lang
        }
    };
}

export function getExecutor(editorId: number, lang: string): ComponentConfig {
    return {
        type: 'component',
        componentName: 'executor',
        componentState: {
            source: editorId,
            lang: lang
        }
    };
}

export function getExecutorWith(editorId: number, lang: string, compilerId: string, libraries: LibraryItem[], compilerArgs, treeId: number): ComponentConfig<CompilerComponentState & TreeComponentState> {
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
            lang: lang
        }
    };
}

export function getEditor(id: number, langId: string): ComponentConfig<IdComponentState & LangComponentState> {
    return {
        type: 'component',
        componentName: 'codeEditor',
        componentState: {
            id: id,
            lang: langId
        }
    };
}

export function getEditorWith(id: number, source: number, options): ComponentConfig<IdComponentState & OptionsComponentState & SourceComponentState> {
    return {
        type: 'component',
        componentName: 'codeEditor',
        componentState: {
            id: id,
            source: source,
            options: options
        }
    };
}

export function getTree(id) {
    return {
        type: 'component',
        componentName: 'tree',
        componentState: {id: id},
    };
}

export function getOutput(compiler, editor, tree) {
    return {
        type: 'component',
        componentName: 'output',
        componentState: {compiler: compiler, editor: editor, tree: tree},
    };
}

export function getToolViewWith(compiler, editor, toolId, args, monacoStdin, tree) {
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

export function getToolInputView() {
    return {
        type: 'component',
        componentName: 'toolInputView',
        componentState: {},
    };
}

export function getToolInputViewWith(compilerId, toolId, toolName) {
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

export function getDiff() {
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

export function getOptViewWith(id, source, optimization, compilerName, editorid) {
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

export function getFlagsView() {
    return {
        type: 'component',
        componentName: 'flags',
        componentState: {},
    };
}

export function getFlagsViewWith(id, compilerName, compilerFlags) {
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

export function getAstView() {
    return {
        type: 'component',
        componentName: 'ast',
        componentState: {},
    };
}

export function getAstViewWith(id, source, astOutput, compilerName, editorid) {
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

export function getGccDumpView() {
    return {
        type: 'component',
        componentName: 'gccdump',
        componentState: {},
    };
}

export function getGccDumpViewWith(id, compilerName, editorid, gccDumpOutput) {
    var ret = {
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

export function getCfgView() {
    return {
        type: 'component',
        componentName: 'cfg',
        componentState: {},
    };
}

export function getCfgViewWith(id, editorid) {
    return {
        type: 'component',
        componentName: 'cfg',
        componentState: {
            id: id,
            editorid: editorid,
        },
    };
}

export function getConformanceView(editorid, source, langId) {
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

export function getIrView() {
    return {
        type: 'component',
        componentName: 'ir',
        componentState: {},
    };
}

export function getIrViewWith(id, source, irOutput, compilerName, editorid) {
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

export function getRustMirView() {
    return {
        type: 'component',
        componentName: 'rustmir',
        componentState: {},
    };
}

export function getRustMirViewWith(id, source, rustMirOutput, compilerName, editorid) {
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

export function getRustMacroExpView() {
    return {
        type: 'component',
        componentName: 'rustmacroexp',
        componentState: {},
    };
}

export function getRustMacroExpViewWith(id, source, rustMacroExpOutput, compilerName, editorid) {
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

export function getDeviceView() {
    return {
        type: 'component',
        componentName: 'device',
        componentState: {},
    };
}

export function getDeviceViewWith(id, source, deviceOutput, compilerName, editorid) {
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
