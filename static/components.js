// Copyright (c) 2016, Compiler Explorer Authors
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
// here instead of in the editor.js and compiler.js etc. to prevent circular dependencies.
module.exports = {
    getCompiler: function (editorId, lang) {
        return {
            type: 'component',
            componentName: 'compiler',
            componentState: {source: editorId, lang: lang},
        };
    },
    getCompilerWith: function (editorId, filters, options, compilerId, langId, libs) {
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
    },
    getCompilerForTree: function (treeId, lang) {
        return {
            type: 'component',
            componentName: 'compiler',
            componentState: {tree: treeId, lang: lang},
        };
    },
    getExecutor: function (editorId, lang) {
        return {
            type: 'component',
            componentName: 'executor',
            componentState: {source: editorId, lang: lang},
        };
    },
    getExecutorWith: function (editorId, lang, compilerId, libraries, compilerArgs, treeId) {
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
    },
    getExecutorForTree: function (treeId, lang) {
        return {
            type: 'component',
            componentName: 'executor',
            componentState: {tree: treeId, lang: lang},
        };
    },
    getEditor: function (id, langId) {
        return {
            type: 'component',
            componentName: 'codeEditor',
            componentState: {id: id, lang: langId},
        };
    },
    getEditorWith: function (id, source, options) {
        return {
            type: 'component',
            componentName: 'codeEditor',
            componentState: {id: id, source: source, options: options},
        };
    },
    getTree: function (id) {
        return {
            type: 'component',
            componentName: 'tree',
            componentState: {id: id},
        };
    },
    getOutput: function (compiler, editor, tree) {
        return {
            type: 'component',
            componentName: 'output',
            componentState: {compiler: compiler, editor: editor, tree: tree},
        };
    },
    getToolViewWith: function (compiler, editor, toolId, args, monacoStdin, tree) {
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
    },
    getToolInputView: function () {
        return {
            type: 'component',
            componentName: 'toolInputView',
            componentState: {},
        };
    },
    getToolInputViewWith: function (compilerId, toolId, toolName) {
        return {
            type: 'component',
            componentName: 'toolInputView',
            componentState: {
                compilerId: compilerId,
                toolId: toolId,
                toolName: toolName,
            },
        };
    },
    getDiff: function () {
        return {
            type: 'component',
            componentName: 'diff',
            componentState: {},
        };
    },
    getOptView: function () {
        return {
            type: 'component',
            componentName: 'opt',
            componentState: {},
        };
    },
    getOptViewWith: function (id, source, optimization, compilerName, editorid, treeid) {
        return {
            type: 'component',
            componentName: 'opt',
            componentState: {
                id: id,
                source: source,
                optOutput: optimization,
                compilerName: compilerName,
                editorid: editorid,
                treeid: treeid,
            },
        };
    },
    getFlagsView: function () {
        return {
            type: 'component',
            componentName: 'flags',
            componentState: {},
        };
    },
    getFlagsViewWith: function (id, compilerName, compilerFlags) {
        return {
            type: 'component',
            componentName: 'flags',
            componentState: {
                id: id,
                compilerName: compilerName,
                compilerFlags: compilerFlags,
            },
        };
    },
    getPpView: function () {
        return {
            type: 'component',
            componentName: 'pp',
            componentState: {},
        };
    },
    getPpViewWith: function (id, source, ppOutput, compilerName, editorid, treeid) {
        return {
            type: 'component',
            componentName: 'pp',
            componentState: {
                id: id,
                source: source,
                ppOutput: ppOutput,
                compilerName: compilerName,
                editorid: editorid,
                treeid: treeid,
            },
        };
    },
    getDiffView: function () {
        return {
            type: 'component',
            componentName: 'diff',
            componentState: {},
        };
    },
    getDiffViewWith: function (lhs, rhs) {
        return {
            type: 'component',
            componentName: 'diff',
            componentState: {lhs: lhs, rhs: rhs},
        };
    },
    getAstView: function () {
        return {
            type: 'component',
            componentName: 'ast',
            componentState: {},
        };
    },
    getAstViewWith: function (id, source, astOutput, compilerName, editorid, treeid) {
        return {
            type: 'component',
            componentName: 'ast',
            componentState: {
                id: id,
                source: source,
                astOutput: astOutput,
                compilerName: compilerName,
                editorid: editorid,
                treeid: treeid,
            },
        };
    },
    getGccDumpView: function () {
        return {
            type: 'component',
            componentName: 'gccdump',
            componentState: {},
        };
    },
    getGccDumpViewWith: function (id, compilerName, editorid, treeid, gccDumpOutput) {
        var ret = {
            type: 'component',
            componentName: 'gccdump',
            componentState: {
                _compilerid: id,
                _compilerName: compilerName,
                _editorid: editorid,
                _treeid: treeid,
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
    },

    getCfgView: function () {
        return {
            type: 'component',
            componentName: 'cfg',
            componentState: {},
        };
    },
    getCfgViewWith: function (id, editorid, treeid) {
        return {
            type: 'component',
            componentName: 'cfg',
            componentState: {
                id: id,
                editorid: editorid,
                treeid: treeid,
            },
        };
    },
    getConformanceView: function (editorid, treeid, source, langId) {
        return {
            type: 'component',
            componentName: 'conformance',
            componentState: {
                editorid: editorid,
                treeid: treeid,
                source: source,
                langId: langId,
            },
        };
    },
    getIrView: function () {
        return {
            type: 'component',
            componentName: 'ir',
            componentState: {},
        };
    },
    getIrViewWith: function (id, source, irOutput, compilerName, editorid, treeid) {
        return {
            type: 'component',
            componentName: 'ir',
            componentState: {
                id: id,
                source: source,
                irOutput: irOutput,
                compilerName: compilerName,
                editorid: editorid,
                treeid: treeid,
            },
        };
    },
    getRustMirView: function () {
        return {
            type: 'component',
            componentName: 'rustmir',
            componentState: {},
        };
    },
    getRustMirViewWith: function (id, source, rustMirOutput, compilerName, editorid, treeid) {
        return {
            type: 'component',
            componentName: 'rustmir',
            componentState: {
                id: id,
                source: source,
                rustMirOutput: rustMirOutput,
                compilerName: compilerName,
                editorid: editorid,
                treeid: treeid,
            },
        };
    },
    getHaskellCoreView: function () {
        return {
            type: 'component',
            componentName: 'haskellCore',
            componentState: {},
        };
    },
    getHaskellCoreViewWith: function (id, source, haskellCoreOutput, compilerName, editorid, treeid) {
        return {
            type: 'component',
            componentName: 'haskellCore',
            componentState: {
                id: id,
                source: source,
                haskellCoreOutput: haskellCoreOutput,
                compilerName: compilerName,
                editorid: editorid,
                treeid: treeid,
            },
        };
    },
    getHaskellStgView: function () {
        return {
            type: 'component',
            componentName: 'haskellStg',
            componentState: {},
        };
    },
    getHaskellStgViewWith: function (id, source, haskellStgOutput, compilerName, editorid, treeid) {
        return {
            type: 'component',
            componentName: 'haskellStg',
            componentState: {
                id: id,
                source: source,
                haskellStgOutput: haskellStgOutput,
                compilerName: compilerName,
                editorid: editorid,
                treeid: treeid,
            },
        };
    },
    getHaskellCmmView: function () {
        return {
            type: 'component',
            componentName: 'haskellCmm',
            componentState: {},
        };
    },
    getHaskellCmmViewWith: function (id, source, haskellCmmOutput, compilerName, editorid, treeid) {
        return {
            type: 'component',
            componentName: 'haskellCmm',
            componentState: {
                id: id,
                source: source,
                haskellCmmOutput: haskellCmmOutput,
                compilerName: compilerName,
                editorid: editorid,
                treeid: treeid,
            },
        };
    },

    getGnatDebugTreeView: function () {
        return {
            type: 'component',
            componentName: 'gnatdebugtree',
            componentState: {},
        };
    },
    getGnatDebugTreeViewWith: function (id, source, gnatDebugTreeOutput, compilerName, editorid, treeid) {
        return {
            type: 'component',
            componentName: 'gnatdebugtree',
            componentState: {
                id: id,
                source: source,
                gnatDebugTreeOutput: gnatDebugTreeOutput,
                compilerName: compilerName,
                editorid: editorid,
                treeid: treeid,
            },
        };
    },

    getGnatDebugView: function () {
        return {
            type: 'component',
            componentName: 'gnatdebug',
            componentState: {},
        };
    },
    getGnatDebugViewWith: function (id, source, gnatDebugOutput, compilerName, editorid, treeid) {
        return {
            type: 'component',
            componentName: 'gnatdebug',
            componentState: {
                id: id,
                source: source,
                gnatDebugOutput: gnatDebugOutput,
                compilerName: compilerName,
                editorid: editorid,
                treeid: treeid,
            },
        };
    },
    getRustMacroExpView: function () {
        return {
            type: 'component',
            componentName: 'rustmacroexp',
            componentState: {},
        };
    },
    getRustMacroExpViewWith: function (id, source, rustMacroExpOutput, compilerName, editorid, treeid) {
        return {
            type: 'component',
            componentName: 'rustmacroexp',
            componentState: {
                id: id,
                source: source,
                rustMacroExpOutput: rustMacroExpOutput,
                compilerName: compilerName,
                editorid: editorid,
                treeid: treeid,
            },
        };
    },
    getRustHirView: function () {
        return {
            type: 'component',
            componentName: 'rusthir',
            componentState: {},
        };
    },
    getRustHirViewWith: function (id, source, rustHirOutput, compilerName, editorid, treeid) {
        return {
            type: 'component',
            componentName: 'rusthir',
            componentState: {
                id: id,
                source: source,
                rustHirOutput: rustHirOutput,
                compilerName: compilerName,
                editorid: editorid,
                treeid: treeid,
            },
        };
    },
    getDeviceView: function () {
        return {
            type: 'component',
            componentName: 'device',
            componentState: {},
        };
    },
    getDeviceViewWith: function (id, source, deviceOutput, compilerName, editorid, treeid) {
        return {
            type: 'component',
            componentName: 'device',
            componentState: {
                id: id,
                source: source,
                deviceOutput: deviceOutput,
                compilerName: compilerName,
                editorid: editorid,
                treeid: treeid,
            },
        };
    },
};
