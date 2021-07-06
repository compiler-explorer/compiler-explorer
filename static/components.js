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
// here instead of in the editor.js and compiler.js etc to prevent circular dependencies.
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
    getExecutor: function (editorId, lang) {
        return {
            type: 'component',
            componentName: 'executor',
            componentState: {source: editorId, lang: lang},
        };
    },
    getExecutorWith: function (editorId, lang, compilerId, libraries, compilerArgs) {
        return {
            type: 'component',
            componentName: 'executor',
            componentState: {
                source: editorId,
                lang: lang,
                compiler: compilerId,
                libs: libraries,
                options: compilerArgs,
            },
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
    getOutput: function (compiler, editor) {
        return {
            type: 'component',
            componentName: 'output',
            componentState: {compiler: compiler, editor: editor},
        };
    },
    getToolViewWith: function (compiler, editor, toolId, args) {
        return {
            type: 'component',
            componentName: 'tool',
            componentState: {
                compiler: compiler,
                editor: editor,
                toolId: toolId,
                args: args,
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
    getOptViewWith: function (id, source, optimization, compilerName, editorid) {
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
    },
    getAstView: function () {
        return {
            type: 'component',
            componentName: 'ast',
            componentState: {},
        };
    },
    getAstViewWith: function (id, source, astOutput, compilerName, editorid) {
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
    },
    getGccDumpView: function () {
        return {
            type: 'component',
            componentName: 'gccdump',
            componentState: {},
        };
    },
    getGccDumpViewWith: function (id, compilerName, editorid, gccDumpOutput) {
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
    },

    getCfgView: function () {
        return {
            type: 'component',
            componentName: 'cfg',
            componentState: {},
        };
    },
    getCfgViewWith: function (id, editorid) {
        return {
            type: 'component',
            componentName: 'cfg',
            componentState: {
                id: id,
                editorid: editorid,
            },
        };
    },
    getConformanceView: function (editorid, source, langId) {
        return {
            type: 'component',
            componentName: 'conformance',
            componentState: {
                editorid: editorid,
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
    getIrViewWith: function (id, source, irOutput, compilerName, editorid) {
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
    },
};
