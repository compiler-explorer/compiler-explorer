// Copyright (c) 2012-2017, Matt Godbolt
//
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

define(function (require) {
    "use strict";
    // here instead of in the editor.js and compiler.js etc to prevent circular dependencies.
    return {
        getCompiler: function (editorId) {
            return {
                type: 'component',
                componentName: 'compiler',
                componentState: {source: editorId}
            };
        },
        getCompilerWith: function (editorId, filters, options, compilerId) {
            return {
                type: 'component',
                componentName: 'compiler',
                componentState: {
                    source: editorId,
                    filters: filters,
                    options: options,
                    compiler: compilerId
                }
            };
        },
        getEditor: function (id) {
            return {
                type: 'component',
                componentName: 'codeEditor',
                componentState: {id: id}
            };
        },
        getEditorWith: function (id, source, options) {
            return {
                type: 'component',
                componentName: 'codeEditor',
                componentState: {id: id, source: source, options: options}
            };
        },
        getOutput: function (compiler, editor) {
            return {
                type: 'component',
                componentName: 'output',
                componentState: {compiler: compiler, editor: editor},
            };
        },
        getDiff: function () {
            return {
                type: 'component',
                componentName: 'diff',
                componentState: {}
            };
        },
        getOptView: function() {
            return {
                type: 'component',
                componentName: 'opt',
                componentState: {}
            };
        },
        getOptViewWith: function (id, source, optimization, compilerName, editorid) {
            return {
                type: 'component',
                componentName: 'opt',
                componentState: {id: id, source: source, optOutput: optimization, compilerName: compilerName, editorid: editorid}
            };
        }
    };

});