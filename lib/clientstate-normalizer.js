// Copyright (c) 2018, Compiler Explorer Authors
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
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ,
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
"use strict";

let ClientState = require('./clientstate');

class ClientStateNormalizer {
    constructor() {
        this.normalized = new ClientState.State();
    }

    fromGoldenLayoutContent(content) {
        content.forEach((component) => {
            this.fromGoldenLayoutComponent(component);
        });
    }

    fromGoldenLayoutComponent(component) {
        if (component.componentName === "codeEditor") {
            const session = this.normalized.findOrCreateSession(component.componentState.id);
            session.language = component.componentState.lang;
            session.source = component.componentState.source;
        } else if (component.componentName === "compiler") {
            const session = this.normalized.findOrCreateSession(component.componentState.source);

            const compiler = new ClientState.Compiler();
            compiler.id = component.componentState.compiler;
            compiler.options = component.componentState.options;
            compiler.libs = component.componentState.libs;
            compiler.filters.binary = component.componentState.filters.binary;
            compiler.filters.labels = component.componentState.filters.labels;
            compiler.filters.directives = component.componentState.filters.directives;
            compiler.filters.commentOnly = component.componentState.filters.commentOnly;
            compiler.filters.trim = component.componentState.filters.trim;
            compiler.filters.intel = component.componentState.filters.intel;
            compiler.filters.demangle = component.componentState.filters.demangle;

            session.compilers.push(compiler);
        } else if (component.componentName === "ast") {
            const session = this.normalized.findOrCreateSession(component.componentState.editorid);
            const compiler = session.findOrCreateCompiler(component.componentState.id);

            compiler.specialoutputs.push("ast");
        } else if (component.componentName === "opt") {
            const session = this.normalized.findOrCreateSession(component.componentState.editorid);
            const compiler = session.findOrCreateCompiler(component.componentState.id);

            compiler.specialoutputs.push("opt");
        } else if (component.componentName === "cfg") {
            const session = this.normalized.findOrCreateSession(component.componentState.editorid);
            const compiler = session.findOrCreateCompiler(component.componentState.id);

            compiler.specialoutputs.push("cfg");
        } else if (component.componentName === "gccdump") {
            const session = this.normalized.findOrCreateSession(component.componentState._editorid);
            const compiler = session.findOrCreateCompiler(component.componentState._compilerid);

            compiler.specialoutputs.push("gccdump");
        } else if (component.content) {
            this.fromGoldenLayoutContent(component.content);
        }
    }

    fromGoldenLayout(globj) {
        if (globj.content) {
            this.fromGoldenLayoutContent(globj.content);
        }
    }
}

class ClientStateGoldenifier {
    constructor() {
        this.golden = {};
    }

    newSourceStackFromSession(session, width) {
        return {
            type: "stack",
            width: width,
            content: [
                {
                    type: "component",
                    componentName: "codeEditor",
                    componentState: {
                        id: session.id,
                        source: session.source,
                        lang: session.language,
                    },
                    isClosable: true,
                    reorderEnabled: true
                }
            ]
        };
    }

    newAstStackFromCompiler(session, compiler, compilerIndex, width) {
        return {
            type: "stack",
            width: width,
            content: [
                {
                    type: "component",
                    componentName: "ast",
                    componentState: {
                        id: compilerIndex,
                        editorid: session.id
                    },
                    isClosable: true,
                    reorderEnabled: true
                }
            ]
        };
    }

    newOptStackFromCompiler(session, compiler, compilerIndex, width) {
        return {
            type: "stack",
            width: width,
            content: [
                {
                    type: "component",
                    componentName: "opt",
                    componentState: {
                        id: compilerIndex,
                        editorid: session.id
                    },
                    isClosable: true,
                    reorderEnabled: true
                }
            ]
        };
    }

    newCfgStackFromCompiler(session, compiler, compilerIndex, width) {
        return {
            type: "stack",
            width: width,
            content: [
                {
                    type: "component",
                    componentName: "opt",
                    componentState: {
                        id: compilerIndex,
                        editorid: session.id,
                        options: {
                            navigation: false,
                            physics: false
                        }
                    },
                    isClosable: true,
                    reorderEnabled: true
                }
            ]
        };
    }

    newGccDumpStackFromCompiler(session, compiler, compilerIndex, width) {
        return {
            type: "stack",
            width: width,
            content: [
                {
                    type: "component",
                    componentName: "gccdump",
                    componentState: {
                        _compilerid: compilerIndex,
                        _editorid: session.id
                    },
                    isClosable: true,
                    reorderEnabled: true
                }
            ]
        };
    }

    newCompilerStackFromSession(session, compiler, width) {
        return {
            type: "stack",
            width: width,
            content: [
                {
                    type: "component",
                    componentName: "compiler",
                    componentState: {
                        compiler: compiler.id,
                        source: session.id,
                        options: compiler.options,
                        filters: {
                            binary: compiler.filters.binary,
                            execute: compiler.filters.execute,
                            labels: compiler.filters.labels,
                            directives: compiler.filters.directives,
                            commentOnly: compiler.filters.commentOnly,
                            trim: compiler.filters.trim,
                            intel: compiler.filters.intel,
                            demangle: compiler.filters.demangle
                        },
                        libs: compiler.libs,
                        lang: session.language
                    },
                    isClosable: true,
                    reorderEnabled: true
                }
            ]
        };
    }

    fromClientState(state) {
        this.golden = {
            settings: {
                hasHeaders: true,
                constrainDragToContainer: false,
                reorderEnabled: true,
                selectionEnabled: false,
                popoutWholeStack: false,
                blockedPopoutsThrowError: true,
                closePopoutsOnUnload: true,
                showPopoutIcon: false,
                showMaximiseIcon: true,
                showCloseIcon: true,
                responsiveMode: "onload",
                tabOverlapAllowance: 0,
                reorderOnTabMenuClick: true,
                tabControlOffset: 10
            },
            dimensions: {
                borderWidth: 5,
                borderGrabWidth: 15,
                minItemHeight: 10,
                minItemWidth: 10,
                headerHeight: 20,
                dragProxyWidth: 300,
                dragProxyHeight: 200
            },
            labels: {
                close: "close",
                maximise: "maximise",
                minimise: "minimise",
                popout: "open in new window",
                popin: "pop in",
                tabDropdown: "additional tabs"
            },
            content: [
                {
                    type: "row",
                    content: [
                    ]
                }
            ]
        };

        if (state.sessions.length > 1) {
            const sessionWidth = 100.0 / state.sessions.length;
                
            for (let idxSession = 0; idxSession < state.sessions.length; idxSession++) {
                const session = state.sessions[idxSession];

                this.golden.content[0].content[idxSession] = {
                    type: "column",
                    isClosable: true,
                    reorderEnabled: true,
                    width: sessionWidth,
                    content: [
                        {
                            type: "row",
                            content: [
                            ]
                        },
                        {
                            type: "row",
                            content: [
                            ]
                        }
                    ]
                };
    
                let stack = this.newSourceStackFromSession(session, 100.0);
                this.golden.content[0].content[idxSession].content[0].content.push(stack);

                const compilerWidth = 100.0 / session.compilers.length;
                session.compilers.forEach((compiler) => {
                    let stack = this.newCompilerStackFromSession(session, compiler, compilerWidth);
                    this.golden.content[0].content[idxSession].content[1].content.push(stack);
                });
            }
        } else if (state.sessions.length === 1) {
            const session = state.sessions[0];
            this.golden.content[0] = {
                type: "row",
                content: [
                ]
            };

            const width = 100.0 / (1 + session.compilers.length + session.countNumberOfSpecialOutputs());
            this.golden.content[0].content.push(this.newSourceStackFromSession(session, width));

            for (let idxCompiler = 0; idxCompiler < session.compilers.length; idxCompiler++) {
                const compiler = session.compilers[idxCompiler];
                let stack = this.newCompilerStackFromSession(session, compiler, width);
                this.golden.content[0].content.push(stack);

                compiler.specialoutputs.forEach((viewtype) => {
                    let stack = null;
                    if (viewtype === "ast") {
                        stack = this.newAstStackFromCompiler(session, compiler, idxCompiler+1, width);
                    } else if (viewtype === "opt") {
                        stack = this.newOptStackFromCompiler(session, compiler, idxCompiler+1, width);
                    } else if (viewtype === "cfg") {
                        stack = this.newCfgStackFromCompiler(session, compiler, idxCompiler+1, width);
                    } else if (viewtype === "gccdump") {
                        stack = this.newGccDumpStackFromCompiler(session, compiler, idxCompiler+1, width);
                    }

                    if (stack) {
                        this.golden.content[0].content.push(stack);
                    }
                });
            }
        }
    }
}

module.exports = {
    ClientStateNormalizer: ClientStateNormalizer,
    ClientStateGoldenifier: ClientStateGoldenifier
};
