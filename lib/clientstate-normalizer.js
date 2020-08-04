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
'use strict';

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

    setFilterSettingsFromComponent(compiler, component) {
        compiler.filters.binary = component.componentState.filters.binary;
        compiler.filters.labels = component.componentState.filters.labels;
        compiler.filters.directives = component.componentState.filters.directives;
        compiler.filters.commentOnly = component.componentState.filters.commentOnly;
        compiler.filters.trim = component.componentState.filters.trim;
        compiler.filters.intel = component.componentState.filters.intel;
        compiler.filters.demangle = component.componentState.filters.demangle;
    }

    fromGoldenLayoutComponent(component) {
        if (component.componentName === 'codeEditor') {
            const session = this.normalized.findOrCreateSession(component.componentState.id);
            session.language = component.componentState.lang;
            session.source = component.componentState.source;
        } else if (component.componentName === 'compiler') {
            const session = this.normalized.findOrCreateSession(component.componentState.source);

            const compiler = new ClientState.Compiler();
            compiler.id = component.componentState.compiler;
            compiler.options = component.componentState.options;
            compiler.libs = component.componentState.libs;
            this.setFilterSettingsFromComponent(compiler, component);

            session.compilers.push(compiler);
        } else if (component.componentName === 'executor') {
            const session = this.normalized.findOrCreateSession(component.componentState.source);

            const executor = new ClientState.Executor();
            executor.compiler.id = component.componentState.compiler;
            executor.compiler.options = component.componentState.options;
            executor.compiler.libs = component.componentState.libs;
            executor.compilerVisible = component.componentState.compilationPanelShown;
            executor.compilerOutputVisible = component.componentState.compilerOutShown;
            executor.arguments = component.componentState.execArgs;
            executor.argumentsVisible = component.componentState.argsPanelShown;
            executor.stdin = component.componentState.execStdin;
            executor.stdinVisible = component.componentState.stdinPanelShown;

            session.executors.push(executor);
        } else if (component.componentName === 'ast') {
            const session = this.normalized.findOrCreateSession(component.componentState.editorid);
            const compiler = session.findOrCreateCompiler(component.componentState.id);

            compiler.specialoutputs.push('ast');
        } else if (component.componentName === 'opt') {
            const session = this.normalized.findOrCreateSession(component.componentState.editorid);
            const compiler = session.findOrCreateCompiler(component.componentState.id);

            compiler.specialoutputs.push('opt');
        } else if (component.componentName === 'cfg') {
            const session = this.normalized.findOrCreateSession(component.componentState.editorid);
            const compiler = session.findOrCreateCompiler(component.componentState.id);

            compiler.specialoutputs.push('cfg');
        } else if (component.componentName === 'gccdump') {
            const session = this.normalized.findOrCreateSession(component.componentState._editorid);
            const compiler = session.findOrCreateCompiler(component.componentState._compilerid);

            compiler.specialoutputs.push('gccdump');
        } else if (component.componentName === 'output') {
            const session = this.normalized.findOrCreateSession(component.componentState.editor);
            const compiler = session.findOrCreateCompiler(component.componentState.compiler);

            compiler.specialoutputs.push('compilerOutput');
        } else if (component.componentName === 'conformance') {
            const session = this.normalized.findOrCreateSession(component.componentState.editorid);
            session.conformanceview = new ClientState.ConformanceView(component.componentState);
        } else if (component.componentName === 'tool') {
            const session = this.normalized.findOrCreateSession(component.componentState.editor);
            const compiler = session.findOrCreateCompiler(component.componentState.compiler);

            compiler.tools.push({
                id: component.componentState.toolId,
                args: component.componentState.args,
            });
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

class GoldenLayoutComponents {
    createSourceComponent(session, customSessionId) {
        return {
            type: 'component',
            componentName: 'codeEditor',
            componentState: {
                id: customSessionId ? customSessionId : session.id,
                source: session.source,
                lang: session.language,
            },
            isClosable: true,
            reorderEnabled: true,
        };
    }

    createAstComponent(session, compilerIndex, customSessionId) {
        return {
            type: 'component',
            componentName: 'ast',
            componentState: {
                id: compilerIndex,
                editorid: customSessionId ? customSessionId : session.id,
            },
            isClosable: true,
            reorderEnabled: true,
        };
    }

    createOptComponent(session, compilerIndex, customSessionId) {
        return {
            type: 'component',
            componentName: 'opt',
            componentState: {
                id: compilerIndex,
                editorid: customSessionId ? customSessionId : session.id,
            },
            isClosable: true,
            reorderEnabled: true,
        };
    }

    createCfgComponent(session, compilerIndex, customSessionId) {
        return {
            type: 'component',
            componentName: 'opt',
            componentState: {
                id: compilerIndex,
                editorid: customSessionId ? customSessionId : session.id,
                options: {
                    navigation: false,
                    physics: false,
                },
            },
            isClosable: true,
            reorderEnabled: true,
        };
    }

    createGccDumpComponent(session, compilerIndex, customSessionId) {
        return {
            type: 'component',
            componentName: 'gccdump',
            componentState: {
                _compilerid: compilerIndex,
                _editorid: customSessionId ? customSessionId : session.id,
            },
            isClosable: true,
            reorderEnabled: true,
        };
    }

    createCompilerOutComponent(session, compilerIndex, customSessionId) {
        return {
            type: 'component',
            componentName: 'output',
            componentState: {
                compiler: compilerIndex,
                editor: customSessionId ? customSessionId : session.id,
                wrap: false,
                fontScale: 14,
            },
            isClosable: true,
            reorderEnabled: true,
        };
    }

    createToolComponent(session, compilerIndex, toolId, args, customSessionId) {
        return {
            type: 'component',
            componentName: 'tool',
            componentState: {
                editor: customSessionId ? customSessionId : session.id,
                compiler: compilerIndex,
                toolId: toolId,
                args: args,
            },
            isClosable: true,
            reorderEnabled: true,
        };
    }

    createConformanceViewComponent(session, conformanceview, customSessionId) {
        return {
            type: 'component',
            componentName: 'conformance',
            componentState: {
                editorid: customSessionId ? customSessionId : session.id,
                langId: session.language,
                compilers: [],
                libs: conformanceview.libs,
            },
            isClosable: true,
            reorderEnabled: true,
        };
    }

    createCompilerComponent(session, compiler, customSessionId) {
        return {
            type: 'component',
            componentName: 'compiler',
            componentState: {
                compiler: compiler.id,
                source: customSessionId ? customSessionId : session.id,
                options: compiler.options,
                filters: {
                    binary: compiler.filters.binary,
                    execute: compiler.filters.execute,
                    labels: compiler.filters.labels,
                    directives: compiler.filters.directives,
                    commentOnly: compiler.filters.commentOnly,
                    trim: compiler.filters.trim,
                    intel: compiler.filters.intel,
                    demangle: compiler.filters.demangle,
                },
                libs: compiler.libs,
                lang: session.language,
            },
            isClosable: true,
            reorderEnabled: true,
        };
    }

    createExecutorComponent(session, executor, customSessionId) {
        return {
            type: 'component',
            componentName: 'executor',
            componentState: {
                compiler: executor.compiler.id,
                source: customSessionId ? customSessionId : session.id,
                options: executor.compiler.options,
                execArgs: executor.arguments,
                execStdin: executor.stdin,
                libs: executor.compiler.libs,
                lang: session.language,
                compilationPanelShown: executor.compilerVisible,
                compilerOutShown: executor.compilerOutputVisible,
                argsPanelShown: executor.argumentsVisible,
                stdinPanelShown: executor.stdinVisible,
            },
            isClosable: true,
            reorderEnabled: true,
        };
    }

    createDiffComponent(left, right) {
        return {
            type: 'component',
            componentName: 'diff',
            componentState:
            {
                lhs: left,
                rhs: right,
                lhsdifftype: 0,
                rhsdifftype: 0,
                fontScale: 14,
            },
        };
    }

    createSpecialOutputComponent(viewtype, session, idxCompiler, customSessionId) {
        let component = null;
        if (viewtype === 'ast') {
            component = this.createAstComponent(session, idxCompiler + 1, customSessionId);
        } else if (viewtype === 'opt') {
            component = this.createOptComponent(session, idxCompiler + 1, customSessionId);
        } else if (viewtype === 'cfg') {
            component = this.createCfgComponent(session, idxCompiler + 1, customSessionId);
        } else if (viewtype === 'gccdump') {
            component = this.createGccDumpComponent(session, idxCompiler + 1, customSessionId);
        } else if (viewtype === 'compilerOutput') {
            component = this.createCompilerOutComponent(session, idxCompiler + 1, customSessionId);
        }

        return component;
    }
}

class ClientStateGoldenifier extends GoldenLayoutComponents {
    constructor() {
        super();
        this.golden = {};
    }

    newStackWithOneComponent(width, component) {
        return {
            type: 'stack',
            width: width,
            content: [component],
        };
    }

    newSourceStackFromSession(session, width) {
        return this.newStackWithOneComponent(width,
            this.createSourceComponent(session),
        );
    }

    newAstStackFromCompiler(session, compilerIndex, width) {
        return this.newStackWithOneComponent(width,
            this.createAstComponent(session, compilerIndex),
        );
    }

    newOptStackFromCompiler(session, compilerIndex, width) {
        return this.newStackWithOneComponent(width,
            this.createOptComponent(session, compilerIndex),
        );
    }

    newCfgStackFromCompiler(session, compilerIndex, width) {
        return this.newStackWithOneComponent(width,
            this.createCfgComponent(session, compilerIndex),
        );
    }

    newGccDumpStackFromCompiler(session, compilerIndex, width) {
        return this.newStackWithOneComponent(width,
            this.createGccDumpComponent(session, compilerIndex),
        );
    }

    newCompilerOutStackFromCompiler(session, compilerIndex, width) {
        return this.newStackWithOneComponent(width,
            this.createCompilerOutComponent(session, compilerIndex),
        );
    }

    newToolStackFromCompiler(session, compilerIndex, toolId, args, width) {
        return this.newStackWithOneComponent(width,
            this.createToolComponent(session, compilerIndex, toolId, args),
        );
    }

    newConformanceViewStack(session, width, conformanceview) {
        const stack = this.newStackWithOneComponent(width,
            this.createConformanceViewComponent(session, conformanceview),
        );

        conformanceview.compilers.forEach((compiler) => {
            const compjson = {
                compilerId: compiler.id,
                options: compiler.options,
            };

            stack.content[0].componentState.compilers.push(compjson);
        });

        return stack;
    }

    newCompilerStackFromSession(session, compiler, width) {
        return this.newStackWithOneComponent(width,
            this.createCompilerComponent(session, compiler),
        );
    }

    newExecutorStackFromSession(session, executor, width) {
        return this.newStackWithOneComponent(width,
            this.createExecutorComponent(session, executor),
        );
    }

    createSourceContentArray(state, left, right) {
        if (left.session === right.session) {
            return [this.createPresentationModeComponents(state.sessions[left.session], 1, 100)];
        } else {
            return [
                this.createPresentationModeComponents(state.sessions[left.session], 1),
                this.createPresentationModeComponents(state.sessions[right.session], 2),
            ];
        }
    }

    getPresentationModeEmptyLayout() {
        return {
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
                showCloseIcon: false,
                responsiveMode: 'onload',
                tabOverlapAllowance: 0,
                reorderOnTabMenuClick: true,
                tabControlOffset: 10,
            },
            dimensions:
            {
                borderWidth: 5,
                borderGrabWidth: 15,
                minItemHeight: 10,
                minItemWidth: 10,
                headerHeight: 20,
                dragProxyWidth: 300,
                dragProxyHeight: 200,
            },
            labels:
            {
                close: 'close',
                maximise: 'maximise',
                minimise: 'minimise',
                popout: 'open in new window',
                popin: 'pop in',
                tabDropdown: 'additional tabs',
            },
            content: [
                {
                    type: 'column',
                    content: [],
                },
            ],
        };
    }

    getPresentationModeLayoutForSource(state, left) {
        const gl = this.getPresentationModeEmptyLayout();
        gl.content[0].content = [
            {
                type: 'column',
                width: 100,
                content: this.createSourceContentArray(state, left, left),
            }];

        return gl;
    }

    getPresentationModeLayoutForComparisonSlide(state, left, right) {
        const gl = this.getPresentationModeEmptyLayout();
        gl.content[0].content = [
            {
                type: 'row',
                height: 50,
                content: this.createSourceContentArray(state, left, right),
            },
            {
                type: 'row',
                height: 50,
                content: [
                    {
                        type: 'stack',
                        width: 100,
                        content: [
                            this.createDiffComponent(left.compiler + 1, right.compiler + 1),
                        ],
                    },
                ],
            },
        ];

        return gl;
    }

    createPresentationModeComponents(session, customSessionId, customWidth) {
        const stack = {
            type: 'stack',
            width: customWidth ? customWidth : 50,
            activeItemIndex: 0,
            content: [
                this.createSourceComponent(session, customSessionId),
            ],
        };

        for (let idxCompiler = 0; idxCompiler < session.compilers.length; idxCompiler++) {
            const compiler = session.compilers[idxCompiler];
            stack.content.push(
                this.createCompilerComponent(session, compiler, customSessionId),
            );

            compiler.specialoutputs.forEach((viewtype) => {
                stack.content.push(
                    this.createSpecialOutputComponent(viewtype, session, idxCompiler, customSessionId),
                );
            });

            compiler.tools.forEach((tool) => {
                stack.content.push(
                    this.createToolComponent(session, idxCompiler + 1, tool.id, tool.args, customSessionId),
                );
            });
        }

        for (let idxExecutor = 0; idxExecutor < session.executors.length; idxExecutor++) {
            stack.content.push(
                this.createExecutorComponent(session, session.executors[idxExecutor], customSessionId),
            );
        }

        return stack;
    }

    generatePresentationModeMobileViewerSlides(state) {
        const slides = [];

        for (var idxSession = 0; idxSession < state.sessions.length; idxSession++) {
            const gl = this.getPresentationModeLayoutForSource(state, {
                session: idxSession,
                compiler: 0,
            });
            slides.push(gl);
        }

        return slides;
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
                responsiveMode: 'onload',
                tabOverlapAllowance: 0,
                reorderOnTabMenuClick: true,
                tabControlOffset: 10,
            },
            dimensions: {
                borderWidth: 5,
                borderGrabWidth: 15,
                minItemHeight: 10,
                minItemWidth: 10,
                headerHeight: 20,
                dragProxyWidth: 300,
                dragProxyHeight: 200,
            },
            labels: {
                close: 'close',
                maximise: 'maximise',
                minimise: 'minimise',
                popout: 'open in new window',
                popin: 'pop in',
                tabDropdown: 'additional tabs',
            },
            content: [
                {
                    type: 'row',
                    content: [
                    ],
                },
            ],
        };

        if (state.sessions.length > 1) {
            const sessionWidth = 100 / state.sessions.length;

            for (let idxSession = 0; idxSession < state.sessions.length; idxSession++) {
                const session = state.sessions[idxSession];

                this.golden.content[0].content[idxSession] = {
                    type: 'column',
                    isClosable: true,
                    reorderEnabled: true,
                    width: sessionWidth,
                    content: [
                        {
                            type: 'row',
                            content: [
                            ],
                        },
                        {
                            type: 'row',
                            content: [
                            ],
                        },
                    ],
                };

                let stack = this.newSourceStackFromSession(session, 100);
                this.golden.content[0].content[idxSession].content[0].content.push(stack);

                const compilerWidth = 100 /
                    (1 + session.compilers.length + session.executors.length +
                        session.countNumberOfSpecialOutputsAndTools());

                if (session.conformanceview) {
                    const stack = this.newConformanceViewStack(session, compilerWidth, session.conformanceview);
                    this.golden.content[0].content[idxSession].content[1].content.push(stack);
                }

                for (let idxCompiler = 0; idxCompiler < session.compilers.length; idxCompiler++) {
                    const compiler = session.compilers[idxCompiler];
                    let stack = this.newCompilerStackFromSession(session, compiler, compilerWidth);
                    this.golden.content[0].content[idxSession].content[1].content.push(stack);

                    compiler.specialoutputs.forEach((viewtype) => {
                        let stack = this.newStackWithOneComponent(compilerWidth,
                            this.createSpecialOutputComponent(viewtype, session, idxCompiler));

                        if (stack) {
                            this.golden.content[0].content[idxSession].content[1].content.push(stack);
                        }
                    });

                    compiler.tools.forEach((tool) => {
                        let stack = this.newToolStackFromCompiler(session, idxCompiler + 1,
                            tool.id, tool.args, compilerWidth);
                        this.golden.content[0].content[idxSession].content[1].content.push(stack);
                    });
                }

                for (let idxExecutor = 0; idxExecutor < session.executors.length; idxExecutor++) {
                    const executor = session.executors[idxExecutor];
                    let stack = this.newExecutorStackFromSession(session, executor, compilerWidth);
                    this.golden.content[0].content[idxSession].content[1].content.push(stack);
                }
            }
        } else if (state.sessions.length === 1) {
            const session = state.sessions[0];
            this.golden.content[0] = {
                type: 'row',
                content: [
                ],
            };

            const width = 100 / (1 + session.compilers.length + session.executors.length +
                session.countNumberOfSpecialOutputsAndTools());
            this.golden.content[0].content.push(this.newSourceStackFromSession(session, width));

            if (session.conformanceview) {
                const stack = this.newConformanceViewStack(session, width, session.conformanceview);
                this.golden.content[0].content.push(stack);
            }

            for (let idxCompiler = 0; idxCompiler < session.compilers.length; idxCompiler++) {
                const compiler = session.compilers[idxCompiler];
                let stack = this.newCompilerStackFromSession(session, compiler, width);
                this.golden.content[0].content.push(stack);

                compiler.specialoutputs.forEach((viewtype) => {
                    let stack = this.newStackWithOneComponent(width,
                        this.createSpecialOutputComponent(viewtype, session, idxCompiler));

                    if (stack) {
                        this.golden.content[0].content.push(stack);
                    }
                });

                compiler.tools.forEach((tool) => {
                    let stack = this.newToolStackFromCompiler(session, compiler, idxCompiler + 1,
                        tool.id, tool.args, width);
                    this.golden.content[0].content.push(stack);
                });
            }

            session.executors.forEach((executor) => {
                let stack = this.newExecutorStackFromSession(session, executor, width);
                this.golden.content[0].content.push(stack);
            });
        }
    }
}

module.exports = {
    ClientStateNormalizer: ClientStateNormalizer,
    ClientStateGoldenifier: ClientStateGoldenifier,
};
