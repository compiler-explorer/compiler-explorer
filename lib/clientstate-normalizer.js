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
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

import {ClientState, ClientStateCompiler, ClientStateConformanceView, ClientStateExecutor} from './clientstate';

export class ClientStateNormalizer {
    constructor() {
        this.normalized = new ClientState();
        this.rootContent = null;
    }

    fromGoldenLayoutContent(content) {
        if (!this.rootContent) this.rootContent = content;

        for (const component of content) {
            this.fromGoldenLayoutComponent(component);
        }
    }

    setFilterSettingsFromComponent(compiler, component) {
        compiler.filters.binary = component.componentState.filters.binary;
        compiler.filters.execute = component.componentState.filters.execute;
        compiler.filters.labels = component.componentState.filters.labels;
        compiler.filters.directives = component.componentState.filters.directives;
        compiler.filters.commentOnly = component.componentState.filters.commentOnly;
        compiler.filters.trim = component.componentState.filters.trim;
        compiler.filters.intel = component.componentState.filters.intel;
        compiler.filters.demangle = component.componentState.filters.demangle;
    }

    findCompilerInGoldenLayout(content, id) {
        let result;

        for (const component of content) {
            if (component.componentName === 'compiler') {
                if (component.componentState.id === id) {
                    return component;
                }
            } else if (component.content && component.content.length > 0) {
                result = this.findCompilerInGoldenLayout(component.content, id);
                if (result) break;
            }
        }

        return result;
    }

    findOrCreateSessionFromEditorOrCompiler(editorId, compilerId) {
        let session;
        if (editorId) {
            session = this.normalized.findOrCreateSession(editorId);
        } else {
            const glCompiler = this.findCompilerInGoldenLayout(this.rootContent, compilerId);
            if (glCompiler) {
                if (glCompiler.componentState.source) {
                    session = this.normalized.findOrCreateSession(glCompiler.componentState.source);
                }
            }
        }
        return session;
    }

    addSpecialOutputToCompiler(compilerId, name, editorId) {
        const glCompiler = this.findCompilerInGoldenLayout(this.rootContent, compilerId);
        if (glCompiler) {
            let compiler;
            if (glCompiler.componentState.source) {
                const session = this.normalized.findOrCreateSession(glCompiler.componentState.source);
                compiler = session.findOrCreateCompiler(compilerId);
            } else if (glCompiler.componentState.tree) {
                const tree = this.normalized.findOrCreateTree(glCompiler.componentState.tree);
                compiler = tree.findOrCreateCompiler(compilerId);
            }

            compiler.specialoutputs.push(name);
        } else if (editorId) {
            const session = this.normalized.findOrCreateSession(editorId);
            const compiler = session.findOrCreateCompiler(compilerId);
            compiler.specialoutputs.push(name);
        }
    }

    addToolToCompiler(compilerId, editorId, toolId, args, stdin) {
        const glCompiler = this.findCompilerInGoldenLayout(this.rootContent, compilerId);
        if (glCompiler) {
            let compiler;
            if (glCompiler.componentState.source) {
                const session = this.normalized.findOrCreateSession(glCompiler.componentState.source);
                compiler = session.findOrCreateCompiler(compilerId);
            } else if (glCompiler.componentState.tree) {
                const tree = this.normalized.findOrCreateTree(glCompiler.componentState.tree);
                compiler = tree.findOrCreateCompiler(compilerId);
            }

            compiler.tools.push({
                id: toolId,
                args: args,
                stdin: stdin,
            });
        }
    }

    fromGoldenLayoutComponent(component) {
        if (component.componentName === 'tree') {
            const tree = this.normalized.findOrCreateTree(component.componentState.id);
            tree.fromJsonData(component.componentState);
        } else if (component.componentName === 'codeEditor') {
            const session = this.normalized.findOrCreateSession(component.componentState.id);
            session.language = component.componentState.lang;
            session.source = component.componentState.source;
            if (component.componentState.filename) session.filename = component.componentState.filename;
        } else if (component.componentName === 'compiler') {
            let compiler;
            if (component.componentState.id) {
                if (component.componentState.source) {
                    const session = this.normalized.findOrCreateSession(component.componentState.source);
                    compiler = session.findOrCreateCompiler(component.componentState.id);
                } else if (component.componentState.tree) {
                    const tree = this.normalized.findOrCreateTree(component.componentState.tree);
                    compiler = tree.findOrCreateCompiler(component.componentState.id);
                } else {
                    return;
                }
            } else {
                compiler = new ClientStateCompiler();

                if (component.componentState.source) {
                    const session = this.normalized.findOrCreateSession(component.componentState.source);
                    session.compilers.push(compiler);

                    this.normalized.numberCompilersIfNeeded(session, this.normalized.getNextCompilerId());
                } else if (component.componentState.tree) {
                    const tree = this.normalized.findOrCreateTree(component.componentState.tree);
                    tree.compilers.push(compiler);
                } else {
                    return;
                }
            }

            compiler.id = component.componentState.compiler;
            compiler.options = component.componentState.options;
            compiler.libs = component.componentState.libs;
            this.setFilterSettingsFromComponent(compiler, component);
        } else if (component.componentName === 'executor') {
            const executor = new ClientStateExecutor();
            executor.compiler.id = component.componentState.compiler;
            executor.compiler.options = component.componentState.options;
            executor.compiler.libs = component.componentState.libs;
            executor.compilerVisible = component.componentState.compilationPanelShown;
            executor.compilerOutputVisible = component.componentState.compilerOutShown;
            executor.arguments = component.componentState.execArgs;
            executor.argumentsVisible = component.componentState.argsPanelShown;
            executor.stdin = component.componentState.execStdin;
            executor.stdinVisible = component.componentState.stdinPanelShown;
            if (component.componentState.wrap) executor.wrap = true;

            if (component.componentState.source) {
                const session = this.normalized.findOrCreateSession(component.componentState.source);

                session.executors.push(executor);
            } else if (component.componentState.tree) {
                const tree = this.normalized.findOrCreateTree(component.componentState.tree);

                tree.executors.push(executor);
            }
        } else if (component.componentName === 'ast') {
            this.addSpecialOutputToCompiler(component.componentState.id, 'ast', component.componentState.editorid);
        } else if (component.componentName === 'opt') {
            this.addSpecialOutputToCompiler(component.componentState.id, 'opt', component.componentState.editorid);
        } else if (component.componentName === 'cfg') {
            this.addSpecialOutputToCompiler(component.componentState.id, 'cfg', component.componentState.editorid);
        } else if (component.componentName === 'gccdump') {
            this.addSpecialOutputToCompiler(
                component.componentState._compilerid,
                'gccdump',
                component.componentState._editorid
            );
        } else if (component.componentName === 'output') {
            this.addSpecialOutputToCompiler(
                component.componentState.compiler,
                'compilerOutput',
                component.componentState.editor
            );
        } else if (component.componentName === 'conformance') {
            const session = this.normalized.findOrCreateSession(component.componentState.editorid);
            session.conformanceview = new ClientStateConformanceView(component.componentState);
        } else if (component.componentName === 'tool') {
            this.addToolToCompiler(
                component.componentState.compiler,
                component.componentState.editor,
                component.componentState.toolId,
                component.componentState.args,
                component.componentState.stdin
            );
        } else if (component.content) {
            this.fromGoldenLayoutContent(component.content);
        }

        this.fixFilesInTrees();
    }

    fixFilesInTrees() {
        for (const tree of this.normalized.trees) {
            tree.files = tree.files.filter(file => file.isIncluded);

            for (const file of tree.files) {
                if (file.editorId) {
                    const session = this.normalized.findSessionById(file.editorId);
                    if (session) {
                        file.content = session.source;
                        file.filename = session.filename;
                    }
                }
            }
        }
    }

    fromGoldenLayout(globj) {
        this.rootContent = globj.content;

        if (globj.content) {
            this.fromGoldenLayoutContent(globj.content);
        }
    }
}

class GoldenLayoutComponents {
    createSourceComponent(session, customSessionId) {
        const editor = {
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

        if (session.filename) {
            editor.title = session.filename;
            editor.componentState.filename = session.filename;
        }

        return editor;
    }

    createTreeComponent(tree, customTreeId) {
        const treeComponent = {
            type: 'component',
            componentName: 'tree',
            componentState: {
                id: tree.id,
                cmakeArgs: tree.cmakeArgs,
                customOutputFilename: tree.customOutputFilename,
                isCMakeProject: tree.isCMakeProject,
                compilerLanguageId: tree.compilerLanguageId,
                files: tree.files,
                newFileId: tree.newFileId,
            },
            isClosable: true,
            reorderEnabled: true,
        };

        if (customTreeId) {
            treeComponent.componentState.id = customTreeId;
        }

        return treeComponent;
    }

    createAstComponent(session, compilerIndex, customSessionId) {
        return {
            type: 'component',
            componentName: 'ast',
            componentState: {
                id: compilerIndex,
                editorid: customSessionId ? customSessionId : session ? session.id : undefined,
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
                editorid: customSessionId ? customSessionId : session ? session.id : undefined,
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
                editorid: customSessionId ? customSessionId : session ? session.id : undefined,
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
                _editorid: customSessionId ? customSessionId : session ? session.id : undefined,
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
                editor: customSessionId ? customSessionId : session ? session.id : undefined,
                wrap: false,
                fontScale: 14,
            },
            isClosable: true,
            reorderEnabled: true,
        };
    }

    createToolComponent(session, compilerIndex, toolId, args, stdin, customSessionId) {
        return {
            type: 'component',
            componentName: 'tool',
            componentState: {
                editor: customSessionId ? customSessionId : session ? session.id : undefined,
                compiler: compilerIndex,
                toolId: toolId,
                args: args,
                stdin: stdin,
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

    copyCompilerFilters(filters) {
        return {...filters};
    }

    createCompilerComponent(session, compiler, customSessionId, idxCompiler) {
        return {
            type: 'component',
            componentName: 'compiler',
            componentState: {
                id: idxCompiler,
                compiler: compiler.id,
                source: customSessionId ? customSessionId : session.id,
                options: compiler.options,
                filters: this.copyCompilerFilters(compiler.filters),
                libs: compiler.libs,
                lang: session.language,
            },
            isClosable: true,
            reorderEnabled: true,
        };
    }

    createCompilerComponentForTree(tree, compiler, customTreeId, idxCompiler) {
        return {
            type: 'component',
            componentName: 'compiler',
            componentState: {
                id: idxCompiler,
                compiler: compiler.id,
                source: undefined,
                tree: customTreeId ? customTreeId : tree.id,
                options: compiler.options,
                filters: this.copyCompilerFilters(compiler.filters),
                libs: compiler.libs,
                lang: tree.compilerLanguageId,
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
                wrap: executor.wrap,
            },
            isClosable: true,
            reorderEnabled: true,
        };
    }

    createExecutorComponentForTree(tree, executor, customTreeId) {
        return {
            type: 'component',
            componentName: 'executor',
            componentState: {
                compiler: executor.compiler.id,
                source: undefined,
                tree: customTreeId ? customTreeId : tree.id,
                options: executor.compiler.options,
                execArgs: executor.arguments,
                execStdin: executor.stdin,
                libs: executor.compiler.libs,
                lang: tree.compilerLanguageId,
                compilationPanelShown: executor.compilerVisible,
                compilerOutShown: executor.compilerOutputVisible,
                argsPanelShown: executor.argumentsVisible,
                stdinPanelShown: executor.stdinVisible,
                wrap: executor.wrap,
            },
            isClosable: true,
            reorderEnabled: true,
        };
    }

    createDiffComponent(left, right) {
        return {
            type: 'component',
            componentName: 'diff',
            componentState: {
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

    createSpecialOutputComponentForTreeCompiler(viewtype, idxCompiler) {
        let component = null;
        if (viewtype === 'ast') {
            component = this.createAstComponent(null, idxCompiler + 1, false);
        } else if (viewtype === 'opt') {
            component = this.createOptComponent(null, idxCompiler + 1, false);
        } else if (viewtype === 'cfg') {
            component = this.createCfgComponent(null, idxCompiler + 1, false);
        } else if (viewtype === 'gccdump') {
            component = this.createGccDumpComponent(null, idxCompiler + 1, false);
        } else if (viewtype === 'compilerOutput') {
            component = this.createCompilerOutComponent(null, idxCompiler + 1, false);
        }

        return component;
    }

    createToolComponentForTreeCompiler(tree, compilerIndex, toolId, args, stdin, customTreeId) {
        return {
            type: 'component',
            componentName: 'tool',
            componentState: {
                tree: customTreeId ? customTreeId : tree ? tree.id : undefined,
                compiler: compilerIndex,
                toolId: toolId,
                args: args,
                stdin: stdin,
            },
            isClosable: true,
            reorderEnabled: true,
        };
    }
}

export class ClientStateGoldenifier extends GoldenLayoutComponents {
    constructor() {
        super();
        this.golden = {};
    }

    newEmptyStack(width) {
        return {
            type: 'stack',
            width: width,
            content: [],
        };
    }

    newStackWithOneComponent(width, component) {
        return {
            type: 'stack',
            width: width,
            content: [component],
        };
    }

    newTreeFromTree(tree, width) {
        return this.newStackWithOneComponent(width, this.createTreeComponent(tree));
    }

    newSourceStackFromSession(session, width) {
        return this.newStackWithOneComponent(width, this.createSourceComponent(session));
    }

    newAstStackFromCompiler(session, compilerIndex, width) {
        return this.newStackWithOneComponent(width, this.createAstComponent(session, compilerIndex));
    }

    newOptStackFromCompiler(session, compilerIndex, width) {
        return this.newStackWithOneComponent(width, this.createOptComponent(session, compilerIndex));
    }

    newCfgStackFromCompiler(session, compilerIndex, width) {
        return this.newStackWithOneComponent(width, this.createCfgComponent(session, compilerIndex));
    }

    newGccDumpStackFromCompiler(session, compilerIndex, width) {
        return this.newStackWithOneComponent(width, this.createGccDumpComponent(session, compilerIndex));
    }

    newCompilerOutStackFromCompiler(session, compilerIndex, width) {
        return this.newStackWithOneComponent(width, this.createCompilerOutComponent(session, compilerIndex));
    }

    newToolStackFromCompiler(session, compilerIndex, toolId, args, stdin, width) {
        return this.newStackWithOneComponent(
            width,
            this.createToolComponent(session, compilerIndex, toolId, args, stdin)
        );
    }

    newConformanceViewStack(session, width, conformanceview) {
        const stack = this.newStackWithOneComponent(
            width,
            this.createConformanceViewComponent(session, conformanceview)
        );

        for (const compiler of conformanceview.compilers) {
            const compjson = {
                compilerId: compiler.id,
                options: compiler.options,
            };

            stack.content[0].componentState.compilers.push(compjson);
        }

        return stack;
    }

    newCompilerStackFromSession(session, compiler, width) {
        return this.newStackWithOneComponent(
            width,
            this.createCompilerComponent(session, compiler, false, compiler._internalId)
        );
    }

    newExecutorStackFromSession(session, executor, width) {
        return this.newStackWithOneComponent(width, this.createExecutorComponent(session, executor));
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
            },
        ];

        return gl;
    }

    closeAllEditors(tree) {
        for (const file of tree.files) {
            file.editorId = -1;
            file.isOpen = false;
        }
    }

    getPresentationModeLayoutForTree(state, left) {
        const gl = this.getPresentationModeEmptyLayout();
        const tree = state.trees[left.tree];
        const customTreeId = 1;

        this.closeAllEditors(tree);

        gl.content[0].content = [
            {
                type: 'column',
                width: 100,
                content: [
                    {
                        type: 'row',
                        height: 50,
                        content: [this.createTreeComponent(tree, customTreeId)],
                    },
                    {
                        type: 'row',
                        height: 50,
                        content: [
                            {
                                type: 'stack',
                                width: 100,
                                content: [],
                            },
                        ],
                    },
                ],
            },
        ];

        let stack = gl.content[0].content[0].content[1].content[0];

        for (let idxCompiler = 0; idxCompiler < tree.compilers.length; idxCompiler++) {
            const compiler = tree.compilers[idxCompiler];
            stack.content.push(this.createCompilerComponentForTree(tree, compiler, customTreeId, idxCompiler + 1));

            for (const viewtype of compiler.specialoutputs) {
                stack.content.push(this.createSpecialOutputComponentForTreeCompiler(viewtype, idxCompiler));
            }

            for (const tool of compiler.tools) {
                stack.content.push(
                    this.createToolComponentForTreeCompiler(tree, idxCompiler + 1, tool.id, tool.args, tool.stdin)
                );
            }
        }

        for (let idxExecutor = 0; idxExecutor < tree.executors.length; idxExecutor++) {
            stack.content.push(this.createExecutorComponentForTree(tree, tree.executors[idxExecutor], customTreeId));
        }

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
                        content: [this.createDiffComponent(left.compiler + 1, right.compiler + 1)],
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
            content: [this.createSourceComponent(session, customSessionId)],
        };

        for (let idxCompiler = 0; idxCompiler < session.compilers.length; idxCompiler++) {
            const compiler = session.compilers[idxCompiler];
            stack.content.push(this.createCompilerComponent(session, compiler, customSessionId));

            for (const viewtype of compiler.specialoutputs) {
                stack.content.push(this.createSpecialOutputComponent(viewtype, session, idxCompiler, customSessionId));
            }

            for (const tool of compiler.tools) {
                stack.content.push(
                    this.createToolComponent(false, idxCompiler + 1, tool.id, tool.args, tool.stdin, customSessionId)
                );
            }
        }

        for (let idxExecutor = 0; idxExecutor < session.executors.length; idxExecutor++) {
            stack.content.push(this.createExecutorComponent(session, session.executors[idxExecutor], customSessionId));
        }

        return stack;
    }

    generatePresentationModeMobileViewerSlides(state) {
        const slides = [];

        if (state.trees.length > 0) {
            for (var idxTree = 0; idxTree < state.trees.length; idxTree++) {
                const gl = this.getPresentationModeLayoutForTree(state, {
                    tree: idxTree,
                });
                slides.push(gl);
            }

            return slides;
        }

        for (var idxSession = 0; idxSession < state.sessions.length; idxSession++) {
            const gl = this.getPresentationModeLayoutForSource(state, {
                session: idxSession,
                compiler: 0,
            });
            slides.push(gl);
        }

        return slides;
    }

    treeLayoutFromClientstate(state) {
        const firstTree = state.trees[0];
        const leftSide = this.newTreeFromTree(firstTree, 25);
        const middle = this.newEmptyStack(40);
        for (const session of state.sessions) {
            middle.content.push(this.createSourceComponent(session));
        }

        const rightSide = this.newEmptyStack(40);
        let idxCompiler = 0;
        for (const compiler of firstTree.compilers) {
            rightSide.content.push(this.createCompilerComponentForTree(firstTree, compiler));

            for (const specialOutput of compiler.specialoutputs) {
                rightSide.content.push(this.createSpecialOutputComponentForTreeCompiler(specialOutput, idxCompiler));
            }

            idxCompiler++;
        }

        this.golden.content[0].content.push(leftSide, middle, rightSide);
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
                    content: [],
                },
            ],
        };

        if (state.trees.length > 0) {
            this.treeLayoutFromClientstate(state);
            return;
        }

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
                            content: [],
                        },
                        {
                            type: 'row',
                            content: [],
                        },
                    ],
                };

                let stack = this.newSourceStackFromSession(session, 100);
                this.golden.content[0].content[idxSession].content[0].content.push(stack);

                const compilerWidth =
                    100 /
                    (1 +
                        session.compilers.length +
                        session.executors.length +
                        session.countNumberOfSpecialOutputsAndTools());

                if (session.conformanceview) {
                    const stack = this.newConformanceViewStack(session, compilerWidth, session.conformanceview);
                    this.golden.content[0].content[idxSession].content[1].content.push(stack);
                }

                for (let idxCompiler = 0; idxCompiler < session.compilers.length; idxCompiler++) {
                    const compiler = session.compilers[idxCompiler];
                    let stack = this.newCompilerStackFromSession(session, compiler, compilerWidth);
                    this.golden.content[0].content[idxSession].content[1].content.push(stack);

                    for (const viewtype of compiler.specialoutputs) {
                        let stack = this.newStackWithOneComponent(
                            compilerWidth,
                            this.createSpecialOutputComponent(viewtype, session, idxCompiler)
                        );

                        if (stack) {
                            this.golden.content[0].content[idxSession].content[1].content.push(stack);
                        }
                    }

                    for (const tool of compiler.tools) {
                        let stack = this.newToolStackFromCompiler(
                            session,
                            idxCompiler + 1,
                            tool.id,
                            tool.args,
                            tool.stdin,
                            compilerWidth
                        );
                        this.golden.content[0].content[idxSession].content[1].content.push(stack);
                    }
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
                content: [],
            };

            const width =
                100 /
                (1 +
                    session.compilers.length +
                    session.executors.length +
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

                for (const viewtype of compiler.specialoutputs) {
                    let stack = this.newStackWithOneComponent(
                        width,
                        this.createSpecialOutputComponent(viewtype, session, idxCompiler)
                    );

                    if (stack) {
                        this.golden.content[0].content.push(stack);
                    }
                }

                for (const tool of compiler.tools) {
                    let stack = this.newToolStackFromCompiler(
                        session,
                        compiler,
                        idxCompiler + 1,
                        tool.id,
                        tool.args,
                        tool.stdin,
                        width
                    );
                    this.golden.content[0].content.push(stack);
                }
            }

            for (const executor of session.executors) {
                let stack = this.newExecutorStackFromSession(session, executor, width);
                this.golden.content[0].content.push(stack);
            }
        }
    }
}
