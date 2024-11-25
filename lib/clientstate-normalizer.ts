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

import {assert} from './assert.js';
import {
    ClientState,
    ClientStateCompiler,
    ClientStateConformanceView,
    ClientStateExecutor,
    ClientStateSession,
    ClientStateTree,
} from './clientstate.js';

type BasicGoldenLayoutStruct = {
    type: string;
    width?: number;
    height?: number;
    isClosable?: boolean;
    reorderEnabled?: boolean;
    content: Array<BasicGoldenLayoutStruct | GoldenLayoutComponentStruct>;
};

type GoldenLayoutComponentStruct = {
    type: string;
    title?: string;
    componentName: string;
    componentState: any;
    isClosable: boolean;
    reorderEnabled: boolean;
};

export type GoldenLayoutRootStruct = {
    settings?: any;
    dimensions?: Record<string, number>;
    labels?: Record<string, string>;
    content?: Array<BasicGoldenLayoutStruct>;
};

export class ClientStateNormalizer {
    normalized = new ClientState();
    rootContent: Array<BasicGoldenLayoutStruct> | undefined;

    fromGoldenLayoutContent(content: BasicGoldenLayoutStruct[]) {
        if (!this.rootContent) this.rootContent = content;

        for (const component of content) {
            this.fromGoldenLayoutComponent(component);
        }
    }

    setFilterSettingsFromComponentState(compiler: ClientStateCompiler, componentState) {
        compiler.filters.binary = componentState.filters.binary;
        compiler.filters.binaryObject = componentState.filters.binaryObject;
        compiler.filters.execute = componentState.filters.execute;
        compiler.filters.labels = componentState.filters.labels;
        compiler.filters.libraryCode = componentState.filters.libraryCode;
        compiler.filters.directives = componentState.filters.directives;
        compiler.filters.commentOnly = componentState.filters.commentOnly;
        compiler.filters.trim = componentState.filters.trim;
        compiler.filters.intel = componentState.filters.intel;
        compiler.filters.demangle = componentState.filters.demangle;
        compiler.filters.debugCalls = componentState.filters.debugCalls;
    }

    // UNUSED, not deleting yet
    // setFilterSettingsFromComponent(compiler: ClientStateCompiler, component) {
    //     this.setFilterSettingsFromComponentState(compiler, component.componentState);
    // }

    findCompilerInGoldenLayout(
        content: Array<BasicGoldenLayoutStruct | GoldenLayoutComponentStruct>,
        id: number,
    ): GoldenLayoutComponentStruct | null {
        let result: GoldenLayoutComponentStruct | null = null;

        for (const component of content) {
            if ('componentName' in component && component.componentName === 'compiler') {
                if (component.componentState.id === id) {
                    return component;
                }
            } else if ('content' in component && component.content.length > 0) {
                result = this.findCompilerInGoldenLayout(component.content, id);
                if (result) break;
            }
        }

        return result;
    }

    // UNUSED. not deleting yet
    // findOrCreateSessionFromEditorOrCompiler(editorId, compilerId) {
    //     let session;
    //     if (editorId) {
    //         session = this.normalized.findOrCreateSession(editorId);
    //     } else {
    //         const glCompiler = this.findCompilerInGoldenLayout(this.rootContent, compilerId);
    //         if (glCompiler) {
    //             if (glCompiler.componentState.source) {
    //                 session = this.normalized.findOrCreateSession(glCompiler.componentState.source);
    //             }
    //         }
    //     }
    //     return session;
    // }

    addSpecialOutputToCompiler(compilerId: number, name: string, editorId: number) {
        const glCompiler = this.findCompilerInGoldenLayout(this.rootContent!, compilerId);
        if (glCompiler) {
            let compiler;
            if (glCompiler.componentState.source) {
                const session = this.normalized.findOrCreateSession(glCompiler.componentState.source);
                compiler = session.findOrCreateCompiler(compilerId);
            } else {
                assert(glCompiler.componentState.tree);
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

    addToolToCompiler(compilerId: number, toolId: string, args: string[], stdin: string) {
        const glCompiler = this.findCompilerInGoldenLayout(this.rootContent!, compilerId);
        if (glCompiler) {
            let compiler: ClientStateCompiler;
            if (glCompiler.componentState.source) {
                const session = this.normalized.findOrCreateSession(glCompiler.componentState.source);
                compiler = session.findOrCreateCompiler(compilerId);
            } else {
                assert(glCompiler.componentState.tree);
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

    addExecutorFromComponentState(componentState) {
        const executor = new ClientStateExecutor();
        executor.compiler.id = componentState.compiler;
        executor.compiler.options = componentState.options;
        executor.compiler.libs = componentState.libs;
        executor.compilerVisible = componentState.compilationPanelShown;
        executor.compilerOutputVisible = componentState.compilerOutShown;
        executor.arguments = componentState.execArgs;
        executor.argumentsVisible = componentState.argsPanelShown;
        executor.stdin = componentState.execStdin;
        executor.stdinVisible = componentState.stdinPanelShown;
        if (componentState.overrides) {
            executor.compiler.overrides = componentState.overrides;
        }
        if (componentState.runtimeTools) {
            executor.runtimeTools = componentState.runtimeTools;
        }
        if (componentState.wrap) executor.wrap = true;

        if (componentState.source) {
            const session = this.normalized.findOrCreateSession(componentState.source);

            session.executors.push(executor);
        } else if (componentState.tree) {
            const tree = this.normalized.findOrCreateTree(componentState.tree);

            tree.executors.push(executor);
        }
    }

    addCompilerFromComponentState(componentState: ClientStateCompiler) {
        let compiler;
        if (componentState.id) {
            if (componentState.source) {
                const session = this.normalized.findOrCreateSession(componentState.source);
                compiler = session.findOrCreateCompiler(componentState.id as number);
            } else if (componentState.tree) {
                const tree = this.normalized.findOrCreateTree(componentState.tree);
                compiler = tree.findOrCreateCompiler(componentState.id as number);
            } else {
                return;
            }
        } else {
            // Ofek: do we ever get here?

            compiler = new ClientStateCompiler();
            if (componentState.source) {
                const session = this.normalized.findOrCreateSession(componentState.source);
                session.compilers.push(compiler);

                this.normalized.numberCompilersIfNeeded(session, this.normalized.getNextCompilerId());
            } else if (componentState.tree) {
                const tree = this.normalized.findOrCreateTree(componentState.tree);
                tree.compilers.push(compiler);
            } else {
                return;
            }
        }

        // Ofek: id and compiler get mixed up in Compiler.getCurrentState?
        compiler.id = componentState.compiler;
        compiler.options = componentState.options;
        compiler.libs = componentState.libs;
        if (componentState.overrides) {
            compiler.overrides = componentState.overrides;
        }
        this.setFilterSettingsFromComponentState(compiler, componentState);
    }

    fromGoldenLayoutComponent(component: BasicGoldenLayoutStruct | GoldenLayoutComponentStruct) {
        if ('componentName' in component) {
            // handle GoldenLayoutComponentStruct
            switch (component.componentName) {
                case 'tree': {
                    const tree = this.normalized.findOrCreateTree(component.componentState.id);
                    tree.fromJsonData(component.componentState);
                    break;
                }
                case 'codeEditor': {
                    const session = this.normalized.findOrCreateSession(component.componentState.id);
                    session.language = component.componentState.lang;
                    session.source = component.componentState.source;
                    if (component.componentState.filename) session.filename = component.componentState.filename;
                    break;
                }
                case 'compiler': {
                    this.addCompilerFromComponentState(component.componentState);
                    break;
                }
                case 'executor': {
                    this.addExecutorFromComponentState(component.componentState);
                    break;
                }
                case 'ast': {
                    this.addSpecialOutputToCompiler(
                        component.componentState.id,
                        'ast',
                        component.componentState.editorid,
                    );
                    break;
                }
                case 'opt': {
                    this.addSpecialOutputToCompiler(
                        component.componentState.id,
                        'opt',
                        component.componentState.editorid,
                    );
                    break;
                }
                case 'stackusage': {
                    this.addSpecialOutputToCompiler(
                        component.componentState.id,
                        'stackusage',
                        component.componentState.editorid,
                    );
                    break;
                }
                case 'cfg': {
                    this.addSpecialOutputToCompiler(
                        component.componentState.id,
                        'cfg',
                        component.componentState.editorid,
                    );
                    break;
                }
                case 'gccdump': {
                    this.addSpecialOutputToCompiler(
                        component.componentState._compilerid,
                        'gccdump',
                        component.componentState._editorid,
                    );
                    break;
                }
                case 'output': {
                    this.addSpecialOutputToCompiler(
                        component.componentState.compiler,
                        'compilerOutput',
                        component.componentState.editor,
                    );
                    break;
                }
                case 'conformance': {
                    const session = this.normalized.findOrCreateSession(component.componentState.editorid);
                    session.conformanceview = new ClientStateConformanceView(component.componentState);
                    break;
                }
                case 'tool': {
                    this.addToolToCompiler(
                        component.componentState.id,
                        component.componentState.toolId,
                        component.componentState.args,
                        component.componentState.stdin,
                    );
                    break;
                }
            }
        } else if ('content' in component) {
            // handle BasicGoldenLayoutStruct
            this.fromGoldenLayoutContent(component.content as unknown as BasicGoldenLayoutStruct[]);
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

    fromGoldenLayout(globj: GoldenLayoutRootStruct) {
        this.rootContent = globj.content;

        if (globj.content) {
            this.fromGoldenLayoutContent(globj.content);
        }
    }
}

class GoldenLayoutComponents {
    createSourceComponent(session: ClientStateSession, customSessionId?: number): GoldenLayoutComponentStruct {
        const editor = {
            type: 'component',
            componentName: 'codeEditor',
            componentState: {
                id: customSessionId || session.id,
                source: session.source,
                lang: session.language,
            },
            isClosable: true,
            reorderEnabled: true,
        };

        if (session.filename) {
            return {
                ...editor,
                title: session.filename,
                componentState: {
                    ...editor.componentState,
                    filename: session.filename,
                },
            };
        } else {
            return editor;
        }
    }

    createTreeComponent(tree: ClientStateTree, customTreeId?: number): GoldenLayoutComponentStruct {
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

    createAstComponent(
        session: ClientStateSession | null,
        compilerIndex: number,
        customSessionId?: number,
    ): GoldenLayoutComponentStruct {
        return {
            type: 'component',
            componentName: 'ast',
            componentState: {
                id: compilerIndex,
                editorid: customSessionId || (session ? session.id : undefined),
            },
            isClosable: true,
            reorderEnabled: true,
        };
    }

    createOptComponent(
        session: ClientStateSession | null,
        compilerIndex: number,
        customSessionId?: number,
    ): GoldenLayoutComponentStruct {
        return {
            type: 'component',
            componentName: 'opt',
            componentState: {
                id: compilerIndex,
                editorid: customSessionId || (session ? session.id : undefined),
            },
            isClosable: true,
            reorderEnabled: true,
        };
    }
    createStackUsageComponent(
        session: ClientStateSession,
        compilerIndex: number,
        customSessionId?: number,
    ): GoldenLayoutComponentStruct {
        return {
            type: 'component',
            componentName: 'stackusage',
            componentState: {
                id: compilerIndex,
                editorid: customSessionId || (session ? session.id : undefined),
            },
            isClosable: true,
            reorderEnabled: true,
        };
    }

    createCfgComponent(
        session: ClientStateSession | null,
        compilerIndex: number,
        customSessionId?: number,
    ): GoldenLayoutComponentStruct {
        return {
            type: 'component',
            componentName: 'opt',
            componentState: {
                id: compilerIndex,
                editorid: customSessionId || (session ? session.id : undefined),
                options: {
                    navigation: false,
                    physics: false,
                },
            },
            isClosable: true,
            reorderEnabled: true,
        };
    }

    createGccDumpComponent(
        session: ClientStateSession | null,
        compilerIndex: number,
        customSessionId?: number,
    ): GoldenLayoutComponentStruct {
        return {
            type: 'component',
            componentName: 'gccdump',
            componentState: {
                _compilerid: compilerIndex,
                _editorid: customSessionId || (session ? session.id : undefined),
            },
            isClosable: true,
            reorderEnabled: true,
        };
    }

    createCompilerOutComponent(
        session: ClientStateSession | null,
        compilerIndex: number,
        customSessionId?: number,
    ): GoldenLayoutComponentStruct {
        return {
            type: 'component',
            componentName: 'output',
            componentState: {
                compiler: compilerIndex,
                editor: customSessionId || (session ? session.id : undefined),
                wrap: false,
                fontScale: 14,
            },
            isClosable: true,
            reorderEnabled: true,
        };
    }

    createToolComponent(
        session: ClientStateSession | null,
        compilerIndex: number,
        toolId: string,
        args: string[],
        stdin: string,
        customSessionId?,
    ): GoldenLayoutComponentStruct {
        return {
            type: 'component',
            componentName: 'tool',
            componentState: {
                editor: customSessionId || (session ? session.id : undefined),
                compiler: compilerIndex,
                toolId: toolId,
                args: args,
                stdin: stdin,
            },
            isClosable: true,
            reorderEnabled: true,
        };
    }

    createConformanceViewComponent(
        session: ClientStateSession,
        conformanceview: ClientStateConformanceView,
    ): GoldenLayoutComponentStruct {
        return {
            type: 'component',
            componentName: 'conformance',
            componentState: {
                editorid: session.id,
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

    createSourceCompilerComponent(
        session: ClientStateSession,
        compiler,
        customSessionId?: number,
        idxCompiler?: number,
    ) {
        return {
            type: 'component',
            componentName: 'compiler',
            componentState: {
                id: idxCompiler,
                compiler: compiler.id,
                source: customSessionId || session.id,
                options: compiler.options,
                filters: this.copyCompilerFilters(compiler.filters),
                libs: compiler.libs,
                lang: session.language,
                overrides: compiler.overrides,
            },
            isClosable: true,
            reorderEnabled: true,
        };
    }

    createCompilerComponentForTree(
        tree: ClientStateTree,
        compiler: ClientStateCompiler,
        customTreeId: number | null,
        idxCompiler?: number,
    ) {
        return {
            type: 'component',
            componentName: 'compiler',
            componentState: {
                id: idxCompiler,
                compiler: compiler.id,
                source: undefined,
                tree: customTreeId || tree.id,
                options: compiler.options,
                filters: this.copyCompilerFilters(compiler.filters),
                libs: compiler.libs,
                lang: tree.compilerLanguageId,
                overrides: compiler.overrides,
            },
            isClosable: true,
            reorderEnabled: true,
        };
    }

    createExecutorComponent(session: ClientStateSession, executor: ClientStateExecutor, customSessionId?: number) {
        return {
            type: 'component',
            componentName: 'executor',
            componentState: {
                compiler: executor.compiler.id,
                source: customSessionId || session.id,
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
                overrides: executor.overrides,
                runtimeTools: executor.runtimeTools,
            },
            isClosable: true,
            reorderEnabled: true,
        };
    }

    createExecutorComponentForTree(tree: ClientStateTree, executor: ClientStateExecutor, customTreeId?: number) {
        return {
            type: 'component',
            componentName: 'executor',
            componentState: {
                compiler: executor.compiler.id,
                source: undefined,
                tree: customTreeId || tree.id,
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
                overrides: executor.overrides,
                runtimeTools: executor.runtimeTools,
            },
            isClosable: true,
            reorderEnabled: true,
        };
    }

    // UNUSED. not deleting yet
    // createDiffComponent(left, right) {
    //     return {
    //         type: 'component',
    //         componentName: 'diff',
    //         componentState: {
    //             lhs: left,
    //             rhs: right,
    //             lhsdifftype: 0,
    //             rhsdifftype: 0,
    //             fontScale: 14,
    //         },
    //     };
    // }

    createSpecialOutputComponent(
        viewtype: string,
        session: ClientStateSession,
        idxCompiler: number,
        customSessionId?: number,
    ): GoldenLayoutComponentStruct {
        switch (viewtype) {
            case 'ast': {
                return this.createAstComponent(session, idxCompiler + 1, customSessionId);
            }
            case 'opt': {
                return this.createOptComponent(session, idxCompiler + 1, customSessionId);
            }
            case 'stackusage': {
                return this.createStackUsageComponent(session, idxCompiler + 1, customSessionId);
            }
            case 'cfg': {
                return this.createCfgComponent(session, idxCompiler + 1, customSessionId);
            }
            case 'gccdump': {
                return this.createGccDumpComponent(session, idxCompiler + 1, customSessionId);
            }
            case 'compilerOutput': {
                return this.createCompilerOutComponent(session, idxCompiler + 1, customSessionId);
            }
            default: {
                throw new Error(`Unknown viewtype for compiler(${idxCompiler + 1}) ${viewtype}`);
            }
        }
    }

    createSpecialOutputComponentForTreeCompiler(viewtype: string, idxCompiler: number): GoldenLayoutComponentStruct {
        switch (viewtype) {
            case 'ast': {
                return this.createAstComponent(null, idxCompiler + 1);
            }
            case 'opt': {
                return this.createOptComponent(null, idxCompiler + 1);
            }
            case 'stackusage': {
                return this.createOptComponent(null, idxCompiler + 1);
            }
            case 'cfg': {
                return this.createCfgComponent(null, idxCompiler + 1);
            }
            case 'gccdump': {
                return this.createGccDumpComponent(null, idxCompiler + 1);
            }
            case 'compilerOutput': {
                return this.createCompilerOutComponent(null, idxCompiler + 1);
            }
            default: {
                throw new Error(`Unknown viewtype (for tree compiler ${idxCompiler + 1}) ${viewtype}`);
            }
        }
    }

    createToolComponentForTreeCompiler(
        tree: ClientStateTree,
        compilerIndex: number,
        toolId: number,
        args: string,
        stdin: string,
    ): GoldenLayoutComponentStruct {
        return {
            type: 'component',
            componentName: 'tool',
            componentState: {
                tree: tree ? tree.id : undefined,
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
    golden: GoldenLayoutRootStruct = {};

    newEmptyStack(width: number): BasicGoldenLayoutStruct {
        return {
            type: 'stack',
            width: width,
            content: [],
        };
    }

    newEmptyRow(height: number): BasicGoldenLayoutStruct {
        return {
            type: 'row',
            height: height,
            content: [],
        };
    }

    newEmptyColumn(): BasicGoldenLayoutStruct {
        return {
            type: 'column',
            content: [],
        };
    }

    newStackWithOneComponent(width: number, component: GoldenLayoutComponentStruct): BasicGoldenLayoutStruct {
        return {
            type: 'stack',
            width: width,
            content: [component],
        };
    }

    newTreeFromTree(tree: ClientStateTree, width: number): BasicGoldenLayoutStruct {
        return this.newStackWithOneComponent(width, this.createTreeComponent(tree));
    }

    newSourceStackFromSession(session: ClientStateSession, width: number): BasicGoldenLayoutStruct {
        return this.newStackWithOneComponent(width, this.createSourceComponent(session));
    }

    newAstStackFromCompiler(
        session: ClientStateSession,
        compilerIndex: number,
        width: number,
    ): BasicGoldenLayoutStruct {
        return this.newStackWithOneComponent(width, this.createAstComponent(session, compilerIndex));
    }

    newOptStackFromCompiler(
        session: ClientStateSession,
        compilerIndex: number,
        width: number,
    ): BasicGoldenLayoutStruct {
        return this.newStackWithOneComponent(width, this.createOptComponent(session, compilerIndex));
    }

    newCfgStackFromCompiler(
        session: ClientStateSession,
        compilerIndex: number,
        width: number,
    ): BasicGoldenLayoutStruct {
        return this.newStackWithOneComponent(width, this.createCfgComponent(session, compilerIndex));
    }

    // UNUSED. not deleting yet
    // newGccDumpStackFromCompiler(session, compilerIndex, width: number): BasicGoldenLayoutStruct {
    //     return this.newStackWithOneComponent(width, this.createGccDumpComponent(session, compilerIndex));
    // }

    // UNUSED. not deleting yet
    // newCompilerOutStackFromCompiler(session, compilerIndex, width: number): BasicGoldenLayoutStruct {
    //     return this.newStackWithOneComponent(width, this.createCompilerOutComponent(session, compilerIndex));
    // }

    newToolStackFromCompiler(
        session: ClientStateSession,
        compilerIndex: number,
        toolId: string,
        args: string[],
        stdin: string,
        width: number,
    ): BasicGoldenLayoutStruct {
        return this.newStackWithOneComponent(
            width,
            this.createToolComponent(session, compilerIndex, toolId, args, stdin),
        );
    }

    newConformanceViewStack(
        session: ClientStateSession,
        width: number,
        conformanceview: ClientStateConformanceView,
    ): BasicGoldenLayoutStruct {
        const component = this.createConformanceViewComponent(session, conformanceview);

        for (const compiler of conformanceview.compilers) {
            const compjson = {
                compilerId: compiler.id,
                options: compiler.options,
            };

            component.componentState.compilers.push(compjson);
        }

        return this.newStackWithOneComponent(width, component);
    }

    newCompilerStackFromSession(
        session: ClientStateSession,
        compiler: ClientStateCompiler,
        width: number,
    ): BasicGoldenLayoutStruct {
        return this.newStackWithOneComponent(
            width,
            this.createSourceCompilerComponent(session, compiler, undefined, compiler._internalid),
        );
    }

    newExecutorStackFromSession(
        session: ClientStateSession,
        executor: ClientStateExecutor,
        width: number,
    ): BasicGoldenLayoutStruct {
        return this.newStackWithOneComponent(width, this.createExecutorComponent(session, executor));
    }

    createSourceContentArray(state: ClientState, leftSession: number, rightSession: number): BasicGoldenLayoutStruct[] {
        if (leftSession === rightSession) {
            return [this.createPresentationModeComponents(state.sessions[leftSession], 1, 100)];
        } else {
            return [
                this.createPresentationModeComponents(state.sessions[leftSession], 1),
                this.createPresentationModeComponents(state.sessions[rightSession], 2),
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
                    content: [] as any[],
                },
            ],
        };
    }

    getPresentationModeLayoutForSource(state: ClientState, sessionIdx: number) {
        const gl = this.getPresentationModeEmptyLayout();
        gl.content[0].content = [
            {
                type: 'column',
                width: 100,
                content: this.createSourceContentArray(state, sessionIdx, sessionIdx),
            },
        ];

        return gl;
    }

    closeAllEditors(tree: ClientStateTree) {
        for (const file of tree.files) {
            file.editorId = -1;
            file.isOpen = false;
        }
    }

    getPresentationModeLayoutForTree(state: ClientState, leftTreeIdx: number) {
        const gl = this.getPresentationModeEmptyLayout();
        const tree = state.trees[leftTreeIdx];
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

        const stack = gl.content[0].content[0].content[1].content[0];

        for (let idxCompiler = 0; idxCompiler < tree.compilers.length; idxCompiler++) {
            const compiler = tree.compilers[idxCompiler];
            stack.content.push(this.createCompilerComponentForTree(tree, compiler, customTreeId, idxCompiler + 1));

            for (const viewtype of compiler.specialoutputs) {
                stack.content.push(this.createSpecialOutputComponentForTreeCompiler(viewtype, idxCompiler));
            }

            for (const tool of compiler.tools) {
                stack.content.push(
                    this.createToolComponentForTreeCompiler(tree, idxCompiler + 1, tool.id, tool.args, tool.stdin),
                );
            }
        }

        for (let idxExecutor = 0; idxExecutor < tree.executors.length; idxExecutor++) {
            stack.content.push(this.createExecutorComponentForTree(tree, tree.executors[idxExecutor], customTreeId));
        }

        return gl;
    }

    // UNUSED. not deleting yet
    // getPresentationModeLayoutForComparisonSlide(state, left, right) {
    //     const gl = this.getPresentationModeEmptyLayout();
    //     gl.content[0].content = [
    //         {
    //             type: 'row',
    //             height: 50,
    //             content: this.createSourceContentArray(state, left, right),
    //         },
    //         {
    //             type: 'row',
    //             height: 50,
    //             content: [
    //                 {
    //                     type: 'stack',
    //                     width: 100,
    //                     content: [this.createDiffComponent(left.compiler + 1, right.compiler + 1)],
    //                 },
    //             ],
    //         },
    //     ];

    //     return gl;
    // }

    createPresentationModeComponents(session: ClientStateSession, customSessionId?: number, customWidth?: number) {
        const stack = {
            type: 'stack',
            width: customWidth || 50,
            activeItemIndex: 0,
            content: [this.createSourceComponent(session, customSessionId)] as any[],
        };

        for (let idxCompiler = 0; idxCompiler < session.compilers.length; idxCompiler++) {
            const compiler = session.compilers[idxCompiler];
            stack.content.push(this.createSourceCompilerComponent(session, compiler, customSessionId));

            for (const viewtype of compiler.specialoutputs) {
                stack.content.push(this.createSpecialOutputComponent(viewtype, session, idxCompiler, customSessionId));
            }

            for (const tool of compiler.tools) {
                stack.content.push(
                    this.createToolComponent(null, idxCompiler + 1, tool.id, tool.args, tool.stdin, customSessionId),
                );
            }
        }

        for (let idxExecutor = 0; idxExecutor < session.executors.length; idxExecutor++) {
            stack.content.push(this.createExecutorComponent(session, session.executors[idxExecutor], customSessionId));
        }

        return stack;
    }

    generatePresentationModeMobileViewerSlides(state: ClientState) {
        const slides: any[] = [];

        if (state.trees.length > 0) {
            for (let idxTree = 0; idxTree < state.trees.length; idxTree++) {
                const gl = this.getPresentationModeLayoutForTree(state, idxTree);
                slides.push(gl);
            }

            return slides;
        }

        for (let idxSession = 0; idxSession < state.sessions.length; idxSession++) {
            const gl = this.getPresentationModeLayoutForSource(state, idxSession);
            slides.push(gl);
        }

        return slides;
    }

    treeLayoutFromClientstate(state: ClientState, leaveSomeSpace: boolean): BasicGoldenLayoutStruct | undefined {
        const firstTree = state.trees[0];
        const leftSide = this.newTreeFromTree(firstTree, 25);
        const middle = this.newEmptyStack(40);
        for (const session of state.sessions) {
            middle.content.push(this.createSourceComponent(session));
        }

        let rightSide;
        let contentRow;
        let extraRow;

        if (leaveSomeSpace) {
            contentRow = this.newEmptyRow(50);
            extraRow = this.newEmptyRow(50);
            rightSide = this.newEmptyColumn();
            rightSide.content.push(contentRow, extraRow);
        } else {
            rightSide = this.newEmptyStack(40);
            contentRow = this.newEmptyRow(100);
        }

        let idxCompiler = 0;
        for (const compiler of firstTree.compilers) {
            contentRow.content.push(this.createCompilerComponentForTree(firstTree, compiler, null, idxCompiler + 1));

            for (const specialOutput of compiler.specialoutputs) {
                contentRow.content.push(
                    this.createSpecialOutputComponentForTreeCompiler(specialOutput, idxCompiler + 1),
                );
            }

            idxCompiler++;
        }

        rightSide.content.push(contentRow);

        assert(this.golden.content);
        this.golden.content[0].content.push(leftSide, middle, rightSide);

        return extraRow;
    }

    hasEditorCompilersOrExecutors(state: ClientState) {
        for (const session of state.sessions) {
            if (session.compilers.length > 0 || session.executors.length > 0) {
                return true;
            }
        }

        return false;
    }

    fromClientState(state: ClientState) {
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
            const hasOtherPanes = this.hasEditorCompilersOrExecutors(state);
            const extraRow = this.treeLayoutFromClientstate(state, hasOtherPanes);

            if (!hasOtherPanes || !extraRow) return;

            const treeCompilerCount = state.trees[0].compilers.length;
            const treeExecutorCount = state.trees[0].executors.length;

            for (let idxSession = 0; idxSession < state.sessions.length; idxSession++) {
                const session = state.sessions[idxSession];

                if (session.compilers.length > 0 || session.executors.length > 0) {
                    const rightCol: BasicGoldenLayoutStruct = {
                        type: 'column',
                        content: [],
                    };
                    const rightStack = this.newEmptyStack(100);

                    for (let idxCompiler = 0; idxCompiler < session.compilers.length; idxCompiler++) {
                        const compiler = session.compilers[idxCompiler];
                        const compilerComponent = this.createSourceCompilerComponent(
                            session,
                            compiler,
                            undefined,
                            treeCompilerCount + idxCompiler + 1,
                        );
                        rightStack.content.push(compilerComponent);

                        for (const viewtype of compiler.specialoutputs) {
                            rightStack.content.push(
                                this.createSpecialOutputComponent(
                                    viewtype,
                                    session,
                                    treeCompilerCount + idxCompiler + 1,
                                ),
                            );
                        }

                        for (const tool of compiler.tools) {
                            rightStack.content.push(
                                this.createToolComponent(
                                    session,
                                    treeCompilerCount + idxCompiler + 1,
                                    tool.id,
                                    tool.args,
                                    tool.stdin,
                                ),
                            );
                        }
                    }

                    for (let idxExecutor = 0; idxExecutor < session.executors.length; idxExecutor++) {
                        const executor = session.executors[idxExecutor];
                        const executorComponent = this.createExecutorComponent(
                            session,
                            executor,
                            treeExecutorCount + idxExecutor + 1,
                        );
                        rightStack.content.push(executorComponent);
                    }

                    rightCol.content.push(rightStack);

                    extraRow.content.push({
                        type: 'row',
                        content: [rightCol],
                    });
                }
            }

            return;
        }

        assert(this.golden.content);

        if (state.sessions.length > 1) {
            const sessionWidth = 100 / state.sessions.length;

            for (let idxSession = 0; idxSession < state.sessions.length; idxSession++) {
                const session = state.sessions[idxSession];

                const topRow: BasicGoldenLayoutStruct = {
                    type: 'row',
                    content: [],
                };
                const bottomRow: BasicGoldenLayoutStruct = {
                    type: 'row',
                    content: [],
                };

                const stack = this.newSourceStackFromSession(session, 100);
                topRow.content.push(stack);

                const compilerWidth =
                    100 /
                    (1 +
                        session.compilers.length +
                        session.executors.length +
                        session.countNumberOfSpecialOutputsAndTools());

                if (session.conformanceview) {
                    const stack = this.newConformanceViewStack(session, compilerWidth, session.conformanceview);
                    bottomRow.content.push(stack);
                }

                for (let idxCompiler = 0; idxCompiler < session.compilers.length; idxCompiler++) {
                    const compiler = session.compilers[idxCompiler];
                    const stack = this.newCompilerStackFromSession(session, compiler, compilerWidth);
                    bottomRow.content.push(stack);

                    for (const viewtype of compiler.specialoutputs) {
                        const stack = this.newStackWithOneComponent(
                            compilerWidth,
                            this.createSpecialOutputComponent(viewtype, session, idxCompiler),
                        );

                        if (stack) {
                            bottomRow.content.push(stack);
                        }
                    }

                    for (const tool of compiler.tools) {
                        const stack = this.newToolStackFromCompiler(
                            session,
                            idxCompiler + 1,
                            tool.id,
                            tool.args,
                            tool.stdin,
                            compilerWidth,
                        );
                        bottomRow.content.push(stack);
                    }
                }

                for (let idxExecutor = 0; idxExecutor < session.executors.length; idxExecutor++) {
                    const executor = session.executors[idxExecutor];
                    const stack = this.newExecutorStackFromSession(session, executor, compilerWidth);
                    bottomRow.content.push(stack);
                }

                const sessionColumn: BasicGoldenLayoutStruct = {
                    type: 'column',
                    isClosable: true,
                    reorderEnabled: true,
                    width: sessionWidth,
                    content: [topRow, bottomRow],
                };

                this.golden.content[0].content[idxSession] = sessionColumn;
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
                const stack = this.newCompilerStackFromSession(session, compiler, width);
                this.golden.content[0].content.push(stack);

                for (const viewtype of compiler.specialoutputs) {
                    const stack = this.newStackWithOneComponent(
                        width,
                        this.createSpecialOutputComponent(viewtype, session, idxCompiler),
                    );

                    if (stack) {
                        this.golden.content[0].content.push(stack);
                    }
                }

                for (const tool of compiler.tools) {
                    const stack = this.newToolStackFromCompiler(
                        session,
                        idxCompiler + 1,
                        tool.id,
                        tool.args,
                        tool.stdin,
                        width,
                    );
                    this.golden.content[0].content.push(stack);
                }
            }

            for (const executor of session.executors) {
                const stack = this.newExecutorStackFromSession(session, executor, width);
                this.golden.content[0].content.push(stack);
            }
        }
    }
}
