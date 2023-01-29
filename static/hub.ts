// Copyright (c) 2022, Compiler Explorer Authors
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

import GoldenLayout, {ContentItem} from 'golden-layout';

import {CompilerService} from './compiler-service';
import {
    AST_VIEW_COMPONENT_NAME,
    CFG_VIEW_COMPONENT_NAME,
    COMPILER_COMPONENT_NAME,
    CONFORMANCE_VIEW_COMPONENT_NAME,
    DEVICE_VIEW_COMPONENT_NAME,
    DIFF_VIEW_COMPONENT_NAME,
    EDITOR_COMPONENT_NAME,
    EXECUTOR_COMPONENT_NAME,
    FLAGS_VIEW_COMPONENT_NAME,
    GCC_DUMP_VIEW_COMPONENT_NAME,
    GNAT_DEBUG_TREE_VIEW_COMPONENT_NAME,
    GNAT_DEBUG_VIEW_COMPONENT_NAME,
    HASKELL_CMM_VIEW_COMPONENT_NAME,
    HASKELL_CORE_VIEW_COMPONENT_NAME,
    HASKELL_STG_VIEW_COMPONENT_NAME,
    IR_VIEW_COMPONENT_NAME,
    LLVM_OPT_PIPELINE_VIEW_COMPONENT_NAME,
    OPT_VIEW_COMPONENT_NAME,
    OUTPUT_COMPONENT_NAME,
    PP_VIEW_COMPONENT_NAME,
    RUST_HIR_VIEW_COMPONENT_NAME,
    RUST_MACRO_EXP_VIEW_COMPONENT_NAME,
    RUST_MIR_VIEW_COMPONENT_NAME,
    TOOL_COMPONENT_NAME,
    TOOL_INPUT_VIEW_COMPONENT_NAME,
    TREE_COMPONENT_NAME,
} from './components.interfaces';
import {EventHub} from './event-hub';
import {Editor} from './panes/editor';
import {Tree} from './panes/tree';
import {Compiler} from './panes/compiler';
import {Executor} from './panes/executor';
import {Output} from './panes/output';
import {Tool} from './panes/tool';
import {Diff} from './panes/diff';
import {ToolInputView} from './panes/tool-input-view';
import {Opt as OptView} from './panes/opt-view';
import {Flags as FlagsView} from './panes/flags-view';
import {PP as PreProcessorView} from './panes/pp-view';
import {Ast as AstView} from './panes/ast-view';
import {Ir as IrView} from './panes/ir-view';
import {LLVMOptPipeline} from './panes/llvm-opt-pipeline';
import {DeviceAsm as DeviceView} from './panes/device-view';
import {GnatDebug as GnatDebugView} from './panes/gnatdebug-view';
import {RustMir as RustMirView} from './panes/rustmir-view';
import {RustHir as RustHirView} from './panes/rusthir-view';
import {HaskellCore as HaskellCoreView} from './panes/haskellcore-view';
import {HaskellStg as HaskellStgView} from './panes/haskellstg-view';
import {HaskellCmm as HaskellCmmView} from './panes/haskellcmm-view';
import {GccDump as GCCDumpView} from './panes/gccdump-view';
import {Cfg as CfgView} from './panes/cfg-view';
import {Conformance as ConformanceView} from './panes/conformance-view';
import {GnatDebugTree as GnatDebugTreeView} from './panes/gnatdebugtree-view';
import {RustMacroExp as RustMacroExpView} from './panes/rustmacroexp-view';
import {IdentifierSet} from './identifier-set';

export class Hub {
    public readonly editorIds: IdentifierSet = new IdentifierSet();
    public readonly compilerIds: IdentifierSet = new IdentifierSet();
    public readonly executorIds: IdentifierSet = new IdentifierSet();
    public readonly treeIds: IdentifierSet = new IdentifierSet();

    public trees: Tree[] = [];
    public editors: any[] = []; // typeof Editor

    public readonly compilerService: CompilerService;

    public deferred = true;
    public deferredEmissions: unknown[][] = [];

    public lastOpenedLangId: string | null;
    public subdomainLangId: string | undefined;
    public defaultLangId: string;

    public constructor(public readonly layout: GoldenLayout, subLangId: string | undefined, defaultLangId: string) {
        this.lastOpenedLangId = null;
        this.subdomainLangId = subLangId;
        this.defaultLangId = defaultLangId;
        this.compilerService = new CompilerService(this.layout.eventHub);

        layout.registerComponent(EDITOR_COMPONENT_NAME, (c, s) => this.codeEditorFactory(c, s));
        layout.registerComponent(COMPILER_COMPONENT_NAME, (c, s) => this.compilerFactory(c, s));
        layout.registerComponent(TREE_COMPONENT_NAME, (c, s) => this.treeFactory(c, s));
        layout.registerComponent(EXECUTOR_COMPONENT_NAME, (c, s) => this.executorFactory(c, s));
        layout.registerComponent(OUTPUT_COMPONENT_NAME, (c, s) => this.outputFactory(c, s));
        layout.registerComponent(TOOL_COMPONENT_NAME, (c, s) => this.toolFactory(c, s));
        layout.registerComponent(TOOL_INPUT_VIEW_COMPONENT_NAME, (c, s) => this.toolInputViewFactory(c, s));
        layout.registerComponent(DIFF_VIEW_COMPONENT_NAME, (c, s) => this.diffFactory(c, s));
        layout.registerComponent(OPT_VIEW_COMPONENT_NAME, (c, s) => this.optViewFactory(c, s));
        layout.registerComponent(FLAGS_VIEW_COMPONENT_NAME, (c, s) => this.flagsViewFactory(c, s));
        layout.registerComponent(PP_VIEW_COMPONENT_NAME, (c, s) => this.ppViewFactory(c, s));
        layout.registerComponent(AST_VIEW_COMPONENT_NAME, (c, s) => this.astViewFactory(c, s));
        layout.registerComponent(IR_VIEW_COMPONENT_NAME, (c, s) => this.irViewFactory(c, s));
        layout.registerComponent(LLVM_OPT_PIPELINE_VIEW_COMPONENT_NAME, (c, s) => this.llvmOptPipelineFactory(c, s));
        layout.registerComponent(DEVICE_VIEW_COMPONENT_NAME, (c, s) => this.deviceViewFactory(c, s));
        layout.registerComponent(RUST_MIR_VIEW_COMPONENT_NAME, (c, s) => this.rustMirViewFactory(c, s));
        layout.registerComponent(HASKELL_CORE_VIEW_COMPONENT_NAME, (c, s) => this.haskellCoreViewFactory(c, s));
        layout.registerComponent(HASKELL_STG_VIEW_COMPONENT_NAME, (c, s) => this.haskellStgViewFactory(c, s));
        layout.registerComponent(HASKELL_CMM_VIEW_COMPONENT_NAME, (c, s) => this.haskellCmmViewFactory(c, s));
        layout.registerComponent(GNAT_DEBUG_TREE_VIEW_COMPONENT_NAME, (c, s) => this.gnatDebugTreeViewFactory(c, s));
        layout.registerComponent(GNAT_DEBUG_VIEW_COMPONENT_NAME, (c, s) => this.gnatDebugViewFactory(c, s));
        layout.registerComponent(RUST_MACRO_EXP_VIEW_COMPONENT_NAME, (c, s) => this.rustMacroExpViewFactory(c, s));
        layout.registerComponent(RUST_HIR_VIEW_COMPONENT_NAME, (c, s) => this.rustHirViewFactory(c, s));
        layout.registerComponent(GCC_DUMP_VIEW_COMPONENT_NAME, (c, s) => this.gccDumpViewFactory(c, s));
        layout.registerComponent(CFG_VIEW_COMPONENT_NAME, (c, s) => this.cfgViewFactory(c, s));
        layout.registerComponent(CONFORMANCE_VIEW_COMPONENT_NAME, (c, s) => this.conformanceViewFactory(c, s));

        layout.eventHub.on(
            'editorOpen',
            function (this: Hub, id: number) {
                this.editorIds.add(id);
            },
            this
        );
        layout.eventHub.on(
            'editorClose',
            function (this: Hub, id: number) {
                this.editorIds.remove(id);
            },
            this
        );
        layout.eventHub.on(
            'compilerOpen',
            function (this: Hub, id: number) {
                this.compilerIds.add(id);
            },
            this
        );
        layout.eventHub.on(
            'compilerClose',
            function (this: Hub, id: number) {
                this.compilerIds.remove(id);
            },
            this
        );
        layout.eventHub.on(
            'treeOpen',
            function (this: Hub, id: number) {
                this.treeIds.add(id);
            },
            this
        );
        layout.eventHub.on(
            'treeClose',
            function (this: Hub, id: number) {
                this.treeIds.remove(id);
            },
            this
        );
        layout.eventHub.on(
            'executorOpen',
            function (this: Hub, id: number) {
                this.executorIds.add(id);
            },
            this
        );
        layout.eventHub.on(
            'executorClose',
            function (this: Hub, id: number) {
                this.executorIds.remove(id);
            },
            this
        );
        layout.eventHub.on(
            'languageChange',
            function (this: Hub, editorId: number, langId: string) {
                this.lastOpenedLangId = langId;
            },
            this
        );

        layout.init();
        this.undefer();
        layout.eventHub.emit('initialised');
    }

    public nextTreeId(): number {
        return this.treeIds.next();
    }

    public nextEditorId(): number {
        return this.editorIds.next();
    }

    public nextCompilerId(): number {
        return this.compilerIds.next();
    }

    public nextExecutorId(): number {
        return this.executorIds.next();
    }

    public createEventHub(): EventHub {
        return new EventHub(this, this.layout.eventHub);
    }

    public undefer(): void {
        this.deferred = false;
        const eventHub = this.layout.eventHub;
        const compilerEmissions: unknown[][] = [];
        const nonCompilerEmissions: unknown[][] = [];

        for (const [eventName, ...args] of this.deferredEmissions) {
            if (eventName === 'compiler') {
                compilerEmissions.push([eventName, ...args]);
            } else {
                nonCompilerEmissions.push([eventName, ...args]);
            }
        }

        for (const args of nonCompilerEmissions) {
            // @ts-expect-error
            // eslint-disable-next-line prefer-spread
            eventHub.emit.apply(eventHub, args);
        }

        for (const args of compilerEmissions) {
            // @ts-expect-error
            // eslint-disable-next-line prefer-spread
            eventHub.emit.apply(eventHub, args);
        }

        this.deferredEmissions = [];
    }

    public getTreeById(id: number): Tree | undefined {
        return this.trees.find(t => t.id === id);
    }

    public removeTree(id: number) {
        this.trees = this.trees.filter(t => t.id !== id);
    }

    public hasTree(): boolean {
        return this.trees.length > 0;
    }

    public getTreesWithEditorId(editorId: number) {
        return this.trees.filter(tree => tree.multifileService.isEditorPartOfProject(editorId));
    }

    public getTrees(): Tree[] {
        return this.trees;
    }

    public getEditorById(id: number): Editor | undefined {
        return this.editors.find(e => e.id === id);
    }

    public removeEditor(id: number) {
        this.editors = this.editors.filter(e => e.id !== id);
    }

    // Layout getters

    public findParentRowOrColumn(elem: GoldenLayout.ContentItem): GoldenLayout.ContentItem | null {
        let currentElem: GoldenLayout.ContentItem | null = elem;
        while (currentElem) {
            if (currentElem.isRow || currentElem.isColumn) return currentElem;
            // currentElem.parent may be null, this is reflected in newer GoldenLayout versions but not the version
            // we're using. Making a cast here just to be precise about what's going on.
            currentElem = currentElem.parent as GoldenLayout.ContentItem | null;
        }
        return null;
    }

    public findParentRowOrColumnOrStack(elem: GoldenLayout.ContentItem): GoldenLayout.ContentItem | null {
        let currentElem: GoldenLayout.ContentItem | null = elem;
        while (currentElem) {
            if (currentElem.isRow || currentElem.isColumn || currentElem.isStack) return currentElem;
            // currentElem.parent may be null, this is reflected in newer GoldenLayout versions but not the version
            // we're using. Making a cast here just to be precise about what's going on.
            currentElem = currentElem.parent as GoldenLayout.ContentItem | null;
        }
        return null;
    }

    public findEditorInChildren(elem: GoldenLayout.ContentItem): GoldenLayout.ContentItem | boolean | null {
        const count = elem.contentItems.length;
        let index = 0;
        while (index < count) {
            const child = elem.contentItems[index];
            // @ts-expect-error -- GoldenLayout's types are messed up here. This
            // is a ContentItem, which can be a Component which has a componentName
            // property
            if (child.componentName === 'codeEditor') {
                return this.findParentRowOrColumnOrStack(child);
            } else {
                if (child.isRow || child.isColumn || child.isStack) {
                    const editor = this.findEditorInChildren(child);
                    if (editor) return editor;
                }
            }
            index++;
        }
        return false;
    }

    public findEditorParentRowOrColumn(): GoldenLayout.ContentItem | boolean | null {
        return this.findEditorInChildren(this.layout.root);
    }

    public addInEditorStackIfPossible(elem: GoldenLayout.ContentItem): void {
        const insertionPoint = this.findEditorParentRowOrColumn();
        // required not-true check because findEditorParentRowOrColumn returns
        // false if there is no editor parent
        if (insertionPoint && insertionPoint !== true) {
            insertionPoint.addChild(elem);
        } else {
            this.addAtRoot(elem);
        }
    }

    public addAtRoot(elem: GoldenLayout.ContentItem): void {
        if (this.layout.root.contentItems.length > 0) {
            const rootFirstItem = this.layout.root.contentItems[0];
            if (rootFirstItem.isRow || rootFirstItem.isColumn) {
                rootFirstItem.addChild(elem);
            } else {
                // @ts-expect-error -- GoldenLayout's types are messed up here?
                const newRow: ContentItem = this.layout.createContentItem(
                    {
                        type: 'row',
                    },
                    this.layout.root
                );
                this.layout.root.replaceChild(rootFirstItem, newRow);
                newRow.addChild(rootFirstItem);
                newRow.addChild(elem);
            }
        } else {
            this.layout.root.addChild({
                type: 'row',
                content: [elem],
            });
        }
    }

    public activateTabForContainer(container?: GoldenLayout.Container) {
        if (container && (container.tab as typeof container.tab | null)) {
            container.tab.header.parent.setActiveContentItem(container.tab.contentItem);
        }
    }

    // Component Factories

    private codeEditorFactory(container: GoldenLayout.Container, state: any): void {
        // Ensure editors are closable: some older versions had 'isClosable' false.
        // NB there doesn't seem to be a better way to do this than reach into the config and rely on the fact nothing
        // has used it yet.
        container.parent.config.isClosable = true;
        const editor = new Editor(this, state, container);
        this.editors.push(editor);
    }

    private treeFactory(container: GoldenLayout.Container, state: ConstructorParameters<typeof Tree>[2]): Tree {
        const tree = new Tree(this, container, state);
        this.trees.push(tree);
        return tree;
    }

    public compilerFactory(container: GoldenLayout.Container, state: any): any /* typeof Compiler */ {
        return new Compiler(this, container, state);
    }

    public executorFactory(container: GoldenLayout.Container, state: any): any /*typeof Executor */ {
        return new Executor(this, container, state);
    }

    public outputFactory(container: GoldenLayout.Container, state: ConstructorParameters<typeof Output>[2]): Output {
        return new Output(this, container, state);
    }

    public toolFactory(container: GoldenLayout.Container, state: any): any /* typeof Tool */ {
        return new Tool(this, container, state);
    }

    public diffFactory(container: GoldenLayout.Container, state: any): any /* typeof Diff */ {
        return new Diff(this, container, state);
    }

    public toolInputViewFactory(
        container: GoldenLayout.Container,
        state: ConstructorParameters<typeof ToolInputView>[2]
    ): ToolInputView {
        return new ToolInputView(this, container, state);
    }

    public optViewFactory(container: GoldenLayout.Container, state: ConstructorParameters<typeof OptView>[2]): OptView {
        return new OptView(this, container, state);
    }

    public flagsViewFactory(
        container: GoldenLayout.Container,
        state: ConstructorParameters<typeof FlagsView>[2]
    ): FlagsView {
        return new FlagsView(this, container, state);
    }

    public ppViewFactory(
        container: GoldenLayout.Container,
        state: ConstructorParameters<typeof PreProcessorView>[2]
    ): PreProcessorView {
        return new PreProcessorView(this, container, state);
    }

    public astViewFactory(container: GoldenLayout.Container, state: ConstructorParameters<typeof AstView>[2]): AstView {
        return new AstView(this, container, state);
    }

    public irViewFactory(container: GoldenLayout.Container, state: ConstructorParameters<typeof IrView>[2]): IrView {
        return new IrView(this, container, state);
    }

    public llvmOptPipelineFactory(
        container: GoldenLayout.Container,
        state: ConstructorParameters<typeof LLVMOptPipeline>[2]
    ): LLVMOptPipeline {
        return new LLVMOptPipeline(this, container, state);
    }

    public deviceViewFactory(
        container: GoldenLayout.Container,
        state: ConstructorParameters<typeof DeviceView>[2]
    ): DeviceView {
        return new DeviceView(this, container, state);
    }

    public gnatDebugTreeViewFactory(
        container: GoldenLayout.Container,
        state: ConstructorParameters<typeof GnatDebugTreeView>[2]
    ): GnatDebugTreeView {
        return new GnatDebugTreeView(this, container, state);
    }

    public gnatDebugViewFactory(
        container: GoldenLayout.Container,
        state: ConstructorParameters<typeof GnatDebugView>[2]
    ): GnatDebugView {
        return new GnatDebugView(this, container, state);
    }

    public rustMirViewFactory(
        container: GoldenLayout.Container,
        state: ConstructorParameters<typeof RustMirView>[2]
    ): RustMirView {
        return new RustMirView(this, container, state);
    }

    public rustMacroExpViewFactory(
        container: GoldenLayout.Container,
        state: ConstructorParameters<typeof RustMacroExpView>[2]
    ): RustMacroExpView {
        return new RustMacroExpView(this, container, state);
    }

    public rustHirViewFactory(
        container: GoldenLayout.Container,
        state: ConstructorParameters<typeof RustHirView>[2]
    ): RustHirView {
        return new RustHirView(this, container, state);
    }

    public haskellCoreViewFactory(
        container: GoldenLayout.Container,
        state: ConstructorParameters<typeof HaskellCoreView>[2]
    ): HaskellCoreView {
        return new HaskellCoreView(this, container, state);
    }

    public haskellStgViewFactory(
        container: GoldenLayout.Container,
        state: ConstructorParameters<typeof HaskellStgView>[2]
    ): HaskellStgView {
        return new HaskellStgView(this, container, state);
    }
    public haskellCmmViewFactory(
        container: GoldenLayout.Container,
        state: ConstructorParameters<typeof HaskellCmmView>[2]
    ): HaskellCmmView {
        return new HaskellCmmView(this, container, state);
    }

    public gccDumpViewFactory(container: GoldenLayout.Container, state: any): any /* typeof GccDumpView */ {
        return new GCCDumpView(this, container, state);
    }

    public cfgViewFactory(container: GoldenLayout.Container, state: ConstructorParameters<typeof CfgView>[2]): CfgView {
        return new CfgView(this, container, state);
    }

    public conformanceViewFactory(container: GoldenLayout.Container, state: any): any /* typeof ConformanceView */ {
        return new ConformanceView(this, container, state);
    }
}
