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
// THIS SOFTWARE IS PROVIDED BY THE COcomponentNamePYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
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
import {IdentifierSet} from './identifier-set';
import {EventHub} from './event-hub';
import * as Components from './components';

import {Tree} from './panes/tree';
import {Editor} from './panes/editor';
import {Compiler} from './panes/compiler';
import {Executor} from './panes/executor';
import {Output} from './panes/output';
import {Tool} from './panes/tool';
import {Diff, getComponent as getDiffComponent} from './panes/diff';
import {ToolInputView} from './panes/tool-input-view';
import {Opt as OptView} from './panes/opt-view';
import {Flags as FlagsView} from './panes/flags-view';
import {PP as PreProcessorView} from './panes/pp-view';
import {Ast as AstView} from './panes/ast-view';
import {Ir as IrView} from './panes/ir-view';
import {DeviceAsm as DeviceView} from './panes/device-view';
import {GnatDebug as GnatDebugView} from './panes/gnatdebug-view';
import {RustMir as RustMirView} from './panes/rustmir-view';
import {RustHir as RustHirView} from './panes/rusthir-view';
import {GccDump as GCCDumpView} from './panes/gccdump-view';
import {Cfg as CfgView} from './panes/cfg-view';
import {Conformance as ConformanceView} from './panes/conformance-view';
import {GnatDebugTree as GnatDebugTreeView} from './panes/gnatdebugtree-view';
import {RustMacroExp as RustMacroExpView} from './panes/rustmacroexp-view';
import Sentry from '@sentry/browser';

export class Hub {
    public readonly editorIds: IdentifierSet = new IdentifierSet();
    public readonly compilerIds: IdentifierSet = new IdentifierSet();
    public readonly executorIds: IdentifierSet = new IdentifierSet();
    public readonly treeIds: IdentifierSet = new IdentifierSet();

    public trees: Tree[] = [];
    public editors: any[] = []; // typeof Editor

    public readonly compilerService: any; // typeof CompilerService

    public deferred = true;
    public deferredEmissions: unknown[][] = [];

    public lastOpenedLangId: string | null;
    public subdomainLangId: string | undefined;
    public defaultLangId: string;

    public constructor(public readonly layout: GoldenLayout, subLangId: string, defaultLangId: string) {
        this.lastOpenedLangId = null;
        this.subdomainLangId = subLangId || undefined;
        this.defaultLangId = defaultLangId;
        this.compilerService = new CompilerService(this.layout.eventHub);

        layout.registerComponent(Components.getEditor().componentName, (c, s) => this.codeEditorFactory(c, s));
        layout.registerComponent(Components.getCompiler().componentName, (c, s) => this.compilerFactory(c, s));
        layout.registerComponent(Components.getTree().componentName, (c, s) => this.treeFactory(c, s));
        layout.registerComponent(Components.getExecutor().componentName, (c, s) => this.executorFactory(c, s));
        layout.registerComponent(Components.getOutput().componentName, (c, s) => this.outputFactory(c, s));
        layout.registerComponent(Components.getToolViewWith().componentName, (c, s) => this.toolFactory(c, s));
        // eslint-disable-next-line max-len
        layout.registerComponent(Components.getToolInputView().componentName, (c, s) =>
            this.toolInputViewFactory(c, s)
        );
        layout.registerComponent(getDiffComponent().componentName, (c, s) => this.diffFactory(c, s));
        layout.registerComponent(Components.getOptView().componentName, (c, s) => this.optViewFactory(c, s));
        layout.registerComponent(Components.getFlagsView().componentName, (c, s) => this.flagsViewFactory(c, s));
        layout.registerComponent(Components.getPpView().componentName, (c, s) => this.ppViewFactory(c, s));
        layout.registerComponent(Components.getAstView().componentName, (c, s) => this.astViewFactory(c, s));
        layout.registerComponent(Components.getIrView().componentName, (c, s) => this.irViewFactory(c, s));
        layout.registerComponent(Components.getDeviceView().componentName, (c, s) => this.deviceViewFactory(c, s));
        layout.registerComponent(Components.getRustMirView().componentName, (c, s) => this.rustMirViewFactory(c, s));
        // eslint-disable-next-line max-len
        layout.registerComponent(Components.getGnatDebugTreeView().componentName, (c, s) =>
            this.gnatDebugTreeViewFactory(c, s)
        );
        // eslint-disable-next-line max-len
        layout.registerComponent(Components.getGnatDebugView().componentName, (c, s) =>
            this.gnatDebugViewFactory(c, s)
        );
        // eslint-disable-next-line max-len
        layout.registerComponent(Components.getRustMacroExpView().componentName, (c, s) =>
            this.rustMacroExpViewFactory(c, s)
        );
        layout.registerComponent(Components.getRustHirView().componentName, (c, s) => this.rustHirViewFactory(c, s));
        layout.registerComponent(Components.getGccDumpView().componentName, (c, s) => this.gccDumpViewFactory(c, s));
        layout.registerComponent(Components.getCfgView().componentName, (c, s) => this.cfgViewFactory(c, s));
        // eslint-disable-next-line max-len
        layout.registerComponent(Components.getConformanceView().componentName, (c, s) =>
            this.conformanceViewFactory(c, s)
        );

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

    public findParentRowOrColumn(elem: GoldenLayout.ContentItem): GoldenLayout.ContentItem {
        let currentElem: GoldenLayout.ContentItem | null = elem;
        while (currentElem) {
            if (currentElem.isRow || currentElem.isColumn) return currentElem;
            currentElem = currentElem.parent as GoldenLayout.ContentItem | null;
        }
        Sentry.captureMessage('findParentRowOrColumn: Unable to find parent row or column');
        return null as any;
    }

    public findParentRowOrColumnOrStack(elem: GoldenLayout.ContentItem): GoldenLayout.ContentItem {
        let currentElem: GoldenLayout.ContentItem | null = elem;
        while (currentElem) {
            if (currentElem.isRow || currentElem.isColumn || currentElem.isStack) return currentElem;
            currentElem = currentElem.parent as GoldenLayout.ContentItem | null;
        }
        Sentry.captureMessage('#findParentRowOrColumnOrStack: Unable to find parent row, column, or stack');
        return null as any;
    }

    public findEditorInChildren(elem: GoldenLayout.ContentItem): GoldenLayout.ContentItem | boolean {
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

    public findEditorParentRowOrColumn(): GoldenLayout.ContentItem | boolean {
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
        const rootFirstItem = this.layout.root.contentItems[0] as typeof this.layout.root.contentItems[0] | undefined;
        if (rootFirstItem) {
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

    private treeFactory(container: GoldenLayout.Container, state: any): Tree {
        const tree = new Tree(this, state, container);
        this.trees.push(tree);
        return tree;
    }

    public compilerFactory(container: GoldenLayout.Container, state: any): any /* typeof Compiler */ {
        return new Compiler(this, container, state);
    }

    public executorFactory(container: GoldenLayout.Container, state: any): any /* typeof Executor */ {
        return new Executor(this, container, state);
    }

    public outputFactory(container: GoldenLayout.Container, state: any): any /* typeof Output */ {
        return new Output(this, container, state);
    }

    public toolFactory(container: GoldenLayout.Container, state: any): any /* typeof Tool */ {
        return new Tool(this, container, state);
    }

    public diffFactory(container: GoldenLayout.Container, state: any): any /* typeof Diff */ {
        return new Diff(this, container, state);
    }

    public toolInputViewFactory(container: GoldenLayout.Container, state: any): any /* typeof ToolInputView */ {
        return new ToolInputView(this, container, state);
    }

    public optViewFactory(container: GoldenLayout.Container, state: any): OptView {
        return new OptView(this, container, state);
    }

    public flagsViewFactory(container: GoldenLayout.Container, state: any): any /* typeof FlagsView */ {
        return new FlagsView(this, container, state);
    }

    public ppViewFactory(container: GoldenLayout.Container, state: any): PreProcessorView {
        return new PreProcessorView(this, container, state);
    }

    public astViewFactory(container: GoldenLayout.Container, state: any): AstView {
        return new AstView(this, container, state);
    }

    public irViewFactory(container: GoldenLayout.Container, state: any): IrView {
        return new IrView(this, container, state);
    }

    public deviceViewFactory(container: GoldenLayout.Container, state: any): any /* typeof DeviceView */ {
        return new DeviceView(this, container, state);
    }

    public gnatDebugTreeViewFactory(container: GoldenLayout.Container, state: any): GnatDebugTreeView {
        return new GnatDebugTreeView(this, container, state);
    }

    public gnatDebugViewFactory(container: GoldenLayout.Container, state: any): GnatDebugView {
        return new GnatDebugView(this, container, state);
    }

    public rustMirViewFactory(container: GoldenLayout.Container, state: any): RustMirView {
        return new RustMirView(this, container, state);
    }

    public rustMacroExpViewFactory(container: GoldenLayout.Container, state: any): RustMacroExpView {
        return new RustMacroExpView(this, container, state);
    }

    public rustHirViewFactory(container: GoldenLayout.Container, state: any): RustHirView {
        return new RustHirView(this, container, state);
    }

    public gccDumpViewFactory(container: GoldenLayout.Container, state: any): any /* typeof GccDumpView */ {
        return new GCCDumpView(this, container, state);
    }

    public cfgViewFactory(container: GoldenLayout.Container, state: any): any /* typeof CfgView */ {
        return new CfgView(this, container, state);
    }

    public conformanceViewFactory(container: GoldenLayout.Container, state: any): any /* typeof ConformanceView */ {
        return new ConformanceView(this, container, state);
    }
}
