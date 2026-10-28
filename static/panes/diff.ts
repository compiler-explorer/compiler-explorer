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

import {Container} from 'golden-layout';
import $ from 'jquery';
import * as monaco from 'monaco-editor';
import TomSelect from 'tom-select';

import {CompilationResult} from '../../types/compilation/compilation.interfaces.js';
import {CompilerInfo} from '../../types/compiler.interfaces.js';
import {ResultLine} from '../../types/resultline/resultline.interfaces.js';
import {ToolResult} from '../../types/tool.interfaces.js';
import {Hub} from '../hub.js';
import {replaceOptions} from '../tom-select-utils.js';
import {DiffState, DiffType} from './diff.interfaces.js';
import {MonacoPaneState} from './pane.interfaces.js';
import {MonacoPane} from './pane.js';

type DiffTypeAndExtra = {
    difftype: DiffType;
    extraoption?: string;
};

function encodeSelectizeValue(value: DiffTypeAndExtra): string {
    if (value.extraoption) {
        return value.difftype.toString() + `:${value.extraoption}`;
    }
    return value.difftype.toString();
}

function decodeSelectizeValue(value: string): DiffTypeAndExtra {
    const opts = value.split(':');
    if (opts.length > 1) {
        return {
            difftype: Number.parseInt(opts[0], 10),
            extraoption: opts[1],
        };
    }
    return {
        difftype: Number.parseInt(value, 10),
        extraoption: '',
    };
}

function getToolResults(result: CompilationResult | undefined): ToolResult[] {
    return result?.tools ?? result?.result?.tools ?? [];
}

function isSourceEntryId(id: number | string | undefined): boolean {
    return typeof id === 'string' && id.startsWith('source_');
}

function getInitialDiffType(id: number | string | undefined, difftype?: DiffType): DiffType {
    if (isSourceEntryId(id)) return DiffType.Source;

    return difftype && difftype !== DiffType.Source ? difftype : DiffType.ASM;
}

type DiffOption = {
    id: string;
    name: string;
};

export class DiffStateObject {
    // can be undefined if there are no compilers / executors
    id?: number | string;
    model: monaco.editor.ITextModel;
    compiler: CompilerEntry | null;
    source: SourceEntry | null;
    result?: CompilationResult;
    difftype: DiffType;
    extraoption?: string;

    constructor(
        id: number | string | undefined,
        model: monaco.editor.ITextModel,
        difftype: DiffType,
        extraoption?: string,
    ) {
        this.id = id;
        this.model = model;
        this.compiler = null;
        this.source = null;
        this.result = undefined;
        this.difftype = difftype;
        this.extraoption = extraoption;
    }

    update(id: number | string, compiler: CompilerInfo, result: CompilationResult) {
        if (this.id !== id) return false;

        // Handle the case where compiler hasn't been initialized yet
        // In a race condition, we might receive results before the compiler is registered
        if (!this.compiler) {
            // Ignore the update - the result will be requested again when the compiler is registered
            return false;
        }

        this.compiler.compiler = compiler;
        this.result = result;
        this.refresh();

        return true;
    }

    refresh() {
        if (this.difftype === DiffType.Source) {
            this.model.setValue(this.source?.source ?? '');
            return;
        }

        let output: {text: string}[] = [];
        if (this.result) {
            switch (this.difftype) {
                case DiffType.ASM:
                    output = this.result.asm
                        ? (this.result.asm as ResultLine[])
                        : this.result.result?.asm
                          ? (this.result.result?.asm as ResultLine[])
                          : [];
                    break;
                case DiffType.CompilerStdOut:
                    output = this.result.stdout;
                    break;
                case DiffType.CompilerStdErr:
                    output = this.result.stderr;
                    break;
                case DiffType.ExecStdOut:
                    if (this.result.execResult) {
                        output = this.result.execResult.stdout;
                    } else {
                        output = [{text: "<activate 'Output...' → 'Execute the code' in this compiler's pane>"}];
                    }
                    break;
                case DiffType.ExecStdErr:
                    if (this.result.execResult) {
                        output = this.result.execResult.stderr;
                    } else {
                        output = [{text: "<activate 'Output...' → 'Execute the code' in this compiler's pane>"}];
                    }
                    break;
                case DiffType.GNAT_ExpandedCode:
                    output = this.result.gnatDebugOutput || [
                        {text: "<select 'Add new...' → 'GNAT Debug Expanded Code' in this compiler's pane>"},
                    ];
                    break;
                case DiffType.GNAT_Tree:
                    output = this.result.gnatDebugTreeOutput || [
                        {text: "<select 'Add new...' → 'GNAT Debug Tree' in this compiler's pane>"},
                    ];
                    break;
                case DiffType.DeviceView:
                    if (this.result.devices && this.extraoption && this.extraoption in this.result.devices) {
                        output = this.result.devices[this.extraoption].asm as ResultLine[];
                    }
                    break;
                case DiffType.AstOutput:
                    output = this.result.astOutput || [{text: "<select 'Add new...' → 'AST' in this compiler's pane>"}];
                    break;
                case DiffType.IrOutput:
                    output = this.result.irOutput?.asm || [
                        {text: "<select 'Add new...' → 'LLVM IR' in this compiler's pane>"},
                    ];
                    break;
                case DiffType.RustMirOutput:
                    output = this.result.rustMirOutput || [
                        {text: "<select 'Add new...' → 'Rust MIR' in this compiler's pane>"},
                    ];
                    break;
                case DiffType.RustMacroExpOutput:
                    output = this.result.rustMacroExpOutput || [
                        {text: "<select 'Add new...' → 'Rust Macro Expansion' in this compiler's pane>"},
                    ];
                    break;
                case DiffType.RustHirOutput:
                    output = this.result.rustHirOutput || [
                        {text: "<select 'Add new...' → 'Rust HIR' in this compiler's pane>"},
                    ];
                    break;
                case DiffType.ClojureMacroExpOutput:
                    output = this.result.clojureMacroExpOutput || [
                        {text: "<select 'Add new...' → 'Clojure Macro Expansion' in this compiler's pane>"},
                    ];
                    break;
                case DiffType.YulOutput:
                    output = this.result.yulOutput || [
                        {text: "<select 'Add new...' → 'Yul (Solidity IR)' in this compiler's pane>"},
                    ];
                    break;
                case DiffType.LeanCOutput:
                    output = this.result.leanCOutput || [
                        {text: "<select 'Add new...' → 'Lean C Output' in this compiler's pane>"},
                    ];
                    break;
                case DiffType.ToolOutput: {
                    const tool = getToolResults(this.result).find(result => result.id === this.extraoption);
                    output = tool
                        ? [...(tool.stdout ?? []), ...(tool.stderr ?? [])]
                        : [{text: "<select 'Add tool...' → this tool in this compiler's pane>"}];
                    break;
                }
            }
        }
        this.model.setValue(output.map(x => x.text).join('\n'));
    }
}

function getItemDisplayTitle(item) {
    if (typeof item.id === 'string') {
        const p = item.id.indexOf('_exec');
        if (p !== -1) {
            return 'Executor #' + item.id.substr(0, p);
        }
    }

    return 'Compiler #' + item.id;
}

type CompilerEntry = {
    kind: 'compiler';
    id: number | string;
    name: string;
    options: unknown;
    editorId: number;
    treeId: number;
    sourceName: string;
    compiler: CompilerInfo;
};

type SourceEntry = {
    kind: 'source';
    id: string;
    editorId: number;
    name: string;
    source: string;
};

type DiffEntry = CompilerEntry | SourceEntry;

type SelectizeType = {
    lhs: TomSelect;
    rhs: TomSelect;
    lhsdifftype: TomSelect;
    rhsdifftype: TomSelect;
};

export class Diff extends MonacoPane<monaco.editor.IStandaloneDiffEditor, DiffState> {
    compilers: Record<string | number, CompilerEntry> = {};
    sources: Record<string, SourceEntry> = {};
    lhs: DiffStateObject;
    rhs: DiffStateObject;
    selectize: SelectizeType;
    constructor(hub: Hub, container: Container, state: MonacoPaneState & DiffState) {
        super(hub, container, state);

        // note: keep this hacky line, properties will be filled in later (1 by 1)
        this.selectize = {} as any;

        this.lhs = new DiffStateObject(
            state.lhs,
            monaco.editor.createModel('', 'asm'),
            getInitialDiffType(state.lhs, state.lhsdifftype),
            state.lhsextraoption,
        );
        this.rhs = new DiffStateObject(
            state.rhs,
            monaco.editor.createModel('', 'asm'),
            getInitialDiffType(state.rhs, state.rhsdifftype),
            state.rhsextraoption,
        );
        this.editor.setModel({original: this.lhs.model, modified: this.rhs.model});

        this.domRoot[0].querySelectorAll('.difftype-picker').forEach(picker => {
            if (!(picker instanceof HTMLSelectElement)) {
                throw new Error('.difftype-picker is not an HTMLSelectElement');
            }

            const side = picker.classList.contains('lhsdifftype') ? this.lhs : this.rhs;

            const instance = new TomSelect(picker, {
                sortField: 'name',
                valueField: 'id',
                labelField: 'name',
                searchField: ['name'],
                options: this.getDiffTypeOptions(side),
                items: [],
                render: <any>{
                    option: (item, escapeHtml) => {
                        return `<div>${escapeHtml(item.name)}</div>`;
                    },
                },
                dropdownParent: 'body',
                plugins: ['input_autogrow'],
                onChange: value => {
                    const options = decodeSelectizeValue(value as string);
                    if (picker.classList.contains('lhsdifftype')) {
                        this.lhs.difftype = options.difftype;
                        this.lhs.extraoption = options.extraoption;
                        this.lhs.refresh();
                    } else {
                        this.rhs.difftype = options.difftype;
                        this.rhs.extraoption = options.extraoption;
                        this.rhs.refresh();
                    }
                    this.updateState();
                },
            });

            if (picker.classList.contains('lhsdifftype')) {
                this.selectize.lhsdifftype = instance;
            } else {
                this.selectize.rhsdifftype = instance;
            }
        });

        this.domRoot[0].querySelectorAll('.diff-picker').forEach(picker => {
            if (!(picker instanceof HTMLSelectElement)) {
                throw new Error('.difftype-picker is not an HTMLSelectElement');
            }
            const instance = new TomSelect(picker, {
                sortField: 'name',
                valueField: 'id',
                labelField: 'name',
                searchField: ['name'],
                options: [],
                items: [],
                render: <any>{
                    item: (item, escapeHtml) => {
                        return `<div>${escapeHtml(item.name)}</div>`;
                    },
                    option: (item, escapeHtml) => {
                        if (item.kind === 'source') {
                            return (
                                '<div>' +
                                `<span class="compiler">${escapeHtml(item.name)}</span>` +
                                '<ul class="meta">' +
                                `<li class="editor">Editor #${escapeHtml(item.editorId.toString())}</li>` +
                                '<li class="compilerId">Source</li>' +
                                '</ul>' +
                                '</div>'
                            );
                        }

                        const origin =
                            item.sourceName || (item.editorId ? 'Editor #' + item.editorId : 'Tree #' + item.treeId);
                        return (
                            '<div>' +
                            `<span class="source">${escapeHtml(origin)}</span>` +
                            `<span class="options">${escapeHtml(item.options || '')}</span>` +
                            '<ul class="meta">' +
                            `<li class="compilerName">${escapeHtml(item.compiler.name)}</li>` +
                            `<li class="compilerId">${escapeHtml(getItemDisplayTitle(item))}</li>` +
                            '</ul>' +
                            '</div>'
                        );
                    },
                },
                dropdownParent: 'body',
                plugins: ['input_autogrow'],
                onChange: value => {
                    const entry = this.getDiffEntry(value as string);
                    if (!entry) return;
                    if (picker.classList.contains('lhs')) {
                        this.setSelectedEntry(this.lhs, this.selectize.lhsdifftype, entry);
                    } else {
                        this.setSelectedEntry(this.rhs, this.selectize.rhsdifftype, entry);
                    }
                    this.onDiffSelect(entry);
                },
            });

            if (picker.classList.contains('lhs')) {
                this.selectize.lhs = instance;
            } else {
                this.selectize.rhs = instance;
            }
        });

        if (this.lhs.id) this.requestResendResult(this.lhs.id);
        if (this.rhs.id) this.requestResendResult(this.rhs.id);

        this.eventHub.emit('findCompilers');
        this.eventHub.emit('findExecutors');
        this.eventHub.emit('findEditors');

        this.eventHub.emit('requestTheme');
        this.eventHub.emit('requestSettings');

        this.updateTitle();
        this.updateCompilers();
    }

    getDiffTypeOptions(side: DiffStateObject): DiffOption[] {
        if (side.source || side.difftype === DiffType.Source) {
            return [{id: DiffType.Source.toString(), name: 'Source'}];
        }

        const options: DiffOption[] = [
            {id: DiffType.ASM.toString(), name: 'Assembly'},
            {id: DiffType.CompilerStdOut.toString(), name: 'Compiler stdout'},
            {id: DiffType.CompilerStdErr.toString(), name: 'Compiler stderr'},
            {id: DiffType.ExecStdOut.toString(), name: 'Execution stdout'},
            {id: DiffType.ExecStdErr.toString(), name: 'Execution stderr'},
        ];

        const compiler = side.compiler?.compiler;
        if (compiler?.supportsGnatDebugViews) {
            options.push({id: DiffType.GNAT_ExpandedCode.toString(), name: 'GNAT Expanded Code'});
            options.push({id: DiffType.GNAT_Tree.toString(), name: 'GNAT Tree Code'});
        }
        if (compiler?.supportsAstView) {
            options.push({id: DiffType.AstOutput.toString(), name: 'AST'});
        }
        if (compiler?.supportsIrView) {
            options.push({id: DiffType.IrOutput.toString(), name: 'LLVM IR'});
        }
        if (compiler?.supportsRustMirView) {
            options.push({id: DiffType.RustMirOutput.toString(), name: 'Rust MIR'});
        }
        if (compiler?.supportsRustMacroExpView) {
            options.push({id: DiffType.RustMacroExpOutput.toString(), name: 'Rust Macro Expansion'});
        }
        if (compiler?.supportsRustHirView) {
            options.push({id: DiffType.RustHirOutput.toString(), name: 'Rust HIR'});
        }
        if (compiler?.supportsClojureMacroExpView) {
            options.push({id: DiffType.ClojureMacroExpOutput.toString(), name: 'Clojure Macro Expansion'});
        }
        if (compiler?.supportsYulView) {
            options.push({id: DiffType.YulOutput.toString(), name: 'Yul (Solidity IR)'});
        }
        if (compiler?.supportsLeanCView) {
            options.push({id: DiffType.LeanCOutput.toString(), name: 'Lean C Output'});
        }

        // Devices are only known once a result arrives, so keep the selected one on offer until then
        // (e.g. when restoring a layout).
        const devices = Object.keys(side.result?.devices ?? {});
        if (side.difftype === DiffType.DeviceView && side.extraoption && !devices.includes(side.extraoption)) {
            devices.push(side.extraoption);
        }
        for (const device of devices) {
            options.push({
                id: encodeSelectizeValue({difftype: DiffType.DeviceView, extraoption: device}),
                name: device,
            });
        }

        // Like devices, tools are only known once a result arrives.
        const tools = getToolResults(side.result).map(tool => ({id: tool.id, name: tool.name ?? tool.id}));
        if (
            side.difftype === DiffType.ToolOutput &&
            side.extraoption &&
            !tools.some(tool => tool.id === side.extraoption)
        ) {
            tools.push({id: side.extraoption, name: side.extraoption});
        }
        for (const tool of tools) {
            options.push({
                id: encodeSelectizeValue({difftype: DiffType.ToolOutput, extraoption: tool.id}),
                name: tool.name,
            });
        }

        return options;
    }

    isDiffTypeOffered(side: DiffStateObject): boolean {
        const selected = encodeSelectizeValue({difftype: side.difftype, extraoption: side.extraoption});
        return this.getDiffTypeOptions(side).some(option => option.id === selected);
    }

    getDiffEntry(id: string): DiffEntry | undefined {
        return this.compilers[id] ?? this.sources[id];
    }

    setSelectedEntry(side: DiffStateObject, difftypePicker: TomSelect, entry: DiffEntry) {
        const idChanged = side.id !== entry.id;
        const previousKind = side.source ? 'source' : side.compiler ? 'compiler' : undefined;
        const kindChanged = previousKind !== entry.kind;
        side.id = entry.id;
        if (idChanged) {
            side.result = undefined;
        }

        let difftypeChanged = false;
        if (entry.kind === 'source') {
            side.compiler = null;
            side.source = entry;
            side.difftype = DiffType.Source;
            side.extraoption = '';
        } else {
            side.compiler = entry;
            side.source = null;
            if (side.difftype === DiffType.Source) {
                side.difftype = DiffType.ASM;
            }
            if (!this.isDiffTypeOffered(side)) {
                side.difftype = DiffType.ASM;
                side.extraoption = '';
                difftypeChanged = true;
            }
        }

        if (idChanged || kindChanged || difftypeChanged) {
            this.updateDiffTypeOptionsFor(side, difftypePicker);
        }
        side.refresh();
    }

    updateDiffTypeOptionsFor(side: DiffStateObject, difftypePicker: TomSelect) {
        // Drop every option, including the selected one the default filter would keep, so a type the
        // new compiler can't produce doesn't linger in the list.
        replaceOptions(difftypePicker, this.getDiffTypeOptions(side), () => false);
        difftypePicker.setValue(
            encodeSelectizeValue({
                difftype: side.difftype || DiffType.ASM,
                extraoption: side.extraoption || '',
            }),
            true,
        );
    }

    clearSelectedEntry(side: DiffStateObject, difftypePicker: TomSelect, picker: TomSelect) {
        side.id = undefined;
        side.compiler = null;
        side.source = null;
        side.result = undefined;
        side.difftype = DiffType.ASM;
        side.extraoption = '';
        side.model.setValue('');
        picker.clear(true);
        this.updateDiffTypeOptionsFor(side, difftypePicker);
    }

    override getInitialHTML() {
        return $('#diff').html();
    }

    override createEditor(editorRoot: HTMLElement) {
        this.editor = monaco.editor.createDiffEditor(editorRoot, {
            fontFamily: this.settings.editorsFFont,
            fontLigatures: this.settings.editorsFLigatures,
            scrollBeyondLastLine: true,
            readOnly: true,
            fixedOverflowWidgets: true,
        });
    }

    override getPrintName() {
        return '<Unimplemented>';
    }

    onDiffSelect(entry: DiffEntry) {
        if (entry.kind === 'compiler') {
            this.requestResendResult(entry.id);
        }
        this.updateTitle();
        this.updateState();
    }

    override onCompileResult(id: number | string, compiler: CompilerInfo, result: CompilationResult) {
        // both sides must be updated, don't be tempted to rewrite this as
        // var changes = lhs.update() || rhs.update();
        const lhsChanged = this.lhs.update(id, compiler, result);
        const rhsChanged = this.rhs.update(id, compiler, result);
        if (lhsChanged) this.updateDiffTypeOptionsFor(this.lhs, this.selectize.lhsdifftype);
        if (rhsChanged) this.updateDiffTypeOptionsFor(this.rhs, this.selectize.rhsdifftype);
        if (lhsChanged || rhsChanged) {
            this.updateTitle();
        }
    }

    onExecuteResult(id: number, compiler: CompilerInfo, result: CompilationResult) {
        const compileResult: any = Object.assign({}, result.buildResult);
        compileResult.execResult = {
            code: result.code,
            stdout: result.stdout,
            stderr: result.stderr,
        };

        this.onCompileResult(id + '_exec', compiler, compileResult);
    }

    override registerCallbacks() {
        this.eventHub.on('editor', this.onEditor, this);
        this.eventHub.on('editorChange', this.onEditorChange, this);
        this.eventHub.on('editorClose', this.onEditorClose, this);
        this.eventHub.on('executeResult', this.onExecuteResult, this);
        this.eventHub.on('executor', this.onExecutor, this);
        this.eventHub.on('executorClose', this.onExecutorClose, this);
        this.eventHub.on('renamePane', this.onRenamePane, this);
    }

    requestResendResult(id: number | string) {
        if (typeof id === 'string') {
            const p = id.indexOf('_exec');
            if (p !== -1) {
                const execId = Number.parseInt(id.substr(0, p), 10);
                this.eventHub.emit('resendExecution', execId);
            }
        } else {
            this.eventHub.emit('resendCompilation', id);
        }
    }

    override onCompiler(
        id: number | string,
        compiler: CompilerInfo | null,
        options: unknown,
        editorId: number,
        treeId: number,
        sourceName?: string,
    ) {
        if (!compiler) return;
        options = options || '';
        const source = sourceName || (editorId ? 'Editor #' + editorId : 'Tree #' + treeId);
        const name = [source, compiler.name, options].filter(Boolean).join(' ');
        const compilerEntry: CompilerEntry = {
            kind: 'compiler',
            id: id,
            name: name,
            options: options,
            editorId: editorId,
            treeId: treeId,
            sourceName: source,
            compiler: compiler,
        };
        this.compilers[id] = compilerEntry;

        const lhsWasSelected = this.lhs.id === id;
        const rhsWasSelected = this.rhs.id === id;
        const lhsWasUnselected = this.lhs.id === undefined;
        const rhsWasUnselected = this.rhs.id === undefined;
        if (lhsWasSelected || lhsWasUnselected) {
            this.setSelectedEntry(this.lhs, this.selectize.lhsdifftype, compilerEntry);
        }
        if (rhsWasSelected || rhsWasUnselected) {
            this.setSelectedEntry(this.rhs, this.selectize.rhsdifftype, compilerEntry);
        }

        if (lhsWasUnselected || rhsWasUnselected) {
            this.onDiffSelect(compilerEntry);
        } else if (lhsWasSelected || rhsWasSelected) {
            this.updateTitle();
            this.updateState();
        }
        this.updateCompilers();
    }

    getSourceId(editorId: number): string {
        return `source_${editorId}`;
    }

    onEditor(editorId: number, source: string, name: string) {
        const id = this.getSourceId(editorId);
        this.sources[id] = {
            kind: 'source',
            id,
            editorId,
            name,
            source,
        };
        const entry = this.sources[id];

        if (this.lhs.id === id) {
            this.setSelectedEntry(this.lhs, this.selectize.lhsdifftype, entry);
        }
        if (this.rhs.id === id) {
            this.setSelectedEntry(this.rhs, this.selectize.rhsdifftype, entry);
        }

        this.updateCompilers();
        this.updateTitle();
    }

    onEditorChange(editorId: number, source: string) {
        const id = this.getSourceId(editorId);
        const entry = this.sources[id];
        if (!entry) return;

        entry.source = source;

        if (this.lhs.id === id) {
            this.lhs.refresh();
        }
        if (this.rhs.id === id) {
            this.rhs.refresh();
        }
    }

    onEditorClose(editorId: number) {
        const id = this.getSourceId(editorId);
        delete this.sources[id];

        if (this.lhs.id === id) {
            this.clearSelectedEntry(this.lhs, this.selectize.lhsdifftype, this.selectize.lhs);
        }
        if (this.rhs.id === id) {
            this.clearSelectedEntry(this.rhs, this.selectize.rhsdifftype, this.selectize.rhs);
        }

        this.updateCompilers();
        this.updateTitle();
        this.updateState();
    }

    onRenamePane() {
        this.eventHub.emit('findCompilers');
        this.eventHub.emit('findExecutors');
        this.eventHub.emit('findEditors');
    }

    onExecutor(
        id: number,
        compiler: CompilerInfo | null,
        options: string,
        editorId: number,
        treeId: number,
        sourceName?: string,
    ) {
        this.onCompiler(id + '_exec', compiler, options, editorId, treeId, sourceName);
    }

    override onCompilerClose(id: number | string) {
        delete this.compilers[id];
        this.updateCompilers();
    }

    onExecutorClose(id: number) {
        this.onCompilerClose(id + '_exec');
    }

    override getDefaultPaneName() {
        return 'Diff Viewer';
    }

    override getPaneTag() {
        // this gets called during the super's constructor before lhs/rhs have been initialized
        if ((this.lhs as any) !== undefined && (this.rhs as any) !== undefined) {
            const lhsName = this.lhs.compiler?.name ?? this.lhs.source?.name;
            const rhsName = this.rhs.compiler?.name ?? this.rhs.source?.name;
            if (lhsName && rhsName) {
                return `${lhsName} vs ${rhsName}`;
            }
        }
        return '';
    }

    updateCompilersFor(selectize: TomSelect, id?: number | string) {
        selectize.clear(true);
        selectize.clearOptions();
        for (const [_, source] of Object.entries(this.sources)) {
            const optionId = source.id.toString();
            selectize.addOption(source);
            selectize.updateOption(optionId, source);
        }
        for (const [_, compiler] of Object.entries(this.compilers)) {
            const optionId = compiler.id.toString();
            selectize.addOption(compiler);
            selectize.updateOption(optionId, compiler);
        }
        selectize.refreshOptions(false);
        if (id !== undefined) {
            const selectedId = id.toString();
            if (this.getDiffEntry(selectedId)) {
                selectize.setValue(selectedId, true);
            }
        }
    }

    updateCompilers() {
        this.updateCompilersFor(this.selectize.lhs, this.lhs.id);
        this.updateCompilersFor(this.selectize.rhs, this.rhs.id);

        this.updateDiffTypeValue(this.lhs, this.selectize.lhsdifftype);
        this.updateDiffTypeValue(this.rhs, this.selectize.rhsdifftype);
    }

    updateDiffTypeValue(side: DiffStateObject, difftypePicker: TomSelect) {
        if (side.source || side.difftype === DiffType.Source) {
            this.updateDiffTypeOptionsFor(side, difftypePicker);
            return;
        }

        difftypePicker.setValue(
            encodeSelectizeValue({
                difftype: side.difftype || DiffType.ASM,
                extraoption: side.extraoption || '',
            }),
            true,
        );
    }

    override getCurrentState() {
        const parent = super.getCurrentState();
        return {
            lhs: this.lhs.id,
            rhs: this.rhs.id,
            lhsdifftype: this.lhs.difftype,
            rhsdifftype: this.rhs.difftype,
            lhsextraoption: this.lhs.extraoption,
            rhsextraoption: this.rhs.extraoption,
            ...parent,
        };
    }

    close() {
        this.eventHub.unsubscribe();
        this.editor.dispose();
    }
}
