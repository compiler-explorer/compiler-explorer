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

import $ from 'jquery';
import * as monaco from 'monaco-editor';
import TomSelect from 'tom-select';

import {ga} from '../analytics.js';
import {Hub} from '../hub.js';
import {Container} from 'golden-layout';
import {MonacoPane} from './pane.js';
import {MonacoPaneState} from './pane.interfaces.js';
import {DiffState, DiffType} from './diff.interfaces.js';
import {ResultLine} from '../../types/resultline/resultline.interfaces.js';
import {CompilationResult} from '../../types/compilation/compilation.interfaces.js';
import {CompilerInfo} from '../../types/compiler.interfaces.js';

type DiffTypeAndExtra = {
    difftype: DiffType;
    extraoption?: string;
};

function encodeSelectizeValue(value: DiffTypeAndExtra): string {
    if (value.extraoption) {
        return value.difftype.toString() + `:${value.extraoption}`;
    } else {
        return value.difftype.toString();
    }
}

function decodeSelectizeValue(value: string): DiffTypeAndExtra {
    const opts = value.split(':');
    if (opts.length > 1) {
        return {
            difftype: parseInt(opts[0]),
            extraoption: opts[1],
        };
    } else {
        return {
            difftype: parseInt(value),
            extraoption: '',
        };
    }
}

type DiffOption = {
    id: string;
    name: string;
};

class DiffStateObject {
    // can be undefined if there are no compilers / executors
    id?: number | string;
    model: monaco.editor.ITextModel;
    compiler: CompilerEntry | null;
    result: CompilationResult | null;
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
        this.result = null;
        this.difftype = difftype;
        this.extraoption = extraoption;
    }

    update(id: number | string, compiler, result: CompilationResult) {
        if (this.id !== id) return false;
        this.compiler = compiler;
        this.result = result;
        this.refresh();

        return true;
    }

    refresh() {
        let output: ResultLine[] = [];
        if (this.result) {
            switch (this.difftype) {
                case DiffType.ASM:
                    output = this.result.asm || [];
                    break;
                case DiffType.CompilerStdOut:
                    output = this.result.stdout;
                    break;
                case DiffType.CompilerStdErr:
                    output = this.result.stderr;
                    break;
                case DiffType.ExecStdOut:
                    if (this.result.execResult) output = this.result.execResult.stdout || [];
                    break;
                case DiffType.ExecStdErr:
                    if (this.result.execResult) output = this.result.execResult.stderr || [];
                    break;
                case DiffType.GNAT_ExpandedCode:
                    if (this.result.hasGnatDebugOutput) output = this.result.gnatDebugOutput || [];
                    break;
                case DiffType.GNAT_Tree:
                    if (this.result.hasGnatDebugTreeOutput) output = this.result.gnatDebugTreeOutput || [];
                    break;
                case DiffType.DeviceView:
                    if (this.result.devices && this.extraoption && this.extraoption in this.result.devices) {
                        output = this.result.devices[this.extraoption].asm || [];
                    }
                    break;
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
    id: number | string;
    name: string;
    options: unknown;
    editorId: number;
    treeId: number;
    compiler: CompilerInfo;
};

type SelectizeType = {
    lhs: TomSelect;
    rhs: TomSelect;
    lhsdifftype: TomSelect;
    rhsdifftype: TomSelect;
};

export class Diff extends MonacoPane<monaco.editor.IStandaloneDiffEditor, DiffState> {
    compilers: Record<string | number, CompilerEntry> = {};
    lhs: DiffStateObject;
    rhs: DiffStateObject;
    selectize: SelectizeType = {} as any; // will be filled in by the constructor
    constructor(hub: Hub, container: Container, state: MonacoPaneState & DiffState) {
        super(hub, container, state);

        this.lhs = new DiffStateObject(
            state.lhs,
            monaco.editor.createModel('', 'asm'),
            state.lhsdifftype || DiffType.ASM,
            state.lhsextraoption,
        );
        this.rhs = new DiffStateObject(
            state.rhs,
            monaco.editor.createModel('', 'asm'),
            state.rhsdifftype || DiffType.ASM,
            state.rhsextraoption,
        );
        this.editor.setModel({original: this.lhs.model, modified: this.rhs.model});

        this.domRoot[0].querySelectorAll('.difftype-picker').forEach(picker => {
            if (!(picker instanceof HTMLSelectElement)) {
                throw new Error('.difftype-picker is not an HTMLSelectElement');
            }

            const diffableOptions = this.getDiffableOptions(picker);

            const instance = new TomSelect(picker, {
                sortField: 'name',
                valueField: 'id',
                labelField: 'name',
                searchField: ['name'],
                options: diffableOptions,
                items: [],
                render: <any>{
                    option: (item, escape) => {
                        return `<div>${escape(item.name)}</div>`;
                    },
                },
                dropdownParent: 'body',
                plugins: ['input_autogrow'],
                onChange: value => {
                    const options = decodeSelectizeValue(value as any as string);
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
                    option: function (item, escape) {
                        const origin = item.editorId !== false ? 'Editor #' + item.editorId : 'Tree #' + item.treeId;
                        return (
                            '<div>' +
                            `<span class="compiler">${escape(item.compiler.name)}</span>` +
                            `<span class="options">${escape(item.options)}</span>` +
                            '<ul class="meta">' +
                            `<li class="editor">${escape(origin)}</li>` +
                            `<li class="compilerId">${escape(getItemDisplayTitle(item))}</li>` +
                            '</ul>' +
                            '</div>'
                        );
                    },
                },
                dropdownParent: 'body',
                plugins: ['input_autogrow'],
                onChange: value => {
                    if (!((value as any as string) in this.compilers)) return;
                    const compiler = this.compilers[value as any as string];
                    if (picker.classList.contains('lhs')) {
                        this.lhs.compiler = compiler;
                        this.lhs.id = compiler.id;
                    } else {
                        this.rhs.compiler = compiler;
                        this.rhs.id = compiler.id;
                    }
                    this.onDiffSelect(compiler.id);
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

        this.eventHub.emit('requestTheme');
        this.eventHub.emit('requestSettings');

        this.updateTitle();
        this.updateCompilers();
    }

    getDiffableOptions(picker?, extraoptions?: DiffOption[]): any[] {
        const options: DiffOption[] = [
            {id: DiffType.ASM.toString(), name: 'Assembly'},
            {id: DiffType.CompilerStdOut.toString(), name: 'Compiler stdout'},
            {id: DiffType.CompilerStdErr.toString(), name: 'Compiler stderr'},
            {id: DiffType.ExecStdOut.toString(), name: 'Execution stdout'},
            {id: DiffType.ExecStdErr.toString(), name: 'Execution stderr'},
            {id: DiffType.GNAT_ExpandedCode.toString(), name: 'GNAT Expanded Code'},
            {id: DiffType.GNAT_Tree.toString(), name: 'GNAT Tree Code'},
        ];

        if (picker && picker.classList) {
            if (picker.classList.contains('lhsdifftype')) {
                if (this.lhs.difftype === DiffType.DeviceView && this.lhs.extraoption) {
                    options.push({
                        id: encodeSelectizeValue({
                            difftype: DiffType.DeviceView,
                            extraoption: this.lhs.extraoption,
                        }),
                        name: this.lhs.extraoption,
                    });
                }
            } else {
                if (this.rhs.difftype === DiffType.DeviceView && this.rhs.extraoption) {
                    options.push({
                        id: encodeSelectizeValue({
                            difftype: DiffType.DeviceView,
                            extraoption: this.rhs.extraoption,
                        }),
                        name: this.rhs.extraoption,
                    });
                }
            }
        }

        if (extraoptions) {
            for (const option of extraoptions) {
                if (!options.find(existing => existing.id === option.id)) options.push(option);
            }
        }

        return options;
    }

    override registerOpeningAnalyticsEvent(): void {
        ga.proxy('send', {
            hitType: 'event',
            eventCategory: 'OpenViewPane',
            eventAction: 'Diff',
        });
    }

    override getInitialHTML() {
        return $('#diff').html();
    }

    override createEditor(editorRoot: HTMLElement) {
        return monaco.editor.createDiffEditor(editorRoot, {
            fontFamily: this.settings.editorsFFont,
            fontLigatures: this.settings.editorsFLigatures,
            scrollBeyondLastLine: true,
            readOnly: true,
        });
    }

    override getPrintName() {
        return '<Unimplemented>';
    }

    onDiffSelect(id: number | string) {
        this.requestResendResult(id);
        this.updateTitle();
        this.updateState();
    }

    override onCompileResult(id: number | string, compiler: CompilerInfo, result: CompilationResult) {
        // both sides must be updated, don't be tempted to rewrite this as
        // var changes = lhs.update() || rhs.update();
        const lhsChanged = this.lhs.update(id, compiler, result);
        const rhsChanged = this.rhs.update(id, compiler, result);
        if (lhsChanged || rhsChanged) {
            this.refreshDiffOptions(id, compiler, result);
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
        this.eventHub.on('executeResult', this.onExecuteResult, this);
        this.eventHub.on('executor', this.onExecutor, this);
        this.eventHub.on('executorClose', this.onExecutorClose, this);
    }

    requestResendResult(id: number | string) {
        if (typeof id === 'string') {
            const p = id.indexOf('_exec');
            if (p !== -1) {
                const execId = parseInt(id.substr(0, p));
                this.eventHub.emit('resendExecution', execId);
            }
        } else {
            this.eventHub.emit('resendCompilation', id);
        }
    }

    refreshDiffOptions(id: number | string, compiler: CompilerInfo, result: CompilationResult) {
        const lhsextraoptions: DiffOption[] = [];
        const rhsextraoptions: DiffOption[] = [];

        if (result.devices) {
            for (const devicename in result.devices) {
                lhsextraoptions.push({
                    id: encodeSelectizeValue({difftype: DiffType.DeviceView, extraoption: devicename}),
                    name: devicename,
                });
                rhsextraoptions.push({
                    id: encodeSelectizeValue({difftype: DiffType.DeviceView, extraoption: devicename}),
                    name: devicename,
                });
            }
        }

        const lhsoptions = this.getDiffableOptions(this.selectize.lhs, lhsextraoptions);
        this.selectize.lhsdifftype.addOptions(lhsoptions);

        const rhsoptions = this.getDiffableOptions(this.selectize.rhs, rhsextraoptions);
        this.selectize.rhsdifftype.addOptions(rhsoptions);
    }

    override onCompiler(
        id: number | string,
        compiler: CompilerInfo | null,
        options: unknown,
        editorId: number,
        treeId: number,
    ) {
        if (!compiler) return;
        options = options || '';
        let name = compiler.name + ' ' + options;
        // TODO: tomselect doesn't play nicely with CSS tricks for truncation; this is the best I can do
        const maxLength = 30;
        if (name.length > maxLength - 3) name = name.substring(0, maxLength - 3) + '...';
        this.compilers[id] = {
            id: id,
            name: name,
            options: options,
            editorId: editorId,
            treeId: treeId,
            compiler: compiler,
        };
        if (this.lhs.id === undefined) {
            this.lhs.compiler = this.compilers[id];
            this.lhs.id = id;
            this.onDiffSelect(id);
        }
        if (this.rhs.id === undefined) {
            this.rhs.compiler = this.compilers[id];
            this.rhs.id = id;
            this.onDiffSelect(id);
        }
        this.updateCompilers();
    }

    onExecutor(id: number, compiler: CompilerInfo | null, options: string, editorId: number, treeId: number) {
        this.onCompiler(id + '_exec', compiler, options, editorId, treeId);
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
            if (this.lhs.compiler && this.rhs.compiler) {
                return `${this.lhs.compiler.name} vs ${this.rhs.compiler.name}`;
            }
        }
        return '';
    }

    updateCompilersFor(selectize: TomSelect, id: number | string) {
        selectize.clearOptions();
        for (const [_, compiler] of Object.entries(this.compilers)) {
            selectize.addOption(compiler);
        }
        if (id in this.compilers) {
            selectize.setValue(id.toString());
        }
    }

    updateCompilers() {
        if (this.lhs.id) this.updateCompilersFor(this.selectize.lhs, this.lhs.id);
        if (this.rhs.id) this.updateCompilersFor(this.selectize.rhs, this.rhs.id);

        this.selectize.lhsdifftype.setValue(
            encodeSelectizeValue({
                difftype: this.lhs.difftype || DiffType.ASM,
                extraoption: this.lhs.extraoption || '',
            }),
        );
        this.selectize.rhsdifftype.setValue(
            encodeSelectizeValue({
                difftype: this.rhs.difftype || DiffType.ASM,
                extraoption: this.rhs.extraoption || '',
            }),
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
