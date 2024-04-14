// Copyright (c) 2017, Jared Wyles
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
import _ from 'underscore';
import * as monaco from 'monaco-editor';
import {Container} from 'golden-layout';

import {MonacoPane} from './pane.js';
import {OptState, OptCodeEntry} from './opt-view.interfaces.js';
import {MonacoPaneState} from './pane.interfaces.js';

import {ga} from '../analytics.js';
import {extendConfig} from '../monaco-config.js';
import {Hub} from '../hub.js';
import {CompilationResult} from '../compilation/compilation.interfaces.js';
import {CompilerInfo} from '../compiler.interfaces.js';
import {unwrap} from '../assert.js';
import {Toggles} from '../widgets/toggles.js';

export class Opt extends MonacoPane<monaco.editor.IStandaloneCodeEditor, OptState> {
    // Note: bool | undef here instead of just bool because of an issue with field initialization order
    isCompilerSupported?: boolean;
    filters: Toggles;

    // Keep optRemarks as state, to avoid triggerring a recompile when options change
    optRemarks: OptCodeEntry[];

    constructor(hub: Hub, container: Container, state: OptState & MonacoPaneState) {
        super(hub, container, state);
        if (state.optOutput) {
            this.optRemarks = state.optOutput;
            this.showOptRemarks();
        }

        this.eventHub.emit('optViewOpened', this.compilerInfo.compilerId);
    }

    override getInitialHTML(): string {
        return $('#opt-view').html();
    }

    override createEditor(editorRoot: HTMLElement): void {
        this.editor = monaco.editor.create(
            editorRoot,
            extendConfig({
                language: 'plaintext',
                readOnly: true,
                glyphMargin: true,
            }),
        );
        this.initDecorations();
    }

    override getPrintName() {
        return 'Opt Remarks';
    }

    override registerOpeningAnalyticsEvent() {
        ga.proxy('send', {
            hitType: 'event',
            eventCategory: 'OpenViewPane',
            eventAction: 'Opt',
        });
    }

    override registerButtons(state: OptState) {
        super.registerButtons(state);
        this.filters = new Toggles(this.domRoot.find('.filters'), state as unknown as Record<string, boolean>);
        this.filters.on('change', this.showOptRemarks.bind(this));
    }

    override registerCallbacks() {
        this.eventHub.emit('requestSettings');
        this.eventHub.emit('findCompilers');

        this.container.on('shown', this.resize, this);

        const cursorSelectionThrottledFunction = _.throttle(this.onDidChangeCursorSelection.bind(this), 500);
        this.editor.onDidChangeCursorSelection(e => {
            cursorSelectionThrottledFunction(e);
        });
    }

    override onCompileResult(id: number, compiler: CompilerInfo, result: CompilationResult) {
        if (this.compilerInfo.compilerId !== id || !this.isCompilerSupported) return;
        this.editor.setValue(unwrap(result.source));
        if (result.hasOptOutput) {
            this.optRemarks = result.optOutput;
            this.showOptRemarks();
        }
        // TODO: This is inelegant again. Previously took advantage of fourth argument for the compileResult event.
        const lang = compiler.lang === 'c++' ? 'cpp' : compiler.lang;
        const model = this.editor.getModel();
        if (model != null && this.getCurrentEditorLanguage() !== lang) {
            monaco.editor.setModelLanguage(model, lang);
        }

        if (!this.isAwaitingInitialResults) {
            if (this.selection) {
                this.editor.setSelection(this.selection);
                this.editor.revealLinesInCenter(this.selection.startLineNumber, this.selection.endLineNumber);
            }
            this.isAwaitingInitialResults = true;
        }
    }

    // Monaco language id of the current editor
    getCurrentEditorLanguage() {
        return this.editor.getModel()?.getLanguageId();
    }

    override getDefaultPaneName() {
        return 'Opt Viewer';
    }

    getDisplayableOpt(optResult: OptCodeEntry) {
        return {
            value: '**' + optResult.optType + '** - ' + optResult.displayString,
            isTrusted: false,
        };
    }

    showOptRemarks() {
        const optDecorations: monaco.editor.IModelDeltaDecoration[] = [];

        const filters = this.filters.get();
        const includeMissed: boolean = filters['filter-missed'];
        const includePassed: boolean = filters['filter-passed'];
        const includeAnalysis: boolean = filters['filter-analysis'];

        const groupedResults = _.groupBy(
            /* eslint-disable-next-line @typescript-eslint/no-unnecessary-condition */ // TODO
            this.optRemarks.filter(x => x.DebugLoc !== undefined),
            x => x.DebugLoc.Line,
        );

        for (const [key, value] of Object.entries(groupedResults)) {
            const linenumber = Number(key);
            const className = value.reduce((acc, x) => {
                if (x.optType === 'Missed' || acc === 'Missed') {
                    return 'Missed';
                } else if (x.optType === 'Passed' || acc === 'Passed') {
                    return 'Passed';
                }
                return x.optType;
            }, '');

            if (className === 'Missed' && !includeMissed) continue;
            if (className === 'Passed' && !includePassed) continue;
            if (className === 'Analysis' && !includeAnalysis) continue;

            const contents = value.map(this.getDisplayableOpt);
            optDecorations.push({
                range: new monaco.Range(linenumber, 1, linenumber, Infinity),
                options: {
                    isWholeLine: true,
                    glyphMarginClassName: 'opt-decoration.' + className.toLowerCase(),
                    hoverMessage: contents,
                    glyphMarginHoverMessage: contents,
                },
            });
        }

        this.editorDecorations.set(optDecorations);
    }

    override onCompiler(id: number, compiler: CompilerInfo | null, options: string, editorId: number, treeId: number) {
        if (id === this.compilerInfo.compilerId) {
            this.compilerInfo.compilerName = compiler ? compiler.name : '';
            this.updateTitle();
            this.isCompilerSupported = compiler ? compiler.supportsOptOutput : false;
            if (!this.isCompilerSupported) {
                this.editor.setValue('<OPT remarks are not supported for this compiler>');
            }
        }
    }

    // Don't do anything for this pane
    override sendPrintData() {}

    override close() {
        this.eventHub.unsubscribe();
        this.eventHub.emit('optViewClosed', this.compilerInfo.compilerId);
        this.editor.dispose();
    }
}
