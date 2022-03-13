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

import _ from 'underscore';
import * as monaco from 'monaco-editor';
import { Container } from 'golden-layout';

import { MonacoPane } from './pane';
import { OptState } from './opt-view.interfaces';
import { MonacoPaneState } from './pane.interfaces';

import { ga } from '../analytics';
import { extendConfig } from '../monaco-config';
import { applyColours } from '../colour';

import { PaneRenaming } from '../widgets/pane-renaming';

export class Opt extends MonacoPane<monaco.editor.IStandaloneCodeEditor, OptState> {
    linkedFadeTimeoutId = -1;
    irCode: any[] = [];
    colours: any[] = [];
    decorations: any = {};
    currentDecorations: string[] = [];
    isCompilerSupported = false;
    cursorSelectionThrottledFunction: (e: any) => void;
    source: any;

    constructor(hub: any, container: Container, state: OptState & MonacoPaneState) {
        console.log(":P4");
        super(hub, container, state);
        this.source = state.source || '';
        this.currentDecorations = [];
    
        this.isAwaitingInitialResults = false;
    
        if (state && state.optOutput) {
            this.showOptResults(state.optOutput);
        }
        this.eventHub.emit('optViewOpened', this.compilerInfo.compilerId);
    }

    override getInitialHTML(): string {
        return $('#opt').html();
    }

    override createEditor(editorRoot: HTMLElement): monaco.editor.IStandaloneCodeEditor {
        return monaco.editor.create(editorRoot, extendConfig({
            language: 'plaintext',
            readOnly: true,
            glyphMargin: true,
        }));
    }

    override registerOpeningAnalyticsEvent() {
        ga.proxy('send', {
            hitType: 'event',
            eventCategory: 'OpenViewPane',
            eventAction: 'Opt',
        });
    }
    
    override registerCallbacks() {
        this.fontScale.on('change', _.bind(this.updateState, this));
        this.paneRenaming.on('renamePane', this.updateState.bind(this));
    
        this.eventHub.on('compileResult', this.onCompileResult, this);
        this.eventHub.on('compiler', this.onCompiler, this);
        this.eventHub.on('settingsChange', this.onSettingsChange, this);
        this.eventHub.on('resize', this.resize, this);
        this.container.on('destroy', this.close, this);
        this.eventHub.emit('requestSettings');
        this.eventHub.emit('findCompilers');
    
        this.container.on('resize', this.resize, this);
        this.container.on('shown', this.resize, this);
    
        this.cursorSelectionThrottledFunction =
            _.throttle(_.bind(this.onDidChangeCursorSelection, this), 500);
        this.editor.onDidChangeCursorSelection(e => {
            this.cursorSelectionThrottledFunction(e);
        });
    }
    
    onCompileResult(id, compiler, result) {
        console.log("onCompileResult", id, compiler, result, this.compilerInfo.compilerId, this.isCompilerSupported);
        if (this.compilerInfo.compilerId !== id || !this.isCompilerSupported) return;
        this.source = result.source;
        this.editor.setValue(this.source);
        if (result.hasOptOutput) {
            this.showOptResults(result.optOutput);
        }
        // TODO: This is unelegant again. Previously took advantage of fourth argument for the compileResult event.
        const lang = compiler.lang === 'c++' ? 'cpp' : compiler.lang;
        const model = this.editor.getModel();
        if (model != null && this.getCurrentEditorLanguage() !== lang) {
            monaco.editor.setModelLanguage(model, lang);
        }
    
        if (!this.isAwaitingInitialResults) {
            if (this.selection) {
                this.editor.setSelection(this.selection);
                this.editor.revealLinesInCenter(this.selection.startLineNumber,
                    this.selection.endLineNumber);
            }
            this.isAwaitingInitialResults = true;
        }
    }
    
    // Monaco language id of the current editor
    getCurrentEditorLanguage() {
        return this.editor.getModel()?.getLanguageId();
    }
    
    getDefaultPaneName() {
        return 'Opt Viewer';
    }
    
    getPaneTag() {
        if(this.compilerInfo.editorId) {
            return this.compilerInfo.compilerName + ' (Editor #' + this.compilerInfo.editorId + ', Compiler #' + this.compilerInfo.compilerId + ')';
        } else {
            return this.compilerInfo.compilerName + ' (Tree #' + this.compilerInfo.treeId + ', Compiler #' + this.compilerInfo.compilerId + ')';
        }
    }
    
    getPaneName() {
        return this.paneName ? this.paneName : this.getDefaultPaneName() + ' ' + this.getPaneTag();
    }
    
    updateTitle() {
        this.container.setTitle(_.escape(this.getPaneName()));
    }
    
    getDisplayableOpt(optResult) {
        return {
            value: '**' + optResult.optType + '** - ' + optResult.displayString,
            isTrusted: false,
        };
    }
    
    showOptResults(results) {
        console.log("showOptResults");
        var opt: any[] = [];
    
        results = _.filter(results, function (x) {
            return x.DebugLoc !== undefined;
        });
    
        results = _.groupBy(results, function (x) {
            return x.DebugLoc.Line;
        });
    
        _.mapObject(results, (value, key) => {
            var linenumber = Number(key);
            var className = value.reduce(function (acc, x) {
                if (x.optType === 'Missed' || acc === 'Missed') {
                    return 'Missed';
                } else if (x.optType === 'Passed' || acc === 'Passed') {
                    return 'Passed';
                }
                return x.optType;
            }, '');
            var contents = value.map(this.getDisplayableOpt);
            opt.push({
                range: new monaco.Range(linenumber, 1, linenumber, Infinity),
                options: {
                    isWholeLine: true,
                    glyphMarginClassName: 'opt-decoration.' + className.toLowerCase(),
                    hoverMessage: contents,
                    glyphMarginHoverMessage: contents,
                },
            });
        });
    
        this.currentDecorations = this.editor.deltaDecorations(this.currentDecorations, opt);
    }
    
    onCompiler(id, compiler) {
        console.log("onCompiler", id, this.compilerInfo.compilerId);
        if (id === this.compilerInfo.compilerId) {
            this.compilerInfo.compilerName = compiler ? compiler.name : '';
            this.updateTitle();
            this.isCompilerSupported = compiler ? compiler.supportsOptOutput : false;
            console.log(this.isCompilerSupported, compiler);
            if (!this.isCompilerSupported) {
                this.editor.setValue('<OPT output is not supported for this compiler>');
            }
        }
    }
    
    updateState() {
        this.container.setState(this.currentState());
    }
    
    currentState() {
        var state = {
            id: this.compilerInfo.compilerId,
            editorid: this.compilerInfo.editorId,
            treeid: this.compilerInfo.treeId,
            selection: this.selection,
        };
        this.paneRenaming.addState(state);
        this.fontScale.addState(state);
        return state;
    }
    
    close() {
        this.eventHub.unsubscribe();
        this.eventHub.emit('optViewClosed', this.compilerInfo.compilerId);
        this.editor.dispose();
    }
    
    onSettingsChange(newSettings) {
        this.editor.updateOptions({
            contextmenu: newSettings.useCustomContextMenu,
            minimap: {
                enabled: newSettings.showMinimap,
            },
            fontFamily: newSettings.editorsFFont,
            fontLigatures: newSettings.editorsFLigatures,
        });
    }
    
    onDidChangeCursorSelection(e) {
        if (this.isAwaitingInitialResults) {
            this.selection = e.selection;
            this.updateState();
        }
    }
};
