// Copyright (c) 2017, Simon Brand
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

'use strict';

import _ from 'underscore';
import $ from 'jquery';

import * as monaco from 'monaco-editor';
import { Container } from 'golden-layout';

import { MonacoPane } from './pane';
import { AstState } from './ast-view.interfaces';
import { MonacoPaneState } from './pane.interfaces';

import { ga } from '../analytics';
import { extendConfig } from '../monaco-config';
import { applyColours } from '../colour';

import { PaneRenaming } from '../widgets/pane-renaming';

import * as colour from '../colour';
import * as monacoConfig from '../monaco-config';

type decorationEntry = {
    linkedCode: any[];
}

type sourceLocation = {
    line: number;
    col: number;
};

type astCodeEntry = {
    source: {
        from: sourceLocation,
        to: sourceLocation,
    }
};

export class Ast extends MonacoPane<monaco.editor.IStandaloneCodeEditor, AstState> {
    decorations: decorationEntry = { linkedCode: [] };
    prevDecorations: any[] = [];
    colours: string[] = [];
    astCode: astCodeEntry[] = [];
    linkedFadeTimeoutId: -1 | NodeJS.Timeout;
    constructor(hub: any, container: Container, state: AstState & MonacoPaneState) {
        super(hub, container, state);

        if (state && state.astOutput) {
            this.showAstResults(state.astOutput);
        }
    }

    override getInitialHTML(): string {
        return $('#ast').html();
    }

    override registerOpeningAnalyticsEvent(): void {
        ga.proxy('send', {
            hitType: 'event',
            eventCategory: 'OpenViewPane',
            eventAction: 'Ast',
        });
    }

    override registerCallbacks(): void {
        this.linkedFadeTimeoutId = -1;
        const mouseMoveThrottledFunction = _.throttle(this.onMouseMove.bind(this), 50);
        this.editor.onMouseMove(e => mouseMoveThrottledFunction(e));

        this.fontScale.on('change', _.bind(this.updateState, this));
        this.paneRenaming.on('renamePane', this.updateState.bind(this));

        this.container.on('destroy', this.close, this);

        const onColoursOnCompile = this.eventHub.mediateDependentCalls(this.onColours, this.onCompileResult);

        this.eventHub.on('compileResult', onColoursOnCompile.dependencyProxy, this);
        this.eventHub.on('compiler', this.onCompiler, this);
        this.eventHub.on('colours', onColoursOnCompile.dependentProxy, this);
        this.eventHub.on('panesLinkLine', this.onPanesLinkLine, this);
        this.eventHub.on('compilerClose', this.onCompilerClose, this);
        this.eventHub.on('settingsChange', this.onSettingsChange, this);
        this.eventHub.emit('astViewOpened', this.compilerInfo.compilerId);
        this.eventHub.emit('requestSettings');

        this.container.on('resize', this.resize, this);
        this.container.on('shown', this.resize, this);

        const cursorSelectionThrottledFunction = _.throttle(this.onDidChangeCursorSelection.bind(this), 500);
        this.editor.onDidChangeCursorSelection(_.bind(function (e) {
            cursorSelectionThrottledFunction(e);
        }, this));
    }

    override createEditor(editorRoot: HTMLElement): monaco.editor.IStandaloneCodeEditor {
        return monaco.editor.create(editorRoot, monacoConfig.extendConfig({
            language: 'plaintext',
            readOnly: true,
            glyphMargin: true,
            lineNumbersMinChars: 3,
        }));
    }

    getCurrentEditorLanguage() {
        return this.editor.getModel()?.getLanguageId();
    }
    
    override getDefaultPaneName() {
        return 'Ast Viewer';
    }
    
    getDisplayableAst(astResult) {
        return '**' + astResult.astType + '** - ' + astResult.displayString;
    }

    onMouseMove(e: monaco.editor.IEditorMouseEvent) {
        if (e === null || e.target === null || e.target.position === null) return;
        if (this.settings.hoverShowSource === true && this.astCode) {
            this.clearLinkedLines();
            const hoverCode = this.astCode[e.target.position.lineNumber - 1];
            if (hoverCode) {
                let sourceLine = -1;
                let colBegin = -1;
                let colEnd = -1;
                // We check that we actually have something to show at this point!
                if (hoverCode.source && hoverCode.source.from) {
                    sourceLine = hoverCode.source.from.line;
                    // Highlight part of a line corresponding to the node if it fits on one line
                    if (hoverCode.source.to && hoverCode.source.from.line === hoverCode.source.to.line) {
                        colBegin = hoverCode.source.from.col;
                        colEnd = hoverCode.source.to.col;
                    }
                }
                this.eventHub.emit('editorLinkLine', this.compilerInfo.editorId, sourceLine, colBegin, colEnd, false);
                this.eventHub.emit('panesLinkLine', this.compilerInfo.compilerId, sourceLine,
                    colBegin, colEnd, false, this.getPaneName());
            }
        }
    }

    override onCompileResult(id, compiler, result) {
        if (this.compilerInfo.compilerId !== id) return;
    
        if (result.hasAstOutput) {
            this.showAstResults(result.astOutput);
        } else if (compiler.supportsAstView) {
            this.showAstResults([{text: '<No output>'}]);
        }
    
        // TODO: This is unelegant. Previously took advantage of fourth argument for the compileResult event.
        // I'm guessing it's not part of the TS rewrite because it's not always passed by the emitter.
        const lang = compiler.lang === 'c++' ? 'cpp' : compiler.lang;
        const model = this.editor.getModel();
        if (model != null && this.getCurrentEditorLanguage() !== lang) {
            monaco.editor.setModelLanguage(model, lang);
        }
    }

    showAstResults(results) {
        const fullText = results.map(function (x) {
            return x.text;
        }).join('\n');
        this.editor.setValue(fullText);
        this.astCode = results;
    
        if (!this.isAwaitingInitialResults) {
            if (this.selection) {
                this.editor.setSelection(this.selection);
                this.editor.revealLinesInCenter(this.selection.startLineNumber,
                    this.selection.endLineNumber);
            }
            this.isAwaitingInitialResults = true;
        }
    }

    onCompiler(id, compiler, options, editorid, treeid) {
        if (id === this.compilerInfo.compilerId) {
            this.compilerInfo.compilerName = compiler ? compiler.name : '';
            this.compilerInfo.editorId = editorid;
            this.compilerInfo.treeId = treeid;
            this.updateTitle();
            if (compiler && !compiler.supportsAstView) {
                this.editor.setValue('<AST output is not supported for this compiler>');
            }
        }
    }
    
    onColours(id, colours, scheme) {
        if (id === this.compilerInfo.compilerId) {
            const astColours = {};
            _.each(this.astCode, function (x, index) {
                if (x.source && x.source.from.line && x.source.to.line &&
                    x.source.from.line <= x.source.to.line && x.source.to.line < x.source.from.line + 100) {
                    let i;
                    for (i = x.source.from.line; i <= x.source.to.line; ++i) {
                        if (colours[i - 1] !== undefined) {
                            astColours[index] = colours[i - 1];
                            break;
                        }
                    }
                }
            });
            this.colours = colour.applyColours(this.editor, astColours, scheme, this.colours);
        }
    }
    
    onCompilerClose(id) {
        if (id === this.compilerInfo.compilerId) {
            // We can't immediately close as an outer loop somewhere in GoldenLayout is iterating over
            // the hierarchy. We can't modify while it's being iterated over.
            this.close();
            _.defer(function (self) {
                self.container.close();
            }, this);
        }
    }
    
    onSettingsChange(newSettings) {
        this.settings = newSettings;
        this.editor.updateOptions({
            contextmenu: newSettings.useCustomContextMenu,
            minimap: {
                enabled: newSettings.showMinimap,
            },
            fontFamily: newSettings.editorsFFont,
            fontLigatures: newSettings.editorsFLigatures,
        });
    }

    onDidChangeCursorSelection(e: monaco.editor.ICursorSelectionChangedEvent) {
        if (this.isAwaitingInitialResults) {
            this.selection = e.selection;
            this.updateState();
        }
    }
    
    updateDecorations() {
        this.prevDecorations = this.editor.deltaDecorations(
            this.prevDecorations, _.flatten(_.values(this.decorations)));
    }
    
    clearLinkedLines() {
        this.decorations.linkedCode = [];
        this.updateDecorations();
    }
    
    onPanesLinkLine(compilerId, lineNumber, colBegin, colEnd, revealLine, sender) {
        if (Number(compilerId) === this.compilerInfo.compilerId) {
            const lineNums: number[] = [];
            const singleNodeLines: number[] = [];
            const signalFromAnotherPane = sender !== this.getPaneName();
            _.each(this.astCode, function (astLine, i) {
                if (astLine.source
                    && astLine.source.from.line <= lineNumber && lineNumber <= astLine.source.to.line) {
                    const line = i + 1;
                    lineNums.push(line);
                    if (signalFromAnotherPane &&
                        astLine.source.from.line === lineNumber && astLine.source.to.line === lineNumber &&
                        astLine.source.from.col <= colEnd && colBegin <= astLine.source.to.col) {
                        singleNodeLines.push(line);
                    }
                }
            });
            if (revealLine && lineNums[0]) this.editor.revealLineInCenter(lineNums[0]);
            const lineClass = signalFromAnotherPane ? 'linked-code-decoration-line' : '';
            const linkedLineDecorations = _.map(lineNums, function (line) {
                return {
                    range: new monaco.Range(line, 1, line, 1),
                    options: {
                        isWholeLine: true,
                        linesDecorationsClassName: 'linked-code-decoration-margin',
                        className: lineClass,
                    },
                };
            });
            const directlyLinkedLineDecorations = _.map(singleNodeLines, function (line) {
                return {
                    range: new monaco.Range(line, 1, line, 1),
                    options: {
                        isWholeLine: true,
                        inlineClassName: 'linked-code-decoration-column',
                    },
                };
            });
            this.decorations.linkedCode = [...linkedLineDecorations, ...directlyLinkedLineDecorations];
            if (this.linkedFadeTimeoutId !== -1) {
                clearTimeout(this.linkedFadeTimeoutId as NodeJS.Timeout);
            }
            this.linkedFadeTimeoutId = setTimeout(() => {
                this.clearLinkedLines();
                this.linkedFadeTimeoutId = -1;
            }, 5000);
            this.updateDecorations();
        }
    }
    
    close() {
        this.eventHub.unsubscribe();
        this.eventHub.emit('astViewClosed', this.compilerInfo.compilerId);
        this.editor.dispose();
    }
}
