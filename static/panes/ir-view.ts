// Copyright (c) 2021, Compiler Explorer Authors
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

import {editor} from 'monaco-editor';
import IEditorMouseEvent = editor.IEditorMouseEvent;

import {MonacoPane} from './pane.js';
import {IrState} from './ir-view.interfaces.js';
import {MonacoPaneState} from './pane.interfaces.js';

import {extendConfig} from '../monaco-config.js';
import {applyColours} from '../colour.js';

import {Hub} from '../hub.js';
import * as Components from '../components.js';
import {unwrap} from '../assert.js';
import {Toggles} from '../widgets/toggles.js';

import {LLVMIrBackendOptions} from '../../types/compilation/ir.interfaces.js';
import {CompilationResult} from '../compilation/compilation.interfaces.js';
import {CompilerInfo} from '../compiler.interfaces.js';
import {Compiler} from './compiler.js';
import {Alert} from '../widgets/alert.js';
import {SentryCapture} from '../sentry.js';

export class Ir extends MonacoPane<monaco.editor.IStandaloneCodeEditor, IrState> {
    private linkedFadeTimeoutId: NodeJS.Timeout | null = null;
    private irCode?: any[] = undefined;
    private srcColours?: Record<number, number | undefined> = undefined;
    private colourScheme?: string = undefined;
    private alertSystem: Alert;
    private isAsmKeywordCtxKey: monaco.editor.IContextKey<boolean>;

    // TODO: eliminate deprecated deltaDecorations monaco API
    private decorations: any = {};
    private previousDecorations: string[] = [];

    private options: Toggles;
    private filters: Toggles;
    private toggleWrapButton: Toggles;
    private lastOptions: LLVMIrBackendOptions = {
        filterDebugInfo: true,
        filterIRMetadata: true,
        filterAttributes: true,
        filterComments: true,
        noDiscardValueNames: true,
        demangle: true,
    };
    private cfgButton: JQuery;
    private wrapButton: JQuery<HTMLElement>;
    private wrapTitle: JQuery<HTMLElement>;

    constructor(hub: Hub, container: Container, state: IrState & MonacoPaneState) {
        super(hub, container, state);
        if (state.irOutput) {
            this.showIrResults(state.irOutput ?? []);
        }

        this.onOptionsChange(true);
        this.alertSystem = new Alert();
        this.alertSystem.prefixMessage = 'LLVM IR';
    }

    override getInitialHTML(): string {
        return $('#ir').html();
    }

    override createEditor(editorRoot: HTMLElement): void {
        this.editor = monaco.editor.create(
            editorRoot,
            extendConfig({
                language: 'llvm-ir',
                readOnly: true,
                glyphMargin: true,
                lineNumbersMinChars: 3,
            }),
        );
    }

    override getPrintName() {
        return 'Ir Output';
    }

    override getDefaultPaneName(): string {
        return 'LLVM IR Viewer';
    }

    override registerButtons(state: IrState) {
        super.registerButtons(state);
        this.options = new Toggles(this.domRoot.find('.options'), state as unknown as Record<string, boolean>);
        this.options.on('change', this.onOptionsChange.bind(this));
        this.filters = new Toggles(this.domRoot.find('.filters'), state as unknown as Record<string, boolean>);
        this.filters.on('change', this.onOptionsChange.bind(this));

        this.cfgButton = this.domRoot.find('.cfg');
        const createCfgView = () => {
            return Components.getCfgViewWith(
                this.compilerInfo.compilerId,
                this.compilerInfo.editorId ?? 0,
                this.compilerInfo.treeId ?? 0,
                true,
            );
        };
        this.container.layoutManager.createDragSource(this.cfgButton, createCfgView as any);
        this.cfgButton.on('click', () => {
            const insertPoint =
                this.hub.findParentRowOrColumn(this.container.parent) ||
                this.container.layoutManager.root.contentItems[0];
            insertPoint.addChild(createCfgView());
        });

        this.toggleWrapButton = new Toggles(this.domRoot.find('.wrap'), state as unknown as Record<string, boolean>);
        this.toggleWrapButton.on('change', this.onToggleWrapChange.bind(this));
        this.wrapButton = this.domRoot.find('.wrap-lines');
        this.wrapTitle = this.wrapButton.prop('title');

        if (state.wrap === true) {
            this.wrapButton.prop('title', '[ON] ' + this.wrapTitle);
        } else {
            this.wrapButton.prop('title', '[OFF] ' + this.wrapTitle);
        }
    }

    async onAsmToolTip(ed: monaco.editor.ICodeEditor) {
        const pos = ed.getPosition();
        if (!pos || !ed.getModel()) return;
        const word = ed.getModel()?.getWordAtPosition(pos);
        if (!word || !word.word) return;
        const opcode = word.word.toUpperCase();

        function newGitHubIssueUrl(): string {
            return (
                'https://github.com/compiler-explorer/compiler-explorer/issues/new?title=' +
                encodeURIComponent('[BUG] Problem with ' + opcode + ' opcode')
            );
        }

        function appendInfo(url: string): string {
            return (
                '<br><br>For more information, visit <a href="' +
                url +
                '" target="_blank" rel="noopener noreferrer">the ' +
                opcode +
                ' documentation <sup><small class="fas fa-external-link-alt opens-new-window"' +
                ' title="Opens in a new window"></small></sup></a>.' +
                '<br>If the documentation for this opcode is wrong or broken in some way, ' +
                'please feel free to <a href="' +
                newGitHubIssueUrl() +
                '" target="_blank" rel="noopener noreferrer">' +
                'open an issue on GitHub <sup><small class="fas fa-external-link-alt opens-new-window" ' +
                'title="Opens in a new window"></small></sup></a>.'
            );
        }

        try {
            const asmHelp = await Compiler.getAsmInfo(word.word, 'llvm');
            if (asmHelp) {
                this.alertSystem.alert(opcode + ' help', asmHelp.html + appendInfo(asmHelp.url), {
                    onClose: () => {
                        ed.focus();
                        ed.setPosition(pos);
                    },
                });
            }
        } catch (error) {
            this.alertSystem.notify('There was an error fetching the documentation for this opcode (' + error + ').', {
                group: 'notokenindocs',
                alertClass: 'notification-error',
                dismissTime: 5000,
            });
        }
    }

    onToggleWrapChange(): void {
        const state = this.getCurrentState();
        if (state.wrap) {
            this.editor.updateOptions({wordWrap: 'on'});
            this.wrapButton.prop('title', '[ON] ' + this.wrapTitle);
        } else {
            this.editor.updateOptions({wordWrap: 'off'});
            this.wrapButton.prop('title', '[OFF] ' + this.wrapTitle);
        }

        this.updateState();
    }

    override getCurrentState() {
        return {
            ...this.options.get(),
            ...this.filters.get(),
            ...super.getCurrentState(),
            wrap: this.toggleWrapButton.get().wrap,
        };
    }

    override registerEditorActions(): void {
        this.isAsmKeywordCtxKey = this.editor.createContextKey('isAsmKeyword', true);
        this.editor.addAction({
            id: 'viewsource',
            label: 'Scroll to source',
            keybindings: [monaco.KeyMod.CtrlCmd | monaco.KeyCode.F10],
            contextMenuGroupId: 'navigation',
            contextMenuOrder: 1.5,
            run: editor => {
                const position = editor.getPosition();
                if (position != null && this.irCode) {
                    const desiredLine = position.lineNumber - 1;
                    const source = this.irCode[desiredLine].source;
                    if (source !== null && source.file !== null) {
                        this.eventHub.emit(
                            'editorLinkLine',
                            unwrap(this.compilerInfo.editorId),
                            source.line,
                            -1,
                            -1,
                            true,
                        );
                    }
                }
            },
        });
        this.editor.addAction({
            id: 'viewasmdoc',
            label: 'View IR documentation',
            keybindings: [monaco.KeyMod.CtrlCmd | monaco.KeyCode.F8],
            keybindingContext: undefined,
            precondition: 'isAsmKeyword',
            contextMenuGroupId: 'help',
            contextMenuOrder: 1.5,
            run: this.onAsmToolTip.bind(this),
        });

        // This returns a vscode's ContextMenuController, but that type is not exposed in Monaco
        const contextMenuContrib = this.editor.getContribution<any>('editor.contrib.contextmenu');

        // This is hacked this way to be able to update the precondition keys before the context menu is shown.
        // Right now Monaco does not expose a proper way to update those preconditions before the menu is shown,
        // because the editor.onContextMenu callback fires after it's been shown, so it's of little use here
        // The original source is src/vs/editor/contrib/contextmenu/browser/contextmenu.ts in vscode
        const originalOnContextMenu: ((e: IEditorMouseEvent) => void) | undefined = contextMenuContrib._onContextMenu;
        if (originalOnContextMenu) {
            contextMenuContrib._onContextMenu = (e: IEditorMouseEvent) => {
                if (e.target.position) {
                    const currentWord = this.editor.getModel()?.getWordAtPosition(e.target.position);
                    if (currentWord?.word) {
                        this.isAsmKeywordCtxKey.set(this.isWordAsmKeyword(e.target.position.lineNumber, currentWord));
                    }

                    // And call the original method now that we've updated the context keys
                    originalOnContextMenu.apply(contextMenuContrib, [e]);
                }
            };
        } else {
            // In case this ever stops working, we'll be notified
            SentryCapture(new Error('Context menu hack did not return valid original method'));
        }
    }

    override registerCallbacks(): void {
        const onMouseMove = _.throttle(this.onMouseMove.bind(this), 50);
        const onDidChangeCursorSelection = _.throttle(this.onDidChangeCursorSelection.bind(this), 500);

        this.eventHub.on('renamePane', this.updateState.bind(this));

        this.eventHub.on('compileResult', this.onCompileResult.bind(this));
        this.eventHub.on('colours', this.onColours.bind(this));
        this.eventHub.on('panesLinkLine', this.onPanesLinkLine.bind(this));

        this.editor.onMouseMove(event => onMouseMove(event));
        this.editor.onDidChangeCursorSelection(event => onDidChangeCursorSelection(event));

        this.eventHub.emit('irViewOpened', this.compilerInfo.compilerId);
        this.eventHub.emit('requestSettings');
    }

    onOptionsChange(force = false) {
        const options = this.options.get();
        const filters = this.filters.get();
        const newOptions: LLVMIrBackendOptions = {
            filterDebugInfo: filters['filter-debug-info'],
            filterIRMetadata: filters['filter-instruction-metadata'],
            filterAttributes: filters['filter-attributes'],
            filterComments: filters['filter-comments'],
            noDiscardValueNames: options['-fno-discard-value-names'],
            demangle: options['demangle-symbols'],
        };
        let changed = false;
        for (const k in newOptions) {
            if (newOptions[k as keyof LLVMIrBackendOptions] !== this.lastOptions[k as keyof LLVMIrBackendOptions]) {
                changed = true;
                break;
            }
        }
        this.lastOptions = newOptions;
        if (changed || force) {
            this.eventHub.emit('llvmIrViewOptionsUpdated', this.compilerInfo.compilerId, newOptions, true);
        }
    }

    override onCompileResult(compilerId: number, compiler: CompilerInfo, result: CompilationResult): void {
        if (this.compilerInfo.compilerId !== compilerId) return;
        if (result.irOutput) {
            this.showIrResults(unwrap(result.irOutput).asm);
            this.tryApplyIrColours();
        } else if (compiler.supportsIrView) {
            this.showIrResults([{text: '<No output>'}]);
        }
    }

    override onCompiler(
        compilerId: number,
        compiler: CompilerInfo | null,
        options: string,
        editorId: number,
        treeId: number,
    ): void {
        if (this.compilerInfo.compilerId !== compilerId) return;
        this.compilerInfo.compilerName = compiler ? compiler.name : '';
        this.compilerInfo.editorId = editorId;
        this.compilerInfo.treeId = treeId;
        this.updateTitle();
        if (compiler && !compiler.supportsIrView) {
            this.editor.setValue('<LLVM IR output is not supported for this compiler>');
        }
    }

    showIrResults(result: any): void {
        if (result && Array.isArray(result)) {
            this.irCode = result;
            this.editor
                .getModel()
                ?.setValue(result.length ? _.pluck(result, 'text').join('\n') : '<No LLVM IR generated>');
        } else {
            this.irCode = [];
        }

        if (!this.isAwaitingInitialResults) {
            if (this.selection) {
                this.editor.setSelection(this.selection);
                this.editor.revealLinesInCenter(this.selection.startLineNumber, this.selection.endLineNumber);
            }
            this.isAwaitingInitialResults = true;
        }
    }

    tryApplyIrColours(): void {
        if (!this.srcColours || !this.colourScheme || !this.irCode || this.irCode.length === 0) return;

        const irColours: Record<number, number> = {};
        for (const [index, code] of this.irCode.entries()) {
            if (
                code.source &&
                code.source.file === null &&
                code.source.line > 0 &&
                this.srcColours[code.source.line - 1] !== undefined
            ) {
                irColours[index] = this.srcColours[code.source.line - 1]!;
            }
        }
        applyColours(irColours, this.colourScheme, this.editorDecorations);
    }

    onColours(editorId: number, srcColours: Record<number, number>, scheme: string): void {
        if (editorId !== this.compilerInfo.editorId) return;
        this.colourScheme = scheme;
        this.srcColours = srcColours;

        this.tryApplyIrColours();
    }

    getLineTokens = (line: number): monaco.Token[] => {
        const model = this.editor.getModel();
        if (!model || line > model.getLineCount()) return [];
        const flavour = model.getLanguageId();
        const tokens = monaco.editor.tokenize(model.getLineContent(line), flavour);
        return tokens.length > 0 ? tokens[0] : [];
    };

    isWordAsmKeyword = (lineNumber: number, word: monaco.editor.IWordAtPosition): boolean => {
        return this.getLineTokens(lineNumber).some(t => {
            return (
                t.offset + 1 === word.startColumn && (t.type === 'keyword.llvm-ir' || t.type === 'operators.llvm-ir')
            );
        });
    };

    async onMouseMove(e: monaco.editor.IEditorMouseEvent): Promise<void> {
        if (e.target.position === null) return;
        if (this.settings.hoverShowSource === true) {
            this.clearLinkedLines();
            if (this.irCode && e.target.position.lineNumber - 1 in this.irCode) {
                const hoverCode = this.irCode[e.target.position.lineNumber - 1];
                let sourceLine = -1;
                let sourceColumnBegin = -1;
                let sourceColumnEnd = -1;

                if (hoverCode.source && !hoverCode.source.file) {
                    sourceLine = hoverCode.source.line;
                    if (hoverCode.source.column) {
                        sourceColumnBegin = hoverCode.source.column;
                        sourceColumnEnd = sourceColumnBegin;
                    }
                }

                this.eventHub.emit(
                    'editorLinkLine',
                    unwrap(this.compilerInfo.editorId),
                    sourceLine,
                    sourceColumnBegin,
                    sourceColumnEnd,
                    false,
                );
                this.eventHub.emit(
                    'panesLinkLine',
                    this.compilerInfo.compilerId,
                    sourceLine,
                    sourceColumnBegin,
                    sourceColumnEnd,
                    false,
                    this.getPaneName(),
                );
            }
        }

        const currentWord = this.editor.getModel()?.getWordAtPosition(e.target.position);
        if (currentWord?.word) {
            let word = currentWord.word;
            let startColumn = currentWord.startColumn;
            // Avoid throwing an exception if somehow (How?) we have a non-existent lineNumber.
            // c.f. https://sentry.io/matt-godbolt/compiler-explorer/issues/285270358/
            if (e.target.position.lineNumber <= (this.editor.getModel()?.getLineCount() ?? 0)) {
                // Hacky workaround to check for negative numbers.
                // c.f. https://github.com/compiler-explorer/compiler-explorer/issues/434
                const lineContent = this.editor.getModel()?.getLineContent(e.target.position.lineNumber);
                if (lineContent && lineContent[currentWord.startColumn - 2] === '-') {
                    word = '-' + word;
                    startColumn -= 1;
                }
            }
            const range = new monaco.Range(
                e.target.position.lineNumber,
                Math.max(startColumn, 1),
                e.target.position.lineNumber,
                currentWord.endColumn,
            );

            if (this.settings.hoverShowAsmDoc && this.isWordAsmKeyword(e.target.position.lineNumber, currentWord)) {
                try {
                    const response = await Compiler.getAsmInfo(currentWord.word, 'llvm');
                    if (!response) return;
                    this.decorations.asmToolTip = [
                        {
                            range: range,
                            options: {
                                isWholeLine: false,
                                hoverMessage: [
                                    {
                                        value: response.tooltip + '\n\nMore information available in the context menu.',
                                        isTrusted: true,
                                    },
                                ],
                            },
                        },
                    ];
                    this.updateDecorations();
                } catch {
                    // ignore errors fetching tooltips
                }
            }
        }
    }

    onPanesLinkLine(
        compilerId: number,
        lineNumber: number,
        columnBegin: number,
        columnEnd: number,
        revealLinesInEditor: boolean,
        sender: string,
    ): void {
        if (compilerId !== this.compilerInfo.compilerId) return;
        const lineNumbers: number[] = [];
        const directlyLinkedLineNumbers: number[] = [];
        const isSignalFromAnotherPane = sender !== this.getPaneName();

        if (this.irCode) {
            for (const [index, irLine] of this.irCode.entries()) {
                if (irLine.source && irLine.source.file === null && irLine.source.line === lineNumber) {
                    const line = index + 1;
                    const currentColumn = irLine.source.column;
                    lineNumbers.push(line);
                    if (
                        isSignalFromAnotherPane &&
                        currentColumn &&
                        columnBegin <= currentColumn &&
                        currentColumn <= columnEnd
                    ) {
                        directlyLinkedLineNumbers.push(line);
                    }
                }
            }
        }

        if (revealLinesInEditor && lineNumbers[0]) {
            // Just make sure that the mapped line is in view!
            this.editor.revealLinesInCenter(lineNumbers[0], lineNumbers[0]);
        }

        const lineClassName = isSignalFromAnotherPane ? 'linked-code-decoration-line' : '';
        const linkedLineDecorations: monaco.editor.IModelDeltaDecoration[] = lineNumbers.map(line => ({
            range: new monaco.Range(line, 1, line, 1),
            options: {
                isWholeLine: true,
                linesDecorationsClassName: 'linked-code-decoration-margin',
                className: lineClassName,
            },
        }));
        const directlyLinkedLineDecorations: monaco.editor.IModelDeltaDecoration[] = directlyLinkedLineNumbers.map(
            line => ({
                range: new monaco.Range(line, 1, line, 1),
                options: {
                    isWholeLine: true,
                    inlineClassName: 'linked-code-decoration-column',
                },
            }),
        );
        this.decorations.linkedCode = [...linkedLineDecorations, ...directlyLinkedLineDecorations];

        if (!this.settings.indefiniteLineHighlight) {
            if (this.linkedFadeTimeoutId !== null) {
                clearTimeout(this.linkedFadeTimeoutId);
            }

            this.linkedFadeTimeoutId = setTimeout(() => {
                this.clearLinkedLines();
                this.linkedFadeTimeoutId = null;
            }, 5000);
        }
        this.updateDecorations();
    }

    updateDecorations(): void {
        this.previousDecorations = this.editor.deltaDecorations(
            this.previousDecorations,
            _.flatten(_.values(this.decorations)),
        );
    }

    clearLinkedLines(): void {
        this.decorations.linkedCode = [];
        this.updateDecorations();
    }

    override close(): void {
        this.eventHub.unsubscribe();
        this.eventHub.emit('irViewClosed', this.compilerInfo.compilerId);
        this.editor.dispose();
    }
}
