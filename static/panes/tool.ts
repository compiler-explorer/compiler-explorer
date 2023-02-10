// Copyright (c) 2023, Compiler Explorer Authors
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
import $ from 'jquery';
import {ga} from '../analytics';
import * as AnsiToHtml from '../ansi-to-html';
import {Toggles} from '../widgets/toggles';
import * as Components from '../components';
import * as monaco from 'monaco-editor';
import * as monacoConfig from '../monaco-config';
import {options as ceoptions} from '../options';
import * as utils from '../utils';
import * as fileSaver from 'file-saver';
import {MonacoPane} from './pane';
import {Hub} from '../hub';
import {Container} from 'golden-layout';
import {MonacoPaneState} from './pane.interfaces';
import {CompilerService} from '../compiler-service';
import {ComponentConfig, PopulatedToolInputViewState} from '../components.interfaces';
import {unwrap, unwrapString} from '../assert';

function makeAnsiToHtml(color?: string) {
    return new AnsiToHtml.Filter({
        fg: color ?? '#333',
        bg: '#f5f5f5',
        stream: true,
        escapeXML: true,
    });
}

export class Tool extends MonacoPane<monaco.editor.IStandaloneCodeEditor, ToolState> {
    toolId: any;
    toolName = 'Tool';
    compilerService: CompilerService;
    // todo: re-evaluate all these
    editorContentRoot: JQuery;
    plainContentRoot: JQuery;
    optionsToolbar: JQuery;
    badLangToolbar: JQuery;
    monacoStdin: boolean;
    monacoEditorOpen: boolean;
    monacoEditorHasBeenAutoOpened: boolean;
    monacoStdinField = '';
    normalAnsiToHtml: AnsiToHtml.Filter;
    optionsField: JQuery;
    localStdinField: JQuery;
    createToolInputView: () => ComponentConfig<PopulatedToolInputViewState>;

    wrapButton: JQuery;
    wrapTitle: JQuery;
    panelArgs: JQuery;
    panelStdin: JQuery;

    toggleArgs: JQuery;
    toggleStdin: JQuery;
    artifactBtn: JQuery;
    artifactText: JQuery;

    options: Toggles;

    constructor(hub: Hub, container: Container, state: ToolState & MonacoPaneState) {
        // canonicalize state
        if ((state as any).compiler) state.id = (state as any).compiler;
        if ((state as any).editor) state.editorid = (state as any).editor;
        if ((state as any).tree) state.treeid = (state as any).tree;
        super(hub, container, state);

        this.toolId = state.toolId;
        this.compilerService = hub.compilerService;

        this.monacoStdin = state.monacoStdin || false;
        this.monacoEditorOpen = state.monacoEditorOpen || false;
        this.monacoEditorHasBeenAutoOpened = state.monacoEditorHasBeenAutoOpened || false;
        this.normalAnsiToHtml = makeAnsiToHtml();

        this.createToolInputView = () =>
            Components.getToolInputViewWith(this.compilerInfo.compilerId, this.toolId, this.toolName);

        this.options = new Toggles(this.domRoot.find('.options'), state as any as Record<string, boolean>);
        this.options.on('change', this.onOptionsChange.bind(this));

        this.initArgs(state);

        this.onOptionsChange();

        this.updateTitle();

        this.eventHub.emit('toolOpened', this.compilerInfo.compilerId, this.getCurrentState());
        this.eventHub.emit('requestSettings');
    }

    override getInitialHTML() {
        return $('#tool-output').html();
    }

    override registerOpeningAnalyticsEvent() {
        ga.proxy('send', {
            hitType: 'event',
            eventCategory: 'OpenViewPane',
            eventAction: 'Tool',
        });
    }

    override createEditor(editorRoot: HTMLElement) {
        return monaco.editor.create(
            editorRoot,
            monacoConfig.extendConfig({
                readOnly: true,
                language: 'text',
                fontFamily: 'courier new',
                lineNumbersMinChars: 5,
                guides: {
                    bracketPairs: false,
                    bracketPairsHorizontal: false,
                    highlightActiveBracketPair: false,
                    highlightActiveIndentation: false,
                    indentation: false,
                },
            })
        );
    }

    override registerDynamicElements(state: ToolState) {
        super.registerDynamicElements(state);
        this.editorContentRoot = this.domRoot.find('.monaco-placeholder');
        this.plainContentRoot = this.domRoot.find('pre.content');
        this.optionsToolbar = this.domRoot.find('.options-toolbar');
        this.badLangToolbar = this.domRoot.find('.bad-lang');
        this.optionsField = this.domRoot.find('input.options');
        this.localStdinField = this.domRoot.find('textarea.tool-stdin');
    }

    override registerCallbacks() {
        super.registerCallbacks();
        this.eventHub.on('languageChange', this.onLanguageChange, this);
        this.eventHub.on('toolInputChange', this.onToolInputChange, this);
        this.eventHub.on('toolInputViewClosed', this.onToolInputViewClosed, this);

        this.toggleArgs.on('click', () => this.togglePanel(this.toggleArgs, this.panelArgs));

        this.toggleStdin.on('click', () => {
            if (!this.monacoStdin) {
                this.togglePanel(this.toggleStdin, this.panelStdin);
            } else {
                if (!this.monacoEditorOpen) {
                    this.openMonacoEditor();
                } else {
                    this.monacoEditorOpen = false;
                    this.toggleStdin.removeClass('active');
                    this.eventHub.emit('toolInputViewCloseRequest', this.compilerInfo.compilerId, this.toolId);
                }
            }
        });

        if ('MutationObserver' in window) {
            new MutationObserver(this.resize.bind(this)).observe(this.localStdinField[0], {
                attributes: true,
                attributeFilter: ['style'],
            });
        }
    }

    onLanguageChange(editorId, newLangId) {
        if (this.compilerInfo.editorId && this.compilerInfo.editorId === editorId) {
            const tools = ceoptions.tools[newLangId];
            this.toggleUsable(tools && tools[this.toolId]);
        }
    }

    toggleUsable(isUsable) {
        if (isUsable) {
            this.plainContentRoot.css('opacity', '1');
            this.badLangToolbar.hide();
            this.optionsToolbar.show();
        } else {
            this.plainContentRoot.css('opacity', '0.5');
            this.optionsToolbar.hide();
            this.badLangToolbar.show();
        }
    }

    initArgs(state: ToolState & MonacoPaneState) {
        const optionsChange = _.debounce(e => {
            this.onOptionsChange();

            this.eventHub.emit('toolSettingsChange', this.compilerInfo.compilerId);
        }, 800);

        this.optionsField.on('change', optionsChange).on('keyup', optionsChange);

        if (state.args) {
            this.optionsField.val(state.args);
        }

        this.localStdinField.on('change', optionsChange).on('keyup', optionsChange);

        if (state.stdin) {
            if (!this.monacoStdin) {
                this.localStdinField.val(state.stdin);
            } else {
                this.eventHub.emit('setToolInput', this.compilerInfo.compilerId, this.toolId, state.stdin);
            }
        }
    }

    getInputArgs() {
        return unwrapString(this.optionsField.val());
    }

    onToolInputChange(compilerId: number, toolId: string, input: string) {
        if (this.compilerInfo.compilerId === compilerId && this.toolId === toolId) {
            this.monacoStdinField = input;
            this.onOptionsChange();
            this.eventHub.emit('toolSettingsChange', this.compilerInfo.compilerId);
        }
    }

    onToolInputViewClosed(compilerId: number, toolId: string, input: string) {
        if (this.compilerInfo.compilerId === compilerId && this.toolId === toolId) {
            // Duplicate close messages have been seen, with the second having no value.
            // If we have a current value and the new value is empty, ignore the message.
            if (this.monacoStdinField && input) {
                this.monacoStdinField = input;
                this.monacoEditorOpen = false;
                this.toggleStdin.removeClass('active');

                this.onOptionsChange();
                this.eventHub.emit('toolSettingsChange', this.compilerInfo.compilerId);
            }
        }
    }

    getInputStdin() {
        if (!this.monacoStdin) {
            return unwrapString(this.localStdinField.val());
        } else {
            return this.monacoStdinField;
        }
    }

    openMonacoEditor() {
        this.monacoEditorHasBeenAutoOpened = true; // just in case we get here in an unexpected way
        this.monacoEditorOpen = true;
        this.toggleStdin.addClass('active');
        const insertPoint =
            this.hub.findParentRowOrColumn(this.container.parent) || this.container.layoutManager.root.contentItems[0];
        insertPoint.addChild(this.createToolInputView());
        this.onOptionsChange();
        this.eventHub.emit('setToolInput', this.compilerInfo.compilerId, this.toolId, this.monacoStdinField);
    }

    getEffectiveOptions() {
        return this.options.get();
    }

    override resize() {
        utils.updateAndCalcTopBarHeight(this.domRoot, this.optionsToolbar, this.hideable);
        let barsHeight = unwrap(this.optionsToolbar.outerHeight()) + 2;
        if (!this.panelArgs.hasClass('d-none')) {
            barsHeight += unwrap(this.panelArgs.outerHeight());
        }
        if (!this.panelStdin.hasClass('d-none')) {
            barsHeight += unwrap(this.panelStdin.outerHeight());
        }

        this.editor.layout({
            width: unwrap(this.domRoot.width()),
            height: unwrap(this.domRoot.height()) - barsHeight,
        });

        this.plainContentRoot.height(unwrap(this.domRoot.height()) - barsHeight);
    }

    onOptionsChange() {
        const options = this.getEffectiveOptions();
        this.plainContentRoot.toggleClass('wrap', options.wrap);
        this.wrapButton.prop('title', '[' + (options.wrap ? 'ON' : 'OFF') + '] ' + this.wrapTitle);

        this.updateState();
    }

    override registerButtons(state: ToolState & MonacoPaneState) {
        super.registerButtons(state);

        this.wrapButton = this.domRoot.find('.wrap-lines');
        this.wrapTitle = this.wrapButton.prop('title');

        this.panelArgs = this.domRoot.find('.panel-args');
        this.panelStdin = this.domRoot.find('.panel-stdin');

        this.initButtonsVisibility(state);
    }

    initButtonsVisibility(state: ToolState & MonacoPaneState) {
        this.toggleArgs = this.domRoot.find('.toggle-args');
        this.toggleStdin = this.domRoot.find('.toggle-stdin');
        this.artifactBtn = this.domRoot.find('.artifact-btn');
        this.artifactText = this.domRoot.find('.artifact-text');

        if (state.argsPanelShown === true) {
            this.showPanel(this.toggleArgs, this.panelArgs);
        }

        if (state.stdinPanelShown === true) {
            if (!this.monacoStdin) {
                this.showPanel(this.toggleStdin, this.panelStdin);
            } else {
                if (!this.monacoEditorOpen) {
                    this.openMonacoEditor();
                }
            }
        }
        this.artifactBtn.addClass('d-none');
    }

    showPanel(button: JQuery, panel: JQuery) {
        panel.removeClass('d-none');
        button.addClass('active');
        this.resize();
    }

    hidePanel(button: JQuery, panel: JQuery) {
        panel.addClass('d-none');
        button.removeClass('active');
        this.resize();
    }

    togglePanel(button: JQuery, panel: JQuery) {
        if (panel.hasClass('d-none')) {
            this.showPanel(button, panel);
        } else {
            this.hidePanel(button, panel);
        }
        this.updateState();
    }

    override getCurrentState() {
        const options = this.getEffectiveOptions();
        const state: MonacoPaneState & ToolState = {
            ...super.getCurrentState(),
            wrap: options.wrap,
            toolId: this.toolId,
            args: this.getInputArgs(),
            stdin: this.getInputStdin(),
            stdinPanelShown: (this.monacoStdin && this.monacoEditorOpen) || !this.panelStdin.hasClass('d-none'),
            monacoStdin: this.monacoStdin,
            monacoEditorOpen: this.monacoEditorOpen,
            monacoEditorHasBeenAutoOpened: this.monacoEditorHasBeenAutoOpened,
            argsPanelShown: !this.panelArgs.hasClass('d-none'),
        };
        return state as MonacoPaneState;
    }

    setLanguage(languageId: false | string) {
        if (languageId) {
            this.options.enableToggle('wrap', false);
            monaco.editor.setModelLanguage(unwrap(this.editor.getModel()), languageId);
            this.editor.setValue('');
            this.fontScale.setTarget(this.editor);
            $(this.plainContentRoot).hide();
            $(this.editorContentRoot).show();
        } else {
            this.options.enableToggle('wrap', true);
            this.plainContentRoot.empty();
            this.fontScale.setTarget('.content');
            $(this.editorContentRoot).hide();
            $(this.plainContentRoot).show();
        }
    }

    clickableUrls(text: string) {
        return text.replace(
            // URL detection regex grabbed from https://stackoverflow.com/a/3809435
            /(https?:\/\/(www\.)?[-a-zA-Z0-9@:%._+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_+.~#?&//=]*))/,
            '<a href="$1" target="_blank">$1</a>'
        );
    }

    override onCompiler(compilerId: number, compiler: any, options: string, editorId: number, treeId: number) {
        // TODO(jeremy-rifkin): This should probably be done in the base pane / standard across all panes
        if (this.compilerInfo.compilerId !== compilerId) return;
        this.compilerInfo.compilerName = compiler ? compiler.name : '';
        this.compilerInfo.editorId = editorId;
        this.compilerInfo.treeId = treeId;
        this.updateTitle();
    }

    override onCompileResult(id: number, compiler, result) {
        try {
            if (id !== this.compilerInfo.compilerId) return;
            if (compiler) this.compilerInfo.compilerName = compiler.name;

            const foundTool = _.find(compiler.tools, tool => tool.tool.id === this.toolId);

            this.toggleUsable(foundTool);

            // any for now for typing reasons... TODO(jeremy-rifkin)
            let toolResult: any = null;
            if (result && result.tools) {
                toolResult = _.find(result.tools, tool => tool.id === this.toolId);
            } else if (result && result.result && result.result.tools) {
                toolResult = _.find(result.result.tools, tool => tool.id === this.toolId);
            }

            // any for now for typing reasons... TODO(jeremy-rifkin)
            let toolInfo: any = null;
            if (compiler && compiler.tools) {
                toolInfo = _.find(compiler.tools, tool => tool.tool.id === this.toolId);
            }

            if (toolInfo) {
                this.toggleStdin.prop('disabled', false);

                if (this.monacoStdin && !this.monacoEditorOpen && !this.monacoEditorHasBeenAutoOpened) {
                    this.monacoEditorHasBeenAutoOpened = true;
                    this.openMonacoEditor();
                } else if (!this.monacoStdin && toolInfo.tool.stdinHint) {
                    this.localStdinField.prop('placeholder', toolInfo.tool.stdinHint);
                    if (toolInfo.tool.stdinHint === 'disabled') {
                        this.toggleStdin.prop('disabled', true);
                    } else {
                        this.showPanel(this.toggleStdin, this.panelStdin);
                    }
                } else {
                    this.localStdinField.prop('placeholder', 'Tool stdin...');
                }
            }

            // reset stream styles
            this.normalAnsiToHtml.reset();

            if (toolResult) {
                if (toolResult.languageId && toolResult.languageId === 'stderr') {
                    toolResult.languageId = false;
                }

                this.setLanguage(toolResult.languageId);

                if (toolResult.languageId) {
                    this.setEditorContent(_.pluck(toolResult.stdout, 'text').join('\n'));
                } else {
                    this.plainContentRoot.empty();
                    for (const obj of (toolResult.stdout || []).concat(toolResult.stderr || [])) {
                        if (obj.text === '') {
                            this.add('<br/>');
                        } else {
                            this.add(
                                this.clickableUrls(this.normalAnsiToHtml.toHtml(obj.text)),
                                obj.tag ? obj.tag.line : obj.line,
                                obj.tag ? obj.tag.column : 0,
                                obj.tag ? obj.tag.flow : null
                            );
                        }
                    }
                }

                this.toolName = toolResult.name;
                this.updateTitle();

                if (toolResult.sourcechanged && this.compilerInfo.editorId) {
                    this.eventHub.emit('newSource', this.compilerInfo.editorId, toolResult.newsource);
                }
                this.artifactBtn.off('click');
                if (toolResult.artifact) {
                    this.artifactBtn.removeClass('d-none');
                    this.artifactText.text('Download ' + toolResult.artifact.title);
                    this.artifactBtn.on('click', () => {
                        // The artifact content can be passed either as plain text or as a base64 encoded binary file
                        if (toolResult.artifact.type === 'application/octet-stream') {
                            // Fetch is the most convenient non ES6 way to build a binary blob out of a base64 string
                            fetch('data:application/octet-stream;base64,' + toolResult.artifact.content)
                                .then(res => res.blob())
                                .then(blob => fileSaver.saveAs(blob, toolResult.artifact.name));
                        } else {
                            fileSaver.saveAs(
                                new Blob([toolResult.artifact.content], {
                                    type: toolResult.artifact.type,
                                }),
                                toolResult.artifact.name
                            );
                        }
                    });
                } else {
                    this.artifactBtn.addClass('d-none');
                }
            } else {
                this.setEditorContent('No tool result');
            }
        } catch (e: any) {
            this.setLanguage(false);
            this.add('javascript error: ' + e.message);
        }
    }

    add(msg: string, lineNum?: number, column?: number, flow?: number) {
        const elem = $('<div/>').appendTo(this.plainContentRoot);
        if (lineNum && this.compilerInfo.editorId) {
            elem.empty();
            const editorId = unwrap(this.compilerInfo.editorId);
            $('<a></a>')
                .prop('href', 'javascript:;')
                .html(msg)
                .on('click', e => {
                    this.eventHub.emit('editorSetDecoration', editorId, lineNum, true, column);
                    if (flow) {
                        // TODO(jeremy-rifkin): Flow's type does not match what the event expects.
                        this.eventHub.emit('editorDisplayFlow', editorId, flow as any);
                    }
                    e.preventDefault();
                    return false;
                })
                .on('mouseover', () => this.eventHub.emit('editorSetDecoration', editorId, lineNum, false, column))
                .appendTo(elem);
        } else {
            elem.html(msg);
        }
    }

    setEditorContent(content: string) {
        if (!this.editor.getModel()) return;
        const editorModel = this.editor.getModel();
        const visibleRanges = this.editor.getVisibleRanges();
        const currentTopLine = visibleRanges.length > 0 ? visibleRanges[0].startLineNumber : 1;
        unwrap(editorModel).setValue(content);
        this.editor.revealLine(currentTopLine);
        this.setNormalContent();
    }

    setNormalContent() {
        this.editor.updateOptions({
            lineNumbers: 'on',
            codeLens: false,
        });
    }

    override getDefaultPaneName() {
        return this.toolName;
    }

    override close() {
        this.eventHub.emit('toolClosed', this.compilerInfo.compilerId, this.getCurrentState());
        this.eventHub.unsubscribe();
        this.editor.dispose();
    }
}
