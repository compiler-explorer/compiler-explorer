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

import _ from 'underscore';
import {FontScale} from '../widgets/fontscale';
import {Filter as AnsiToHtml} from '../ansi-to-html';
import {Toggles} from '../widgets/toggles';
import {ga} from '../analytics';
import * as Components from '../components';
import * as monaco from 'monaco-editor';
import * as monacoConfig from '../monaco-config';
import {options as ceoptions} from '../options';
import * as utils from '../utils';
import {PaneRenaming} from '../widgets/pane-renaming';
import {saveAs} from 'file-saver';
import {MonacoPane} from './pane';
import {Hub} from '../hub';
import {Container} from 'golden-layout';
import {MonacoPaneState} from './pane.interfaces';
import {SiteSettings} from '../settings';
import {ComponentConfig, PopulatedToolInputViewState} from '../components.interfaces';
import {ToolState} from './tool.interfaces';
import {CompilerInfo} from '../../types/compiler.interfaces';
import {CompilationResult} from '../../types/compilation/compilation.interfaces';
import {ToolInfo} from '../../lib/tooling/base-tool.interface';
import {MessageWithLocation} from '../../types/resultline/resultline.interfaces';

function makeAnsiToHtml(color?: string): AnsiToHtml {
    return new AnsiToHtml({
        fg: color ? color : '#333',
        bg: '#f5f5f5',
        stream: true,
        escapeXML: true,
    });
}

export class Tool extends MonacoPane<monaco.editor.IStandaloneCodeEditor, any> {
    private hub: Hub;
    private editorContentRoot: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private readonly plainContentRoot: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private readonly optionsToolbar: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private badLangToolbar: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private compilerName: string;
    private monacoStdinField: string;
    private normalAnsiToHtml: AnsiToHtml;
    private readonly optionsField?: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private readonly localStdinField?: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private readonly createToolInputView: () => ComponentConfig<PopulatedToolInputViewState>;
    private readonly compilerId: number;
    private readonly toolId: number;
    private toolName?: string;
    private options: Toggles;
    private monacoEditorOpen: boolean;
    private monacoEditorHasBeenAutoOpened: boolean;
    private wrapButton: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private wrapTitle: JQuery<HTMLElement>;
    private panelArgs: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private panelStdin: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private toggleArgs: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private toggleStdin: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private artifactBtn: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private artifactText: JQuery<HTMLElementTagNameMap[keyof HTMLElementTagNameMap]>;
    private readonly editorId?: number;
    private compilerService: any;
    private readonly monacoStdin: string | null;
    private readonly treeId: number;
    private codeLensProvider?: monaco.languages.CodeLensList;

    constructor(hub: Hub, container: Container, state: MonacoPaneState & ToolState) {
        super(hub, container, state);
        this.hub = hub;
        this.compilerId = state.compiler;
        this.editorId = state.editorid;
        this.treeId = state.tree;
        this.toolId = state.toolId;
        this.toolName = 'Tool';
        this.compilerService = hub.compilerService;
        this.editorContentRoot = this.domRoot.find('.monaco-placeholder');
        this.plainContentRoot = this.domRoot.find('pre.content');
        this.optionsToolbar = this.domRoot.find('.options-toolbar');
        this.badLangToolbar = this.domRoot.find('.bad-lang');
        this.compilerName = '';
        this.monacoStdin = state.monacoStdin || null;
        this.monacoEditorOpen = state.monacoEditorOpen || false;
        this.monacoEditorHasBeenAutoOpened = state.monacoEditorHasBeenAutoOpened || false;
        this.monacoStdinField = '';
        this.normalAnsiToHtml = makeAnsiToHtml();

        this.optionsField = this.domRoot.find('input.options');
        this.localStdinField = this.domRoot.find('textarea.tool-stdin');

        this.fontScale = new FontScale(this.domRoot, state, '.content');
        this.fontScale.on('change', () => {
            this.saveState();
        });

        this.createToolInputView = () => {
            return Components.getToolInputViewWith(this.compilerId + '', this.toolId + '', this.toolName ?? '');
        };

        this.initButtons(state);
        this.options = new Toggles(this.domRoot.find('.options'), state as unknown as Record<string, boolean>);
        this.options.on('change', _.bind(this.onOptionsChange, this));

        this.paneRenaming = new PaneRenaming(this, state);

        this.initArgs(state);
        this.initCallbacks();

        this.onOptionsChange();

        this.eventHub.emit('toolOpened', this.compilerId, this.currentState());
        this.eventHub.emit('requestSettings');
    }

    override getInitialHTML(): string {
        return $('#tool-output').html();
    }

    override registerOpeningAnalyticsEvent() {
        ga.proxy('send', {
            hitType: 'event',
            eventCategory: 'OpenViewPane',
            eventAction: 'Tool',
        });
    }

    override createEditor(editorRoot: HTMLElement): monaco.editor.IStandaloneCodeEditor {
        return monaco.editor.create(
            editorRoot,
            monacoConfig.extendConfig({
                readOnly: true,
                language: 'text',
                fontFamily: 'courier new',
                lineNumbersMinChars: 5,
                guides: undefined,
            })
        );
    }

    initCallbacks() {
        this.container.on('resize', this.resize, this);
        this.container.on('shown', this.resize, this);
        this.container.on('destroy', this.close, this);

        this.paneRenaming.on('renamePane', this.saveState.bind(this));

        this.eventHub.on('compileResult', this.onCompileResult, this);
        this.eventHub.on('compilerClose', this.onCompilerClose, this);
        this.eventHub.on('settingsChange', this.onSettingsChange, this);
        this.eventHub.on('languageChange', this.onLanguageChange, this);
        this.eventHub.on('toolInputChange', this.onToolInputChange, this);
        this.eventHub.on('toolInputViewClosed', this.onToolInputViewClosed, this);

        this.toggleArgs.on('click', () => {
            this.togglePanel(this.toggleArgs, this.panelArgs);
        });

        this.toggleStdin.on('click', () => {
            if (!this.monacoStdin) {
                this.togglePanel(this.toggleStdin, this.panelStdin);
            } else {
                if (!this.monacoEditorOpen) {
                    this.openMonacoEditor();
                } else {
                    this.monacoEditorOpen = false;
                    this.toggleStdin.removeClass('active');
                    this.eventHub.emit('toolInputViewCloseRequest', this.compilerId, this.toolId);
                }
            }
        });

        // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
        if (MutationObserver !== undefined && this.localStdinField) {
            new MutationObserver(_.bind(this.resize, this)).observe(this.localStdinField[0], {
                attributes: true,
                attributeFilter: ['style'],
            });
        }
    }

    onLanguageChange(editorId: number | boolean, newLangId: string) {
        if (this.editorId && this.editorId === editorId) {
            // @ts-expect-error: 'tools' is of type 'unknown'
            const tools = ceoptions.tools[newLangId];
            this.toggleUsable(tools && tools[this.toolId]);
        }
    }

    toggleUsable(isUsable: boolean) {
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

    override onSettingsChange(newSettings: SiteSettings) {
        this.editor.updateOptions({
            contextmenu: newSettings.useCustomContextMenu,
            minimap: {
                enabled: newSettings.showMinimap,
            },
            fontFamily: newSettings.editorsFFont,
            codeLensFontFamily: newSettings.editorsFFont,
            fontLigatures: newSettings.editorsFLigatures,
        });
    }

    initArgs(state: MonacoPaneState & ToolState) {
        const optionsChange = _.debounce(() => {
            this.onOptionsChange();

            this.eventHub.emit('toolSettingsChange', Number(this.compilerId));
        }, 800);

        if (this.optionsField) {
            this.optionsField.on('change', optionsChange).on('keyup', optionsChange);

            if (state.args) {
                this.optionsField.val(state.args);
            }
        }

        if (this.localStdinField) {
            this.localStdinField.on('change', optionsChange).on('keyup', optionsChange);

            if (state.stdin) {
                if (!this.monacoStdin) {
                    this.localStdinField.val(state.stdin);
                } else {
                    this.eventHub.emit('setToolInput', this.compilerId, this.toolId, state.stdin);
                }
            }
        }
    }

    getInputArgs(): string {
        if (this.optionsField) {
            return this.optionsField.val() as string;
        } else {
            return '';
        }
    }

    onToolInputChange(compilerId: number, toolId: number, input: string) {
        if (this.compilerId === compilerId && this.toolId === toolId) {
            this.monacoStdinField = input;
            this.onOptionsChange();
            this.eventHub.emit('toolSettingsChange', this.compilerId);
        }
    }

    onToolInputViewClosed(compilerId: number, toolId: number, input: string) {
        if (this.compilerId === compilerId && this.toolId === toolId) {
            // Duplicate close messages have been seen, with the second having no value.
            // If we have a current value and the new value is empty, ignore the message.
            if (this.monacoStdinField && input) {
                this.monacoStdinField = input;
                this.monacoEditorOpen = false;
                this.toggleStdin.removeClass('active');

                this.onOptionsChange();
                this.eventHub.emit('toolSettingsChange', this.compilerId);
            }
        }
    }

    getInputStdin() {
        if (!this.monacoStdin) {
            if (this.localStdinField) {
                return this.localStdinField.val();
            } else {
                return '';
            }
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
        this.eventHub.emit('setToolInput', this.compilerId, this.toolId, this.monacoStdinField);
    }

    getEffectiveOptions() {
        return this.options.get();
    }

    override resize() {
        utils.updateAndCalcTopBarHeight(this.domRoot, this.optionsToolbar, this.hideable);
        let barsHeight = (this.optionsToolbar.outerHeight() ?? 0) + 2;
        if (!this.panelArgs.hasClass('d-none')) {
            barsHeight += this.panelArgs.outerHeight() ?? 0;
        }
        if (!this.panelStdin.hasClass('d-none')) {
            barsHeight += this.panelStdin.outerHeight() ?? 0;
        }

        this.editor.layout({
            width: this.domRoot.width() ?? 0,
            height: (this.domRoot.height() ?? 0) - barsHeight,
        });

        this.plainContentRoot.height((this.domRoot.height() ?? 0) - barsHeight);
    }

    onOptionsChange(): void {
        const options = this.getEffectiveOptions();
        this.plainContentRoot.toggleClass('wrap', options.wrap);
        this.wrapButton.prop('title', '[' + (options.wrap ? 'ON' : 'OFF') + '] ' + this.wrapTitle);

        this.saveState();
    }

    initButtons(state: MonacoPaneState & ToolState) {
        this.wrapButton = this.domRoot.find('.wrap-lines');
        this.wrapTitle = this.wrapButton.prop('title');

        this.panelArgs = this.domRoot.find('.panel-args');
        this.panelStdin = this.domRoot.find('.panel-stdin');

        this.hideable = this.domRoot.find('.hideable');

        this.initButtonsVisibility(state);
    }

    initButtonsVisibility(state: MonacoPaneState & ToolState) {
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

    showPanel(button: JQuery<HTMLElement>, panel: JQuery<HTMLElement>): void {
        panel.removeClass('d-none');
        button.addClass('active');
        this.resize();
    }

    hidePanel(button: JQuery<HTMLElement>, panel: JQuery<HTMLElement>): void {
        panel.addClass('d-none');
        button.removeClass('active');
        this.resize();
    }

    togglePanel(button: JQuery<HTMLElement>, panel: JQuery<HTMLElement>) {
        if (panel.hasClass('d-none')) {
            this.showPanel(button, panel);
        } else {
            this.hidePanel(button, panel);
        }
        this.saveState();
    }

    currentState() {
        const options = this.getEffectiveOptions();
        const state = {
            compiler: this.compilerId,
            editor: this.editorId,
            tree: this.treeId,
            wrap: options.wrap,
            toolId: this.toolId,
            args: this.getInputArgs(),
            stdin: this.getInputStdin(),
            stdinPanelShown:
                // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
                (this.monacoStdin && this.monacoEditorOpen) || (this.panelStdin && !this.panelStdin.hasClass('d-none')),
            monacoStdin: this.monacoStdin,
            monacoEditorOpen: this.monacoEditorOpen,
            monacoEditorHasBeenAutoOpened: this.monacoEditorHasBeenAutoOpened,
            argsPanelShow: !this.panelArgs.hasClass('d-none'),
        };
        this.paneRenaming.addState(state);
        this.fontScale.addState(state);
        return state;
    }

    saveState() {
        this.container.setState(this.currentState());
    }

    setLanguage(languageId) {
        if (languageId) {
            this.options.enableToggle('wrap', false);
            const model = this.editor.getModel();
            if (model) {
                monaco.editor.setModelLanguage(model, languageId);
            }

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

    clickableUrls(text) {
        return text.replace(
            // URL detection regex grabbed from https://stackoverflow.com/a/3809435
            /(https?:\/\/(www\.)?[-a-zA-Z0-9@:%._+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_+.~#?&//=]*))/,
            '<a href="$1" target="_blank">$1</a>'
        );
    }

    onCompileResult(id: number, compiler: CompilerInfo, result: CompilationResult) {
        try {
            if (id !== this.compilerId) return;
            // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
            if (compiler) this.compilerName = compiler.name;

            const foundTool = Object.values(compiler.tools).find(tool => {
                // @ts-expect-error: tool.id is typeof string but this.toolId is typeof number
                return tool.id === this.toolId;
            });

            this.toggleUsable(!!foundTool);

            let toolResult: ToolInfo | null = null;
            // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
            if (result && result.tools) {
                toolResult = _.find(result.tools, tool => {
                    return tool.id === this.toolId;
                });

                // @ts-expect-error: CompilationResult has no member 'result'
                // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
            } else if (result && result.result && result.result.tools) {
                toolResult = _.find(
                    // @ts-expect-error: CompilationResult has no member 'result'
                    result.result.tools,
                    tool => {
                        return tool.id === this.toolId;
                    }
                );
            }

            let toolInfo: ToolInfo | null = null;
            // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
            if (compiler && compiler.tools) {
                toolInfo =
                    Object.values(compiler.tools).find(tool => {
                        // @ts-expect-error: tool.id is typeof string but this.toolId is typeof number
                        return tool.id === this.toolId;
                    }) ?? null;
            }

            if (toolInfo) {
                this.toggleStdin.prop('disabled', false);

                if (this.monacoStdin && !this.monacoEditorOpen && !this.monacoEditorHasBeenAutoOpened) {
                    this.monacoEditorHasBeenAutoOpened = true;
                    this.openMonacoEditor();
                    // @ts-expect-error: ToolInfo has no member 'tool'
                } else if (!this.monacoStdin && toolInfo.tool.stdinHint) {
                    // @ts-expect-error: ToolInfo has no member 'tool'
                    this.localStdinField?.prop('placeholder', toolInfo.tool.stdinHint);
                    // @ts-expect-error: ToolInfo has no member 'tool'
                    if (toolInfo.tool.stdinHint === 'disabled') {
                        this.toggleStdin.prop('disabled', true);
                    } else {
                        this.showPanel(this.toggleStdin, this.panelStdin);
                    }
                } else {
                    this.localStdinField?.prop('placeholder', 'Tool stdin...');
                }
            }

            // reset stream styles
            this.normalAnsiToHtml.reset();

            if (toolResult) {
                // @ts-expect-error: typescript handles this poorly
                if (toolResult.languageId && toolResult.languageId === 'stderr') {
                    toolResult.languageId = undefined;
                }

                this.setLanguage(toolResult.languageId);

                if (toolResult.languageId) {
                    // @ts-expect-error: ToolInfo has no member 'stdout'
                    this.setEditorContent(_.pluck(toolResult.stdout, 'text').join('\n'));
                } else {
                    this.plainContentRoot.empty();
                    _.each(
                        // @ts-expect-error: ToolInfo has no member 'stdout' or 'stderr'
                        (toolResult.stdout || []).concat(toolResult.stderr || []),
                        obj => {
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
                        },
                        this
                    );
                }

                this.toolName = toolResult.name;
                this.updateTitle();

                // @ts-expect-error: ToolInfo has no member 'sourceChanged'
                if (toolResult.sourcechanged && this.editorId) {
                    // @ts-expect-error: ToolInfo has no member 'newsource'
                    this.eventHub.emit('newSource', this.editorId, toolResult.newsource);
                }
                this.artifactBtn.off('click');
                // @ts-expect-error: ToolInfo has no member 'artifact'
                if (toolResult.artifact) {
                    this.artifactBtn.removeClass('d-none');
                    // @ts-expect-error: ToolInfo has no member 'artifact'
                    this.artifactText.text('Download ' + toolResult.artifact.title);
                    this.artifactBtn.on('click', () => {
                        // @ts-expect-error: ToolInfo has no member 'artifact'
                        // The artifact content can be passed either as plain text or as a base64 encoded binary file
                        if (toolResult.artifact.type === 'application/octet-stream') {
                            // @ts-expect-error: ToolInfo has no member 'artifact'
                            // Fetch is the most convenient non ES6 way to build a binary blob out of a base64 string
                            fetch('data:application/octet-stream;base64,' + toolResult.artifact.content)
                                .then(res => res.blob())
                                // @ts-expect-error: ToolInfo has no member 'artifact'
                                .then(blob => saveAs(blob, toolResult.artifact.name));
                        } else {
                            saveAs(
                                // @ts-expect-error: ToolInfo has no member 'artifact'
                                new Blob([toolResult.artifact.content], {
                                    // @ts-expect-error: ToolInfo has no member 'artifact'
                                    type: toolResult.artifact.type,
                                }),
                                // @ts-expect-error: ToolInfo has no member 'artifact'
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

    add(msg: string, lineNum?: number, column?: number, flow?: MessageWithLocation[]): void {
        const elem = $('<div/>').appendTo(this.plainContentRoot);
        if (lineNum && this.editorId) {
            elem.html(
                // @ts-expect-error: JQuery types are wrong
                $('<a></a>')
                    .prop('href', 'javascript:;')
                    .html(msg)
                    .on('click', e => {
                        // @ts-expect-error: 'editorSetDecoration' only accepts 4 arguments
                        this.eventHub.emit('editorSetDecoration', this.editorId, lineNum, true, column);
                        if (flow && this.editorId) {
                            this.eventHub.emit('editorDisplayFlow', this.editorId, flow);
                        }
                        e.preventDefault();
                        return false;
                    })
                    .on('mouseover', () => {
                        // @ts-expect-error: 'editorSetDecoration' only accepts 4 arguments
                        this.eventHub.emit('editorSetDecoration', this.editorId, lineNum, false, column);
                    })
            );
        } else {
            elem.html(msg);
        }
    }

    setEditorContent(content) {
        const editorModel = this.editor.getModel();
        if (!editorModel) return;
        const visibleRanges = this.editor.getVisibleRanges();
        const currentTopLine = visibleRanges.length > 0 ? visibleRanges[0].startLineNumber : 1;
        editorModel.setValue(content);
        this.editor.revealLine(currentTopLine);
        this.setNormalContent();
    }

    setNormalContent() {
        this.editor.updateOptions({
            lineNumbers: 'on',
            codeLens: false,
        });
        if (this.codeLensProvider) {
            this.codeLensProvider.dispose();
        }
    }

    override getDefaultPaneName(): string {
        return '';
    }

    override getPaneName(): string {
        let name = this.toolName + ' #' + this.compilerId;
        if (this.compilerName) name += ' with ' + this.compilerName;
        return name;
    }

    override updateTitle(): void {
        const name = this.paneName ? this.paneName : this.getPaneName();
        this.container.setTitle(_.escape(name));
    }

    override onCompilerClose(id: number): void {
        if (id === this.compilerId) {
            this.close();
            _.defer(() => {
                this.container.close();
            });
        }
    }

    close(): void {
        this.eventHub.emit('toolClosed', this.compilerId, this.currentState());
        this.eventHub.unsubscribe();
        this.editor.dispose();
    }

    onCompiler(compilerId: number, compiler: unknown, options: string, editorId: number, treeId: number): void {}
}
