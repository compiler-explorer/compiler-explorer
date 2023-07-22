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

import * as monaco from 'monaco-editor';
import _ from 'underscore';
import $ from 'jquery';
import * as colour from '../colour.js';
import {ga} from '../analytics.js';
import * as monacoConfig from '../monaco-config.js';
import TomSelect from 'tom-select';
import GoldenLayout from 'golden-layout';
import {Hub} from '../hub.js';
import {MonacoPane} from './pane.js';
import {DeviceAsmState} from './device-view.interfaces.js';
import {MonacoPaneState} from './pane.interfaces.js';
import {CompilerInfo} from '../../types/compiler.interfaces.js';
import {CompilationResult} from '../../types/compilation/compilation.interfaces.js';
import {ResultLine} from '../../types/resultline/resultline.interfaces.js';
import {assert} from '../assert.js';
import {Alert} from '../widgets/alert';
import {Compiler} from './compiler';
import {InstructionSet} from '../instructionsets.js';

export class DeviceAsm extends MonacoPane<monaco.editor.IStandaloneCodeEditor, DeviceAsmState> {
    private decorations: Record<string, monaco.editor.IModelDeltaDecoration[]>;
    private prevDecorations: string[];
    private selectedDevice: string;
    private devices: Record<string, CompilationResult> | null;
    private colours: string[];
    private deviceCode: ResultLine[];
    private lastColours: Record<number, number>;
    private lastColourScheme: string;
    private selectize: TomSelect;
    private linkedFadeTimeoutId: NodeJS.Timeout | null;
    private alertSystem: Alert;

    public constructor(hub: Hub, container: GoldenLayout.Container, state: DeviceAsmState & MonacoPaneState) {
        super(hub, container, state);

        this.prevDecorations = [];
        this.decorations = {
            linkedCode: [],
        };

        this.selection = state.selection;
        this.selectedDevice = state.device || '';
        this.devices = null;

        this.colours = [];
        this.deviceCode = [];
        this.lastColours = [];
        this.lastColourScheme = '';

        if (state.devices) {
            this.devices = state.devices;
        }

        if (state.deviceOutput) {
            this.showDeviceAsmResults(state.deviceOutput);
        } else if (state.devices) {
            this.onDevices(state.devices);
        }
        this.alertSystem = new Alert();
    }

    override getInitialHTML(): string {
        return $('#device').html();
    }

    override createEditor(editorRoot: HTMLElement): monaco.editor.IStandaloneCodeEditor {
        return monaco.editor.create(
            editorRoot,
            monacoConfig.extendConfig({
                language: 'asm',
                readOnly: true,
                glyphMargin: true,
                lineNumbersMinChars: 3,
            }),
        );
    }

    override getPrintName() {
        return 'Device Output';
    }

    override registerOpeningAnalyticsEvent(): void {
        ga.proxy('send', {
            hitType: 'event',
            eventCategory: 'OpenViewPane',
            eventAction: 'DeviceAsm',
        });
    }

    override registerEditorActions(): void {
        this.editor.addAction({
            id: 'viewsource',
            label: 'Scroll to source',
            keybindings: [monaco.KeyMod.CtrlCmd | monaco.KeyCode.F10],
            keybindingContext: undefined,
            contextMenuGroupId: 'navigation',
            contextMenuOrder: 1.5,
            run: ed => {
                const position = ed.getPosition();
                if (position != null) {
                    const desiredLine = position.lineNumber - 1;
                    const source = this.deviceCode[desiredLine].source;
                    if (source && source.file == null && this.compilerInfo.editorId != null) {
                        // a null file means it was the user's source
                        this.eventHub.emit('editorLinkLine', this.compilerInfo.editorId, source.line, -1, -1, true);
                    }
                }
            },
        });
        this.editor.addAction({
            id: 'viewasmdoc',
            label: 'View assembly documentation',
            keybindings: [monaco.KeyMod.CtrlCmd | monaco.KeyCode.F8],
            keybindingContext: undefined,
            // precondition: 'isAsmKeyword',
            contextMenuGroupId: 'help',
            contextMenuOrder: 1.5,
            run: this.onAsmToolTip.bind(this),
        });
    }
    async onAsmToolTip(ed: monaco.editor.ICodeEditor) {
        ga.proxy('send', {
            hitType: 'event',
            eventCategory: 'OpenModalPane',
            eventAction: 'AsmDocs',
        });
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
                '<br><br>If the documentation for this opcode is wrong or broken in some way, ' +
                'please feel free to <a href="' +
                newGitHubIssueUrl() +
                '" target="_blank" rel="noopener noreferrer">' +
                'open an issue on GitHub <sup><small class="fas fa-external-link-alt opens-new-window" ' +
                'title="Opens in a new window"></small></sup></a>.'
            );
        }

        try {
            const asmHelp = await Compiler.getAsmInfo(
                word.word,
                this.selectedDevice.split(' ')[0].toLowerCase() as InstructionSet,
            );
            if (asmHelp) {
                this.alertSystem.alert(opcode + ' help', asmHelp.html + appendInfo(asmHelp.url), {
                    onClose: () => {
                        ed.focus();
                        ed.setPosition(pos);
                    },
                });
            } else {
                this.alertSystem.notify('This token was not found in the documentation. Sorry!', {
                    group: 'notokenindocs',
                    alertClass: 'notification-error',
                    dismissTime: 5000,
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

    override registerButtons(state: DeviceAsmState): void {
        super.registerButtons(state);

        const changeDeviceEl = this.domRoot[0].querySelector('.change-device');
        assert(changeDeviceEl instanceof HTMLSelectElement);
        this.selectize = new TomSelect(changeDeviceEl, {
            sortField: 'name',
            valueField: 'name',
            labelField: 'name',
            searchField: ['name'],
            options: [],
            items: [],
            dropdownParent: 'body',
            plugins: ['input_autogrow'],
        });
    }

    override registerCallbacks(): void {
        this.linkedFadeTimeoutId = null;
        const mouseMoveThrottledFunction = _.throttle(this.onMouseMove.bind(this), 50);
        this.editor.onMouseMove(e => mouseMoveThrottledFunction(e));

        const cursorSelectionThrottledFunction = _.throttle(this.onDidChangeCursorSelection.bind(this), 500);
        this.editor.onDidChangeCursorSelection(e => cursorSelectionThrottledFunction(e));

        this.selectize.on('change', this.onDeviceSelect.bind(this));

        this.eventHub.on('colours', this.onColours, this);
        this.eventHub.on('panesLinkLine', this.onPanesLinkLine, this);
        this.eventHub.emit('deviceViewOpened', this.compilerInfo.compilerId);
        this.eventHub.emit('requestSettings');

        this.container.on('shown', this.resize, this);
    }

    onDevices(devices: Record<string, CompilationResult>): void {
        this.devices = devices;

        let deviceNames: string[] = [];
        // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
        if (!this.devices) {
            this.showDeviceAsmResults([{text: '<No output>'}]);
        } else {
            deviceNames = Object.keys(this.devices);
        }

        this.makeDeviceSelector(deviceNames);
        this.updateDeviceAsm();

        // Why call this explicitly instead of just listening to the "colours" event?
        // Because the recolouring happens before this editors value is set using "showDeviceAsmResults".
        this.onColours(this.compilerInfo.compilerId, this.lastColours, this.lastColourScheme);
    }

    override onCompileResult(id: number, compiler: CompilerInfo, result: CompilationResult): void {
        if (this.compilerInfo.compilerId !== id) return;

        if (result.devices) {
            this.onDevices(result.devices);
        }
    }

    makeDeviceSelector(deviceNames: string[]): void {
        const selectize = this.selectize;

        for (const key in selectize.options) {
            if (!deviceNames.includes(selectize.options[key].name)) {
                selectize.removeOption(selectize.options[key].name);
            }
        }

        deviceNames.forEach(deviceName => {
            selectize.addOption({name: deviceName});
        });

        if (!this.selectedDevice && deviceNames.length > 0) {
            this.selectedDevice = deviceNames[0];
            selectize.setValue(this.selectedDevice, true);
        } else if (this.selectedDevice && !deviceNames.includes(this.selectedDevice)) {
            selectize.clear(true);
            this.showDeviceAsmResults([{text: '<Device ' + this.selectedDevice + ' not found>'}]);
        } else {
            selectize.setValue(this.selectedDevice, true);
            this.updateDeviceAsm();
        }
    }

    override getCurrentState(): DeviceAsmState & MonacoPaneState {
        const state: DeviceAsmState & MonacoPaneState = {
            ...super.getCurrentState(),
            device: this.selectedDevice,
        };

        // note: this is disabled, because that overrides the selection again
        // if (this.devices) state.devices = this.devices;

        return state;
    }

    onDeviceSelect(): void {
        this.selectedDevice = this.selectize.getValue() as string;
        this.updateState();
        this.updateDeviceAsm();
        this.onColours(this.compilerInfo.compilerId, this.lastColours, this.lastColourScheme);
    }

    updateDeviceAsm(): void {
        if (this.selectedDevice && this.devices != null && this.selectedDevice in this.devices) {
            const devOutput = this.devices[this.selectedDevice];
            const languageId = devOutput.languageId;
            if (devOutput.asm) {
                this.showDeviceAsmResults(devOutput.asm, languageId);
            } else {
                this.showDeviceAsmResults(
                    [{text: `<Device ${this.selectedDevice} has errors>`}].concat(devOutput.stderr),
                );
            }
        } else {
            this.showDeviceAsmResults([{text: `<Device ${this.selectedDevice} not found>`}]);
        }
    }

    override getDefaultPaneName(): string {
        return 'Device Viewer';
    }

    showDeviceAsmResults(deviceCode: ResultLine[], languageId?: string): void {
        this.deviceCode = deviceCode;

        if (!languageId) {
            languageId = 'asm';
        }

        const model = this.editor.getModel();
        if (model) {
            monaco.editor.setModelLanguage(model, languageId);
            model.setValue(deviceCode.length ? deviceCode.map(d => d.text).join('\n') : '<No device code>');
        }

        if (!this.isAwaitingInitialResults) {
            if (this.selection) {
                this.editor.setSelection(this.selection);
                this.editor.revealLinesInCenter(this.selection.startLineNumber, this.selection.endLineNumber);
            }
            this.isAwaitingInitialResults = true;
        }
    }

    override onCompiler(
        id: number,
        compiler: CompilerInfo | null,
        options: string,
        editorId: number,
        treeId: number,
    ): void {
        if (id === this.compilerInfo.compilerId) {
            this.compilerInfo.compilerName = compiler ? compiler.name : '';
            this.compilerInfo.editorId = editorId;
            this.compilerInfo.treeId = treeId;
            this.updateTitle();
            if (compiler && !compiler.supportsDeviceAsmView) {
                this.editor.setValue('<Device output is not supported for this compiler>');
            }
        }
    }

    onColours(id: number, colours: Record<number, number>, scheme: string): void {
        this.lastColours = colours;
        this.lastColourScheme = scheme;

        // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
        if (id === this.compilerInfo.compilerId && this.deviceCode) {
            const irColours = {};
            this.deviceCode.forEach((x: ResultLine, index: number) => {
                if (x.source && x.source.file == null && x.source.line > 0 && colours[x.source.line - 1]) {
                    irColours[index] = colours[x.source.line - 1];
                }
            });
            this.colours = colour.applyColours(this.editor, irColours, scheme, this.colours);
        }
    }

    override onCompilerClose(id: number): void {
        if (id === this.compilerInfo.compilerId) {
            // We can't immediately close as an outer loop somewhere in GoldenLayout is iterating over
            // the hierarchy. We can't modify while it's being iterated over.
            this.close();
            _.defer(() => {
                this.container.close();
            });
        }
    }

    async onMouseMove(e: any) {
        if (e === null || e.target === null || e.target.position === null) return;
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
            const numericToolTip = Compiler.getNumericToolTip(word);
            if (numericToolTip) {
                this.decorations.numericToolTip = [
                    {
                        range: range,
                        options: {
                            isWholeLine: false,
                            hoverMessage: [
                                {
                                    // We use double `` as numericToolTip may include a single ` character.
                                    value: '``' + numericToolTip + '``',
                                },
                            ],
                        },
                    },
                ];
                this.updateDecorations();
            }
            const hoverShowAsmDoc = this.settings.hoverShowAsmDoc;
            if (hoverShowAsmDoc) {
                try {
                    const response = await Compiler.getAsmInfo(
                        currentWord.word,
                        this.selectedDevice.split(' ')[0].toLowerCase() as InstructionSet,
                    );
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

    getLineTokens(line: number): monaco.Token[] {
        const model = this.editor.getModel();
        if (!model || line > model.getLineCount()) return [];
        const flavour = model.getLanguageId();
        const tokens = monaco.editor.tokenize(model.getLineContent(line), flavour);
        return tokens.length > 0 ? tokens[0] : [];
    }

    updateDecorations(): void {
        this.prevDecorations = this.editor.deltaDecorations(
            this.prevDecorations,
            Object.values(this.decorations).flatMap(x => x),
        );
    }

    clearLinkedLines(): void {
        this.decorations.linkedCode = [];
        this.updateDecorations();
    }

    onPanesLinkLine(
        compilerId: number,
        lineNumber: number,
        _colBegin: number,
        _colEnd: number,
        revealLine: boolean,
        sender: string,
    ): void {
        // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
        if (Number(compilerId) === this.compilerInfo.compilerId && this.deviceCode) {
            const lineNums: number[] = [];
            this.deviceCode.forEach((line: ResultLine, i: number) => {
                if (line.source && line.source.file == null && line.source.line === lineNumber) {
                    const line = i + 1;
                    lineNums.push(line);
                }
            });
            if (revealLine && lineNums[0]) this.editor.revealLineInCenter(lineNums[0]);
            const lineClass = sender !== this.getPaneName() ? 'linked-code-decoration-line' : '';
            this.decorations.linkedCode = lineNums.map(line => ({
                range: new monaco.Range(line, 1, line, 1),
                options: {
                    isWholeLine: true,
                    linesDecorationsClassName: 'linked-code-decoration-margin',
                    className: lineClass,
                },
            }));

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
    }

    override close(): void {
        this.eventHub.unsubscribe();
        this.eventHub.emit('deviceViewClosed', this.compilerInfo.compilerId);
        this.editor.dispose();
    }
}
