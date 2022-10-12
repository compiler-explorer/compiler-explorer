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
import * as colour from '../colour';
import {ga} from '../analytics';
import * as monacoConfig from '../monaco-config';
import TomSelect from 'tom-select';
import GoldenLayout from 'golden-layout';
import {Hub} from '../hub';
import {MonacoPane} from './pane';
import {DeviceAsmCode, DeviceAsmState} from './device-view.interfaces';
import {MonacoPaneState} from './pane.interfaces';
import {CompilerInfo} from '../../types/compiler.interfaces';
import {CompilationResult} from '../../types/compilation/compilation.interfaces';

type DeviceType = {
    languageId: string;
    asm: DeviceAsmCode[];
};

export class DeviceAsm extends MonacoPane<monaco.editor.IStandaloneCodeEditor, DeviceAsmState> {
    private decorations: Record<'linkedCode', monaco.editor.IModelDeltaDecoration[]>;
    private prevDecorations: string[];
    private selectedDevice: string;
    private devices: Record<string, DeviceType> | null;
    private colours: string[];
    private deviceCode: DeviceAsmCode[];
    private lastColours: Record<number, number>;
    private lastColourScheme: string;
    private selectize: TomSelect;
    private linkedFadeTimeoutId: NodeJS.Timeout | null;

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

        if (state.irOutput) {
            this.showDeviceAsmResults(state.irOutput);
        }
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
            })
        );
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
    }

    override registerButtons(state: DeviceAsmState): void {
        super.registerButtons(state);

        const changeDeviceEl = this.domRoot[0].querySelector('.change-device') as HTMLInputElement;
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

    override onCompileResult(id: number, compiler: CompilerInfo, result: CompilationResult): void {
        if (this.compilerInfo.compilerId !== id) return;
        // @ts-expect-error: CompilationResult does not have the 'devices' type
        this.devices = result.devices;
        let deviceNames: string[] = [];
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

    makeDeviceSelector(deviceNames: string[]): void {
        const selectize = this.selectize;

        for (const key in selectize.options) {
            if (deviceNames.includes(selectize.options[key].name)) {
                selectize.removeOption(selectize.options[key].name);
            }
        }

        deviceNames.forEach(deviceName => {
            selectize.addOption({name: deviceName});
        });

        if (!this.selectedDevice && deviceNames.length > 0) {
            this.selectedDevice = deviceNames[0];
            selectize.setValue(this.selectedDevice, true);
        } else if (this.selectedDevice && deviceNames.indexOf(this.selectedDevice) === -1) {
            selectize.clear(true);
            this.showDeviceAsmResults([{text: '<Device ' + this.selectedDevice + ' not found>'}]);
        } else {
            selectize.setValue(this.selectedDevice, true);
            this.updateDeviceAsm();
        }
    }

    onDeviceSelect(): void {
        this.selectedDevice = this.selectize.getValue() as string;
        this.updateState();
        this.updateDeviceAsm();
    }

    updateDeviceAsm(): void {
        if (this.selectedDevice && this.devices != null && this.selectedDevice in this.devices) {
            const languageId = this.devices[this.selectedDevice].languageId;
            this.showDeviceAsmResults(this.devices[this.selectedDevice].asm, languageId);
        } else {
            this.showDeviceAsmResults([{text: `<Device ${this.selectedDevice} not found>`}]);
        }
    }

    override getDefaultPaneName(): string {
        return 'Device Viewer';
    }

    showDeviceAsmResults(deviceCode: DeviceAsmCode[], languageId?: string): void {
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
        compiler: CompilerInfo | undefined,
        options: string,
        editorId: number,
        treeId: number
    ): void {
        if (id === this.compilerInfo.compilerId) {
            this.compilerInfo.compilerName = compiler ? compiler.name : '';
            this.compilerInfo.editorId = editorId;
            this.compilerInfo.treeId = treeId;
            this.updateTitle();
            // @ts-expect-error: CompilerInfo does not have the 'supportsDeviceAsmView' type
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
            this.deviceCode.forEach((x: DeviceAsmCode, index: number) => {
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

    onMouseMove(e: monaco.editor.IEditorMouseEvent): void {
        if (e.target.position === null) return;
        if (this.settings.hoverShowSource) {
            this.clearLinkedLines();
            const hoverCode = this.deviceCode[e.target.position.lineNumber - 1];
            // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
            if (hoverCode && this.compilerInfo.editorId != null) {
                // We check that we actually have something to show at this point!
                const sourceLine = hoverCode.source && !hoverCode.source.file ? hoverCode.source.line : -1;
                this.eventHub.emit('editorLinkLine', this.compilerInfo.editorId, sourceLine, -1, 0, false);
                this.eventHub.emit(
                    'panesLinkLine',
                    this.compilerInfo.compilerId,
                    sourceLine,
                    -1,
                    0,
                    false,
                    this.getPaneName()
                );
            }
        }
    }

    updateDecorations(): void {
        this.prevDecorations = this.editor.deltaDecorations(
            this.prevDecorations,
            Object.values(this.decorations).flatMap(x => x)
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
        sender: string
    ): void {
        if (Number(compilerId) === this.compilerInfo.compilerId) {
            const lineNums: number[] = [];
            this.deviceCode.forEach((irLine: DeviceAsmCode, i: number) => {
                if (irLine.source && irLine.source.file == null && irLine.source.line === lineNumber) {
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
            if (this.linkedFadeTimeoutId !== null) {
                clearTimeout(this.linkedFadeTimeoutId);
            }
            this.linkedFadeTimeoutId = setTimeout(() => {
                this.clearLinkedLines();
                this.linkedFadeTimeoutId = null;
            }, 5000);
            this.updateDecorations();
        }
    }

    override close(): void {
        this.eventHub.unsubscribe();
        this.eventHub.emit('deviceViewClosed', this.compilerInfo.compilerId);
        this.editor.dispose();
    }
}
