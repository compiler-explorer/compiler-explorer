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

import {FontScale} from '../widgets/fontscale';
import * as monaco from 'monaco-editor';
import _ from 'underscore';
import $ from 'jquery';
import * as colour from '../colour';
import {ga} from '../analytics';
import * as monacoConfig from '../monaco-config';
import {PaneRenaming} from '../widgets/pane-renaming';
import TomSelect from 'tom-select';
import GoldenLayout from 'golden-layout';
import {Hub} from '../hub';
import {MonacoPane} from './pane';
import {DeviceAsmCode, DeviceAsmState} from './device-view.interfaces';
import {MonacoPaneState} from './pane.interfaces';

type DecorationEntry = {
    linkedCode: any[];
};

type DeviceType = {
    languageId: string;
    asm: DeviceAsmCode[];
};

export class DeviceAsm extends MonacoPane<monaco.editor.IStandaloneCodeEditor, DeviceAsmState> {
    private deviceEditor: monaco.editor.IStandaloneCodeEditor;
    private decorations: DecorationEntry;
    private prevDecorations: string[];
    private readonly _compilerId: number;
    private _compilerName: string;
    private _editorId: number | undefined;
    private _treeId: number | undefined;
    private awaitingInitialResults: boolean;
    private selectedDevice: string;
    private devices: Record<string, DeviceType> | null;
    private colours: string[];
    private deviceCode: DeviceAsmCode[];
    private lastColours: Record<number, number>;
    private lastColourScheme: string;
    private readonly selectize: TomSelect;
    private linkedFadeTimeoutId: NodeJS.Timeout | null;

    public constructor(hub: Hub, container: GoldenLayout.Container, state: DeviceAsmState & MonacoPaneState) {
        super(hub, container, state);

        this.prevDecorations = [];

        this._compilerId = state.id;
        this._compilerName = state.compilerName;
        this._editorId = state.editorid;
        this._treeId = state.treeid;

        this.awaitingInitialResults = false;
        this.selection = state.selection;
        this.selectedDevice = state.device || '';
        this.devices = null;

        this.colours = [];
        this.deviceCode = [];
        this.lastColours = [];
        this.lastColourScheme = '';

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

        this.paneRenaming = new PaneRenaming(this, state);

        this.initButtons(state);
        this.initEditorActions();

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

    initEditorActions() {
        this.deviceEditor.addAction({
            id: 'viewsource',
            label: 'Scroll to source',
            keybindings: [monaco.KeyMod.CtrlCmd | monaco.KeyCode.F10],
            keybindingContext: undefined,
            contextMenuGroupId: 'navigation',
            contextMenuOrder: 1.5,
            run: _.bind(ed => {
                const position = ed.getPosition();
                if (position != null) {
                    const desiredLine = position.lineNumber - 1;
                    const source = this.deviceCode[desiredLine].source;
                    if (source && source.file == null && this._editorId != null) {
                        // a null file means it was the user's source
                        this.eventHub.emit('editorLinkLine', this._editorId, source.line, -1, -1, true);
                    }
                }
            }, this),
        });
    }

    initButtons(state) {
        this.fontScale = new FontScale(this.domRoot, state, this.deviceEditor);

        this.topBar = this.domRoot.find('.top-bar');
    }

    override registerCallbacks() {
        this.linkedFadeTimeoutId = null;
        const mouseMoveThrottledFunction = _.throttle(this.onMouseMove.bind(this), 50);
        this.deviceEditor.onMouseMove(e => mouseMoveThrottledFunction(e));

        const cursorSelectionThrottledFunction = _.throttle(this.onDidChangeCursorSelection.bind(this), 500);
        this.deviceEditor.onDidChangeCursorSelection(e => cursorSelectionThrottledFunction(e));

        this.fontScale.on('change', _.bind(this.updateState, this));
        this.selectize.on('change', _.bind(this.onDeviceSelect, this));
        this.paneRenaming.on('renamePane', this.updateState.bind(this));

        this.container.on('destroy', this.close, this);

        this.eventHub.on('compileResult', this.onCompileResult, this);
        this.eventHub.on('compiler', this.onCompiler, this);
        this.eventHub.on('colours', this.onColours, this);
        this.eventHub.on('panesLinkLine', this.onPanesLinkLine, this);
        this.eventHub.on('compilerClose', this.onCompilerClose, this);
        this.eventHub.on('settingsChange', this.onSettingsChange, this);
        this.eventHub.emit('deviceViewOpened', this._compilerId);
        this.eventHub.emit('requestSettings');

        this.container.on('resize', this.resize, this);
        this.container.on('shown', this.resize, this);
    }

    // TODO: de-dupe with compiler etc
    override resize() {
        const topBarHeight = this.topBar.outerHeight(true);
        this.deviceEditor.layout({
            width: this.domRoot.width() ?? 0,
            height: this.domRoot.height() ?? 0 - (topBarHeight ?? 0),
        });
    }

    override onCompileResult(id: number, compiler: any, result: any): void {
        if (this._compilerId !== id) return;
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
        this.onColours(this._compilerId, this.lastColours, this.lastColourScheme);
    }

    makeDeviceSelector(deviceNames: string[]): void {
        const selectize = this.selectize;

        _.each(
            selectize.options,
            function (p) {
                if (deviceNames.indexOf(p.name) === -1) {
                    selectize.removeOption(p.name);
                }
            },
            this
        );

        _.each(
            deviceNames,
            function (p) {
                selectize.addOption({name: p});
            },
            this
        );

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

    onDeviceSelect() {
        this.selectedDevice = this.selectize.getValue() as string;
        this.updateState();
        this.updateDeviceAsm();
    }

    updateDeviceAsm() {
        // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
        if (this.selectedDevice && this.devices != null && this.devices[this.selectedDevice]) {
            const languageId = this.devices[this.selectedDevice].languageId;
            this.showDeviceAsmResults(this.devices[this.selectedDevice].asm, languageId);
        } else {
            this.showDeviceAsmResults([{text: '<Device ' + this.selectedDevice + ' not found>'}]);
        }
    }

    override getPaneTag() {
        if (this._editorId) {
            return this._compilerName + ' (Editor #' + this._editorId + ', Compiler #' + this._compilerId + ')';
        } else {
            return this._compilerName + ' (Tree #' + this._treeId + ', Compiler #' + this._compilerId + ')';
        }
    }

    getDefaultPaneName() {
        return 'Device Viewer';
    }

    override getPaneName() {
        return this.paneName ? this.paneName : this.getDefaultPaneName() + ' ' + this.getPaneTag();
    }

    override updateTitle() {
        this.container.setTitle(_.escape(this.getPaneName()));
    }

    showDeviceAsmResults(deviceCode: DeviceAsmCode[], languageId?: string) {
        this.deviceCode = deviceCode;

        if (!languageId) {
            languageId = 'asm';
        }

        const model = this.deviceEditor.getModel();
        if (model) {
            monaco.editor.setModelLanguage(model, languageId);
            model.setValue(deviceCode.length ? _.pluck(deviceCode, 'text').join('\n') : '<No device code>');
        }

        if (!this.awaitingInitialResults) {
            if (this.selection) {
                this.deviceEditor.setSelection(this.selection);
                this.deviceEditor.revealLinesInCenter(this.selection.startLineNumber, this.selection.endLineNumber);
            }
            this.awaitingInitialResults = true;
        }
    }

    onCompiler(id, compiler, options, editorId, treeId) {
        if (id === this._compilerId) {
            this._compilerName = compiler ? compiler.name : '';
            this._editorId = editorId;
            this._treeId = treeId;
            this.updateTitle();
            if (compiler && !compiler.supportsDeviceAsmView) {
                this.deviceEditor.setValue('<Device output is not supported for this compiler>');
            }
        }
    }

    onColours(id: number, colours: Record<number, number>, scheme: string) {
        this.lastColours = colours;
        this.lastColourScheme = scheme;

        if (id === this._compilerId) {
            const irColours = {};
            _.each(this.deviceCode, function (x, index) {
                if (x.source && x.source.file == null && x.source.line > 0 && colours[x.source.line - 1]) {
                    irColours[index] = colours[x.source.line - 1];
                }
            });
            this.colours = colour.applyColours(this.deviceEditor, irColours, scheme, this.colours);
        }
    }

    override updateState() {
        this.container.setState(this.currentState());
    }

    currentState() {
        const state = {
            id: this._compilerId,
            editorid: this._editorId,
            treeid: this._treeId,
            selection: this.selection,
            device: this.selectedDevice,
        };
        this.paneRenaming.addState(state);
        this.fontScale.addState(state);
        return state;
    }

    override onCompilerClose(id) {
        if (id === this._compilerId) {
            // We can't immediately close as an outer loop somewhere in GoldenLayout is iterating over
            // the hierarchy. We can't modify while it's being iterated over.
            this.close();
            _.defer(function (self) {
                self.container.close();
            }, this);
        }
    }

    override onSettingsChange(newSettings) {
        this.settings = newSettings;
        this.deviceEditor.updateOptions({
            contextmenu: newSettings.useCustomContextMenu,
            minimap: {
                enabled: newSettings.showMinimap,
            },
            fontFamily: newSettings.editorsFFont,
            fontLigatures: newSettings.editorsFLigatures,
        });
    }

    onMouseMove(e: monaco.editor.IEditorMouseEvent) {
        if (e.target.position === null) return;
        if (this.settings.hoverShowSource) {
            this.clearLinkedLines();
            const hoverCode = this.deviceCode[e.target.position.lineNumber - 1];
            // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
            if (hoverCode && this._editorId != null) {
                // We check that we actually have something to show at this point!
                const sourceLine = hoverCode.source && !hoverCode.source.file ? hoverCode.source.line : -1;
                this.eventHub.emit('editorLinkLine', this._editorId, sourceLine, -1, 0, false);
                this.eventHub.emit('panesLinkLine', this._compilerId, sourceLine, -1, 0, false, this.getPaneName());
            }
        }
    }

    override onDidChangeCursorSelection(e) {
        if (this.awaitingInitialResults) {
            this.selection = e.selection;
            this.updateState();
        }
    }

    updateDecorations() {
        this.prevDecorations = this.deviceEditor.deltaDecorations(
            this.prevDecorations,
            _.flatten(_.values(this.decorations))
        );
    }

    clearLinkedLines() {
        this.decorations.linkedCode = [];
        this.updateDecorations();
    }

    onPanesLinkLine(compilerId, lineNumber, revealLine, sender) {
        if (Number(compilerId) === this._compilerId) {
            const lineNums: number[] = [];
            _.each(this.deviceCode, function (irLine, i) {
                if (irLine.source && irLine.source.file == null && irLine.source.line === lineNumber) {
                    const line = i + 1;
                    lineNums.push(line);
                }
            });
            if (revealLine && lineNums[0]) this.deviceEditor.revealLineInCenter(lineNums[0]);
            const lineClass = sender !== this.getPaneName() ? 'linked-code-decoration-line' : '';
            this.decorations.linkedCode = _.map(lineNums, function (line) {
                return {
                    range: new monaco.Range(line, 1, line, 1),
                    options: {
                        isWholeLine: true,
                        linesDecorationsClassName: 'linked-code-decoration-margin',
                        className: lineClass,
                    },
                };
            });
            if (this.linkedFadeTimeoutId !== null) {
                clearTimeout(this.linkedFadeTimeoutId);
            }
            this.linkedFadeTimeoutId = setTimeout(
                _.bind(() => {
                    this.clearLinkedLines();
                    this.linkedFadeTimeoutId = null;
                }, this),
                5000
            );
            this.updateDecorations();
        }
    }

    close() {
        this.eventHub.unsubscribe();
        this.eventHub.emit('deviceViewClosed', this._compilerId);
        this.deviceEditor.dispose();
    }
}
