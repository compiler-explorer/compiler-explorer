// Copyright (c) 2020, Compiler Explorer Authors
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

var FontScale = require('../widgets/fontscale').FontScale;
var monaco = require('monaco-editor');
var _ = require('underscore');
var $ = require('jquery');
var colour = require('../colour');
var ga = require('../analytics').ga;
var monacoConfig = require('../monaco-config');
var PaneRenaming = require('../widgets/pane-renaming').PaneRenaming;

var TomSelect = require('tom-select');

function DeviceAsm(hub, container, state) {
    this.container = container;
    this.eventHub = hub.createEventHub();
    this.domRoot = container.getElement();
    this.domRoot.html($('#device').html());

    this.decorations = {};
    this.prevDecorations = [];
    var root = this.domRoot.find('.monaco-placeholder');

    this.deviceEditor = monaco.editor.create(
        root[0],
        monacoConfig.extendConfig({
            language: 'asm',
            readOnly: true,
            glyphMargin: true,
            lineNumbersMinChars: 3,
        })
    );

    this._compilerId = state.id;
    this._compilerName = state.compilerName;
    this._editorId = state.editorid;
    this._treeId = state.treeid;

    this.awaitingInitialResults = false;
    this.selection = state.selection;
    this.selectedDevice = state.device || '';
    this.devices = null;

    this.settings = {};

    this.colours = [];
    this.deviceCode = [];
    this.lastColours = [];
    this.lastColourScheme = {};

    var changeDeviceEl = this.domRoot[0].querySelector('.change-device');
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
    this.initCallbacks();
    this.initEditorActions();

    if (state && state.irOutput) {
        this.showDeviceAsmResults(state.irOutput);
    }

    ga.proxy('send', {
        hitType: 'event',
        eventCategory: 'OpenViewPane',
        eventAction: 'DeviceAsm',
    });
}

DeviceAsm.prototype.initEditorActions = function () {
    this.deviceEditor.addAction({
        id: 'viewsource',
        label: 'Scroll to source',
        keybindings: [monaco.KeyMod.CtrlCmd | monaco.KeyCode.F10],
        keybindingContext: null,
        contextMenuGroupId: 'navigation',
        contextMenuOrder: 1.5,
        run: _.bind(function (ed) {
            var position = ed.getPosition();
            if (position != null) {
                var desiredLine = position.lineNumber - 1;
                var source = this.deviceCode[desiredLine].source;
                if (source !== null && source.file === null) {
                    // a null file means it was the user's source
                    this.eventHub.emit('editorLinkLine', this._editorId, source.line, -1, true);
                }
            }
        }, this),
    });
};

DeviceAsm.prototype.initButtons = function (state) {
    this.fontScale = new FontScale(this.domRoot, state, this.deviceEditor);

    this.topBar = this.domRoot.find('.top-bar');
};

DeviceAsm.prototype.initCallbacks = function () {
    this.linkedFadeTimeoutId = -1;
    this.mouseMoveThrottledFunction = _.throttle(_.bind(this.onMouseMove, this), 50);
    this.deviceEditor.onMouseMove(
        _.bind(function (e) {
            this.mouseMoveThrottledFunction(e);
        }, this)
    );

    this.cursorSelectionThrottledFunction = _.throttle(_.bind(this.onDidChangeCursorSelection, this), 500);
    this.deviceEditor.onDidChangeCursorSelection(
        _.bind(function (e) {
            this.cursorSelectionThrottledFunction(e);
        }, this)
    );

    this.fontScale.on('change', _.bind(this.updateState, this));
    this.selectize.on('change', _.bind(this.onDeviceSelect, this));
    this.paneRenaming.on('renamePane', this.updateState.bind(this));

    this.container.on('destroy', this.close, this);

    this.eventHub.on('compileResult', this.onCompileResponse, this);
    this.eventHub.on('compiler', this.onCompiler, this);
    this.eventHub.on('colours', this.onColours, this);
    this.eventHub.on('panesLinkLine', this.onPanesLinkLine, this);
    this.eventHub.on('compilerClose', this.onCompilerClose, this);
    this.eventHub.on('settingsChange', this.onSettingsChange, this);
    this.eventHub.emit('deviceViewOpened', this._compilerId);
    this.eventHub.emit('requestSettings');

    this.container.on('resize', this.resize, this);
    this.container.on('shown', this.resize, this);
};

// TODO: de-dupe with compiler etc
DeviceAsm.prototype.resize = function () {
    var topBarHeight = this.topBar.outerHeight(true);
    this.deviceEditor.layout({
        width: this.domRoot.width(),
        height: this.domRoot.height() - topBarHeight,
    });
};

DeviceAsm.prototype.onCompileResponse = function (id, compiler, result) {
    if (this._compilerId !== id) return;
    this.devices = result.devices;
    var deviceNames = [];
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
};

DeviceAsm.prototype.makeDeviceSelector = function (deviceNames) {
    var selectize = this.selectize;

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
};

DeviceAsm.prototype.onDeviceSelect = function () {
    this.selectedDevice = this.selectize.getValue();
    this.updateState();
    this.updateDeviceAsm();
};

DeviceAsm.prototype.updateDeviceAsm = function () {
    if (this.selectedDevice && this.devices[this.selectedDevice])
        this.showDeviceAsmResults(this.devices[this.selectedDevice].asm);
    else this.showDeviceAsmResults([{text: '<Device ' + this.selectedDevice + ' not found>'}]);
};

DeviceAsm.prototype.getPaneTag = function () {
    if (this._editorId) {
        return this._compilerName + ' (Editor #' + this._editorId + ', Compiler #' + this._compilerId + ')';
    } else {
        return this._compilerName + ' (Tree #' + this._treeId + ', Compiler #' + this._compilerId + ')';
    }
};

DeviceAsm.prototype.getDefaultPaneName = function () {
    return 'Device Viewer';
};

DeviceAsm.prototype.getPaneName = function () {
    return this.paneName ? this.paneName : this.getDefaultPaneName() + ' ' + this.getPaneTag();
};

DeviceAsm.prototype.updateTitle = function () {
    this.container.setTitle(_.escape(this.getPaneName()));
};

DeviceAsm.prototype.showDeviceAsmResults = function (deviceCode) {
    if (!this.deviceEditor) return;
    this.deviceCode = deviceCode;
    this.deviceEditor
        .getModel()
        .setValue(deviceCode.length ? _.pluck(deviceCode, 'text').join('\n') : '<No device code>');

    if (!this.awaitingInitialResults) {
        if (this.selection) {
            this.deviceEditor.setSelection(this.selection);
            this.deviceEditor.revealLinesInCenter(this.selection.startLineNumber, this.selection.endLineNumber);
        }
        this.awaitingInitialResults = true;
    }
};

DeviceAsm.prototype.onCompiler = function (id, compiler, options, editorId, treeId) {
    if (id === this._compilerId) {
        this._compilerName = compiler ? compiler.name : '';
        this._editorId = editorId;
        this._treeId = treeId;
        this.updateTitle();
        if (compiler && !compiler.supportsDeviceAsmView) {
            this.deviceEditor.setValue('<Device output is not supported for this compiler>');
        }
    }
};

DeviceAsm.prototype.onColours = function (id, colours, scheme) {
    this.lastColours = colours;
    this.lastColourScheme = scheme;

    if (id === this._compilerId) {
        var irColours = {};
        _.each(this.deviceCode, function (x, index) {
            if (x.source && x.source.file === null && x.source.line > 0 && colours[x.source.line - 1] !== undefined) {
                irColours[index] = colours[x.source.line - 1];
            }
        });
        this.colours = colour.applyColours(this.deviceEditor, irColours, scheme, this.colours);
    }
};

DeviceAsm.prototype.onCompilerClose = function (id) {
    if (id === this._compilerId) {
        // We can't immediately close as an outer loop somewhere in GoldenLayout is iterating over
        // the hierarchy. We can't modify while it's being iterated over.
        _.defer(function (self) {
            self.container.close();
        }, this);
    }
};

DeviceAsm.prototype.updateState = function () {
    this.container.setState(this.currentState());
};

DeviceAsm.prototype.currentState = function () {
    var state = {
        id: this._compilerId,
        editorid: this._editorId,
        treeid: this._treeId,
        selection: this.selection,
        device: this.selectedDevice,
    };
    this.paneRenaming.addState(state);
    this.fontScale.addState(state);
    return state;
};

DeviceAsm.prototype.onCompilerClose = function (id) {
    if (id === this._compilerId) {
        // We can't immediately close as an outer loop somewhere in GoldenLayout is iterating over
        // the hierarchy. We can't modify while it's being iterated over.
        this.close();
        _.defer(function (self) {
            self.container.close();
        }, this);
    }
};

DeviceAsm.prototype.onSettingsChange = function (newSettings) {
    this.settings = newSettings;
    this.deviceEditor.updateOptions({
        contextmenu: newSettings.useCustomContextMenu,
        minimap: {
            enabled: newSettings.showMinimap,
        },
        fontFamily: newSettings.editorsFFont,
        fontLigatures: newSettings.editorsFLigatures,
    });
};

DeviceAsm.prototype.onMouseMove = function (e) {
    if (e === null || e.target === null || e.target.position === null) return;
    if (this.settings.hoverShowSource === true && this.deviceCode) {
        this.clearLinkedLines();
        var hoverCode = this.deviceCode[e.target.position.lineNumber - 1];
        if (hoverCode) {
            // We check that we actually have something to show at this point!
            var sourceLine = hoverCode.source && !hoverCode.source.file ? hoverCode.source.line : -1;
            this.eventHub.emit('editorLinkLine', this._editorId, sourceLine, -1, false);
            this.eventHub.emit('panesLinkLine', this._compilerId, sourceLine, false, this.getPaneName());
        }
    }
};

DeviceAsm.prototype.onDidChangeCursorSelection = function (e) {
    if (this.awaitingInitialResults) {
        this.selection = e.selection;
        this.updateState();
    }
};

DeviceAsm.prototype.updateDecorations = function () {
    this.prevDecorations = this.deviceEditor.deltaDecorations(
        this.prevDecorations,
        _.flatten(_.values(this.decorations))
    );
};

DeviceAsm.prototype.clearLinkedLines = function () {
    this.decorations.linkedCode = [];
    this.updateDecorations();
};

DeviceAsm.prototype.onPanesLinkLine = function (compilerId, lineNumber, revealLine, sender) {
    if (Number(compilerId) === this._compilerId) {
        var lineNums = [];
        _.each(this.deviceCode, function (irLine, i) {
            if (irLine.source && irLine.source.file === null && irLine.source.line === lineNumber) {
                var line = i + 1;
                lineNums.push(line);
            }
        });
        if (revealLine && lineNums[0]) this.deviceEditor.revealLineInCenter(lineNums[0]);
        var lineClass = sender !== this.getPaneName() ? 'linked-code-decoration-line' : '';
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
        if (this.linkedFadeTimeoutId !== -1) {
            clearTimeout(this.linkedFadeTimeoutId);
        }
        this.linkedFadeTimeoutId = setTimeout(
            _.bind(function () {
                this.clearLinkedLines();
                this.linkedFadeTimeoutId = -1;
            }, this),
            5000
        );
        this.updateDecorations();
    }
};

DeviceAsm.prototype.close = function () {
    this.eventHub.unsubscribe();
    this.eventHub.emit('deviceViewClosed', this._compilerId);
    this.deviceEditor.dispose();
};

module.exports = {
    DeviceAsm: DeviceAsm,
};
