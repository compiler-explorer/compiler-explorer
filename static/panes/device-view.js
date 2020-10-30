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

var FontScale = require('../fontscale');
var monaco = require('monaco-editor');
var _ = require('underscore');
var $ = require('jquery');
var colour = require('../colour');
var ga = require('../analytics');
var monacoConfig = require('../monaco-config');

function DeviceAsm(hub, container, state) {
    this.container = container;
    this.eventHub = hub.createEventHub();
    this.domRoot = container.getElement();
    this.domRoot.html($('#device').html());

    this.decorations = {};
    this.prevDecorations = [];
    var root = this.domRoot.find('.monaco-placeholder');

    this.deviceEditor = monaco.editor.create(root[0], monacoConfig.extendConfig({
        language: 'asm',
        readOnly: true,
        glyphMargin: true,
        lineNumbersMinChars: 3,
    }));

    this._compilerid = state.id;
    this._compilerName = state.compilerName;
    this._editorid = state.editorid;

    this.awaitingInitialResults = false;
    this.selection = state.selection;

    this.settings = {};

    this.colours = [];
    this.deviceCode = [];
    this.lastColours = [];
    this.lastColourScheme = {};

    var selectize = this.domRoot.find('.change-device').selectize({
        sortField: 'name',
        valueField: 'name',
        labelField: 'name',
        searchField: ['name'],
        options: [],
        items: [],
    });

    this.selectize = selectize[0].selectize;

    this.initButtons(state);
    this.initCallbacks();
    this.initEditorActions();

    if (state && state.irOutput) {
        this.showDeviceAsmResults(state.irOutput);
    }
    this.setTitle();

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
            var desiredLine = ed.getPosition().lineNumber - 1;
            var source = this.deviceCode[desiredLine].source;
            if (source !== null && source.file === null) {
                // a null file means it was the user's source
                this.eventHub.emit('editorLinkLine', this._editorid, source.line, -1, true);
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
    this.deviceEditor.onMouseMove(_.bind(function (e) {
        this.mouseMoveThrottledFunction(e);
    }, this));

    this.cursorSelectionThrottledFunction =
        _.throttle(_.bind(this.onDidChangeCursorSelection, this), 500);
    this.deviceEditor.onDidChangeCursorSelection(_.bind(function (e) {
        this.cursorSelectionThrottledFunction(e);
    }, this));

    this.fontScale.on('change', _.bind(this.updateState, this));
    this.selectize.on('change', _.bind(this.onDeviceSelect, this));

    this.container.on('destroy', this.close, this);

    this.eventHub.on('compileResult', this.onCompileResponse, this);
    this.eventHub.on('compiler', this.onCompiler, this);
    this.eventHub.on('colours', this.onColours, this);
    this.eventHub.on('panesLinkLine', this.onPanesLinkLine, this);
    this.eventHub.on('compilerClose', this.onCompilerClose, this);
    this.eventHub.on('settingsChange', this.onSettingsChange, this);
    this.eventHub.emit('deviceViewOpened', this._compilerid);
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
    if (this._compilerid !== id) return;
    var devices = result.devices;
    var deviceNames = [];
    if (!devices) {
        this.showDeviceAsmResults([{text: '<No output>'}]);
    } else if (typeof devices == 'string') {
        this.showDeviceAsmResults([{text: 'Device extraction error:\n' + devices}]);
    } else {
        deviceNames = Object.keys(devices);
    }

    this.makeDeviceSelector(deviceNames);
    var selectedDevice = this.selectize.getValue();
    if (selectedDevice)
        this.showDeviceAsmResults(devices[selectedDevice].asm);

    // Why call this explicitly instead of just listening to the "colours" event?
    // Because the recolouring happens before this editors value is set using "showDeviceAsmResults".
    this.onColours(this._compilerid, this.lastColours, this.lastColourScheme);
};

DeviceAsm.prototype.makeDeviceSelector = function (deviceNames) {
    var selectize = this.selectize;
    var selected = this.selectize.getValue();

    _.each(selectize.options, function (p) {
        if (deviceNames.indexOf(p.name) === -1) {
            selectize.removeOption(p.name);
        }
    }, this);

    _.each(deviceNames, function (p) {
        selectize.addOption({
            name: p,
        });
    }, this);

    if (!selected && deviceNames.length > 0) {
        selected = deviceNames[0];
        selectize.setValue(selected, true);
    } else if (selected && deviceNames.indexOf(selected) === -1) {
        selectize.clear(true);
        this.showDeviceAsmResults(
            [{text: '<Device ' + selected + ' not found>'}]);
    }
};

DeviceAsm.prototype.onDeviceSelect = function () {
    this.eventHub.emit('deviceSettingsChanged', this._compilerid);
};

DeviceAsm.prototype.getPaneName = function () {
    return this._compilerName + ' Device Viewer (Editor #' + this._editorid + ', Compiler #' + this._compilerid + ')';
};

DeviceAsm.prototype.setTitle = function () {
    this.container.setTitle(this.getPaneName());
};

DeviceAsm.prototype.showDeviceAsmResults = function (deviceCode) {
    if (!this.deviceEditor) return;
    this.deviceCode = deviceCode;
    this.deviceEditor.getModel().setValue(
        deviceCode.length ? _.pluck(deviceCode, 'text').join('\n') : '<No device code>');

    if (!this.awaitingInitialResults) {
        if (this.selection) {
            this.deviceEditor.setSelection(this.selection);
            this.deviceEditor.revealLinesInCenter(this.selection.startLineNumber,
                this.selection.endLineNumber);
        }
        this.awaitingInitialResults = true;
    }
};

DeviceAsm.prototype.onCompiler = function (id, compiler, options, editorid) {
    if (id === this._compilerid) {
        this._compilerName = compiler ? compiler.name : '';
        this._editorid = editorid;
        this.setTitle();
        if (compiler && !compiler.supportsDeviceAsmView) {
            this.deviceEditor.setValue('<Device output is not supported for this compiler>');
        }
    }
};

DeviceAsm.prototype.onColours = function (id, colours, scheme) {
    this.lastColours = colours;
    this.lastColourScheme = scheme;

    if (id === this._compilerid) {
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
    if (id === this._compilerid) {
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
        id: this._compilerid,
        editorid: this._editorid,
        selection: this.selection,
    };
    this.fontScale.addState(state);
    return state;
};

DeviceAsm.prototype.onCompilerClose = function (id) {
    if (id === this._compilerid) {
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
            this.eventHub.emit('editorLinkLine', this._editorid, sourceLine, -1, false);
            this.eventHub.emit('panesLinkLine', this._compilerid, sourceLine, false, this.getPaneName());
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
        this.prevDecorations, _.flatten(_.values(this.decorations)));
};

DeviceAsm.prototype.clearLinkedLines = function () {
    this.decorations.linkedCode = [];
    this.updateDecorations();
};

DeviceAsm.prototype.onPanesLinkLine = function (compilerId, lineNumber, revealLine, sender) {
    if (Number(compilerId) === this._compilerid) {
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
        this.linkedFadeTimeoutId = setTimeout(_.bind(function () {
            this.clearLinkedLines();
            this.linkedFadeTimeoutId = -1;
        }, this), 5000);
        this.updateDecorations();
    }
};

DeviceAsm.prototype.close = function () {
    this.eventHub.unsubscribe();
    this.eventHub.emit('deviceViewClosed', this._compilerid);
    this.deviceEditor.dispose();
};

module.exports = {
    DeviceAsm: DeviceAsm,
};
