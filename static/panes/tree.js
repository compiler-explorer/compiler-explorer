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

'use strict';
var _ = require('underscore');
var $ = require('jquery');
var Alert = require('../alert');
var Components = require('../components');
var local = require('../local');
var ga = require('../analytics');
var mf = require('../multifile-service');
var lc = require('../line-colouring');
var TomSelect = require('tom-select');
var Toggles = require('../toggles');
var LineColouring = lc.LineColouring;
var MultifileService = mf.MultifileService;
var options = require('../options');
var languages = options.languages;

function Tree(hub, state, container) {
    this.id = state.id || hub.nextTreeId();
    this.container = container;
    this.domRoot = container.getElement();
    this.domRoot.html($('#filelisting').html());
    this.hub = hub;
    this.eventHub = hub.createEventHub();
    this.settings = JSON.parse(local.get('settings', '{}'));

    this.httpRoot = window.httpRoot;

    this.alertSystem = new Alert();
    this.alertSystem.prefixMessage = 'Tree #' + this.id + ': ';

    this.root = this.domRoot.find('.tree');
    this.rowTemplate = $('#filelisting-editor-tpl');
    this.namedItems = this.domRoot.find('.named-editors');
    this.unnamedItems = this.domRoot.find('.unnamed-editors');

    this.langKeys = _.keys(languages);

    var usableLanguages = _.filter(languages, function (language) {
        return hub.compilerService.compilersByLang[language.id];
    });

    if (state) {
        if (!state.compilerLanguageId) {
            state.compilerLanguageId = this.settings.defaultLanguage;
        }
    } else {
        state = {
            compilerLanguageId: this.settings.defaultLanguage,
        };
    }

    this.multifileService = new MultifileService(this.hub, this.eventHub, this.alertSystem, state);
    this.lineColouring = new LineColouring(this.multifileService);
    this.ourCompilers = {};
    this.busyCompilers = {};
    this.asmByCompiler = {};

    this.initButtons(state);
    this.initCallbacks();
    this.onSettingsChange(this.settings);

    this.selectize = new TomSelect(this.languageBtn, {
        sortField: 'name',
        valueField: 'id',
        labelField: 'name',
        searchField: ['name'],
        options: _.map(usableLanguages, _.identity),
        items: [this.multifileService.getLanguageId()],
        dropdownParent: 'body',
        plugins: ['input_autogrow'],
        onChange: _.bind(this.onLanguageChange, this),
    });

    this.updateTitle();
    this.updateState();
    this.onLanguageChange(this.multifileService.getLanguageId());

    ga.proxy('send', {
        hitType: 'event',
        eventCategory: 'OpenViewPane',
        eventAction: 'Tree',
    });

    this.refresh();
    this.eventHub.emit('findEditors');
}

Tree.prototype.updateState = function () {
    var state = Object.assign({
        id: this.id,
    }, this.multifileService.getState());
    this.container.setState(state);

    this.updateButtons();
};

Tree.prototype.initCallbacks = function () {
    this.container.on('resize', this.resize, this);
    this.container.on('shown', this.resize, this);
    this.container.on('open', _.bind(function () {
        this.eventHub.emit('treeOpen', this.id);
    }, this));
    this.container.on('destroy', this.close, this);

    this.eventHub.on('editorOpen', this.onEditorOpen, this);
    this.eventHub.on('editorClose', this.onEditorClose, this);
    this.eventHub.on('compilerOpen', this.onCompilerOpen, this);
    this.eventHub.on('compilerClose', this.onCompilerClose, this);

    this.eventHub.on('compileResult', this.onCompileResponse, this);

    this.toggleCMakeButton.on('change', _.bind(this.onToggleCMakeChange, this));
};

Tree.prototype.onToggleCMakeChange = function () {
    var isOn = this.toggleCMakeButton.state.isCMakeProject;
    this.multifileService.setAsCMakeProject(isOn);

    this.domRoot.find('.cmake-project').prop('title',
        '[' + (isOn ? 'ON' : 'OFF') + '] CMake project');
    this.updateState();
};

Tree.prototype.onLanguageChange = function (newLangId) {
    if (languages[newLangId]) {
        this.multifileService.setLanguageId(newLangId);
        this.eventHub.emit('languageChange', false, newLangId, this.id);
    }

    this.toggleCMakeButton.enableToggle('isCMakeProject', this.multifileService.isCompatibleWithCMake());

    this.refresh();
};

Tree.prototype.sendCompilerChangesToEditor = function (compilerId) {
    this.multifileService.forEachOpenFile(_.bind(function (file) {
        if (file.isIncluded) {
            this.eventHub.emit('treeCompilerEditorIncludeChange', this.id, file.editorId, compilerId);
        } else {
            this.eventHub.emit('treeCompilerEditorExcludeChange', this.id, file.editorId, compilerId);
        }
    }, this));
};

Tree.prototype.isEditorPartOfProject = function (editorId) {
    return this.multifileService.isEditorPartOfProject(editorId);
};

Tree.prototype.getMainSource = function () {
    return this.multifileService.getMainSource();
};

Tree.prototype.getFiles = function () {
    return this.multifileService.getFiles();
};

Tree.prototype.getEditorIdByFilename = function (filename) {
    return this.multifileService.getEditorIdByFilename(filename);
};

Tree.prototype.sendChangesToAllEditors = function () {
    _.each(this.ourCompilers, _.bind(function (unused, compilerId) {
        this.sendCompilerChangesToEditor(Number(compilerId));
    }, this));
};

Tree.prototype.onCompilerOpen = function (compilerId, unused, treeId) {
    if (treeId === this.id) {
        this.ourCompilers[compilerId] = true;
        this.sendCompilerChangesToEditor(Number(compilerId));
    }
};

Tree.prototype.onCompilerClose = function (compilerId, unused, treeId) {
    if (treeId === this.id) {
        delete this.ourCompilers[compilerId];
    }
};

Tree.prototype.onEditorOpen = function (editorId) {
    var file = this.multifileService.getFileByEditorId(editorId);
    if (file) return;

    this.multifileService.addFileForEditorId(editorId);
    this.refresh();
    this.sendChangesToAllEditors();
};

Tree.prototype.onEditorClose = function (editorId) {
    var file = this.multifileService.getFileByEditorId(editorId);

    if (file) {
        file.isOpen = false;
        var editor = this.hub.getEditorById(editorId);
        file.langId = editor.currentLanguage.id;
        file.content = editor.getSource();
        file.editorId = -1;
    }

    this.refresh();
};

Tree.prototype.removeFile = function (fileId) {
    var file = this.multifileService.removeFileByFileId(fileId);
    if (file) {
        if (file.isOpen) {
            var editor = this.hub.getEditorById(file.editorId);
            if (editor) {
                editor.container.close();
            }
        }
    }

    this.refresh();
};

Tree.prototype.addRowToTreelist = function (file) {
    var item = $(this.rowTemplate.children()[0].cloneNode(true));
    var stagingButton = item.find('.stage-file');
    var editButton = item.find('.edit-file');
    var renameButton = item.find('.rename-file');
    var deleteButton = item.find('.delete-file');

    item.data('fileId', file.fileId);
    if (file.filename) {
        item.find('.filename').html(file.filename);
    } else if (file.editorId > 0) {
        var editor = this.hub.getEditorById(file.editorId);
        if (editor) {
            item.find('.filename').html(editor.getPaneName());
        } else {
            // wait for editor to appear first
            return;
        }
    } else {
        item.find('.filename').html('Unknown file');
    }

    editButton.on('click', _.bind(function (e) {
        var fileId = $(e.currentTarget).parent('li').data('fileId');
        this.editFile(fileId);
    }, this));

    renameButton.on('click', _.bind(function (e) {
        var fileId = $(e.currentTarget).parent('li').data('fileId');
        this.multifileService.renameFile(fileId).then(_.bind(function () {
            this.refresh();
        }, this));
    }, this));

    deleteButton.on('click', _.bind(function (e) {
        var fileId = $(e.currentTarget).parent('li').data('fileId');
        this.removeFile(fileId);
    }, this));

    if (file.isIncluded) {
        stagingButton.removeClass('fa-plus').addClass('fa-minus');
        stagingButton.on('click', _.bind(function (e) {
            var fileId = $(e.currentTarget).parent('li').data('fileId');
            this.multifileService.excludeByFileId(fileId).then(_.bind(function () {
                this.refresh();
            }, this));
        }, this));
        this.namedItems.append(item);
    } else {
        stagingButton.removeClass('fa-minus').addClass('fa-plus');
        stagingButton.on('click', _.bind(function (e) {
            var fileId = $(e.currentTarget).parent('li').data('fileId');
            this.multifileService.includeByFileId(fileId).then(_.bind(function () {
                this.refresh();
            }, this));
        }, this));
        this.unnamedItems.append(item);
    }
};

Tree.prototype.refresh = function () {
    this.updateState();

    this.namedItems.html('');
    this.unnamedItems.html('');

    this.multifileService.forEachFile(_.bind(function (file) {
        this.addRowToTreelist(file);
    }, this));
};

Tree.prototype.editFile = function (fileId) {
    var file = this.multifileService.getFileByFileId(fileId);
    if (!file.isOpen) {
        var dragConfig = this.getConfigForNewEditor(file);
        file.isOpen = true;

        this.hub.addInEditorStackIfPossible(dragConfig);
    } else {
        var editor = this.hub.getEditorById(file.editorId);
        this.hub.activateTabForContainer(editor.container);
    }
};

Tree.prototype.moveToInclude = function (fileId) {
    this.multifileService.includeByFileId(fileId).then(_.bind(function () {
        this.refresh();
        this.sendChangesToAllEditors();
    }, this));
};

Tree.prototype.moveToExclude = function (fileId) {
    this.multifileService.excludeByFileId(fileId).then(_.bind(function () {
        this.refresh();
        this.sendChangesToAllEditors();
    }, this));
};

Tree.prototype.bindClickToOpenPane = function (dragSource, dragConfig) {
    dragSource.on('click', _.bind(function () {
        this.hub.addInEditorStackIfPossible(_.bind(dragConfig, this));
    }, this));
};

Tree.prototype.getConfigForNewCompiler = function () {
    return Components.getCompilerForTree(this.id, this.compilerLanguageId);
};

Tree.prototype.getConfigForNewExecutor = function () {
    return Components.getCompilerForTree(this.id, this.compilerLanguageId);
};

Tree.prototype.getConfigForNewEditor = function (file) {
    var editor;
    var editorId = this.hub.nextEditorId();

    if (file) {
        file.editorId = editorId;
        editor = Components.getEditor(
            editorId,
            file.langId);

        editor.componentState.source = file.content;
        if (file.filename) {
            editor.componentState.customPaneName = file.filename;
        }
    } else {
        editor = Components.getEditor(
            editorId,
            this.multifileService.getLanguageId());
    }

    return editor;
};

Tree.prototype.initButtons = function (state) {
    var addCompilerButton = this.domRoot.find('.add-compiler');
    var addExecutorButton = this.domRoot.find('.add-executor');
    var addEditorButton = this.domRoot.find('.add-editor');

    this.bindClickToOpenPane(addCompilerButton, this.getConfigForNewCompiler);
    this.bindClickToOpenPane(addExecutorButton, this.getConfigForNewExecutor);
    this.bindClickToOpenPane(addEditorButton, this.getConfigForNewEditor);

    this.languageBtn = this.domRoot.find('.change-language');

    if (this.langKeys.length <= 1) {
        this.languageBtn.prop('disabled', true);
    }

    this.toggleCMakeButton = new Toggles(this.domRoot.find('.options'), state);
};

Tree.prototype.numberUsedLines = function () {
    if (_.any(this.busyCompilers)) return;

    if (!this.settings.colouriseAsm) {
        this.updateColoursNone();
        return;
    }

    this.lineColouring.clear();

    _.each(this.asmByCompiler, _.bind(function (asm, compilerId) {
        if (asm) this.lineColouring.addFromAssembly(parseInt(compilerId), asm);
    }, this));

    this.lineColouring.calculate();

    this.updateColours();
};

Tree.prototype.updateColours = function () {
    _.each(this.ourCompilers, _.bind(function (unused, compilerId) {
        var id = parseInt(compilerId);
        this.eventHub.emit('coloursForCompiler', id,
            this.lineColouring.getColoursForCompiler(id), this.settings.colourScheme);
    }, this));

    this.multifileService.forEachOpenFile(_.bind(function (file) {
        this.eventHub.emit('coloursForEditor', file.editorId,
            this.lineColouring.getColoursForEditor(file.editorId), this.settings.colourScheme);
    }, this));
};

Tree.prototype.updateColoursNone = function () {
    _.each(this.ourCompilers, _.bind(function (unused, compilerId) {
        this.eventHub.emit('coloursForCompiler', parseInt(compilerId), {}, this.settings.colourScheme);
    }, this));

    this.multifileService.forEachOpenFile(_.bind(function (file) {
        this.eventHub.emit('coloursForEditor', file.editorId, {}, this.settings.colourScheme);
    }, this));
};

Tree.prototype.onCompileResponse = function (compilerId, compiler, result) {
    if (!this.ourCompilers[compilerId]) return;

    this.busyCompilers[compilerId] = false;

    // todo: parse errors and warnings and relate them to lines in the code
    // note: requires info about the filename, do we currently have that?

    // eslint-disable-next-line max-len
    // {"text":"/tmp/compiler-explorer-compiler2021428-7126-95g4xc.zfo8p/example.cpp:4:21: error: expected ‘;’ before ‘}’ token"}

    if (result.result && result.result.asm) {
        this.asmByCompiler[compilerId] = result.result.asm;
    } else {
        this.asmByCompiler[compilerId] = result.asm;
    }

    this.numberUsedLines();
};

Tree.prototype.updateButtons = function () {
    // this.languageBtn ...
};

Tree.prototype.resize = function () {
};

Tree.prototype.onSettingsChange = function (/*newSettings*/) {
};

Tree.prototype.getPaneName = function () {
    return 'Tree #' + this.id;
};

Tree.prototype.updateTitle = function () {
    this.container.setTitle(this.getPaneName());
};

Tree.prototype.close = function () {
    this.eventHub.unsubscribe();
    this.eventHub.emit('treeClose', this.id);
    this.hub.removeTree(this.id);
};

module.exports = {
    Tree: Tree,
};
