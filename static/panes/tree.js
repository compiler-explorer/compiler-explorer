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
var options = require('../options');
require('selectize');

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

    this.isCMakeProject = state.isCMakeProject || false;
    this.compilerLanguageId = state.compilerLanguageId || '';
    this.files = state.files || [];

    this.initButtons(state);
    this.initCallbacks();
    this.onSettingsChange(this.settings);

    this.updateTitle();
    this.updateState();
    ga.proxy('send', {
        hitType: 'event',
        eventCategory: 'OpenViewPane',
        eventAction: 'Tree',
    });

    this.refresh();
    this.eventHub.emit('findEditors');
}

Tree.prototype.updateState = function () {
    var state = {
        id: this.id,
        isCMakeProject: this.isCMakeProject,
        compilerLanguageId: this.compilerLanguageId,
        files: this.files,
    };
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
    this.eventHub.on('languageChange', this.onLanguageChange, this);
};

Tree.prototype.onLanguageChange = function () {
    this.refresh();
};

Tree.prototype.onEditorOpen = function (editorId) {
    var file = this.getFileByEditorId(editorId);
    if (file) return;

    var newFileId = this.files.length + 1;

    file = {
        fileId: newFileId,
        isIncluded: false,
        isOpen: true,
        isMainSource: false,
        filename: '',
        content: '',
        editorId: editorId,
        langId: '',
    };
    this.files.push(file);

    this.refresh();
};

Tree.prototype.onEditorClose = function (editorId) {
    var file = this.getFileByEditorId(editorId);

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
    this.files = this.files.filter(function (obj) {
        return obj.fileId !== fileId;
    });
};

Tree.prototype.refresh = function () {
    this.updateState();

    var namedItems = this.domRoot.find('.named-editors');
    namedItems.html('');
    var unnamedItems = this.domRoot.find('.unnamed-editors');
    unnamedItems.html('');

    var editorTemplate = $('#filelisting-editor-tpl');
    _.each(this.files, _.bind(function (file) {
        var item = $(editorTemplate.children()[0].cloneNode(true));

        item.data('fileId', file.fileId);
        if (file.filename) {
            item.find('.filename').html(file.filename);
        } else if (file.editorId > 0) {
            var editor = this.hub.getEditorById(file.editorId);
            item.find('.filename').html(editor.getPaneName());
        } else {
            item.find('.filename').html('Unknown file');
        }

        var stagingButton = item.find('.stage-file');
        var editButton = item.find('.edit-file');
        editButton.on('click', _.bind(function (e) {
            var fileId = $(e.currentTarget).parent('li').data('fileId');
            this.editFile(fileId);
        }, this));

        if (file.isIncluded) {
            stagingButton.removeClass('fa-plus').addClass('fa-minus');
            stagingButton.on('click', _.bind(function (e) {
                var fileId = $(e.currentTarget).parent('li').data('fileId');
                this.moveToExclude(fileId);
            }, this));
            namedItems.append(item);
        } else {
            stagingButton.removeClass('fa-minus').addClass('fa-plus');
            stagingButton.on('click', _.bind(function (e) {
                var fileId = $(e.currentTarget).parent('li').data('fileId');
                this.moveToInclude(fileId);
            }, this));
            unnamedItems.append(item);
        }
    }, this));
};

Tree.prototype.editFile = function (fileId) {
    var file = this.getFileByFileId(fileId);
    if (file.isOpen) {

    } else {
        var dragConfig = this.getConfigForNewEditor(file);
        file.isOpen = true;

        var insertPoint = this.hub.findParentRowOrColumn(this.container) ||
            this.container.layoutManager.root.contentItems[0];
        insertPoint.addChild(dragConfig);
    }
};

Tree.prototype.getFileByFileId = function (fileId) {
    return _.find(this.files, function (file) {
        return file.fileId === fileId;
    });
};

Tree.prototype.getFileByEditorId = function (editorId) {
    return _.find(this.files, function (file) {
        return file.editorId === editorId;
    });
};

Tree.prototype.setAsMainSource = function (mainFileId) {
    _.each(this.files, function (file) {
        file.isMainSource = false;
    });

    var mainfile = this.getFileByFileId(mainFileId);
    mainfile.isMainSource = true;
};

Tree.prototype.setAsCMakeProject = function () {
    this.compilerLanguageId = 'c++';
    this.isCMakeProject = true;
};

Tree.prototype.moveToInclude = function (fileId) {
    var file = _.find(this.files, function (obj) {
        return obj.fileId === fileId;
    });

    if (file.filename === '') {
        var editor = this.hub.getEditorById(file.editorId);
        var langId = file.isOpen ? editor.currentLanguage.id : file.langId;
        var lang = languages[langId];
        var ext0 = lang.extensions[0];

        this.alertSystem.enterSomething('Filename', 'Please enter a filename for this editor', 'example' + ext0, {
            yes: _.bind(function (value) {
                file.filename = value;
                this.moveToInclude(fileId);
            }, this),
        });
    } else {
        file.isIncluded = true;

        if (file.filename === 'CMakeLists.txt') {
            this.setAsCMakeProject();
            this.setAsMainSource(fileId);
        }

        this.refresh();
    }
};

Tree.prototype.moveToExclude = function (fileId) {
    var file = this.getFileByFileId(fileId);
    file.isIncluded = false;
    this.refresh();
};

Tree.prototype.getFileContents = function (file) {
    if (file.isOpen) {
        var editor = this.hub.getEditorById(file.editorId);
        return editor.getSource();
    } else {
        return file.content;
    }
};

Tree.prototype.getMainSource = function () {
    var mainFile = _.find(this.files, function (file) {
        return file.isMainSource && file.isIncluded;
    });

    if (mainFile) {
        return this.getFileContents(mainFile);
    } else {
        return '';
    }
};

Tree.prototype.getFiles = function () {
    var filtered = _.filter(this.files, function (file) {
        return !file.isMainSource && file.isIncluded;
    });

    return _.map(filtered, _.bind(function (file) {
        return {
            filename: file.filename,
            contents: this.getFileContents(file),
        };
    }, this));
};

Tree.prototype.bindClickToOpenPane = function (dragSource, dragConfig) {
    dragSource.on('click', _.bind(function () {
        var insertPoint = this.hub.findParentRowOrColumn(this.container) ||
            this.container.layoutManager.root.contentItems[0];
        insertPoint.addChild(_.bind(dragConfig, this));
    }, this));
};

Tree.prototype.getConfigForNewCompiler = function () {
    return Components.getCompilerForTree(this.id, this.compilerLanguageId);
};

Tree.prototype.getConfigForNewExecutor = function () {
    return Components.getCompilerForTree(this.id, this.compilerLanguageId);
};

Tree.prototype.getConfigForNewEditor = function (file) {
    file.editorId = this.hub.nextEditorId();
    var editor = Components.getEditor(
        file.editorId,
        file.compilerLanguageId
    );

    editor.componentState.source = file.content;

    return editor;
};

Tree.prototype.initButtons = function (/*state*/) {
    var addCompilerButton = this.domRoot.find('.add-compiler');
    var addExecutorButton = this.domRoot.find('.add-executor');

    this.bindClickToOpenPane(addCompilerButton, this.getConfigForNewCompiler);
    this.bindClickToOpenPane(addExecutorButton, this.getConfigForNewExecutor);
};

Tree.prototype.updateButtons = function () {
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
