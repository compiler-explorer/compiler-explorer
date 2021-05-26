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
    this.rowTemplate = $('#filelisting-editor-tpl');
    this.namedItems = this.domRoot.find('.named-editors');
    this.unnamedItems = this.domRoot.find('.unnamed-editors');

    this.ourCompilers = {};

    this.isCMakeProject = state.isCMakeProject || false;
    this.compilerLanguageId = state.compilerLanguageId || '';
    this.files = state.files || [];
    this.newFileId = state.newFileId || 1;

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
        newFileId: this.newFileId,
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
    this.eventHub.on('compilerOpen', this.onCompilerOpen, this);
    this.eventHub.on('compilerClose', this.onCompilerClose, this);
};

Tree.prototype.onLanguageChange = function () {
    this.refresh();
};

Tree.prototype.sendCompilerChangesToEditor = function (compilerId) {
    _.each(this.files, _.bind(function (file) {
        if (file.isOpen && file.editorId > 0) {
            if (file.isIncluded) {
                this.eventHub.emit('treeCompilerEditorIncludeChange', this.id, file.editorId, compilerId);
            } else {
                this.eventHub.emit('treeCompilerEditorExcludeChange', this.id, file.editorId, compilerId);
            }
        }
    }, this));
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
    var file = this.getFileByEditorId(editorId);
    if (file) return;

    file = {
        fileId: this.newFileId,
        isIncluded: false,
        isOpen: true,
        isMainSource: false,
        filename: '',
        content: '',
        editorId: editorId,
        langId: '',
    };
    this.newFileId++;
    this.files.push(file);

    this.refresh();

    this.sendChangesToAllEditors();
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
    var file = this.getFileByFileId(fileId);
    if (file) {
        this.files = this.files.filter(function (obj) {
            return obj.fileId !== fileId;
        });

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
        item.find('.filename').html(editor.getPaneName());
    } else {
        item.find('.filename').html('Unknown file');
    }

    editButton.on('click', _.bind(function (e) {
        var fileId = $(e.currentTarget).parent('li').data('fileId');
        this.editFile(fileId);
    }, this));

    renameButton.on('click', _.bind(function (e) {
        var fileId = $(e.currentTarget).parent('li').data('fileId');
        this.renameFile(fileId, _.bind(function () {
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
            this.moveToExclude(fileId);
        }, this));
        this.namedItems.append(item);
    } else {
        stagingButton.removeClass('fa-minus').addClass('fa-plus');
        stagingButton.on('click', _.bind(function (e) {
            var fileId = $(e.currentTarget).parent('li').data('fileId');
            this.moveToInclude(fileId);
        }, this));
        this.unnamedItems.append(item);
    }
};

Tree.prototype.refresh = function () {
    this.updateState();

    this.namedItems.html('');
    this.unnamedItems.html('');

    _.each(this.files, _.bind(function (file) {
        this.addRowToTreelist(file);
    }, this));
};

Tree.prototype.editFile = function (fileId) {
    var file = this.getFileByFileId(fileId);
    if (!file.isOpen) {
        var dragConfig = this.getConfigForNewEditor(file);
        file.isOpen = true;

        this.hub.addInEditorStackIfPossible(dragConfig);
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

Tree.prototype.getEditorIdByFilename = function (filename) {
    if (!filename) return false;

    var found = _.find(this.files, function (file) {
        if (filename.includes('/')) {
            if (filename.endsWith('/' + file.filename)) {
                return true;
            }
        } else if (filename === file.filename) {
            return true;
        }
        return false;
    });

    if (found) {
        return found.editorId;
    } else {
        return false;
    }
};

Tree.prototype.isEditorPartOfProject = function (editorId) {
    var found = _.find(this.files, function (file) {
        return (file.isIncluded) && file.isOpen && (editorId === file.editorId);
    });

    return !!found;
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

Tree.prototype.renameFile = function (fileId, callback) {
    var file = this.getFileByFileId(fileId);

    var suggestedFilename = file.filename;
    if (file.filename === '') {
        var langId = file.langId;
        if (file.isOpen && file.editorId > 0) {
            var editor = this.hub.getEditorById(file.editorId);
            langId = editor.currentLanguage.id;
            if (editor.customPaneName) {
                suggestedFilename = editor.customPaneName;
            }
        }

        if (!suggestedFilename) {
            var lang = languages[langId];
            var ext0 = lang.extensions[0];
            suggestedFilename = 'example' + ext0;
        }
    }

    this.alertSystem.enterSomething('Rename file', 'Please enter a filename', suggestedFilename, {
        yes: _.bind(function (value) {
            file.filename = value;

            if (file.isOpen && file.editorId > 0) {
                var editor = this.hub.getEditorById(file.editorId);
                editor.setCustomPaneName(file.filename);
            }

            callback();
        }, this),
    });
};

Tree.prototype.moveToInclude = function (fileId) {
    var file = this.getFileByFileId(fileId);

    if (file.filename === '') {
        this.renameFile(fileId, _.bind(function () {
            this.moveToInclude(fileId);
        }, this));
    } else {
        file.isIncluded = true;

        if (file.filename === 'CMakeLists.txt') {
            this.setAsCMakeProject();
            this.setAsMainSource(fileId);
        }

        this.refresh();
    }

    this.sendChangesToAllEditors();
};

Tree.prototype.moveToExclude = function (fileId) {
    var file = this.getFileByFileId(fileId);
    file.isIncluded = false;
    this.refresh();

    this.sendChangesToAllEditors();
};

Tree.prototype.getFileContents = function (file) {
    if (file.isOpen) {
        var editor = this.hub.getEditorById(file.editorId);
        if (editor) {
            return editor.getSource();
        } else {
            file.isOpen = false;
            file.editorId = -1;
        }
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

Tree.prototype.getEditorIdForMainsource = function () {
    var mainFile = null;
    if (this.isCMakeProject) {
        mainFile = _.find(this.files, function (file) {
            return file.isIncluded && (file.filename === 'example.cpp');
        });

        if (mainFile) return mainFile.editorId;
    } else {
        mainFile = _.find(this.files, function (file) {
            return file.isMainSource && file.isIncluded;
        });

        if (mainFile) return mainFile.editorId;
    }

    return false;
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
    if (file.filename) {
        editor.componentState.customPaneName = file.filename;
    }

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
