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

    this.isCMakeProject = false;
    this.compilerLanguageId = '';
    this.files = [];

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

    this.eventHub.emit('findEditors');
}

Tree.prototype.updateState = function () {
    var state = {
        id: this.id,
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

Tree.prototype.onEditorOpen = function (editorId, editor) {
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
        editor: editor,
        langId: '',
    };
    this.files.push(file);

    this.refresh();
};

Tree.prototype.onEditorClose = function (editorId) {
    var file = this.getFileByEditorId(editorId);

    if (file) {
        file.isOpen = false;
        file.langId = file.editor.currentLanguage.id;
        file.content = file.editor.getSource();
        file.editor = null;
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
    var namedItems = this.domRoot.find('.named-editors');
    namedItems.html('');
    var unnamedItems = this.domRoot.find('.unnamed-editors');
    unnamedItems.html('');

    var editorTemplate = $('#filelisting-editor-tpl');
    _.each(this.files, _.bind(function (file) {
        var item = $(editorTemplate.children()[0].cloneNode(true));
        if (file.isIncluded) {
            item.data('fileId', file.fileId);
            item.find('.filename').html(file.filename);
            item.removeClass('fa-plus').addClass('fa-minus');
            item.on('click', _.bind(function (e) {
                var fileId = $(e.currentTarget).data('fileId');
                this.moveToExclude(fileId);
            }, this));
            namedItems.append(item);
        } else {
            item.data('fileId', file.fileId);
            if (file.filename) {
                item.find('.filename').html(file.filename);
            } else {
                item.find('.filename').html(file.editor.getPaneName());
            }
            item.removeClass('fa-minus').addClass('fa-plus');
            item.on('click', _.bind(function (e) {
                var fileId = $(e.currentTarget).data('fileId');
                this.moveToInclude(fileId);
            }, this));
            unnamedItems.append(item);
        }
    }, this));
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
        var langId = file.isOpen ? file.editor.currentLanguage.id : file.langId;
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
        return file.editor.getSource();
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

Tree.prototype.initButtons = function (/*state*/) {
    var addCompilerButton = this.domRoot.find('.add-compiler');
    var addExecutorButton = this.domRoot.find('.add-executor');

    var getCompilerConfig = _.bind(function () {
        return Components.getCompilerForTree(this.id, this.compilerLanguageId);
    }, this);

    var getExecutorConfig = _.bind(function () {
        return Components.getExecutorForTree(this.id, this.compilerLanguageId);
    }, this);

    var bindClickEvent = _.bind(function (dragSource, dragConfig) {
        dragSource.click(_.bind(function () {
            var insertPoint = this.hub.findParentRowOrColumn(this.container) ||
                this.container.layoutManager.root.contentItems[0];
            insertPoint.addChild(dragConfig);
        }, this));
    }, this);

    bindClickEvent(addCompilerButton, getCompilerConfig);
    bindClickEvent(addExecutorButton, getExecutorConfig);
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
};

module.exports = {
    Tree: Tree,
};
