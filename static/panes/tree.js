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

    this.currentEditors = [];

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

Tree.prototype.onEditorOpen = function (id, editor) {
    if (!editor) return;
    if (_.find(this.currentEditors, function (obj) {
        return obj.id === id;
    })) return;

    this.currentEditors.push({
        id: id,
        isIncluded: false,
        isOpen: true,
        filename: '',
        content: '',
        editor: editor,
        langId: '',
    });

    this.refresh();
};

Tree.prototype.onEditorClose = function (id) {
    var editor = _.find(this.currentEditors, function (obj) {
        return obj.id === id;
    });

    if (editor) {
        editor.isOpen = false;
        editor.langId = editor.editor.currentLanguage.id;
        editor.content = editor.editor.getSource();
    }
};

Tree.prototype.removeEditor = function (id) {
    this.currentEditors = this.currentEditors.filter(function (obj) {
        return obj.id !== id;
    });
};

Tree.prototype.refresh = function () {
    var namedItems = this.domRoot.find('.named-editors');
    namedItems.html('');
    var unnamedItems = this.domRoot.find('.unnamed-editors');
    unnamedItems.html('');

    var editorTemplate = $('#filelisting-editor-tpl');
    _.each(this.currentEditors, _.bind(function (editor) {
        var item = $(editorTemplate.children()[0].cloneNode(true));
        if (editor.isIncluded) {
            item.data('id', editor.id);
            item.find('.filename').html(editor.filename);
            item.removeClass('fa-plus').addClass('fa-minus');
            item.on('click', _.bind(function (e) {
                var id = $(e.currentTarget).data('id');
                this.moveToExclude(id);
            }, this));
            namedItems.append(item);
        } else {
            item.data('id', editor.id);
            if (editor.filename) {
                item.find('.filename').html(editor.filename);
            } else {
                item.find('.filename').html(editor.editor.getPaneName());
            }
            item.removeClass('fa-minus').addClass('fa-plus');
            item.on('click', _.bind(function (e) {
                var id = $(e.currentTarget).data('id');
                this.moveToInclude(id);
            }, this));
            unnamedItems.append(item);
        }
    }, this));
};

Tree.prototype.moveToInclude = function (id) {
    var editor = _.find(this.currentEditors, function (obj) {
        return obj.id === id;
    });

    if (editor.filename === '') {
        var langId = editor.isOpen ? editor.editor.currentLanguage.id : editor.langId;
        var lang = languages[langId];
        var ext0 = lang.extensions[0];

        this.alertSystem.enterSomething('Filename', 'Please enter a filename for this editor', 'example' + ext0, {
            yes: _.bind(function (value) {
                editor.filename = value;
                this.moveToInclude(id);
            }, this),
        });
    } else {
        editor.isIncluded = true;
        this.refresh();
    }
};

Tree.prototype.moveToExclude = function (id) {
    var editor = _.find(this.currentEditors, function (obj) {
        return obj.id === id;
    });

    editor.isIncluded = false;
    this.refresh();
};

Tree.prototype.initButtons = function (/*state*/) {
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
