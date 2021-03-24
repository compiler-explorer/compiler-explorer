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
require('selectize');

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
    this.editor.dispose();
};

module.exports = {
    Tree: Tree,
};
