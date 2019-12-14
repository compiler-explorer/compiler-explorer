// Copyright (c) 2019, Compiler Explorer Authors
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

"use strict";

var
    $ = require('jquery'),
    _ = require('underscore'),
    ga = require('analytics'),
    history = require('./history');

function History() {
    this.modal = null;
}

History.prototype.initializeIfNeeded = function () {
    if (this.modal === null) {
        this.modal = $("#history");
    }
};

History.prototype.populateFromLocalStorage = function () {
    this.populate(
        this.modal.find('.historiccode'),
        _.map(history.list(), _.bind(function (data) {
            var dt = new Date(data.dt);
            return {
                name: dt.toString(),
                load: _.bind(function () {
                    this.onLoad(data);
                    this.modal.modal('hide');
                }, this)
            };
        }, this)));
};

History.prototype.populate = function (root, list) {
    root.find('li:not(.template)').remove();
    var template = root.find('.template');
    _.each(list, _.bind(function (elem) {
        template
            .clone()
            .removeClass('template')
            .appendTo(root)
            .find('a')
            .text(elem.name)
            .click(elem.load);
    }, this));
};

History.prototype.run = function (onLoad) {
    this.initializeIfNeeded();
    this.populateFromLocalStorage();
    this.onLoad = onLoad;
    this.modal.modal();
    ga.proxy('send', {
        hitType: 'event',
        eventCategory: 'OpenModalPane',
        eventAction: 'History'
    });
};

module.exports = {
    History: History
};
