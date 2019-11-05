// Copyright (c) 2016, Matt Godbolt
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
var _ = require('underscore');
var $ = require('jquery');
var EventEmitter = require('events');
var options = require('./options');

function makeFontSizeDropdown(elem, obj, buttonDropdown) {
    var onWheelEvent = function (e) {
        e.preventDefault();
        var selectedId = elem.find('.dropdown-item-active').index();
        if (e.originalEvent.deltaY >= 0 && selectedId < elem.children().length - 1) {
            selectedId++;
        } else if (e.originalEvent.deltaY < 0 && selectedId > 0) {
            selectedId--;
        }
        elem.children().eq(selectedId).trigger('click');
    };

    var onClickEvent = function () {
        // Toggle off the selection of the others
        $(this)
            .addClass('dropdown-item-active')
            .siblings().removeClass('dropdown-item-active');
        // Send the data
        obj.scale = $(this).data('value');
        obj.apply();
        obj.emit('change');
    };

    for (var i = 8; i < 30; i++) {
        var item = $('<button></button>');

        item.attr('data-value', i)
            .addClass('dropdown-item btn btn-sm')
            .text(i)
            .appendTo(elem)
            .click(onClickEvent);

        if (obj.scale === i) {
            item.addClass('dropdown-item-active');
        }
    }

    if (buttonDropdown) {
        buttonDropdown.on('wheel', onWheelEvent);
    }
}

function FontScale(domRoot, state, fontSelectorOrEditor) {
    EventEmitter.call(this);
    this.domRoot = domRoot;
    this.scale = state.fontScale || options.defaultFontScale;
    this.fontSelectorOrEditor = fontSelectorOrEditor;
    this.isFontOfStr = typeof (this.fontSelectorOrEditor) === "string";
    this.apply();
    makeFontSizeDropdown(this.domRoot.find('.font-size-list'), this, this.domRoot.find('.fs-button'));
}

_.extend(FontScale.prototype, EventEmitter.prototype);

FontScale.prototype.apply = function () {
    if (this.isFontOfStr) {
        this.domRoot.find(this.fontSelectorOrEditor).css('font-size', this.scale + "pt");
    } else {
        this.fontSelectorOrEditor.updateOptions({
            fontSize: this.scale
        });
    }
};

FontScale.prototype.addState = function (state) {
    state.fontScale = this.scale;
};

FontScale.prototype.setScale = function (scale) {
    this.scale = scale;
    this.apply();
};

module.exports = FontScale;
