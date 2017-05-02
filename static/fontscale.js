// Copyright (c) 2012-2017, Matt Godbolt
//
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

define(function (require) {
    "use strict";
    const _ = require('underscore');
    const EventEmitter = require('events');

    function makeFontSizeDropdown(elem, interval, obj) {
        const factor = obj.isFontOfStr ? 10 : 14;
        let step = 0.05;
        let found = false;

        const clickHandle = function () {
            // Toggle off the selection of the others
            elem.children().removeClass('font-option-active');
            // Toggle us on
            $(this).addClass("font-option-active");
            // Send the data
            obj.scale = $(this).data("value");
            obj.apply();
            obj.emit('change');
        };

        for (let i = interval[0]; i <= interval[1]; i += step) {
            step *= 1.2;
            let newElementStr = '<li data-value="' + i + '" class="font-option';
            if (!found && (i === obj.scale || Math.floor(i) == obj.scale)) {
                found = true;
                newElementStr += ' font-option-active';
            }
            newElementStr += '">' + Math.round(i * factor) + '</li>';
            elem.append($(newElementStr).click(clickHandle));
        }
    }

    function FontScale(domRoot, state, fontSelectorOrEditor) {
        EventEmitter.call(this);
        this.domRoot = domRoot;
        this.scale = state.fontScale || 1.0;
        this.fontSelectorOrEditor = fontSelectorOrEditor;
        this.isFontOfStr = typeof(this.fontSelectorOrEditor) === "string";
        this.apply();
        makeFontSizeDropdown(this.domRoot.find('.font-size-list'), [0.3, 3], this);
    }

    _.extend(FontScale.prototype, EventEmitter.prototype);

    FontScale.prototype.apply = function () {
        if (this.isFontOfStr) {
            this.domRoot.find(this.fontSelectorOrEditor).css('font-size', (10 * this.scale) + "pt");
        } else {
            this.fontSelectorOrEditor.updateOptions({
                fontSize: 14 * this.scale
            });
        }
    };

    FontScale.prototype.addState = function (state) {
        if (this.scale != 1.0)
            state.fontScale = this.scale;
    };

    FontScale.prototype.setScale = function (scale) {
        this.scale = scale;
        this.apply();
    };

    return FontScale;
});
