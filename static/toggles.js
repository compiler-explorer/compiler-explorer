// Copyright (c) 2018, Compiler Explorer Authors
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


function Togglesv2(root, state) {
    EventEmitter.call(this);
    var buttons = root.find('.button-checkbox');
    var self = this;
    this.buttons = buttons;
    this.state = _.extend({}, state);
    // Based on https://bootsnipp.com/snippets/featured/jquery-checkbox-buttons
    buttons.each(function () {
        // Settings
        var $widget = $(this),
            $button = $widget.find('button'),
            $checkbox = $widget.find('input:checkbox'),
            bind = $button.data('bind'),
            settings = {
                on: {
                    icon: 'far fa-check-square'
                },
                off: {
                    icon: 'far fa-square'
                }
            };

        // Event Handlers
        $button.on('click', function (e) {
            $checkbox.prop('checked', !$checkbox.is(':checked'));
            $checkbox.triggerHandler('change');
            e.stopPropagation();
        });
        $checkbox.on('change', function () {
            updateDisplay();
        });

        // Actions
        function updateDisplay(forcedState) {
            if (forcedState !== undefined) {
                $checkbox.prop('checked', forcedState);
            }
            var isChecked = $checkbox.is(':checked');

            // Set the button's state
            $button.data('state', (isChecked) ? "on" : "off");

            // Set the button's icon
            $button.find('.state-icon')
                .removeClass()
                .addClass('state-icon ' + settings[$button.data('state')].icon);

            // Update the button's color
            $button.toggleClass('active', isChecked);
            if (forcedState === undefined) {
                self.set(bind, isChecked);
            }
        }

        // Initialization
        function init() {
            updateDisplay(self.state[bind]);

            // Inject the icon if applicable
            if ($button.find('.state-icon').length === 0) {
                $button.prepend('<i class="state-icon ' + settings[$button.data('state')].icon + '"></i> ');
            }
        }
        init();
    });
}

_.extend(Togglesv2.prototype, EventEmitter.prototype);

Togglesv2.prototype.get = function () {
    return _.clone(this.state);
};

Togglesv2.prototype.set = function (key, value) {
    this._change(function () {
        this.state[key] = value;
    }.bind(this));
};

Togglesv2.prototype.enableToggle = function (key, enable) {
    this.buttons.each(function () {
        var widget = $(this);
        var button = $(widget.find('button'));
        var bind = button.data('bind');
        if (bind === key) {
            button.prop("disabled", !enable);
        }
    });
};

Togglesv2.prototype._change = function (update) {
    var before = this.get();
    update();
    this.emit('change', before, this.get());
};

module.exports = Togglesv2;
