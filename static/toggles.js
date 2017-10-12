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
    var _ = require('underscore');
    var $ = require('jquery');
    var EventEmitter = require('events');

    function get(domRoot) {
        var result = {};
        _.each(domRoot.find(".btn"), function (a) {
            var obj = $(a);
            result[obj.data().bind] = obj.hasClass("active");
        });
        return result;
    }

    function Toggles(root, state) {
        EventEmitter.call(this);
        this.domRoot = root;
        state = _.extend(get(this.domRoot), state);
        this.domRoot.find('.btn')
            .click(_.bind(this.onClick, this))
            .each(function () {
                $(this).toggleClass('active', !!state[$(this).data().bind]);
            });
        this.state = state;
    }

    _.extend(Toggles.prototype, EventEmitter.prototype);

    Toggles.prototype.get = function () {
        return _.clone(this.state);
    };

    Toggles.prototype.set = function (key, value) {
        this._change(function() {
            this.state[key] = value;
        }.bind(this));
    };

    Toggles.prototype._change = function(update) {
        var before = this.get();
        update();
        this.emit('change', before, this.get()); 
    };

    Toggles.prototype.onClick = function (event) {
        var button = $(event.currentTarget);
        if (button.hasClass("disabled")) return;
        button.toggleClass('active');
        this._change(function() {
            this.state = get(this.domRoot);
        }.bind(this));
    };

    return Toggles;
});
