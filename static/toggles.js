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
        _.each(domRoot.find(".btn.active"), function (a) {
            var obj = $(a);
            result[obj.data().bind] = true;
        });
        return result;
    }

    function Toggles(root, state) {
        EventEmitter.call(this);
        this.domRoot = root;
        state = state || get(this.domRoot);
        this.domRoot.find('.btn')
            .click(_.bind(this.onClick, this))
            .each(function () {
                $(this).toggleClass('active', !!state[$(this).data().bind]);
            });
        this.state = get(this.domRoot);
    }

    _.extend(Toggles.prototype, EventEmitter.prototype);

    Toggles.prototype.get = function () {
        return _.clone(this.state);
    };

    Toggles.prototype.onClick = function (event) {
        var button = $(event.currentTarget);
        if (button.hasClass("disabled")) return;
        button.toggleClass('active');
        var before = this.state;
        var after = this.state = get(this.domRoot);
        this.emit('change', before, after);
    };

    return Toggles;
});
