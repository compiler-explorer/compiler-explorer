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
    var $ = require('jquery');
    var _ = require('underscore');

    function LoadSave() {
        this.modal = $('#load-save');
        this.onLoad = _.identity;

        $.getJSON('/source/builtin/list', _.bind(function (list) {
            this.modal.find('.example:visible').remove();
            var examples = this.modal.find('.examples');
            var template = examples.find('.template.example');
            _.each(list, _.bind(function (elem) {
                template
                    .clone()
                    .removeClass('template')
                    .appendTo(examples)
                    .find('a')
                    .text(elem.name)
                    .click(_.bind(function () {
                        this.doLoad(elem.urlpart);
                    }, this));
            }, this));
        }, this));
    }

    LoadSave.prototype.run = function (onLoad) {
        this.onLoad = onLoad;
        this.modal.modal();
    };

    LoadSave.prototype.doLoad = function (urlpart) {
        // TODO: handle errors. consider promises...
        $.getJSON('/source/builtin/load/' + urlpart, _.bind(function (response) {
            this.onLoad(response.file);
        }, this));
        this.modal.modal('hide');
    };

    return {LoadSave: LoadSave};
});
