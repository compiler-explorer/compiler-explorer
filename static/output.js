// Copyright (c) 2012-2016, Matt Godbolt
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
    var FontScale = require('fontscale');

    function Output(hub, container, state) {
        var self = this;
        this.container = container;
        this.compilerId = state.compiler;
        this.editorId = state.editor;
        this.eventHub = hub.createEventHub();
        this.domRoot = container.getElement();
        this.domRoot.html($('#compiler-output').html());
        this.contentRoot = this.domRoot.find(".content");
        this.compiler = null;
        this.fontScale = new FontScale(this.domRoot, state, "pre");

        this.eventHub.on('compileResult', this.onCompileResult, this);
        this.eventHub.emit('resendCompilation', this.compilerId);
        this.eventHub.on('compilerFontScale', this.onFontScale, this);

        this.updateCompilerName();
    }

    Output.prototype.onCompileResult = function (id, compiler, result) {
        if (id !== this.compilerId) return;
        this.compiler = compiler;

        this.contentRoot.empty();

        _.each((result.stdout || []).concat(result.stderr || []), function (obj) {
            this.add(obj.text, obj.line);
        }, this);

        this.add("Compiler exited with result code " + result.code);

        this.updateCompilerName();
    };

    Output.prototype.onFontScale = function (id, scale) {
        if (id === this.compilerId) this.fontScale.setScale(scale);
    };

    Output.prototype.add = function (msg, lineNum) {
        var elem = $('<div></div>').appendTo(this.contentRoot);
        if (lineNum) {
            elem.html($('<a href="#">').text(lineNum + " : " + msg)).click(_.bind(function (e) {
                this.eventHub.emit('selectLine', this.editorId, lineNum);
                // do not bring user to the top of index.html
                // http://stackoverflow.com/questions/3252730
                e.preventDefault();
                return false;
            }, this));
        } else {
            elem.text(msg);
        }
    };

    Output.prototype.updateCompilerName = function () {
        var name = "#" + this.compilerId;
        if (this.compiler) name += " with " + this.compiler.name;
        this.container.setTitle(name);
    };

    return {
        Output: Output
    };
});
