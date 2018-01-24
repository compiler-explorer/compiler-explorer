// Copyright (c) 2012-2018, Matt Godbolt
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
    var FontScale = require('fontscale');
    var AnsiToHtml = require('ansi-to-html');

    function makeAnsiToHtml(color) {
        return new AnsiToHtml({
            fg: color ? color : '#333',
            bg: '#f5f5f5',
            stream: true,
            escapeXML: true
        });
    }

    function Output(hub, container, state) {
        this.container = container;
        this.compilerId = state.compiler;
        this.editorId = state.editor;
        this.eventHub = hub.createEventHub();
        this.domRoot = container.getElement();
        this.domRoot.html($('#compiler-output').html());
        this.contentRoot = this.domRoot.find(".content");
        this.compilerName = "";
        this.fontScale = new FontScale(this.domRoot, state, "pre");

        this.eventHub.on('compileResult', this.onCompileResult, this);
        this.eventHub.emit('resendCompilation', this.compilerId);
        this.eventHub.on('compilerFontScale', this.onFontScale, this);
        this.eventHub.on('compilerClose', this.onCompilerClose, this);

        this.updateCompilerName();
    }

    Output.prototype.onCompileResult = function (id, compiler, result) {
        if (id !== this.compilerId) return;
        if (compiler) this.compilerName = compiler.name;

        this.contentRoot.empty();

        var ansiToHtml = makeAnsiToHtml();

        _.each((result.stdout || []).concat(result.stderr || []), function (obj) {
            this.add(ansiToHtml.toHtml(obj.text), obj.tag ? obj.tag.line : obj.line);
        }, this);

        if (!result.execResult) {
            this.add("Compiler returned: " + result.code);
        } else {
            this.add("Program returned: " + result.execResult.code);
            if (result.execResult.stderr.length || result.execResult.stdout.length) {
                ansiToHtml = makeAnsiToHtml("red");
                _.each(result.execResult.stderr, function (obj) {
                    this.programOutput(ansiToHtml.toHtml(obj.text), "red");
                }, this);

                ansiToHtml = makeAnsiToHtml();
                _.each(result.execResult.stdout, function (obj) {
                    this.programOutput(ansiToHtml.toHtml(obj.text));
                }, this);
            }
        }

        this.updateCompilerName();
    };

    Output.prototype.onFontScale = function (id, scale) {
        if (id === this.compilerId) this.fontScale.setScale(scale);
    };

    Output.prototype.programOutput = function (msg, color) {
        var elem = $('<div></div>').appendTo(this.contentRoot)
            .html(msg)
            .css('font-family', '"Courier New", Courier, monospace');
        if (color)
            elem.css("color", color);
    };

    Output.prototype.add = function (msg, lineNum) {
        var elem = $('<div></div>').appendTo(this.contentRoot);
        if (lineNum) {
            elem.html($('<a></a>').prop('href', '#').html(msg))
                .click(_.bind(function (e) {
                    this.eventHub.emit('editorSetDecoration', this.editorId, lineNum, true);
                    // do not bring user to the top of index.html
                    // http://stackoverflow.com/questions/3252730
                    e.preventDefault();
                    return false;
                }, this))
                .on('mouseover', _.bind(function () {
                    this.eventHub.emit('editorSetDecoration', this.editorId, lineNum, false);
                }, this));
        } else {
            elem.html(msg);
        }
    };

    Output.prototype.updateCompilerName = function () {
        var name = "#" + this.compilerId;
        if (this.compilerName) name += " with " + this.compilerName;
        this.container.setTitle(name);
    };

    Output.prototype.onCompilerClose = function (id) {
        if (id === this.compilerId) {
            // We can't immediately close as an outer loop somewhere in GoldenLayout is iterating over
            // the hierarchy. We can't modify while it's being iterated over.
            _.defer(function (self) {
                self.container.close();
            }, this);
        }
    };

    return {
        Output: Output
    };
});
