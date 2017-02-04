// Copyright (c) 2012-2017, Matt Godbolt
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

var _ = require('underscore-node');
var logger = require('./logger').logger;
var utils = require('./utils');

var sourceTag = /^;\s*([0-9]+)\s*:/;
var ignoreAll = /^\s*include listing\.inc$/;
var fileFind = /^; File\s+(.*)$/;
var gccExplorerDir = /\\compiler-explorer-compiler/; // has to match part of the path in compile-handler.js (ugly)
// Parse into:
// * optional leading whitespace
// * middle part
// * comment part
var parseRe = /^(\s*)([^;]*)(;.*)?$/;
var isProc = /.*PROC\s*$/;
var isEndp = /.*ENDP\s*$/;
var constDef = /^([a-zA-Z_$@][a-zA-Z_$@0-9.]*)\s*=.*$/;
var labelFind = /[.a-zA-Z_@$][a-zA-Z_$@0-9.]*/g;
var labelDef = /^(.*):\s*$/;
// Anything identifier-looking with a "@@" in the middle, and a comment at the end
// is treated as a mangled name. The comment will be used to replace the identifier.
var mangledIdentifier = /\?[^ ]+@@[^ |]+/;
var commentedLine = /([^;]+);\s*(.*)/;
var numberRe = /^\s+(([0-9a-f]+\b\s*)(([0-9a-f][0-9a-f])+\b\s*)*)(.*)/;

function debug() {
    logger.debug.apply(logger, arguments);
}

function demangle(line) {
    var match, comment;
    if (!(match = line.match(mangledIdentifier))) return line;
    if (!(comment = line.match(commentedLine))) return line;
    return comment[1].trimRight().replace(match[0], comment[2]);
}

function AddrOpcoder() {
    var self = this;
    this.opcodes = [];
    this.offset = null;
    var prevOffset = -1;
    var prevOpcodes = [];
    this.hasOpcodes = function () {
        return self.offset !== null;
    };
    this.onLine = function (line) {
        var match = line.match(numberRe);
        self.opcodes = [];
        self.offset = null;
        if (!match) {
            prevOffset = -1;
            return line;
        }
        var restOfLine = match[5];
        var numbers = _.chain(match[1].split(/\s+/))
            .filter(function (x) {
                return x !== "";
            })
            .value();
        // If restOfLine is empty, we should accumulate offset opcodes...
        if (restOfLine === "") {
            if (prevOffset < 0) {
                // First in a batch of opcodes, so first is the offset
                prevOffset = parseInt(numbers[0], 16);
                prevOpcodes = numbers.splice(1);
            } else {
                prevOpcodes = prevOpcodes.concat(numbers);
            }
        } else {
            if (prevOffset >= 0) {
                // we had something from a prior line
                self.offset = prevOffset;
                self.opcodes = prevOpcodes.concat(numbers);
                prevOffset = -1;
            } else {
                self.offset = parseInt(numbers[0], 16);
                self.opcodes = numbers.splice(1);
            }
        }
        return "  " + restOfLine;
    };
}

function ClParser(filters) {
    this.filters = filters;
    this.opcoder = new AddrOpcoder();
    this.result = [];
    this.inMain = false;
    this.source = null;
    this.labels = {};
    this.currentLabel = null;
    debug("Parser created");
}

ClParser.prototype._add = function (obj) {
    if (obj.text === "") return;
    if (this.currentLabel) obj.label = this.currentLabel;
    obj.text = utils.expandTabs(obj.text);
    if (this.filters.binary && this.opcoder.hasOpcodes()) {
        obj.opcodes = this.opcoder.opcodes;
        obj.address = this.opcoder.offset;
    }
    if (obj.keep) {
        this.markUsed(obj.lineLabels);
    }
    this.result.push(obj);
    debug(obj);
};

// TODO: unify this with asm.js.
// "moose" example, for example has extraneous labels (|$M7|)
ClParser.prototype.markUsed = function (lineLabels) {
    _.each(lineLabels, function (val, label) {
        if (!this.labels[label]) {
            debug("Marking " + label + " as used");
        }
        this.labels[label] = true;
    }, this);
};

ClParser.prototype.addLine = function (line) {
    if (!!line.match(ignoreAll)) return;
    line = this.opcoder.onLine(line);
    if (line.trim() === "") {
        this._add({keep: true, text: "", source: null});
        return;
    }

    var match;
    if (!!(match = line.match(fileFind))) {
        this.inMain = !!match[1].match(gccExplorerDir);
        return;
    }
    if (!!(match = line.match(sourceTag))) {
        if (this.inMain)
            this.source = parseInt(match[1]);
        return;
    }

    line = demangle(line);

    match = line.match(parseRe);
    if (!match) {
        throw new Error("Unable to parse '" + line + "'");
    }

    var isIndented = match[1] !== "";
    var command = match[2];
    var comment = match[3] || "";
    var lineLabels = {};
    _.each(command.match(labelFind), function (label) {
        lineLabels[label] = true;
    }, this);
    if (isIndented && this.opcoder.hasOpcodes()) {
        this._add({keep: true, lineLabels: lineLabels, text: "\t" + command + comment, source: this.source});
    } else {
        var keep = !this.filters.directives;
        if (command.match(isProc))
            keep = true;
        if (command.match(isEndp)) {
            keep = true;
            this.source = null;
            this.currentLabel = null;
        }
        var tempDef = false;
        if (!!(match = command.match(labelDef))) {
            keep = !this.filters.labels;
            this.currentLabel = match[1];
            debug(match, this.currentLabel);
        }
        if (!!(match = command.match(constDef))) {
            keep = !this.filters.labels;
            this.currentLabel = match[1];
            debug(match, this.currentLabel);
            tempDef = true;
        }
        this._add({keep: keep, lineLabels: lineLabels, text: command + comment, source: null});
        if (tempDef) this.currentLabel = null;
    }
};

ClParser.prototype.findUsedInternal = function () {
    var changed = false;
    _.each(this.result, function (obj) {
        if (obj.keep || !this.labels[obj.label]) {
            return;
        }
        debug("Keeping", obj);
        obj.keep = true;
        this.markUsed(obj.lineLabels);
        changed = true;
    }, this);
    debug("changed", changed);
    return changed;
};

ClParser.prototype.findUsed = function () {
    // TODO write tests that cover dependent labels being used.
    var MaxIterations = 100;
    for (var i = 0; i < MaxIterations; ++i) {
        if (!this.findUsedInternal())
            return;
    }
};

ClParser.prototype.get = function () {
    this.findUsed();
    var lastWasEmpty = true;
    return _.chain(this.result)
        .filter(function (elem) {
            if (!elem.keep) return false;
            var thisIsEmpty = elem.text === "";
            if (thisIsEmpty && lastWasEmpty) return false;
            lastWasEmpty = thisIsEmpty;
            return true;
        })
        .map(function (elem) {
            return _.pick(elem, ['opcodes', 'address', 'source', 'text']);
        })
        .value();
};

function AsmParser(compilerProps) {
}

AsmParser.prototype.process = function (asm, filters) {
    var parser = new ClParser(filters);
    utils.eachLine(asm, function (line) {
        parser.addLine(line);
    });
    return parser.get();
};

module.exports = {
    ClParser: ClParser,
    AsmParser: AsmParser
};
