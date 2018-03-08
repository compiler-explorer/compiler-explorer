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

const _ = require('underscore-node');
const logger = require('./logger').logger;
const utils = require('./utils');

const sourceTag = /^;\s*([0-9]+)\s*:/;
const ignoreAll = /^\s*include listing\.inc$/;
const fileFind = /^; File\s+(.*)$/;
const ceErrorLine = /^<.*>$/; // CE internal error line
const gccExplorerDir = /\\(compiler-explorer-compiler|tmp)/; // has to match part of the path in handlers/compile.js (ugly)
// Parse into:
// * optional leading whitespace
// * middle part
// * comment part
const parseRe = /^(\s*)([^;]*)(;.*)?$/;
const isProc = /.*PROC\s*$/;
const isEndp = /.*ENDP\s*$/;
const constDef = /^([a-zA-Z_$@][a-zA-Z_$@0-9.]*)\s*(=|DB|DD).*$/;
const labelFind = /[.a-zA-Z_@$][a-zA-Z_$@0-9.]*/g;
const labelDef = /^(.*):\s*$/;
// Anything identifier-looking with a "@@" in the middle, and a comment at the end
// is treated as a mangled name. The comment will be used to replace the identifier.
const mangledIdentifier = /\?[^ ]+@@[^ |]+/;
const commentedLine = /([^;]+);\s*(.*)/;
const numberRe = /^\s+(([0-9a-f]+\b\s*)(([0-9a-f][0-9a-f])+\b\s*)*)(.*)/;

function debug() {
    logger.debug.apply(logger, arguments);
}

function demangle(line) {
    let match, comment;
    if (!(match = line.match(mangledIdentifier))) return line;
    if (!(comment = line.match(commentedLine))) return line;
    return comment[1].trimRight().replace(match[0], comment[2]);
}

class AddrOpcoder {
    constructor() {
        this.opcodes = [];
        this.offset = null;
        this.prevOffset = -1;
        this.prevOpcodes = [];
    }

    hasOpcodes() {
        return this.offset !== null;
    }

    onLine(line) {
        const match = line.match(numberRe);
        this.opcodes = [];
        this.offset = null;
        if (!match) {
            this.prevOffset = -1;
            return line;
        }
        const restOfLine = match[5];
        const numbers = _.chain(match[1].split(/\s+/))
            .filter(function (x) {
                return x !== "";
            })
            .value();
        // If restOfLine is empty, we should accumulate offset opcodes...
        if (restOfLine === "") {
            if (this.prevOffset < 0) {
                // First in a batch of opcodes, so first is the offset
                this.prevOffset = parseInt(numbers[0], 16);
                this.prevOpcodes = numbers.splice(1);
            } else {
                this.prevOpcodes = this.prevOpcodes.concat(numbers);
            }
        } else {
            if (this.prevOffset >= 0) {
                // we had something from a prior line
                this.offset = this.prevOffset;
                this.opcodes = this.prevOpcodes.concat(numbers);
                this.prevOffset = -1;
            } else {
                this.offset = parseInt(numbers[0], 16);
                this.opcodes = numbers.splice(1);
            }
        }
        return "  " + restOfLine;
    }
}

class ClParser {
    constructor(filters) {
        this.filters = filters;
        this.opcoder = new AddrOpcoder();
        this.result = [];
        this.sourceFilename = null;
        this.source = null;
        this.labels = {};
        this.currentLabel = null;
        debug("Parser created");
    }

    _add(obj) {
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
    }

    // TODO: unify this with asm.js.
    // "moose" example, for example has extraneous labels (|$M7|)
    markUsed(lineLabels) {
        _.each(lineLabels, function (val, label) {
            if (!this.labels[label]) {
                debug("Marking " + label + " as used");
            }
            this.labels[label] = true;
        }, this);
    }

    addLine(line) {
        if (line.match(ignoreAll)) return;
        if (line.match(ceErrorLine)) {
            this._add({keep: true, text: line, source: null});
            return;
        }
        line = this.opcoder.onLine(line);
        if (line.trim() === "") {
            this._add({keep: true, text: "", source: null});
            return;
        }

        let match = line.match(fileFind);
        if (match) {
            if (match[1].match(gccExplorerDir)) {
                this.sourceFilename = null;
            } else {
                this.sourceFilename = match[1];
            }
            return;
        }
        match = line.match(sourceTag);
        if (match) {
            this.source = {file: this.sourceFilename, line: parseInt(match[1])};
            return;
        }

        line = demangle(line);

        match = line.match(parseRe);
        if (!match) {
            throw new Error("Unable to parse '" + line + "'");
        }

        const isIndented = match[1] !== "";
        const command = match[2];
        const comment = match[3] || "";
        const lineLabels = {};
        _.each(command.match(labelFind), function (label) {
            lineLabels[label] = true;
        }, this);
        if (isIndented && this.opcoder.hasOpcodes()) {
            this._add({keep: true, lineLabels: lineLabels, text: "\t" + command + comment, source: this.source});
        } else {
            let keep = !this.filters.directives;
            if (command.match(isProc))
                keep = true;
            if (command.match(isEndp)) {
                keep = true;
                this.source = null;
                this.currentLabel = null;
            }
            let tempDef = false;
            match = command.match(labelDef);
            if (match) {
                keep = !this.filters.labels;
                this.currentLabel = match[1];
                debug(match, this.currentLabel);
            }
            match = command.match(constDef);
            if (match) {
                keep = !this.filters.labels;
                this.currentLabel = match[1];
                debug(match, this.currentLabel);
                tempDef = true;
            }
            this._add({keep: keep, lineLabels: lineLabels, text: command + comment, source: null});
            if (tempDef) this.currentLabel = null;
        }
    }

    findUsedInternal() {
        let changed = false;
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
    }

    findUsed() {
        // TODO write tests that cover dependent labels being used.
        const MaxIterations = 100;
        for (let i = 0; i < MaxIterations; ++i) {
            if (!this.findUsedInternal())
                return;
        }
    }

    get() {
        this.findUsed();
        let lastWasEmpty = true;
        return _.chain(this.result)
            .filter(function (elem) {
                if (!elem.keep) return false;
                const thisIsEmpty = elem.text === "";
                if (thisIsEmpty && lastWasEmpty) return false;
                lastWasEmpty = thisIsEmpty;
                return true;
            })
            .map(function (elem) {
                return _.pick(elem, ['opcodes', 'address', 'source', 'text']);
            })
            .value();
    }
}


class AsmParser {
    process(asm, filters) {
        const parser = new ClParser(filters);
        utils.eachLine(asm, function (line) {
            parser.addLine(line);
        });
        return parser.get();
    }
}


module.exports = {
    ClParser: ClParser,
    AsmParser: AsmParser
};
