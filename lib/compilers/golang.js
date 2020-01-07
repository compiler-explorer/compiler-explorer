// Copyright (c) 2016, Matt Godbolt & Rubén Rincón
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

const BaseCompiler = require('../base-compiler'),
    argumentParsers = require("./argument-parsers"),
    _ = require('underscore'),
    utils = require('../utils');

// Each arch has a list of jump instructions in
// Go source src/cmd/asm/internal/arch.
const jumpPrefixes = [
    'j',
    'b',

    // arm
    'cb',
    'tb',

    // s390x
    'cmpb',
    'cmpub'
];

class GolangCompiler extends BaseCompiler {
    convertNewGoL(code) {
        const re = /^\s+(0[xX]?[0-9A-Za-z]+)?\s?([0-9]+)\s*\(([^:]+):([0-9]+)\)\s*([A-Z]+)(.*)/;
        const reUnknown = /^\s+(0[xX]?[0-9A-Za-z]+)?\s?([0-9]+)\s*\(<unknown line number>\)\s*([A-Z]+)(.*)/;
        const reFunc = /TEXT\s+[".]*(\S+)\(SB\)/;
        let prevLine = null;
        let file = null;
        let fileCount = 0;
        let func = null;
        const funcCollisions = {};
        const labels = {};
        const usedLabels = {};
        const lines = code.map(obj => {
            let pcMatch = null;
            let fileMatch = null;
            let lineMatch = null;
            let ins = null;
            let args = null;

            const line = obj.text;
            let match = line.match(re);
            if (match) {
                pcMatch = match[2];
                fileMatch = match[3];
                lineMatch = match[4];
                ins = match[5];
                args = match[6];
            } else {
                match = line.match(reUnknown);
                if (match) {
                    pcMatch = match[2];
                    ins = match[3];
                    args = match[4];
                } else {
                    return null;
                }
            }

            match = line.match(reFunc);
            if (match) {
                // Normalize function name.
                func = match[1].replace(/[.()*]+/g, "_");

                // It's possible for normalized function names to collide.
                // Keep a count of collisions per function name. Labels get
                // suffixed with _[collisions] when collisions > 0.
                let collisions = funcCollisions[func];
                if (collisions == null) {
                    collisions = 0;
                } else {
                    collisions++;
                }

                funcCollisions[func] = collisions;
            }

            let res = [];
            if (pcMatch && !labels[pcMatch]) {
                // Create pseudo-label.
                let label = pcMatch.replace(/^0{0,4}/, '');
                let suffix = '';
                if (funcCollisions[func] > 0) {
                    suffix = `_${funcCollisions[func]}`;
                }

                label = `${func}_pc${label}${suffix}:`;
                if (!labels[label]) {
                    res.push(label);
                    labels[label] = true;
                }
            }

            if (fileMatch && file !== fileMatch) {
                fileCount++;
                res.push(`\t.file ${fileCount} "${fileMatch}"`);
                file = fileMatch;
            }

            if (lineMatch && prevLine !== lineMatch) {
                res.push(`\t.loc ${fileCount} ${lineMatch} 0`);
                prevLine = lineMatch;
            }

            ins = ins.toLowerCase();
            args = this.replaceJump(func, funcCollisions[func], ins, args, usedLabels);
            res.push(`\t${ins}${args}`);
            return res;
        });

        // Find unused pseudo-labels so they can be filtered out.
        const unusedLabels = _.mapObject(labels, (val, key) => !usedLabels[key]);

        return _.chain(lines)
            .flatten()
            .compact()
            .filter(line => !unusedLabels[line])
            .value()
            .join("\n");
    }

    replaceJump(func, collisions, ins, args, usedLabels) {
        // Check if last argument is a decimal number.
        const re = /(\s+)([0-9]+)(\s?)$/;
        const match = args.match(re);
        if (!match) {
            return args;
        }

        // Check instruction has a jump prefix
        if (_.any(jumpPrefixes, prefix => ins.startsWith(prefix))) {
            let label = `${func}_pc${match[2]}`;
            if (collisions > 0) {
                label += `_${collisions}`;
            }
            usedLabels[label + ":"] = true; // record label use for later filtering
            return `${match[1]}${label}${match[3]}`;
        }

        return args;
    }

    extractLogging(stdout) {
        const reLogging = /^<source>:([0-9]*):([0-9]*):\s(.*)/i;
        return stdout.filter(obj => obj.text.match(reLogging))
            .map(obj => obj.text)
            .join('\n');
    }

    async postProcess(result) {
        const logging = this.extractLogging(result.stdout);
        result.asm = this.convertNewGoL(result.stdout);
        result.stdout = utils.parseOutput(logging, result.inputFilename);
        return [result, ""];
    }

    optionsForFilter(filters, outputFilename) {
        // If we're dealing with an older version...
        if (this.compiler.id === '6g141') {
            return ['tool', '6g', '-g', '-o', outputFilename, '-S'];
        }
        return ['tool', 'compile', '-o', outputFilename, '-S'];
    }

    getDefaultExecOptions() {
        const execOptions = super.getDefaultExecOptions();
        const goroot = this.compilerProps("compiler." + this.compiler.id + ".goroot");
        if (goroot) {
            execOptions.env.GOROOT = goroot;
        }
        return execOptions;
    }

    getArgumentParser() {
        return argumentParsers.Clang;
    }
}

module.exports = GolangCompiler;
