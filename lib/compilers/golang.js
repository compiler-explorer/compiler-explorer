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
    _ = require('underscore'),
    utils = require('../utils');

class GolangCompiler extends BaseCompiler {
    convertNewGoL(code) {
        const re = /^\s+(0[xX]?[0-9A-Za-z]+)?\s?[0-9]+\s*\(([^:]+):([0-9]+)\)\s*([A-Z]+)(.*)/;
        const reUnknown = /^\s+(0[xX]?[0-9A-Za-z]+)?\s?[0-9]+\s*\(<unknown line number>\)\s*([A-Z]+)(.*)/;
        let prevLine = null;
        let file = null;
        let fileCount = 0;
        return _.compact(code.map(obj => {
            const line = obj.text;
            let match = line.match(re);
            if (match) {
                let res = "";
                if (file !== match[2]) {
                    fileCount++;
                    res += "\t.file " + fileCount + ' "' + match[2] + '"\n';
                    file = match[2];
                }
                if (prevLine !== match[3]) {
                    res += "\t.loc " + fileCount + " " + match[3] + " 0\n";
                    prevLine = match[3];
                }
                return res + "\t" + match[4].toLowerCase() + match[5];
            } else {
                match = line.match(reUnknown);
                if (match) {
                    return "\t" + match[2].toLowerCase() + match[3];
                } else {
                    return null;
                }
            }

        })).join("\n");
    }

    extractLogging(stdout) {
        const reLogging = /^\/(.*):([0-9]*):([0-9]*):\s(.*)/i;
        return stdout.filter(obj => obj.text.match(reLogging)).
            map(obj => obj.text).
            join('\n');
    }

    postProcess(result) {
        const logging = this.extractLogging(result.stdout);
        result.asm = this.convertNewGoL(result.stdout);
        result.stdout = utils.parseOutput(logging, result.inputFilename);
        return Promise.resolve(result);
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
}

module.exports = GolangCompiler;
