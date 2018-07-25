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

const BaseCompiler = require('../base-compiler'),
    AsmParser = require('../asm-parser-vc'),
    argumentParsers = require("./argument-parsers");


class CLCompiler extends BaseCompiler {
    constructor(info, env) {
        info.supportsFiltersInBinary = true;
        super(info, env);
        this.asm = new AsmParser();
    }

    exec(command, args, options) {
        options = options || {};
        if ((process.platform === "linux" || process.platform === "darwin") && command.toLowerCase().endsWith(".exe")) {
            args.unshift(command);
            command = this.env.ceProps("wine");
            options.env = options.env || {};
            // Force use of wine vcruntime (See change 45106c382)
            options.env.WINEDLLOVERRIDES = "vcruntime140=b";
            options.env.WINEDEBUG = "-all";
        }
        return super.exec(command, args, options);
    }

    filename(fn) {
        return process.platform === "linux" || process.platform === "darwin" ? 'Z:' + fn : super.filename(fn);
    }

    supportsObjdump() {
        return false;
    }

    getArgumentParser() {
        return argumentParsers.Base;
    }

    optionsForFilter(filters, outputFilename) {
        return [
            '/FA',
            '/c',
            '/Fa' + this.filename(outputFilename),
            '/Fo' + this.filename(outputFilename + '.obj')
        ];
    }

}

module.exports = CLCompiler;
