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
    argumentParsers = require("./argument-parsers"),
    fs = require('fs-extra'),
    logger = require('./../logger').logger,
    semverParser = require('semver');

class LDCCompiler extends BaseCompiler {
    constructor(info, env) {
        super(info, env);
        this.compiler.supportsIntel = true;
    }

    optionsForFilter(filters, outputFilename) {
        let options = ['-g', '-of', this.filename(outputFilename)];
        if (filters.intel && !filters.binary) options = options.concat('-x86-asm-syntax=intel');
        if (!filters.binary) options = options.concat('-output-s');
        return options;
    }

    getArgumentParser() {
        return argumentParsers.Clang;
    }

    filterUserOptions(userOptions) {
        return userOptions.filter(option => option !== '-run');
    }

    isCfgCompiler() {
        return true;
    }

    couldSupportASTDump(version) {
        const versionRegex = /\((\d\.\d+)\.\d+/;
        const versionMatch = versionRegex.exec(version);
        return versionMatch ? semverParser.compare(versionMatch[1] + ".0", "1.4.0") >= 0 : false;
    }

    generateAST(inputFilename, options) {
        // These options make LDC produce an AST dump in a separate file `<inputFilename>.cg`.
        let newOptions = options.concat("-vcg-ast");
        let execOptions = this.getDefaultExecOptions();
        return this.runCompiler(this.compiler.exe, newOptions, this.filename(inputFilename), execOptions)
            .then(this.loadASTOutput);
    }

    loadASTOutput(output) {
        if (output.code !== 0) {
            return `Error generating AST: ${output.code}`;
        }
        // Load the AST output from the `.cg` file.
        // Demangling is not needed.
        const astFilename = output.inputFilename.concat(".cg");
        if (!fs.existsSync(astFilename)) {
            logger.warn(`LDC AST file ${astFilename} requested but it does not exist`);
            return "";
        }
        return fs.readFileSync(astFilename, 'utf-8');
    }
}

module.exports = LDCCompiler;
