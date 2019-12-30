// Copyright (c) 2018, Marc Tiehuis
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
    path = require('path'),
    Semver = require('semver');

class ZigCompiler extends BaseCompiler {
    constructor(info, env) {
        super(info, env);
        this.compiler.supportsIntel = true;
        this.compiler.supportsIrView = true;
        this.compiler.irArg = ['--emit', 'llvm-ir'];
    }

    getSharedLibraryPathsAsArguments() {
        return [];
    }

    preProcess(source) {
        if (this.compiler.semver === '0.2.0') {
            source += '\n';
            source += 'extern fn zig_panic() noreturn;\n';
            source += 'pub fn panic(msg: []const u8, error_return_trace: ' +
                '?&@import("builtin").StackTrace) noreturn {\n';
            source += '    zig_panic();\n';
            source += '}\n';
        } else {
            source += '\n';
            source += 'extern fn zig_panic() noreturn;\n';
            source += 'pub fn panic(msg: []const u8, error_return_trace: ' +
                '?*@import("builtin").StackTrace) noreturn {\n';
            source += '    zig_panic();\n';
            source += '}\n';
        }

        return source;
    }

    optionsForFilter(filters, outputFilename, userOptions) {
        let options = [filters.execute ? 'build-exe' : 'build-obj'];
        if (this.compiler.semver === 'trunk' || (this.compiler.semver && Semver.gt(this.compiler.semver, '0.3.0'))) {
            const outputDir = path.dirname(outputFilename);
            const desiredName = path.basename(outputFilename);
            // strip '.s' if we aren't executing
            const name = filters.execute ? desiredName : desiredName.slice(0, -2);
            options.push('--cache-dir', outputDir,
                '--output-dir', outputDir,
                '--name', name);
        } else {
            // Older versions use a different command line interface (#1304)
            options.push('--cache-dir', path.dirname(outputFilename),
                '--output', this.filename(outputFilename),
                '--output-h', '/dev/null');
        }

        if (!filters.binary) {
            let userRequestedEmit = _.any(userOptions, opt => opt.indexOf("--emit") > -1);
            if (!userRequestedEmit) {
                options = options.concat('--emit', 'asm');
            }
            if (filters.intel) options = options.concat('-mllvm', '--x86-asm-syntax=intel');
        }
        return options;
    }

    getIrOutputFilename(inputFilename) {
        return this.getOutputFilename(path.dirname(inputFilename), this.outputFilebase)
            .replace('.s', '.ll');
    }

    filterUserOptions(userOptions) {
        const blacklist = /^(((--(cache-dir|name|output|verbose))|(-mllvm)).*)$/;
        return _.filter(userOptions, option => !blacklist.test(option));
    }

    isCfgCompiler(/*compilerVersion*/) {
        return true;
    }
}

module.exports = ZigCompiler;
