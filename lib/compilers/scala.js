// Copyright (c) 2021, Compiler Explorer Authors
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

import Semver from 'semver';

import {asSafeVer} from '../utils';

import { ScalaParser } from './argument-parsers';
import { JavaCompiler } from './java';

export class ScalaCompiler extends JavaCompiler {
    static get key() {
        return 'scala';
    }

    constructor(compilerInfo, env) {
        super(compilerInfo, env);
        this.javaHome = this.compilerProps(`compiler.${this.compiler.id}.java_home`);
    }

    getDefaultExecOptions() {
        const execOptions = super.getDefaultExecOptions();
        if (this.javaHome) {
            execOptions.env.JAVA_HOME = this.javaHome;
        }

        return execOptions;
    }

    filterUserOptions(userOptions) {
        // filter options without extra arguments
        userOptions = userOptions.filter(option =>
            option !== '-Xscript');

        const oneArgForbiddenList = new Set([
            // -d directory
            // Destination for generated class files
            '-d',
        ]);

        // filter options with one argument
        return super.filterUserOptionsWithArg(userOptions, oneArgForbiddenList);
    }

    optionsForFilter(filters) {
        // Forcibly enable javap
        filters.binary = true;

        const scala2Opts = [
            '-Xlint:_',
        ];

        const scala3Opts = [
            '-deprecation',
        ];

        return Semver.gte(asSafeVer(this.compiler.semver), '3.0.0', true) ? scala3Opts : scala2Opts;
    }

    getArgumentParser() {
        return ScalaParser;
    }
}
