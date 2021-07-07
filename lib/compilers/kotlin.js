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

import { KotlinParser } from './argument-parsers';
import { JavaCompiler } from './java';

export class KotlinCompiler extends JavaCompiler {
    static get key() {
        return 'kotlin';
    }

    getDefaultExecOptions() {
        const execOptions = super.getDefaultExecOptions();
        const javaHome = this.compilerProps(`compiler.${this.compiler.id}.java_home`);
        if (javaHome) {
            execOptions.env.JAVA_HOME = javaHome;
        }

        return execOptions;
    }

    getMainClassName() {
        return 'ExampleKt';
    }

    filterUserOptions(userOptions) {
        // filter options without extra arguments
        userOptions = (userOptions || []).filter(option =>
            option !== '-script' && option !== '-progressive' && !option.startsWith('-Xjavac'));

        const oneArgForbiddenList = new Set([
            // -d directory
            // Destination for generated class files
            '-d',
            // -jdk-home path
            // Include a custom JDK from the specified location
            // into the classpath instead of the default JAVA_HOME
            '-jdk-home',
            // -kotlin-home path
            // Path to the home directory of Kotlin compiler used for
            // discovery of runtime libraries
            '-kotlin-home',
        ]);

        // filter options with one argument
        return super.filterUserOptionsWithArg(userOptions, oneArgForbiddenList);
    }

    optionsForFilter(filters) {
        // Forcibly enable javap
        filters.binary = true;

        return [
            '-Xjavac-arguments="-Xlint:all"',
        ];
    }

    getArgumentParser() {
        return KotlinParser;
    }
}
