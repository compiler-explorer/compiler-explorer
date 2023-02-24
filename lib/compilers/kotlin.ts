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

import {PreliminaryCompilerInfo} from '../../types/compiler.interfaces';
import {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces';

import {KotlinParser} from './argument-parsers';
import {JavaCompiler} from './java';

export class KotlinCompiler extends JavaCompiler {
    static override get key() {
        return 'kotlin';
    }

    javaHome: string;

    constructor(compilerInfo: PreliminaryCompilerInfo, env) {
        super(compilerInfo, env);
        this.javaHome = this.compilerProps<string>(`compiler.${this.compiler.id}.java_home`);
    }

    override getDefaultExecOptions() {
        const execOptions = super.getDefaultExecOptions();
        if (this.javaHome) {
            execOptions.env.JAVA_HOME = this.javaHome;
        }

        return execOptions;
    }

    override async getMainClassName() {
        return 'ExampleKt';
    }

    override filterUserOptions(userOptions: string[]) {
        // filter options without extra arguments
        userOptions = (userOptions || []).filter(
            option => option !== '-script' && option !== '-progressive' && !option.startsWith('-Xjavac')
        );

        const oneArgForbiddenList = new Set([
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

    override optionsForFilter(filters: ParseFiltersAndOutputOptions) {
        // Forcibly enable javap
        filters.binary = true;

        return ['-Xjavac-arguments="-Xlint:all"'];
    }

    /**
     * Handle Kotlin execution.
     *
     * Kotlin execution differs in the way that Kotlin requires its standard
     * standard library because that's where the runtime libraries such as
     * kotlin.jvm.internal.Intrinsics is.
     *
     * Therefore, we append the -include-runtime and -d flags to specify where
     * to output the jarfile which we will run using `java -jar`
     *
     * TODO(supergrecko): Find a better fix than this bandaid for execution
     */
    override async handleInterpreting(key, executeParameters) {
        const alteredKey = {
            ...key,
            options: ['-include-runtime', '-d', 'example.jar'],
        };
        const compileResult = await this.getOrBuildExecutable(alteredKey);
        executeParameters.args = [
            '-Xss136K', // Reduce thread stack size
            '-XX:CICompilerCount=2', // Reduce JIT compilation threads. 2 is minimum
            '-XX:-UseDynamicNumberOfCompilerThreads',
            '-XX:-UseDynamicNumberOfGCThreads',
            '-XX:+UseSerialGC', // Disable parallell/concurrent garbage collector
            '-cp',
            compileResult.dirPath,
            '-jar',
            'example.jar',
            // -jar <jar> has to be the last java parameter, otherwise it will use
            // our java parameters as program parameters
            ...executeParameters.args,
        ];
        const result = await this.runExecutable(this.javaRuntime, executeParameters, compileResult.dirPath);
        return {
            ...result,
            didExecute: true,
            buildResult: compileResult,
        };
    }

    override getArgumentParser() {
        return KotlinParser;
    }
}
