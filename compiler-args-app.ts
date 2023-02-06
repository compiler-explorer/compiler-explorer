// Copyright (c) 2023, Compiler Explorer Authors
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

/* eslint-disable unicorn/prefer-top-level-await */
/* eslint-disable no-console */

import nopt from 'nopt';
import _ from 'underscore';

import {CompilerArguments} from './lib/compiler-arguments';
import * as Parsers from './lib/compilers/argument-parsers';
import {executeDirect} from './lib/exec';
import {logger} from './lib/logger';
import {padRight} from './lib/utils';

const opts = nopt({
    parser: [String],
    exe: [String],
    padding: [Number],
    debug: [Boolean],
});

if (opts.debug) logger.level = 'debug';

const compilerParsers = {
    gcc: Parsers.GCCParser,
    clang: Parsers.ClangParser,
    ldc: Parsers.LDCParser,
    erlang: Parsers.ErlangParser,
    pascal: Parsers.PascalParser,
    ispc: Parsers.ISPCParser,
    java: Parsers.JavaParser,
    kotlin: Parsers.KotlinParser,
    scala: Parsers.ScalaParser,
    vc: Parsers.VCParser,
    rust: Parsers.RustParser,
    mrustc: Parsers.MrustcParser,
    num: Parsers.NimParser,
    crystal: Parsers.CrystalParser,
    ts: Parsers.TypeScriptNativeParser,
    turboc: Parsers.TurboCParser,
    toit: Parsers.ToitParser,
};

class CompilerArgsApp {
    parserName: string;
    executable: string;
    compiler: any;
    pad: number;

    constructor() {
        this.parserName = opts.parser;
        this.executable = opts.exe;
        this.pad = opts.padding || 40;
        this.compiler = {
            compiler: {
                exe: this.executable,
            },
            possibleArguments: new CompilerArguments(),
            execCompilerCached: async (command: string, args: string[]) => {
                return executeDirect(command, args, {}, fn => fn);
            },
        };

        if (this.parserName === 'juliawrapper') {
            this.compiler.compilerWrapperPath = 'etc/scripts/julia_wrapper.jl';
        }
    }

    async doTheParsing() {
        if (compilerParsers[this.parserName]) {
            const parser = compilerParsers[this.parserName];
            await parser.parse(this.compiler);
        } else {
            console.error('Unknown parser type');
            process.exit(1);
        }
    }

    print() {
        const args = _.keys(this.compiler.possibleArguments.possibleArguments);
        for (const arg of args) {
            console.log(padRight(arg, this.pad) + this.compiler.possibleArguments.possibleArguments[arg].description);
        }
    }
}

if (!opts.parser || !opts.exe) {
    console.error(
        'Usage: ' +
            'node -r esm -r ts-node/register compiler-args-app.ts ' +
            '--parser=<compilertype> --exe=<path> [--padding=<number>]\n' +
            'for example: --parser=clang --exe=/opt/compiler-explorer/clang-15.0.0/bin/clang++ --padding=50',
    );
    process.exit(1);
} else {
    const app = new CompilerArgsApp();
    app.doTheParsing()
        .then(() => {
            app.print();
        })
        .catch(e => console.error(e));
}
