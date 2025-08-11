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

import {Command} from 'commander';
import _ from 'underscore';

import {CompilerArguments} from './lib/compiler-arguments.js';
import * as Parsers from './lib/compilers/argument-parsers.js';
import {executeDirect} from './lib/exec.js';
import {logger} from './lib/logger.js';
import {BaseParser} from './lib/compilers/argument-parsers.js';

const program = new Command();
program
    .name('compiler-args-app.ts')
    .usage('--parser=<compilertype> --exe=<path> [--padding=<number>]')
    .description(
        'Extracts compiler arguments\nFor example: node --no-warnings=ExperimentalWarning --import=tsx compiler-args-app.ts --parser=clang --exe=/opt/compiler-explorer/clang-15.0.0/bin/clang++ --padding=50',
    )
    .requiredOption('--parser <type>', 'Compiler parser type')
    .requiredOption('--exe <path>', 'Path to compiler executable')
    .option('--padding <number>', 'Padding for output formatting', '40')
    .option('--debug', 'Enable debug output')
    .allowUnknownOption(false)
    .configureOutput({
        writeErr: (str) => {
            if (str.includes('too many arguments')) {
                console.error('Error: Unexpected arguments provided.');
                console.error('This tool only accepts the following options: --parser, --exe, --padding, --debug');
                console.error('\nExample usage:');
                console.error('  node --import tsx compiler-args-app.ts --parser gcc --exe /path/to/gcc');
                console.error('\nNote: Do not use shell redirections like "2>&1" directly - they will be interpreted as arguments');
                process.exit(1);
            }
            process.stderr.write(str);
        }
    });

program.parse();
const opts = program.opts();

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
    circle: Parsers.CircleParser,
    ghc: Parsers.GHCParser,
    tendra: Parsers.TendraParser,
    golang: Parsers.GolangParser,
    zig: Parsers.ZigParser,
};

class CompilerArgsApp {
    parserName: string;
    executable: string;
    compiler: any;
    pad: number;

    constructor() {
        this.parserName = opts.parser;
        this.executable = opts.exe;
        this.pad = Number.parseInt(opts.padding, 10);
        this.compiler = {
            compiler: {
                exe: this.executable,
            },
            possibleArguments: new CompilerArguments('some-id'),
            execCompilerCached: async (command: string, args: string[]) => {
                return executeDirect(command, args, {}, fn => fn);
            },
            getDefaultExecOptions: () => {
                return {
                    env: process.env,
                    cwd: process.cwd(),
                    timeout: 10000,
                };
            }
        };

        if (this.parserName === 'juliawrapper') {
            this.compiler.compilerWrapperPath = 'etc/scripts/julia_wrapper.jl';
        }
    }

    async getPossibleStdvers() {
        const parser = this.getParser();
        return await parser.getPossibleStdvers();
    }

    async getPossibleTargets() {
        const parser = this.getParser();
        return await parser.getPossibleTargets();
    }

    async getPossibleEditions() {
        const parser = this.getParser();
        return await parser.getPossibleEditions();
    }

    getParser(): BaseParser {
        if (compilerParsers[this.parserName as keyof typeof compilerParsers]) {
            return new (compilerParsers[this.parserName as keyof typeof compilerParsers])(this.compiler);
        }
        console.error('Unknown parser type');
        process.exit(1);
    }

    async doTheParsing() {
        const parser = this.getParser();
        await parser.parse();
        const options = this.compiler.possibleArguments.possibleArguments;
        if (parser.hasSupportStartsWith(options, '--target=')) {
            console.log('supportsTargetIs');
        } else if (parser.hasSupportStartsWith(options, '--target ')) {
            console.log('supportsTarget');
        } else if (parser.hasSupportStartsWith(options, '-target ')) {
            console.log('supportsHyphenTarget');
        } else if (parser.hasSupportStartsWith(options, '--march=')) {
            console.log('supportsMarch');
        } else {
            console.log('none of the things?');
        }
    }

    async print() {
        const args = _.keys(this.compiler.possibleArguments.possibleArguments);
        for (const arg of args) {
            const description = this.compiler.possibleArguments.possibleArguments[arg].description;
            console.log(`${arg.padEnd(this.pad, ' ')} ${description}`);
        }

        console.log('Stdvers:');
        console.log(await this.getPossibleStdvers());
        console.log('Targets:');
        console.log(await this.getPossibleTargets());
        console.log('Editions:');
        console.log(await this.getPossibleEditions());

        console.log('supportsOptOutput:', !!this.compiler.compiler.supportsOptOutput);
        console.log('supportsStackUsageOutput', !!this.compiler.compiler.supportsStackUsageOutput);
        console.log('optPipeline:', this.compiler.compiler.optPipeline);
        console.log('supportsGccDump', !!this.compiler.compiler.supportsGccDump);
    }
}

const app = new CompilerArgsApp();
app.doTheParsing()
    .then(() => app.print())
    .catch(e => console.error(e));
