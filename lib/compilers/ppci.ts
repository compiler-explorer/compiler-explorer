// Copyright (c) 2017, Windel Bouwman & Compiler Explorer Authors
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

import type {ExecutionOptions} from '../../types/compilation/compilation.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import {assert} from '../assert.js';
import {BaseCompiler} from '../base-compiler.js';
import type {CompilationEnvironment} from '../compilation-env.js';
import * as exec from '../exec.js';

const forbiddenOptions = new Set(['--report', '--text-report', '--html-report']);

export class PPCICompiler extends BaseCompiler {
    static get key() {
        return 'ppci';
    }

    private readonly python: string;

    constructor(info: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(info, env);
        // ppci is run as a Python module. Prefer a compiler- or group-specific interpreter, as
        // older ppci releases do not run on every Python version; fall back to the global python3.
        this.python =
            this.compilerProps<string>(`compiler.${this.compiler.id}.python`, '') ||
            this.compilerProps<string>(`group.${this.compiler.group}.python`, '') ||
            this.env.ceProps<string>('python3');
    }

    override filterUserOptions(args: string[]) {
        return args.filter(item => {
            if (typeof item !== 'string') return true;

            return !forbiddenOptions.has(item.toLowerCase());
        });
    }

    override exec(compiler: string, args: string[], options: ExecutionOptions) {
        if (compiler.endsWith('.py')) {
            options = options || {};

            const matches = compiler.match(/^(.*)(\/ppci\/)(.*).py/);
            if (matches) {
                const pythonPath = matches[1];
                const ppciName = `ppci.${matches[3].replace('/', '.')}`;
                options.env = {PYTHONPATH: pythonPath};
                const python_args = ['-m', ppciName].concat(args);
                return exec.execute(this.python, python_args, options);
            }
            assert(false, `Invalid ppci path ${compiler}`);
        } else {
            return super.exec(compiler, args, options);
        }
    }
}
