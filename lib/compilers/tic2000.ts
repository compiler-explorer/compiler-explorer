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

import path from 'path';

import _ from 'underscore';

import {BaseCompiler} from '../base-compiler';
import {TiC2000AsmParser} from '../parsers/asm-parser-tic2000';

export class TIC2000 extends BaseCompiler {
    static get key() {
        return 'tic2000';
    }

    constructor(info, env) {
        super(info, env);
        this.outputFilebase = this.compileFilename.split('.')[0];
        this.asm = new TiC2000AsmParser(this.compilerProps);
    }

    override optionsForFilter(filters, outputFilename) {
        const options = ['-g', '-c', '-n', '--output_file=' + outputFilename];

        filters.preProcessLines = _.bind(this.preProcessLines, this);

        return options;
    }

    override getOutputFilename(dirPath, outputFilebase) {
        return path.join(dirPath, `${outputFilebase}.asm`);
    }

    override exec(compiler, args, options_) {
        const options = Object.assign({}, options_);
        options.env = Object.assign({}, options.env);

        if (this.compiler.includePath) {
            options.env['INCLUDE'] = this.compiler.includePath;
        }
        if (this.compiler.libPath) {
            options.env['LIB'] = this.compiler.libPath;
        }
        for (const [env, to] of this.compiler.envVars) {
            options.env[env] = to;
        }

        options.env['TMP'] = process.env.TMP;

        return super.exec(compiler, args, options);
    }

    getExtraAsmHint(asm) {
        asm = asm.trim();
        if (asm.startsWith('.dwpsn')) {
            const tokens = asm.split(',');
            const file = tokens[0].split('"')[1];
            const line = tokens[1].split(' ')[1];
            const column = tokens[2].split(' ')[1];

            const retval: string[] = [];

            retval.push('  .file 1 "' + file + '"');

            if (!isNaN(line)) {
                if (isNaN(column)) {
                    retval.push('  .loc 1 ' + line + ' 0');
                } else {
                    retval.push('  .loc 1 ' + line + ' ' + column);
                }
            }
            return retval;
        } else {
            return false;
        }
    }

    preProcessLines(asmLines) {
        let i = 0;

        while (i < asmLines.length) {
            const extraHint = this.getExtraAsmHint(asmLines[i]);
            if (extraHint) {
                if (Array.isArray(extraHint)) {
                    for (const z of extraHint) {
                        i++;
                        asmLines.splice(i, 0, z);
                    }
                } else {
                    i++;
                    asmLines.splice(i, 0, extraHint);
                }
            }

            i++;
        }

        return asmLines;
    }
}
