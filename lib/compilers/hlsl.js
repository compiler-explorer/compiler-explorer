// Copyright (c) 2022, Compiler Explorer Authors
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
import process from 'process';
import url from 'url';

import {BaseCompiler} from '../base-compiler';

// The RGA compiler executes in a secondary pass, leveraging the first listed DXC
// instead of the bundled RGA DXC (which isn't as up-to-date). This is initialized
// when the default DXC compiler instance is initialized.
let default_dxc = null;

export class HLSLCompiler extends BaseCompiler {
    static get key() {
        return 'hlsl';
    }

    constructor(info, env) {
        super(info, env);

        this.compiler.supportsIntel = false;

        if (default_dxc === null && this.compiler.group === 'dxc') {
            default_dxc = this.compiler;
        }

        if (this.rga) {
            // RGA is invoked in two steps. First, DXC is invoked to compile
            // the SPIR-V output of the HLSL file. Next, RGA is invoked to
            // consume the SPIR-V output and produce the requested ISA. To
            // accomplish this two-pass compilation, a helper script rga.js
            // is used to wrap the compilation as a single executable script.
            this.rgaOptions = [
                path.join(path.dirname(url.fileURLToPath(import.meta.url)), 'rga.js'),
                this.compiler.exe,
            ];

            // Replace the RGA executable with the node path
            this.compiler.exe = process.execPath;
        }
    }

    // Returns true if the compiler selected is a Radeon GPU Analyzer
    get rga() {
        return this.compiler.group === 'rga';
    }

    // Returns true if the compiler selected is a DirectX Shader Compiler
    get dxc() {
        return this.compiler.group === 'dxc';
    }

    /* eslint-disable no-unused-vars */
    optionsForFilter(filters, outputFilename, userOptions) {
        // If compiling with RGA, inject the RGA script, RGA executable,
        // DXC executable, and output filename as the initial arguments
        // (to be passed to node). Changes here require changes in the
        // rga.js script.
        if (this.rga) {
            return [...this.rgaOptions, default_dxc.exe, outputFilename, '-Zi', '-Qembed_debug'];
        }

        return [
            '-Zi', // Embed debug information to get DXIL line associations
            '-Qembed_debug', // Silences the warning associated with embedded debug information
            `-Fc ${outputFilename}`, // Output object
        ];
    }
    /* eslint-enable no-unused-vars */

    filterUserOptions(userOptions) {
        // RGA supports a non-standard flag --asic [ASIC] which must be removed when compiling with DXC
        if (this.rga) {
            return super.filterUserOptions(userOptions);
        } else {
            let options = userOptions.slice(0);
            // Scan for the RGA-specific argument --asic and strip it and its corresponding argument
            // Assumes the argument exists at most once (compilation will fail if supplied more than
            // once regardless)
            for (let i = 0; i !== options.length; ++i) {
                const option = options[i];
                if (option === '--asic') {
                    options.splice(i, 2);
                    break;
                }
            }
            return options;
        }
    }

    getIrOutputFilename(inputFilename) {
        if (this.compiler.group === 'dxc') {
            return this.getOutputFilename(path.dirname(inputFilename), this.outputFilebase).replace('.s', '.dxil');
        }
        return inputFilename;
    }
}
