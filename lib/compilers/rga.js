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

// NOTE: This script is intended to be invoked as a standalone script as a
// wrapper around DXC and RGA. AMD ISA for HLSL is compiled in two passes,
// first invoking DXC, then RGA. Note that some HLSL features are not yet
// supported by SPIR-V, so this approach cannot compile all possible HLSL
// that DXC can compile. However, the advantage is that a full pipeline state
// object and root signature isn't required.

/* eslint-disable @typescript-eslint/no-var-requires */
/* eslint-disable unicorn/prefer-module */
const {execFileSync} = require('child_process');
const {readdirSync, renameSync, writeFileSync} = require('fs');
const path = require('path');
const process = require('process');

function debug(output) {
    if (process.env.NODE_ENV === 'DEV') {
        console.log(output);
    }
}

function princASICOptions() {
    const asicOptions = execFileSync(process.argv[2], ['-s', 'vk-spv-txt-offline', '-l'], {encoding: 'utf-8'});
    console.log(asicOptions);
}

debug(process.argv);

// Expected arguments to this script:
//   - node path
//   - script path
//   - RGA path
//   - DXC path
//   - Output filename
//   - ... args to DXC
const DXC_ARG_START = 5;

const outputFilename = process.argv[4];
const outputDir = path.dirname(outputFilename);
const spirvTemp = path.join(outputDir, 'output.spirv.txt');
debug(outputDir);

const dxcArgs = process.argv.slice(DXC_ARG_START);
if (!dxcArgs.includes('-spirv')) {
    dxcArgs.push('-spirv');
}
debug(dxcArgs);

// Scan dxc args for an `--asic` argument that should be stripped and passed
// later to RGA
// Default to RDNA2
let ASIC = 'gfx1030';
let usedDefaultASIC = true;
for (let i = 0; i !== dxcArgs.length; ++i) {
    const arg = dxcArgs[i];
    if (arg === '--asic') {
        // NOTE: the last arguments are the input source file and -spirv, so check
        // if --asic immediately precedes that
        if (i === dxcArgs.length - 3) {
            console.log('--asic flag supplied without subsequent ASIC!');
            process.exit(1);
        }
        ASIC = dxcArgs[i + 1];
        // Do a quick sanity check to determine if a valid ASIC was supplied
        if (!ASIC.startsWith('gfx')) {
            console.log(
                // eslint-disable-next-line max-len
                `The argument immediately following --asic doesn't appear to be a valid ASIC. Please supply an ASIC from the following options:`,
            );
            princASICOptions();
            process.exit(1);
        }
        // Remove these two arguments from the dxcArgs list
        dxcArgs.splice(i, 2);
        usedDefaultASIC = false;
    }
}

// Attempt to compile HLSL-to-SPIR-V using dxc
try {
    const dxcOutput = execFileSync(process.argv[3], dxcArgs, {encoding: 'utf-8'});
    writeFileSync(spirvTemp, dxcOutput);
} catch (e) {
    console.log(e.stderr);
    process.exit(1);
}

// Forward spirv text output to RGA
const rgaOptions = ['-s', 'vk-spv-txt-offline', '-c', ASIC, '--isa', outputFilename, spirvTemp];
debug(rgaOptions);
try {
    const rgaOutput = execFileSync(process.argv[2], rgaOptions, {encoding: 'utf-8'});
    debug(rgaOutput);
} catch (e) {
    console.log(e.stderr);
    process.exit(1);
}

// RGA doesn't emit the exact file we requested. It prepends the requested GPU
// architecture and appends the shader type (with underscore separators). Here,
// we rename the generated file to the output file Compiler Explorer expects.
const files = readdirSync(outputDir);
for (const file of files) {
    if (file.startsWith(ASIC)) {
        renameSync(path.join(outputDir, file), outputFilename);

        if (usedDefaultASIC) {
            console.log(
                // eslint-disable-next-line max-len
                'ISA compiled with the default AMD ASIC (Radeon RX 6800 series RDNA2). To override this, pass --asic [ASIC] to the options above (nonstandard DXC option), where [ASIC] corresponds to one of the following options:',
            );
            princASICOptions();
        }
        process.exit(0);
    }
}

console.log(`RGA didn't emit expected ISA output.`);
process.exit(1);
