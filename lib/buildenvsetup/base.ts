// Copyright (c) 2020, Compiler Explorer Authors
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

import fs from 'fs-extra';
import _ from 'underscore';

import {logger} from '../logger';
import * as utils from '../utils';

import {BuildEnvDownloadInfo} from './buildenv.interfaces';

export class BuildEnvSetupBase {
    protected compiler: any;
    protected env: any;
    protected compilerOptionsArr: string[];
    public compilerArch: string | boolean;
    protected compilerTypeOrGCC: any;
    protected compilerSupportsX86: boolean;

    constructor(compilerInfo, env) {
        this.compiler = compilerInfo;
        this.env = env;

        this.compilerOptionsArr = utils.splitArguments(this.compiler.options);
        this.compilerArch = this.getCompilerArch();
        this.compilerTypeOrGCC = compilerInfo.compilerType || 'gcc';
        if (this.compilerTypeOrGCC === 'clang-intel') this.compilerTypeOrGCC = 'gcc';
        this.compilerSupportsX86 = !this.compilerArch;
    }

    async initialise(execCompilerCachedFunc) {
        if (this.compilerArch) return;
        await this.hasSupportForArch(execCompilerCachedFunc, 'x86')
            .then(res => (this.compilerSupportsX86 = res))
            .catch(error => {
                // Log & keep going, we assume x86 is supported
                logger.error('Could not check for x86 arch support', error);
            });
    }

    async hasSupportForArch(execCompilerCached, arch) {
        let result: any;
        let searchFor = arch;
        if (this.compiler.exe.includes('icpx')) {
            return arch === 'x86' || arch === 'x86_64';
        } else if (this.compiler.group === 'icc') {
            result = await execCompilerCached(this.compiler.exe, ['--help']);
            if (arch === 'x86') {
                searchFor = '-m32';
            } else if (arch === 'x86_64') {
                searchFor = '-m64';
            }
        } else if (this.compilerTypeOrGCC === 'gcc') {
            if (this.compiler.exe.includes('/icpx')) {
                return arch === 'x86' || arch === 'x86_64';
            } else {
                result = await execCompilerCached(this.compiler.exe, ['--target-help']);
            }
        } else if (this.compilerTypeOrGCC === 'clang') {
            const binpath = path.dirname(this.compiler.exe);
            const llc = path.join(binpath, 'llc');
            if (fs.existsSync(llc)) {
                result = await execCompilerCached(llc, ['--version']);
            }
        }

        if (result) {
            return result.stdout ? result.stdout.includes(searchFor) : false;
        }

        return false;
    }

    async setup(key, dirPath, selectedLibraries): Promise<BuildEnvDownloadInfo[]> {
        return [];
    }

    getCompilerArch() {
        let arch = _.find(this.compilerOptionsArr, option => {
            return option.startsWith('-march=');
        });

        let target = _.find(this.compilerOptionsArr, option => {
            return option.startsWith('-target=') || option.startsWith('--target=');
        });

        if (target) {
            target = target.substr(target.indexOf('=') + 1);
        } else {
            const targetIdx = this.compilerOptionsArr.indexOf('-target');
            if (targetIdx !== -1) {
                target = this.compilerOptionsArr[targetIdx + 1];
            }
        }

        if (arch) {
            arch = arch.substr(7);
        }

        if (target && arch) {
            if (arch.length < target.length) {
                return arch;
            } else {
                return target;
            }
        } else {
            if (target) return target;
            if (arch) return arch;
        }

        return false;
    }

    getLibcxx(key) {
        const match = this.compiler.options.match(/-stdlib=(\S*)/i);
        if (match) {
            return match[1];
        } else {
            const stdlibOption = _.find(key.options, option => {
                return option.startsWith('-stdlib=');
            });

            if (stdlibOption) {
                return stdlibOption.substr(8);
            }

            return 'libstdc++';
        }
    }

    getTarget(key) {
        if (!this.compilerSupportsX86) return '';
        if (this.compilerArch) return this.compilerArch;

        if (key.options.includes('-m32')) {
            return 'x86';
        } else {
            const target = _.find(key.options, option => {
                return option.startsWith('-target=') || option.startsWith('--target=');
            });

            if (target) {
                return target.substr(target.indexOf('=') + 1);
            }
        }

        return 'x86_64';
    }
}
