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

import path from 'node:path';

import _ from 'underscore';

import {splitArguments} from '../../shared/common-utils.js';
import {Arch, CacheKey, ExecutionOptionsWithEnv} from '../../types/compilation/compilation.interfaces.js';
import {CompilerInfo} from '../../types/compiler.interfaces.js';
import {UnprocessedExecResult} from '../../types/execution/execution.interfaces.js';
import {CompilationEnvironment} from '../compilation-env.js';
import {logger} from '../logger.js';
import {VersionInfo} from '../options-handler.js';
import * as utils from '../utils.js';
import type {BuildEnvDownloadInfo} from './buildenv.interfaces.js';

export type ExecCompilerCachedFunc = (
    compiler: string,
    args: string[],
    options?: ExecutionOptionsWithEnv,
) => Promise<UnprocessedExecResult>;

export class BuildEnvSetupBase {
    protected compiler: any;
    protected env: any;
    protected compilerOptionsArr: string[];
    public compilerArch: string | false;
    protected compilerTypeOrGCC: any;
    public compilerSupportsX86: boolean;
    public compilerSupportsAMD64: boolean;
    public defaultLibCxx: string;

    constructor(compilerInfo: CompilerInfo, env: CompilationEnvironment) {
        this.compiler = compilerInfo;
        this.env = env;

        this.compilerOptionsArr = splitArguments(this.compiler.options);
        this.compilerArch = this.getCompilerArch();
        this.compilerTypeOrGCC = compilerInfo.compilerType || 'gcc';
        if (this.compilerTypeOrGCC === 'clang-intel') this.compilerTypeOrGCC = 'gcc';
        this.compilerSupportsX86 = !this.compilerArch;
        this.compilerSupportsAMD64 = this.compilerSupportsX86;
        this.defaultLibCxx = 'libstdc++';
    }

    async initialise(execCompilerCachedFunc: ExecCompilerCachedFunc) {
        if (this.compilerArch) return;

        await this.hasSupportForArch(execCompilerCachedFunc, 'x86')
            .then(res => (this.compilerSupportsX86 = res))
            .catch(error => {
                // Log & keep going, we assume x86 is supported
                logger.error('Could not check for x86 arch support', error);
            });

        if (this.compilerTypeOrGCC === 'win32-vc') {
            await this.hasSupportForArch(execCompilerCachedFunc, 'x86_64')
                .then(res => (this.compilerSupportsAMD64 = res))
                .catch(error => {
                    logger.error('Could not check for x86_64 arch support', error);
                });

            if (this.compilerSupportsX86) {
                this.compilerArch = 'x86';
            } else if (this.compilerSupportsAMD64) {
                this.compilerArch = 'x86_64';
            }
        }

        logger.debug(
            `Compiler ${this.compiler.exe} supports x86: ${this.compilerSupportsX86}, x86_64: ${this.compilerSupportsAMD64}`,
        );
    }

    protected get_support_check_text(group: string, compilerType: string, arch: string) {
        if (group.includes('icc')) {
            if (arch === 'x86') return '-m32';
            if (arch === 'x86_64') return '-m64';
        } else if (compilerType === 'win32-vc') {
            if (arch === 'x86') return 'for x86';
            if (arch === 'x86_64') return 'for x64';
            if (arch === 'arm64') return 'for ARM64';
        }

        return arch;
    }

    async hasSupportForArch(execCompilerCached: ExecCompilerCachedFunc, arch: Arch): Promise<boolean> {
        let result: any;
        if (this.compiler.exe.includes('icpx')) {
            return arch === 'x86' || arch === 'x86_64';
        }
        if (this.compiler.exe.includes('circle')) {
            return arch === 'x86' || arch === 'x86_64';
        }
        if (this.compiler.group === 'icc') {
            result = await execCompilerCached(this.compiler.exe, ['--help']);
        } else if (this.compilerTypeOrGCC === 'gcc' || this.compilerTypeOrGCC === 'win32-mingw-gcc') {
            if (this.compiler.exe.includes('/icpx')) {
                return arch === 'x86' || arch === 'x86_64';
            }
            result = await execCompilerCached(this.compiler.exe, ['--target-help']);
        } else if (this.compilerTypeOrGCC === 'clang' || this.compilerTypeOrGCC === 'win32-mingw-clang') {
            const binpath = path.dirname(this.compiler.exe);
            const llc = path.join(binpath, 'llc');
            if (await utils.fileExists(llc)) {
                result = await execCompilerCached(llc, ['--version']);
            }
        } else if (this.compilerTypeOrGCC === 'win32-vc') {
            result = await execCompilerCached(this.compiler.exe, []);
        }

        if (result) {
            const searchFor = this.get_support_check_text(this.compiler.group, this.compilerTypeOrGCC, arch as string);
            if (this.compilerTypeOrGCC === 'win32-vc' && result.stderr) return result.stderr.includes(searchFor);
            if (result.stdout) {
                return result.stdout.includes(searchFor);
            }

            return false;
        }

        return false;
    }

    async setup(
        key: CacheKey,
        dirPath: string,
        selectedLibraries: Record<string, VersionInfo>,
        binary: boolean,
    ): Promise<BuildEnvDownloadInfo[]> {
        return [];
    }

    getCompilerArch(): string | false {
        let arch = _.find(this.compilerOptionsArr, option => {
            return option.startsWith('-march=');
        });

        let target = _.find(this.compilerOptionsArr, option => {
            return option.startsWith('-target=') || option.startsWith('--target=');
        });

        if (target) {
            target = target.substring(target.indexOf('=') + 1);
        } else {
            const targetIdx = this.compilerOptionsArr.indexOf('-target');
            if (targetIdx !== -1) {
                target = this.compilerOptionsArr[targetIdx + 1];
            }
        }

        if (arch) {
            arch = arch.substring(7);
        }

        if (target && arch) {
            if (arch.length < target.length) {
                return arch;
            }
            return target;
        }
        if (target) return target;
        if (arch) return arch;

        return false;
    }

    getLibcxx(key: CacheKey): string {
        const match = this.compiler.options.match(/-stdlib=(\S*)/i);
        if (match) {
            return match[1];
        }
        const stdlibOption: string | undefined = _.find(key.options, option => {
            return option.startsWith('-stdlib=');
        });

        if (stdlibOption) {
            return stdlibOption.substring(8);
        }

        return this.defaultLibCxx;
    }

    getTarget(key: CacheKey): string {
        if (this.compilerTypeOrGCC === 'win32-vc') {
            if (this.compilerSupportsX86) return 'x86';
            if (this.compilerSupportsAMD64) return 'x86_64';
            return '';
        }

        if (!this.compilerSupportsX86) return '';
        if (this.compilerArch) return this.compilerArch;

        if (key.options.includes('-m32')) {
            return 'x86';
        }
        const target: string | undefined = _.find(key.options, option => {
            return option.startsWith('-target=') || option.startsWith('--target=');
        });

        if (target) {
            return target.substring(target.indexOf('=') + 1);
        }

        return 'x86_64';
    }

    hasBinariesToLink(details: VersionInfo) {
        return (
            details.libpath.length === 0 &&
            (details.staticliblink.length > 0 || details.liblink.length > 0) &&
            details.version !== 'autodetect'
        );
    }

    hasPackagedHeaders(details: VersionInfo) {
        return !!details.packagedheaders;
    }

    shouldDownloadPackage(details: VersionInfo) {
        return this.hasPackagedHeaders(details) || this.hasBinariesToLink(details);
    }
}
