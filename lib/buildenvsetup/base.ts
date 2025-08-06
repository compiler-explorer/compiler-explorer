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
        logger.info(
            `BuildEnvSetupBase constructor: compiler.exe=${this.compiler.exe}, compiler.options=${this.compiler.options}`,
        );
        logger.info(`BuildEnvSetupBase constructor: compilerOptionsArr=${JSON.stringify(this.compilerOptionsArr)}`);

        this.compilerArch = this.getCompilerArch();
        logger.info(`BuildEnvSetupBase constructor: compilerArch=${this.compilerArch}`);

        this.compilerTypeOrGCC = compilerInfo.compilerType || 'gcc';
        if (this.compilerTypeOrGCC === 'clang-intel') this.compilerTypeOrGCC = 'gcc';
        logger.info(`BuildEnvSetupBase constructor: compilerTypeOrGCC=${this.compilerTypeOrGCC}`);

        this.compilerSupportsX86 = !this.compilerArch;
        this.compilerSupportsAMD64 = this.compilerSupportsX86;
        logger.info(
            `BuildEnvSetupBase constructor: initial compilerSupportsX86=${this.compilerSupportsX86}, compilerSupportsAMD64=${this.compilerSupportsAMD64}`,
        );

        this.defaultLibCxx = 'libstdc++';
    }

    async initialise(execCompilerCachedFunc: ExecCompilerCachedFunc) {
        logger.info(`BuildEnvSetupBase initialise: Starting for compiler ${this.compiler.exe}`);
        logger.info(
            `BuildEnvSetupBase initialise: compilerArch=${this.compilerArch}, early return=${!!this.compilerArch}`,
        );

        if (this.compilerArch) {
            logger.info(`BuildEnvSetupBase initialise: Early return due to existing compilerArch=${this.compilerArch}`);
            return;
        }

        logger.info(`BuildEnvSetupBase initialise: Checking x86 support via hasSupportForArch`);
        await this.hasSupportForArch(execCompilerCachedFunc, 'x86')
            .then(res => {
                logger.info(`BuildEnvSetupBase initialise: x86 support check result=${res}`);
                this.compilerSupportsX86 = res;
            })
            .catch(error => {
                // Log & keep going, we assume x86 is supported
                logger.info(
                    'BuildEnvSetupBase initialise: Could not check for x86 arch support, keeping default',
                    error,
                );
            });

        if (this.compilerTypeOrGCC === 'win32-vc') {
            logger.info(`BuildEnvSetupBase initialise: win32-vc compiler, checking x86_64 support`);
            await this.hasSupportForArch(execCompilerCachedFunc, 'x86_64')
                .then(res => {
                    logger.info(`BuildEnvSetupBase initialise: x86_64 support check result=${res}`);
                    this.compilerSupportsAMD64 = res;
                })
                .catch(error => {
                    logger.info('BuildEnvSetupBase initialise: Could not check for x86_64 arch support', error);
                });

            logger.info(
                `BuildEnvSetupBase initialise: win32-vc arch selection logic: x86=${this.compilerSupportsX86}, x86_64=${this.compilerSupportsAMD64}`,
            );
            if (this.compilerSupportsX86) {
                this.compilerArch = 'x86';
                logger.info(`BuildEnvSetupBase initialise: Set compilerArch to x86`);
            } else if (this.compilerSupportsAMD64) {
                this.compilerArch = 'x86_64';
                logger.info(`BuildEnvSetupBase initialise: Set compilerArch to x86_64`);
            } else {
                logger.info(`BuildEnvSetupBase initialise: No arch support found, compilerArch remains false`);
            }
        }

        logger.info(
            `BuildEnvSetupBase initialise: Final state - Compiler ${this.compiler.exe} supports x86: ${this.compilerSupportsX86}, x86_64: ${this.compilerSupportsAMD64}, compilerArch: ${this.compilerArch}`,
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
        logger.info(`hasSupportForArch: Starting check for arch=${arch}, compiler=${this.compiler.exe}`);
        logger.info(
            `hasSupportForArch: compiler.group=${this.compiler.group}, compilerTypeOrGCC=${this.compilerTypeOrGCC}`,
        );

        let result: any;

        if (this.compiler.exe.includes('icpx')) {
            const supported = arch === 'x86' || arch === 'x86_64';
            logger.info(`hasSupportForArch: icpx compiler detected, returning ${supported} for arch ${arch}`);
            return supported;
        }

        if (this.compiler.exe.includes('circle')) {
            const supported = arch === 'x86' || arch === 'x86_64';
            logger.info(`hasSupportForArch: circle compiler detected, returning ${supported} for arch ${arch}`);
            return supported;
        }

        if (this.compiler.group === 'icc') {
            logger.info(`hasSupportForArch: icc group, executing --help`);
            result = await execCompilerCached(this.compiler.exe, ['--help']);
        } else if (this.compilerTypeOrGCC === 'gcc' || this.compilerTypeOrGCC === 'win32-mingw-gcc') {
            if (this.compiler.exe.includes('/icpx')) {
                const supported = arch === 'x86' || arch === 'x86_64';
                logger.info(`hasSupportForArch: gcc with icpx path detected, returning ${supported} for arch ${arch}`);
                return supported;
            }
            logger.info(`hasSupportForArch: gcc/mingw-gcc type, executing --target-help`);
            result = await execCompilerCached(this.compiler.exe, ['--target-help']);
        } else if (this.compilerTypeOrGCC === 'clang' || this.compilerTypeOrGCC === 'win32-mingw-clang') {
            const binpath = path.dirname(this.compiler.exe);
            const llc = path.join(binpath, 'llc');
            logger.info(`hasSupportForArch: clang type, checking for llc at ${llc}`);

            if (await utils.fileExists(llc)) {
                logger.info(`hasSupportForArch: llc exists, executing --version`);
                result = await execCompilerCached(llc, ['--version']);
            } else {
                logger.info(`hasSupportForArch: llc does not exist at ${llc}`);
            }
        } else if (this.compilerTypeOrGCC === 'win32-vc') {
            logger.info(`hasSupportForArch: win32-vc type, executing compiler with no args`);
            result = await execCompilerCached(this.compiler.exe, []);
        }

        logger.info(
            `hasSupportForArch: exec result - stdout length: ${result?.stdout?.length || 0}, stderr length: ${result?.stderr?.length || 0}`,
        );
        logger.info(`hasSupportForArch: exec result - stdout: ${result?.stdout || '(none)'}`);
        logger.info(`hasSupportForArch: exec result - stderr: ${result?.stderr || '(none)'}`);

        if (result) {
            const searchFor = this.get_support_check_text(this.compiler.group, this.compilerTypeOrGCC, arch as string);
            logger.info(`hasSupportForArch: searching for text: '${searchFor}'`);

            if (this.compilerTypeOrGCC === 'win32-vc' && result.stderr) {
                const found = result.stderr.includes(searchFor);
                logger.info(`hasSupportForArch: win32-vc stderr search result: ${found}`);
                return found;
            }

            if (result.stdout) {
                const found = result.stdout.includes(searchFor);
                logger.info(`hasSupportForArch: stdout search result: ${found}`);
                return found;
            }

            logger.info(`hasSupportForArch: no stdout available, returning false`);
            return false;
        }

        logger.info(`hasSupportForArch: no result from execution, returning false`);
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
        logger.info(`getCompilerArch: Starting analysis of compiler options`);
        logger.info(`getCompilerArch: compilerOptionsArr=${JSON.stringify(this.compilerOptionsArr)}`);

        let arch = _.find(this.compilerOptionsArr, option => {
            return option.startsWith('-march=');
        });
        logger.info(`getCompilerArch: found -march option: ${arch || '(none)'}`);

        let target = _.find(this.compilerOptionsArr, option => {
            return option.startsWith('-target=') || option.startsWith('--target=');
        });
        logger.info(`getCompilerArch: found -target/--target option: ${target || '(none)'}`);

        if (target) {
            target = target.substring(target.indexOf('=') + 1);
            logger.info(`getCompilerArch: extracted target value: ${target}`);
        } else {
            const targetIdx = this.compilerOptionsArr.indexOf('-target');
            logger.info(`getCompilerArch: checking for separate -target flag at index: ${targetIdx}`);
            if (targetIdx !== -1) {
                target = this.compilerOptionsArr[targetIdx + 1];
                logger.info(`getCompilerArch: found separate -target value: ${target || '(undefined)'}`);
            }
        }

        if (arch) {
            arch = arch.substring(7);
            logger.info(`getCompilerArch: extracted arch value: ${arch}`);
        }

        logger.info(`getCompilerArch: final values - arch: ${arch || '(none)'}, target: ${target || '(none)'}`);

        if (target && arch) {
            const result = arch.length < target.length ? arch : target;
            logger.info(`getCompilerArch: both arch and target present, returning shorter: ${result}`);
            return result;
        }

        if (target) {
            logger.info(`getCompilerArch: returning target: ${target}`);
            return target;
        }

        if (arch) {
            logger.info(`getCompilerArch: returning arch: ${arch}`);
            return arch;
        }

        logger.info(`getCompilerArch: no arch found, returning false`);
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
        logger.info(`getTarget: Starting target determination`);
        logger.info(
            `getTarget: compilerTypeOrGCC=${this.compilerTypeOrGCC}, compilerSupportsX86=${this.compilerSupportsX86}, compilerSupportsAMD64=${this.compilerSupportsAMD64}, compilerArch=${this.compilerArch}`,
        );
        logger.info(`getTarget: key.options=${JSON.stringify(key.options)}`);

        if (this.compilerTypeOrGCC === 'win32-vc') {
            logger.info(`getTarget: win32-vc compiler type`);
            if (this.compilerSupportsX86) {
                logger.info(`getTarget: win32-vc returning x86`);
                return 'x86';
            }
            if (this.compilerSupportsAMD64) {
                logger.info(`getTarget: win32-vc returning x86_64`);
                return 'x86_64';
            }
            logger.info(`getTarget: win32-vc no arch support, returning empty string`);
            return '';
        }

        if (!this.compilerSupportsX86) {
            logger.info(`getTarget: no x86 support, returning empty string`);
            return '';
        }

        if (this.compilerArch) {
            logger.info(`getTarget: using compilerArch: ${this.compilerArch}`);
            return this.compilerArch;
        }

        if (key.options.includes('-m32')) {
            logger.info(`getTarget: found -m32 option, returning x86`);
            return 'x86';
        }

        const target: string | undefined = _.find(key.options, option => {
            return option.startsWith('-target=') || option.startsWith('--target=');
        });

        if (target) {
            const result = target.substring(target.indexOf('=') + 1);
            logger.info(`getTarget: found target option, returning: ${result}`);
            return result;
        }

        logger.info(`getTarget: defaulting to x86_64`);
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
