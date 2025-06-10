// Copyright (c) 2025, Compiler Explorer Authors
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

import fs from 'node:fs/promises';
import JSZip from 'jszip';

import {unwrap} from './assert.js';
import type {BuildEnvDownloadInfo} from './buildenvsetup/buildenv.interfaces.js';
import {logger} from './logger.js';
import type {PropertyGetter} from './properties.interfaces.js';
import {S3Bucket} from './s3-handler.js';
import * as utils from './utils.js';

export interface CmakeCacheModificationFlags {
    compilerFlags: string[];
    includeFlags: string[];
    linkerFlags: string[];
    language: string; // 'C' or 'CXX'
}

export class CmakeCacheDownloader {
    private readonly awsProps: PropertyGetter;

    constructor(awsProps: PropertyGetter) {
        this.awsProps = awsProps;
    }

    async downloadAndExtractCmakeCache(compilerId: string, buildDir: string): Promise<BuildEnvDownloadInfo | null> {
        const region = this.awsProps('region', '');
        const bucket = this.awsProps('storageBucket', '');
        const prefix = 'compiler-cmake-cache';

        if (!region || !compilerId) {
            logger.debug('CMake cache download skipped: missing region or compiler ID');
            return null;
        }

        const startTime = process.hrtime.bigint();
        const cacheFileName = `${compilerId}.zip`;

        try {
            const s3 = new S3Bucket(bucket, region);
            const result = await s3.get(cacheFileName, prefix);

            if (!result.hit) {
                logger.debug(`CMake cache not found for compiler ${compilerId}`);
                return null;
            }

            const zipData = unwrap(result.data);
            const zip = await JSZip.loadAsync(zipData);

            let extractedFiles = 0;
            for (const [fileName, file] of Object.entries(zip.files)) {
                if (file.dir) {
                    continue;
                }

                const filePath = path.join(buildDir, fileName);

                const resolved = path.resolve(path.dirname(filePath));
                if (!resolved.startsWith(buildDir)) {
                    logger.error('Cmake cache is using a zip-slip, skipping file');
                    continue;
                }

                const dirPath = path.dirname(filePath);
                await fs.mkdir(dirPath, {recursive: true});

                const content = await file.async('nodebuffer');
                await fs.writeFile(filePath, content);
                extractedFiles++;
            }

            const endTime = process.hrtime.bigint();
            const downloadInfo: BuildEnvDownloadInfo = {
                step: `CMake cache download for ${compilerId}`,
                packageUrl: `s3://${bucket}/${prefix}/${cacheFileName}`,
                time: utils.deltaTimeNanoToMili(startTime, endTime),
            };

            logger.info(`CMake cache extracted: ${extractedFiles} files for ${compilerId}`);
            return downloadInfo;
        } catch (error) {
            logger.debug(`Failed to download CMake cache for ${compilerId}: ${error}`);
            return null;
        }
    }

    async modifyCmakeCache(buildDir: string, flags: CmakeCacheModificationFlags): Promise<void> {
        const cmakeCachePath = path.join(buildDir, 'CMakeCache.txt');

        try {
            if (!(await utils.fileExists(cmakeCachePath))) {
                logger.debug('CMakeCache.txt not found, skipping modification');
                return;
            }

            const re = utils.getRegexForTempdir();

            const content = await fs.readFile(cmakeCachePath, 'utf-8');
            const lines = content.split('\n');

            const allCompilerFlags = [...flags.compilerFlags, ...flags.includeFlags]
                .map(flag => {
                    if (process.platform === 'win32') return flag;
                    return flag.replace(re, '/app/');
                })
                .join(' ');
            const allLinkerFlags = flags.linkerFlags
                .map(flag => {
                    if (process.platform === 'win32') return flag;
                    return flag.replace(re, '/app/');
                })
                .join(' ');

            const compilerFlagVar = flags.language === 'C' ? 'CMAKE_C_FLAGS' : 'CMAKE_CXX_FLAGS';

            const updates = new Map([
                [compilerFlagVar, allCompilerFlags],
                ['CMAKE_EXE_LINKER_FLAGS', allLinkerFlags],
                ['CMAKE_SHARED_LINKER_FLAGS', allLinkerFlags],
                ['CMAKE_STATIC_LINKER_FLAGS', allLinkerFlags],
            ]);

            for (let i = 0; i < lines.length; i++) {
                const line = lines[i];

                // Update build directory in header comment
                if (line.startsWith('# For build in directory:')) {
                    lines[i] = `# For build in directory: ${buildDir}`;
                    logger.debug('Updated build directory in CMakeCache.txt header');
                    continue;
                }

                // Update CMake variables
                for (const [variable, value] of updates) {
                    const pattern = new RegExp(`^${variable}:STRING=`);
                    if (pattern.test(line)) {
                        lines[i] = `${variable}:STRING=${value}`;
                        logger.debug(`Updated ${variable} in CMakeCache.txt`);
                        break;
                    }
                }
            }

            await fs.writeFile(cmakeCachePath, lines.join('\n'));
            logger.info('CMakeCache.txt updated with compilation flags and paths');
        } catch (error) {
            logger.warn(`Failed to modify CMakeCache.txt: ${error}`);
        }
    }
}
