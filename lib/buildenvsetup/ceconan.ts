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

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {Readable, Transform} from 'node:stream';
import {pipeline} from 'node:stream/promises';
import zlib from 'node:zlib';

import tar from 'tar-stream';
import _ from 'underscore';

import {CacheKey} from '../../types/compilation/compilation.interfaces.js';
import {CompilerInfo} from '../../types/compiler.interfaces.js';
import type {LibraryBuildType} from '../../types/libraries/libraries.interfaces.js';
import {CompilationEnvironment} from '../compilation-env.js';
import {logger} from '../logger.js';
import {VersionInfo} from '../options-handler.js';
import * as utils from '../utils.js';
import {BuildEnvSetupBase} from './base.js';
import type {BuildEnvDownloadInfo} from './buildenv.interfaces.js';

export type ConanBuildProperties = {
    os: string;
    build_type: string;
    compiler: string;
    'compiler.version': string;
    'compiler.libcxx': string;
    arch: string;
    stdver: string;
    flagcollection: string;
};

type LibVerBuild = {
    id: string;
    version: string;
    lookupname: string;
    lookupversion: string;
    buildType: LibraryBuildType;
    possibleBuilds: Promise<{builds: any} | {error: unknown}>;
};

// State shared by all the library downloads of one compilation: the byte cap applies to the request as a
// whole rather than to each package separately, and one failed download stops the rest.
export type LibrarySetupContext = {
    downloadedBytes: number;
    signal?: AbortSignal;
};

// The search for a library that conan doesn't have rejects with this prefix; any other failure means we
// couldn't find out, which must not be mistaken for the library not existing.
const NOT_FOUND_PREFIX = 'Not found';

function isNotFound(error: unknown) {
    return typeof error === 'string' && error.startsWith(NOT_FOUND_PREFIX);
}

export class BuildEnvSetupCeConanDirect extends BuildEnvSetupBase {
    protected host: any;
    protected onlyonstaticliblink: any;
    protected extractAllToRoot: boolean;
    protected conan_os: string;
    // Generous cap on the total declared size of extracted files; real library packages are
    // tens to a few hundred MiB, so anything near this is a packaging error or a gzip bomb.
    protected maxExtractedBytes: number;
    // How long any request to the conan server may wait for a response, and how long a package download
    // may go without making progress.
    protected fetchTimeoutMs: number;
    // How long one package download may take in total.
    protected downloadTimeoutMs: number;
    // Real compilations select a handful of libraries at most; these bound how many packages one compilation
    // fetches and how much it downloads in total. Extracted size is capped per package (maxExtractedBytes).
    protected maxLibraries: number;
    protected maxDownloadBytes: number;

    static get key() {
        return 'ceconan';
    }

    constructor(compilerInfo: CompilerInfo, env: CompilationEnvironment) {
        super(compilerInfo, env);

        this.host = compilerInfo.buildenvsetup!.props('host', '');
        this.onlyonstaticliblink = compilerInfo.buildenvsetup!.props('onlyonstaticliblink', '');
        this.extractAllToRoot = false;
        this.conan_os = os.platform() === 'win32' ? 'Windows' : 'Linux';
        this.maxExtractedBytes = compilerInfo.buildenvsetup!.props('maxExtractedBytes', 2 * 1024 * 1024 * 1024);
        this.fetchTimeoutMs = compilerInfo.buildenvsetup!.props('fetchTimeoutMs', 120_000);
        this.downloadTimeoutMs = compilerInfo.buildenvsetup!.props('downloadTimeoutMs', 300_000);
        this.maxLibraries = compilerInfo.buildenvsetup!.props('maxLibraries', 50);
        this.maxDownloadBytes = compilerInfo.buildenvsetup!.props('maxDownloadBytes', 1024 * 1024 * 1024);
    }

    protected fetchSettings(signal?: AbortSignal): RequestInit {
        const timeout = AbortSignal.timeout(this.fetchTimeoutMs);
        return this.requestInit(signal ? AbortSignal.any([timeout, signal]) : timeout);
    }

    private requestInit(signal: AbortSignal): RequestInit {
        return {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
            },
            signal,
        };
    }

    async getAllPossibleBuilds(libid: string, version: string, signal?: AbortSignal): Promise<any> {
        return new Promise((resolve, reject) => {
            const encLibid = encodeURIComponent(libid);
            const encVersion = encodeURIComponent(version);
            const url = `${this.host}/v1/conans/${encLibid}/${encVersion}/${encLibid}/${encVersion}/search`;

            fetch(url, this.fetchSettings(signal))
                .then(async (response: Response) => {
                    if (response.status === 404) {
                        response.body?.cancel().catch(() => {});
                        reject(`${NOT_FOUND_PREFIX} (${url})`);
                    } else if (!response.ok) {
                        response.body?.cancel().catch(() => {});
                        reject(new Error(`Library search failed with status ${response.status} (${url})`));
                    } else {
                        resolve(await response.json());
                    }
                })
                .catch(err => {
                    // An abort is another library's failure stopping this one; that failure is logged already.
                    if (!signal?.aborted) {
                        logger.error(`Unexpected error during getAllPossibleBuilds(${libid}, ${version}): `, err);
                    }
                    reject(err);
                });
        });
    }

    async getPackageUrl(libid: string, version: string, hash: string, signal?: AbortSignal): Promise<string> {
        const encLibid = encodeURIComponent(libid);
        const encVersion = encodeURIComponent(version);
        const libUrl = `${this.host}/v1/conans/${encLibid}/${encVersion}/${encLibid}/${encVersion}`;
        const url = `${libUrl}/packages/${hash}/download_urls`;

        const response = await fetch(url, this.fetchSettings(signal));
        if (!response.ok) {
            response.body?.cancel().catch(() => {});
            throw new Error(`Package download URL lookup failed with status ${response.status} (${url})`);
        }
        const body = await response.json();
        const packageURL = body['conan_package.tgz'];
        if (!packageURL) {
            throw new Error('Unable to get package download URL from conan.');
        }
        return packageURL;
    }

    getDestinationFilepath(downloadPath: string, zippedPath: string, libId: string): string {
        if (this.extractAllToRoot) {
            const filename = path.basename(zippedPath);
            return path.join(downloadPath, filename);
        }
        return path.join(downloadPath, libId, zippedPath);
    }

    async downloadAndExtractPackage(
        libId: string,
        version: string,
        downloadPath: string,
        packageUrl: string,
        context: LibrarySetupContext = {downloadedBytes: 0},
    ): Promise<BuildEnvDownloadInfo> {
        const protocol = new URL(packageUrl).protocol;
        if (protocol !== 'http:' && protocol !== 'https:') {
            throw new Error(`Unexpected protocol '${protocol}' for conan package URL`);
        }

        const startTime = process.hrtime.bigint();
        const extract = tar.extract();
        const gunzip = zlib.createGunzip();
        let extractedBytes = 0;
        // tar-stream doesn't wait for this handler, so track it: nothing may still be writing files once this
        // method settles.
        let entryInFlight: Promise<void> = Promise.resolve();

        extract.on('entry', (header, stream, next) => {
            entryInFlight = this.extractEntry(header, stream, next, extract, downloadPath, libId, version, () => {
                extractedBytes += header.size ?? 0;
                return extractedBytes;
            });
        });

        // fetchTimeoutMs here resets whenever data arrives, so a stall fails quickly while a large package that is
        // arriving steadily gets up to downloadTimeoutMs.
        const idle = new AbortController();
        const timeoutMs = this.fetchTimeoutMs;
        let idleTimer: NodeJS.Timeout | undefined;
        const resetIdleTimer = () => {
            clearTimeout(idleTimer);
            idleTimer = setTimeout(
                () =>
                    idle.abort(
                        new DOMException(`Package download made no progress for ${timeoutMs} ms`, 'TimeoutError'),
                    ),
                timeoutMs,
            );
        };
        resetIdleTimer();
        try {
            return await this.fetchAndExtract(
                libId,
                version,
                packageUrl,
                context,
                extract,
                gunzip,
                idle.signal,
                resetIdleTimer,
                startTime,
            );
        } finally {
            clearTimeout(idleTimer);
            await entryInFlight;
        }
    }

    private async fetchAndExtract(
        libId: string,
        version: string,
        packageUrl: string,
        context: LibrarySetupContext,
        extract: tar.Extract,
        gunzip: zlib.Gunzip,
        idleSignal: AbortSignal,
        resetIdleTimer: () => void,
        startTime: bigint,
    ): Promise<BuildEnvDownloadInfo> {
        const signals = [idleSignal, AbortSignal.timeout(this.downloadTimeoutMs)];
        if (context.signal) signals.push(context.signal);
        const signal = AbortSignal.any(signals);
        const response = await fetch(packageUrl, this.requestInit(signal));
        if (!response.ok || !response.body) {
            response.body?.cancel().catch(() => {});
            logger.error(`Error requesting package from conan: ${response.status} for ${packageUrl}`);
            throw new Error(`Unable to request library from conan: ${response.status}`);
        }

        const maxDownloadBytes = this.maxDownloadBytes;
        const tooBig = () => new Error(`Libraries for this compilation exceed ${maxDownloadBytes} bytes downloaded`);
        const declaredLength = Number(response.headers.get('content-length'));
        if (declaredLength > 0 && context.downloadedBytes + declaredLength > maxDownloadBytes) {
            response.body.cancel().catch(() => {});
            throw tooBig();
        }
        const countDownloaded = new Transform({
            transform(chunk: Buffer, _encoding, callback) {
                resetIdleTimer();
                context.downloadedBytes += chunk.length;
                if (context.downloadedBytes > maxDownloadBytes) {
                    callback(tooBig());
                } else {
                    callback(null, chunk);
                }
            },
        });

        try {
            // pipeline propagates errors from every stage (including a download severed
            // mid-stream) and guarantees this settles; hand-rolled .pipe() chains do neither.
            await pipeline(
                Readable.fromWeb(response.body as import('node:stream/web').ReadableStream),
                countDownloaded,
                gunzip,
                extract,
                // Also stops decompression and extraction of data already received.
                {signal},
            );
        } catch (error) {
            // pipeline reports an abort as a generic AbortError; the signal's reason says which limit it was.
            const reported = (error as Error)?.name === 'AbortError' && signal.aborted ? signal.reason : error;
            if (!context.signal?.aborted) {
                logger.error(`Error downloading/extracting package from conan (${packageUrl}):`, reported);
            }
            throw reported;
        }

        const endTime = process.hrtime.bigint();
        return {
            step: `Download of ${libId} ${version}`,
            packageUrl: packageUrl,
            time: utils.deltaTimeNanoToMili(startTime, endTime),
        };
    }

    private async extractEntry(
        header: tar.Header,
        stream: tar.ExtractEvents['entry'][1],
        next: () => void,
        extract: tar.Extract,
        downloadPath: string,
        libId: string,
        version: string,
        addExtracted: () => number,
    ): Promise<void> {
        // Drained streams (skipped entries) have no other error listener; without this an
        // entry-stream error would throw as an unhandled 'error' event.
        stream.on('error', (error: any) => extract.destroy(error));
        try {
            const filepath = this.getDestinationFilepath(downloadPath, header.name, libId);

            // Each library extracts under its own root; skip entries that would resolve outside
            // it (and so could overwrite other libraries' files or the compilation's own).
            const extractionRoot = path.resolve(this.extractAllToRoot ? downloadPath : path.join(downloadPath, libId));
            const relative = path.relative(extractionRoot, path.resolve(path.dirname(filepath)));
            if (relative === '..' || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative)) {
                logger.error(`Library ${libId}/${version} is using a zip-slip, skipping file`);
                stream.resume();
                next();
                return;
            }

            if (addExtracted() > this.maxExtractedBytes) {
                throw new Error(`Package ${libId}/${version} exceeds ${this.maxExtractedBytes} bytes extracted`);
            }

            if (header.type !== 'file') {
                // Only regular files are ever written; symlinks, hardlinks, directories etc
                // are drained and skipped. Draining also avoids a hang on malformed entries
                // (e.g. a directory with a non-zero size, whose stream tar-stream never ends).
                stream.resume();
                next();
                return;
            }

            if (!this.extractAllToRoot) {
                await fs.promises.mkdir(path.dirname(filepath), {recursive: true});
            }
            if (extract.destroying) {
                // The download failed or was aborted while we waited; write nothing more.
                stream.resume();
                return;
            }

            if (header.size === 0) {
                // tar-stream emits no data for zero-length entries (see
                // https://github.com/mafintosh/tar-stream/issues/145): create empty files
                // explicitly.
                await fs.promises.writeFile(filepath, '');
                stream.resume();
                next();
            } else {
                await pipeline(stream, fs.createWriteStream(filepath));
                next();
            }
        } catch (error) {
            logger.error(`Error extracting entry ${JSON.stringify(header.name)} of ${libId}/${version}:`, error);
            // Propagate the failure through the extract stream so the outer pipeline rejects.
            extract.destroy(error as Error);
        }
    }

    async getConanBuildProperties(key: CacheKey, buildType: LibraryBuildType): Promise<ConanBuildProperties> {
        const arch = this.getTarget(key);
        const libcxx = this.getLibcxx(key);
        const stdver = '';
        const flagcollection = '';

        return {
            os: this.conan_os,
            build_type: buildType,
            compiler: this.compilerTypeOrGCC,
            'compiler.version': this.compiler.id,
            'compiler.libcxx': libcxx,
            arch: arch,
            stdver: stdver,
            flagcollection: flagcollection,
        };
    }

    async findMatchingHash(buildProperties: ConanBuildProperties, possibleBuilds: any) {
        return _.findKey(possibleBuilds, elem => {
            return _.all(buildProperties, (val, key) => {
                if ((key === 'compiler' || key === 'compiler.version') && elem.settings[key] === 'headeronly') {
                    return true;
                }
                if ((key === 'compiler' || key === 'compiler.version') && elem.settings[key] === 'cshared') {
                    return true;
                }
                if (key === 'compiler.libcxx' && elem.settings['compiler'] === 'headeronly') {
                    return true;
                }
                if (key === 'arch' && elem.settings['compiler'] === 'headeronly') {
                    return true;
                }
                if (key === 'compiler.libcxx' && elem.settings['compiler'] === 'cshared') {
                    return true;
                }
                if (key === 'stdver') {
                    // unless ABI breakage happens in future stdversions, assume they are all cxx11 and compatible
                    //  or if not that the compiler.version has already made sure those won't be matched
                    return true;
                }
                return val === elem.settings[key];
            });
        });
    }

    // Debug unless the request asked for Release for this library; a Release request falls back to
    // Debug for libraries that have no Release package. MSVC packages are mostly Release only, and
    // CE compiles MSVC code against the release runtime, so MSVC always prefers Release.
    buildTypesToTry(buildType: LibraryBuildType): LibraryBuildType[] {
        if (this.compilerTypeOrGCC === 'win32-vc' || buildType === 'Release') return ['Release', 'Debug'];
        return ['Debug'];
    }

    async findBuildHash(
        key: CacheKey,
        possibleBuilds: any,
        buildType: LibraryBuildType,
    ): Promise<{hash: string; buildType: LibraryBuildType} | undefined> {
        for (const candidate of this.buildTypesToTry(buildType)) {
            const buildProperties = await this.getConanBuildProperties(key, candidate);
            const hash = await this.findMatchingHash(buildProperties, possibleBuilds);
            if (hash) return {hash, buildType: candidate};
        }
        return undefined;
    }

    async download(
        key: CacheKey,
        dirPath: string,
        libraryDetails: Record<string, VersionInfo>,
    ): Promise<BuildEnvDownloadInfo[]> {
        const allDownloads: Promise<BuildEnvDownloadInfo>[] = [];
        const allLibraryBuilds: LibVerBuild[] = [];
        const abort = new AbortController();
        const context: LibrarySetupContext = {downloadedBytes: 0, signal: abort.signal};
        let hasFailed = false;
        let firstFailure: unknown;
        const failed = (error: unknown) => {
            if (!hasFailed) {
                hasFailed = true;
                firstFailure = error;
            }
            abort.abort();
        };

        const numLibraries = Object.keys(libraryDetails).length;
        if (numLibraries > this.maxLibraries) {
            throw new Error(
                `Too many libraries selected (${numLibraries}); at most ${this.maxLibraries} can be used in one compilation`,
            );
        }

        _.each(libraryDetails, (details: VersionInfo, libId: string) => {
            if (details.packagedheaders || this.hasBinariesToLink(details)) {
                const lookupname = details.lookupname || libId;
                const lookupversion = details.lookupversion || details.version;
                allLibraryBuilds.push({
                    id: libId,
                    version: details.version,
                    lookupname: details.lookupname as string,
                    lookupversion: details.lookupversion as string,
                    // A request that doesn't name a build type gets Debug, as it always has.
                    buildType: details.buildType ?? 'Debug',
                    // Settled here so a failure is handled even while an earlier library is being awaited.
                    possibleBuilds: this.getAllPossibleBuilds(
                        lookupname as string,
                        lookupversion as string,
                        abort.signal,
                    ).then(
                        builds => ({builds}),
                        (error: unknown) => {
                            // Fail now rather than once the libraries before this one are done.
                            if (!isNotFound(error)) failed(error);
                            return {error};
                        },
                    ),
                });
            }
        });

        try {
            await this.startDownloads(key, dirPath, allLibraryBuilds, context, allDownloads, failed);
        } catch (error) {
            failed(error);
        }
        // Wait for every download to stop, including ones aborted by another's failure, so nothing is still
        // writing into the compilation directory once this returns.
        const results = await Promise.allSettled(allDownloads);
        if (hasFailed) throw firstFailure;
        return results.map(result => (result as PromiseFulfilledResult<BuildEnvDownloadInfo>).value);
    }

    private async startDownloads(
        key: CacheKey,
        dirPath: string,
        allLibraryBuilds: LibVerBuild[],
        context: LibrarySetupContext,
        allDownloads: Promise<BuildEnvDownloadInfo>[],
        failed: (error: unknown) => void,
    ) {
        for (const libVerBuilds of allLibraryBuilds) {
            const lookupname = libVerBuilds.lookupname || libVerBuilds.id;
            const lookupversion = libVerBuilds.lookupversion || libVerBuilds.version;
            const libVer = `${lookupname}/${lookupversion}`;
            const search = await libVerBuilds.possibleBuilds;
            // Another library's download has already failed the compilation.
            if (context.signal?.aborted) return;
            if ('error' in search) {
                if (!isNotFound(search.error)) {
                    throw search.error;
                }
            }
            const possibleBuilds = 'builds' in search ? search.builds : false;
            if (possibleBuilds) {
                const found = await this.findBuildHash(key, possibleBuilds, libVerBuilds.buildType);
                if (found) {
                    logger.debug(`Found conan hash ${found.hash} for ${libVer} with build type ${found.buildType}`);
                    const download = this.getPackageUrl(lookupname, lookupversion, found.hash, context.signal).then(
                        downloadUrl =>
                            this.downloadAndExtractPackage(
                                libVerBuilds.id,
                                lookupversion,
                                dirPath,
                                downloadUrl,
                                context,
                            ),
                    );
                    // Later libraries' searches are awaited before anything else waits on this download.
                    download.catch(failed);
                    allDownloads.push(download);
                } else {
                    const buildProperties = await this.getConanBuildProperties(key, libVerBuilds.buildType);
                    logger.warn(`No build found for ${libVer} matching ${JSON.stringify(buildProperties)}`);
                }
            } else {
                logger.warn(`Library ${libVer} not available`);
            }
        }
    }

    override async setup(
        key: CacheKey,
        dirPath: string,
        libraryDetails: Record<string, VersionInfo>,
        binary: boolean,
    ): Promise<BuildEnvDownloadInfo[]> {
        if (!this.host) return [];

        if (this.onlyonstaticliblink && !binary) return [];

        const librariesToDownload = _.pick(libraryDetails, (details: VersionInfo) => {
            return this.shouldDownloadPackage(details);
        }) as Record<string, VersionInfo>;

        return this.download(key, dirPath, librariesToDownload);
    }
}
