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

import {StorageClass} from '@aws-sdk/client-s3';
import ems from 'enhanced-ms';

import {FiledataPair} from '../types/compilation/compilation.interfaces.js';
import {CompilerOverrideOptions} from '../types/compilation/compiler-overrides.interfaces.js';
import {ConfiguredRuntimeTool} from '../types/execution/execution.interfaces.js';
import {SelectedLibraryVersion} from '../types/libraries/libraries.interfaces.js';

import {ParsedRequest} from './handlers/compile.js';
import {logger} from './logger.js';
import {PropertyGetter} from './properties.interfaces.js';
import {S3Bucket} from './s3-handler.js';
import {getHash} from './utils.js';

export enum KnownBuildMethod {
    Compile = 'compile',
    CMake = 'cmake',
}

export interface IStatsNoter {
    noteCompilation(compilerId: string, request: ParsedRequest, files: FiledataPair[], buildMethod: string): void;
}

class NullStatsNoter implements IStatsNoter {
    noteCompilation(compilerId: string, request: ParsedRequest, files: FiledataPair[], buildMethod: string): void {}
}

// A type for storing only compilation information deemed non-identifying; that is, no source or execution options.
// This started out as a `Omit<ParsedRequest, ...>` but really in order to be more useful it needs to be more specialised.
type CompilationRecord = {
    time: string;
    compilerId: string;
    sourceHash: string;
    executionParamsHash: string;
    options: string[];
    filters: Record<string, boolean>;
    backendOptions: string[];
    bypassCache: boolean;
    libraries: string[];
    tools: string[];
    overrides: string[];
    runtimeTools: string[];
    buildMethod: string;
};

export function filterCompilerOptions(args: string[]): string[] {
    const capturableArg = /^[/-]/;
    const unwantedArg = /^(([/-][DIdi])|(-$))/;
    return args.filter(x => capturableArg.exec(x) && !unwantedArg.test(x));
}

// note: any type on `request` is on purpose, we cannot trust ParsedRequest to be truthful to the type as it is user input

export function makeSafe(
    time: Date,
    compilerId: string,
    request: ParsedRequest,
    files: FiledataPair[],
    buildMethod: string,
): CompilationRecord {
    const sourceHash = getHash(request.source + JSON.stringify(files));
    return {
        time: time.toISOString(),
        compilerId: compilerId,
        sourceHash: sourceHash,
        executionParamsHash: getHash(request.executeParameters),
        options: filterCompilerOptions(request.options),
        filters: Object.fromEntries(
            Object.entries(request.filters)
                .filter(value => typeof value[1] === 'boolean')
                .map(item => [item[0].toLowerCase(), item[1]]),
        ) as Record<string, boolean>,
        backendOptions: Object.entries(
            Object.fromEntries(
                Object.entries(request.backendOptions)
                    .filter(item => item[0] !== 'overrides')
                    .map(item => [item[0].toLowerCase(), item[1]]),
            ),
        ).map(item => `${item[0]}=${item[1] ? '1' : '0'}`),
        bypassCache: !!request.bypassCache,
        libraries: (request.libraries || []).map((lib: SelectedLibraryVersion) => lib.id + '/' + lib.version),
        tools: (request.tools || []).map(tool => tool.id),
        overrides: ((request.backendOptions.overrides || []) as CompilerOverrideOptions)
            .filter(item => item.name !== 'env' && item.value)
            .map(item => `${item.name}=${item.value}`),
        runtimeTools: (request.executeParameters.runtimeTools || [])
            .filter((item: ConfiguredRuntimeTool) => item.name !== 'env')
            .map((item: ConfiguredRuntimeTool) => item.name),
        buildMethod: buildMethod,
    };
}

function makeKey(now: Date): string {
    return `year=${now.getUTCFullYear()}/month=${now.getUTCMonth()}/date=${now.getUTCDate()}/${now.toISOString()}.json`;
}

class StatsNoter implements IStatsNoter {
    private _statsQueue: CompilationRecord[];
    private readonly _flushAfterMs: number;
    private _flushJob: NodeJS.Timeout | undefined;
    private readonly _s3: S3Bucket;
    private readonly _path: string;

    constructor(bucket: string, path?: string, region?: string, flushMs?: number) {
        this._statsQueue = [];
        this._flushAfterMs = flushMs ?? 5 * 60 * 1000;
        this._flushJob = undefined;
        this._s3 = new S3Bucket(bucket, region ?? 'us-east-1');
        this._path = path ?? 'compile-stats';
        logger.info(`Flushing stats to ${bucket}/${this._path} every ${ems(this._flushAfterMs)}`);
    }

    private flush() {
        const toFlush = this._statsQueue;
        this._statsQueue = [];
        if (toFlush) {
            // async write to S3
            const key = makeKey(new Date(Date.now()));
            this._s3
                .put(key, Buffer.from(toFlush.map(x => JSON.stringify(x)).join('\n')), this._path, {
                    redundancy: StorageClass.REDUCED_REDUNDANCY,
                })
                .then(() => {})
                .catch(e => {
                    logger.warn(`Caught exception trying to log compilations to ${key}: ${e}`);
                });
        }
        if (this._flushJob !== undefined) {
            clearTimeout(this._flushJob);
            this._flushJob = undefined;
        }
    }

    noteCompilation(compilerId: string, request: ParsedRequest, files: FiledataPair[], buildMethod: string): void {
        this._statsQueue.push(makeSafe(new Date(), compilerId, request, files, buildMethod));
        if (!this._flushJob) this._flushJob = setTimeout(() => this.flush(), this._flushAfterMs);
    }
}

export function createStatsNoter(props: PropertyGetter): IStatsNoter {
    const config = props('compilationStatsNotifier', 'None()');
    const match = config.match(/^([^(]+)\(([^)]*)\)$/);
    if (!match) throw new Error(`Unable to parse '${config}'`);
    const params = match[2].split(',');

    const type = match[1];
    switch (type) {
        case 'None': {
            if (params.length !== 1) throw new Error(`Bad params: ${config}`);
            return new NullStatsNoter();
        }
        case 'S3': {
            if (params.length === 0 || params.length > 4)
                throw new Error(`Bad params: ${config} - expected S3(bucket, path?, region?, flushTime?)`);
            let durationMs: number | undefined;
            if (params[3]) {
                const parsed = ems(params[3]);
                if (!parsed)
                    throw new Error(
                        `Bad params: ${config} - expected S3(bucket, path?, region?, flushTime?), bad flush time`,
                    );
                durationMs = parsed;
            }
            return new StatsNoter(params[0], params[1], params[2], durationMs);
        }
    }
    throw new Error(`Unknown stats type '${type}'`);
}
