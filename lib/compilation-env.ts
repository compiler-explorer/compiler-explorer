// Copyright (c) 2016, Compiler Explorer Authors
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

import child_process from 'child_process';

import fs from 'fs-extra';
import _ from 'underscore';

import type {CacheableValue} from '../types/cache.interfaces.js';
import {CompilerOverrideOptions} from '../types/compilation/compiler-overrides.interfaces.js';

import {unwrap} from './assert.js';
import type {Cache} from './cache/base.interfaces.js';
import {BaseCache} from './cache/base.js';
import {createCacheFromConfig} from './cache/from-config.js';
import {CompilationQueue, EnqueueOptions, Job} from './compilation-queue.js';
import {FormattingHandler} from './handlers/formatting.js';
import {logger} from './logger.js';
import type {PropertyGetter} from './properties.interfaces.js';
import {CompilerProps} from './properties.js';
import {createStatsNoter, IStatsNoter} from './stats.js';

type PropFunc = (string, any?) => any;

export class CompilationEnvironment {
    ceProps: PropertyGetter;
    compilationQueue: CompilationQueue;
    compilerProps: CompilerProps;
    okOptions: RegExp;
    badOptions: RegExp;
    cache: Cache;
    executableCache: Cache;
    compilerCache: Cache;
    reportCacheEvery: number;
    multiarch: string | null;
    baseEnv: Record<string, string>;
    formatHandler: FormattingHandler;
    possibleToolchains?: CompilerOverrideOptions;
    statsNoter: IStatsNoter;
    private logCompilerCacheAccesses: boolean;

    constructor(compilerProps, compilationQueue, doCache) {
        this.ceProps = compilerProps.ceProps;
        this.compilationQueue = compilationQueue;
        this.compilerProps = compilerProps.get.bind(compilerProps);
        // So people running local instances don't break suddenly when updating
        const deprecatedAllowed = this.ceProps('optionsWhitelistRe', '.*');
        const deprecatedForbidden = this.ceProps('optionsBlacklistRe', '(?!)');

        this.okOptions = new RegExp(this.ceProps('optionsAllowedRe', deprecatedAllowed));
        this.badOptions = new RegExp(this.ceProps('optionsForbiddenRe', deprecatedForbidden));
        this.cache = createCacheFromConfig(
            'default',
            doCache === undefined || doCache ? this.ceProps('cacheConfig', '') : '',
        );
        this.executableCache = createCacheFromConfig(
            'executable',
            doCache === undefined || doCache ? this.ceProps('executableCacheConfig', '') : '',
        );
        this.compilerCache = createCacheFromConfig(
            'compiler',
            doCache === undefined || doCache ? this.ceProps('compilerCacheConfig', '') : '',
        );
        this.reportCacheEvery = this.ceProps('cacheReportEvery', 100);
        this.multiarch = null;
        try {
            const multi = child_process.execSync('gcc -print-multiarch').toString().trim();
            if (multi) {
                logger.info(`Multiarch: ${multi}`);
                this.multiarch = multi;
            } else {
                logger.info('No multiarch');
            }
        } catch (err) {
            logger.warn(`Unable to get multiarch: ${err}`);
        }
        this.baseEnv = {};
        const envs = this.ceProps('environmentPassThrough', 'LD_LIBRARY_PATH,PATH,HOME').split(',');
        _.each(envs, environmentVariable => {
            if (environmentVariable === '') return;
            this.baseEnv[environmentVariable] = process.env[environmentVariable] ?? '';
        });
        // I'm not sure that this is the best design; but each compiler having its own means each constructs its own
        // handler, and passing it in from the outside is a pain as each compiler's constructor needs it.
        this.formatHandler = new FormattingHandler(this.ceProps);
        this.logCompilerCacheAccesses = this.ceProps('logCompilerCacheAccesses', false);
        this.statsNoter = createStatsNoter(this.ceProps);
    }

    getEnv(needsMulti: boolean) {
        const env = {...this.baseEnv};
        if (needsMulti && this.multiarch) {
            env.LIBRARY_PATH = '/usr/lib/' + this.multiarch;
            env.C_INCLUDE_PATH = '/usr/include/' + this.multiarch;
            env.CPLUS_INCLUDE_PATH = '/usr/include/' + this.multiarch;
        }
        return env;
    }

    setPossibleToolchains(toolchains: CompilerOverrideOptions) {
        this.possibleToolchains = toolchains;
    }

    getPossibleToolchains(): CompilerOverrideOptions {
        return this.possibleToolchains || [];
    }

    async cacheGet(object: CacheableValue) {
        const result = await this.cache.get(BaseCache.hash(object));
        if (this.cache.gets % this.reportCacheEvery === 0) {
            this.cache.report();
        }
        if (!result.hit) return null;
        return JSON.parse(unwrap(result.data).toString());
    }

    async cachePut(object: CacheableValue, result: object, creator: string | undefined) {
        const key = BaseCache.hash(object);
        return this.cache.put(key, JSON.stringify(result), creator);
    }

    async compilerCacheGet(object: CacheableValue) {
        const key = BaseCache.hash(object);
        const result = await this.compilerCache.get(key);
        if (this.logCompilerCacheAccesses) {
            logger.info(`Cache get ${JSON.stringify(object)} hash ${key} ${result.hit ? 'hit' : 'miss'}`);
        }
        if (!result.hit) return null;
        return JSON.parse(unwrap(result.data).toString());
    }

    async compilerCachePut(object: CacheableValue, result: object, creator: string | undefined) {
        const key = BaseCache.hash(object);
        if (this.logCompilerCacheAccesses) {
            logger.info(`Cache put ${JSON.stringify(object)} hash ${key}`);
        }
        return this.compilerCache.put(key, JSON.stringify(result), creator);
    }

    async executableGet(object: CacheableValue, destinationFolder: string): Promise<string | null> {
        const key = BaseCache.hash(object) + '_exec';
        const result = await this.executableCache.get(key);
        if (!result.hit) return null;
        const filepath = destinationFolder + '/' + key;
        await fs.writeFile(filepath, unwrap(result.data));
        return filepath;
    }

    async executablePut(object: CacheableValue, filepath: string): Promise<string> {
        const key = BaseCache.hash(object) + '_exec';
        await this.executableCache.put(key, fs.readFileSync(filepath));
        return key;
    }

    enqueue<T>(job: Job<T>, options?: EnqueueOptions) {
        return this.compilationQueue.enqueue(job, options);
    }

    findBadOptions(options: string[]) {
        return options.filter(option => !this.okOptions.test(option) || this.badOptions.test(option));
    }

    getCompilerPropsForLanguage(languageId: string): PropFunc {
        return _.partial(this.compilerProps as any, languageId);
    }
}
