// Copyright (c) 2016, Rubén Rincón & Compiler Explorer Authors
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

import { BaseCache } from './cache/base';
import { createCacheFromConfig } from './cache/from-config';
import { logger } from './logger';

export class CompilationEnvironment {
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
            doCache === undefined || doCache ? this.ceProps('cacheConfig', '') : '');
        this.executableCache = createCacheFromConfig(
            'executable',
            doCache === undefined || doCache ? this.ceProps('executableCacheConfig', '') : '');
        this.compilerCache = createCacheFromConfig(
            'compiler',
            doCache === undefined || doCache ? this.ceProps('compilerCacheConfig', '') : '');
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
            if (!environmentVariable) return;
            this.baseEnv[environmentVariable] = process.env[environmentVariable];
        });
    }

    getEnv(needsMulti) {
        const env = {...this.baseEnv};
        if (needsMulti && this.multiarch) {
            env.LIBRARY_PATH = '/usr/lib/' + this.multiarch;
            env.C_INCLUDE_PATH = '/usr/include/' + this.multiarch;
            env.CPLUS_INCLUDE_PATH = '/usr/include/' + this.multiarch;
        }
        return env;
    }

    async cacheGet(object) {
        const result = await this.cache.get(BaseCache.hash(object));
        if ((this.cache.gets % this.reportCacheEvery) === 0) {
            this.cache.report();
        }
        if (!result.hit) return null;
        return JSON.parse(result.data);
    }

    async cachePut(object, result, creator) {
        const key = BaseCache.hash(object);
        return this.cache.put(key, JSON.stringify(result), creator);
    }

    async compilerCacheGet(object) {
        const key = BaseCache.hash(object);
        const result = await this.compilerCache.get(key);
        if (!result.hit) return null;
        return JSON.parse(result.data);
    }

    async compilerCachePut(object, result, creator) {
        const key = BaseCache.hash(object);
        return this.compilerCache.put(key, JSON.stringify(result), creator);
    }

    async executableGet(object, destinationFolder) {
        const key = BaseCache.hash(object) + '_exec';
        const result = await this.executableCache.get(key);
        if (!result.hit) return null;
        const filepath = destinationFolder + '/' + key;
        await fs.writeFile(filepath, result.data);
        return filepath;
    }

    executablePut(object, filepath) {
        const key = BaseCache.hash(object) + '_exec';
        return this.executableCache.put(key, fs.readFileSync(filepath));
    }

    enqueue(job) {
        return this.compilationQueue.enqueue(job);
    }

    findBadOptions(options) {
        return _.filter(options, option => !this.okOptions.test(option) || option.match(this.badOptions));
    }
}
