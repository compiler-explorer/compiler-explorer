// Copyright (c) 2016, Matt Godbolt & Rubén Rincón
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

const
    Queue = require('promise-queue'),
    child_process = require('child_process'),
    FromConfig = require('./cache/from-config'),
    BaseCache = require('./cache/base'),
    logger = require('./logger').logger,
    _ = require('underscore'),
    Raven = require('raven');

Queue.configure(Promise);

class CompilationEnvironment {
    constructor(compilerProps, doCache) {
        this.ceProps = compilerProps.ceProps;
        this.compilerProps = compilerProps.get.bind(compilerProps);
        this.okOptions = new RegExp(this.ceProps('optionsWhitelistRe', '.*'));
        this.badOptions = new RegExp(this.ceProps('optionsBlacklistRe', '(?!)'));
        this.cache = FromConfig.create(doCache === undefined || doCache ? this.ceProps('cacheConfig', '') : "");
        this.compileQueue = new Queue(this.ceProps("maxConcurrentCompiles", 1), Infinity);
        this.reportCacheEvery = this.ceProps("cacheReportEvery", 100);
        this.multiarch = null;
        try {
            const multi = child_process.execSync("gcc -print-multiarch").toString().trim();
            if (multi) {
                logger.info(`Multiarch: ${multi}`);
                this.multiarch = multi;
            } else {
                logger.info("No multiarch");
            }
        } catch (err) {
            logger.warn(`Unable to get multiarch: ${err}`);
        }
    }

    getEnv(needsMulti) {
        const env = {
            LD_LIBRARY_PATH: process.env.LD_LIBRARY_PATH,
            PATH: process.env.PATH,
            HOME: process.env.HOME
        };
        if (needsMulti && this.multiarch) {
            env.LIBRARY_PATH = '/usr/lib/' + this.multiarch;
            env.C_INCLUDE_PATH = '/usr/include/' + this.multiarch;
            env.CPLUS_INCLUDE_PATH = '/usr/include/' + this.multiarch;
        }
        return env;
    }

    cacheGet(object) {
        const key = BaseCache.hash(object);
        return this.cache.get(key).then((result) => {
            if ((this.cache.gets % this.reportCacheEvery) === 0) {
                this.cache.report();
            }
            if (!result.hit) return null;
            return JSON.parse(result.data);
        });
    }

    cachePut(object, result, creator) {
        const key = BaseCache.hash(object);
        return this.cache.put(key, JSON.stringify(result), creator);
    }

    enqueue(job) {
        const wrappedJob = function () {
            try {
                return job();
            } catch (e) {
                if (e.stack) logger.info(e.stack);
                logger.error(`Caught promise exception ${e}`);
                Raven.captureException(e);
                return Promise.resolve(null);
            }
        };
        return this.compileQueue.add(wrappedJob);
    }

    isBusy() {
        return this.compileQueue.getPendingLength() > 0 || this.compileQueue.getQueueLength() > 0;
    }

    findBadOptions(options) {
        return _.filter(options, option => !option.match(this.okOptions) || option.match(this.badOptions));
    }
}

module.exports = CompilationEnvironment;
