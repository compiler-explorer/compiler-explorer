// Copyright (c) 2012-2017, Matt Godbolt
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

const LRU = require('lru-cache'),
    Promise = require('promise'), // jshint ignore:line
    Queue = require('promise-queue'),
    child_process = require('child_process'),
    logger = require('./logger').logger,
    _ = require('underscore-node'),
    Raven = require('raven');

Queue.configure(Promise);

function CompilationEnvironment(ceProps, compilerPropsL) {
    this.ceProps = ceProps;
    this.compilerPropsL = compilerPropsL;
    this.okOptions = new RegExp(ceProps('optionsWhitelistRe', '.*'));
    this.badOptions = new RegExp(ceProps('optionsBlacklistRe', '(?!)'));
    this.cache = LRU({
        max: ceProps('cacheMb') * 1024 * 1024,
        length: n => JSON.stringify(n).length
    });
    this.cacheHits = 0;
    this.cacheMisses = 0;
    this.compileQueue = new Queue(ceProps("maxConcurrentCompiles", 1), Infinity);
    this.multiarch = null;
    try {
        var multi = child_process.execSync("gcc -print-multiarch").toString().trim();
        if (multi) {
            logger.info("Multiarch: " + multi);
            this.multiarch = multi;
        } else {
            logger.info("No multiarch");
        }
    } catch (err) {
        logger.warn("Unable to get multiarch: " + err);
    }
}

CompilationEnvironment.prototype.getEnv = function (needsMulti) {
    var env = {
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
};

CompilationEnvironment.prototype.cacheStats = function () {
    var total = this.cacheHits + this.cacheMisses;
    if ((total % 100) == 1) {
        var pc = (100 * this.cacheHits) / total;
        logger.info("Cache stats: " + this.cacheHits + " hits, " + this.cacheMisses + " misses (" + pc.toFixed(2) +
            "%), LRU has " + this.cache.itemCount + " item(s) totalling " + this.cache.length + " bytes");
    }
};

CompilationEnvironment.prototype.cacheGet = function (key) {
    var cached = this.cache.get(key);
    if (cached) {
        this.cacheStats();
        this.cacheHits++;
        return cached;
    }
    this.cacheMisses++;
    return undefined;
};

CompilationEnvironment.prototype.cachePut = function (key, result) {
    this.cache.set(key, result);
    this.cacheStats();
};

CompilationEnvironment.prototype.enqueue = function (job) {
    var wrappedJob = function () {
        try {
            return job();
        } catch (e) {
            Raven.captureException(e);
            logger.error("Caught promise exception " + e);
            if (e.stack) logger.info(e.stack);
            return Promise.resolve(null);
        }
    };
    return this.compileQueue.add(wrappedJob);
};

CompilationEnvironment.prototype.isBusy = function () {
    return this.compileQueue.getPendingLength() > 0 || this.compileQueue.getQueueLength() > 0;
};

CompilationEnvironment.prototype.findBadOptions = function (options) {
    return _.filter(options, function (option) {
        return !option.match(this.okOptions) || option.match(this.badOptions);
    }, this);
};

exports.CompilationEnvironment = CompilationEnvironment;
