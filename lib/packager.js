// Copyright (c) 2019, Compiler Explorer Team
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
"use strict";

const
    exec = require('./exec').execute,
    fs = require('fs-extra'),
    path = require('path'),
    utils = require('./utils'),
    logger = require('./logger').logger;

class Packager {
    constructor() {
    }

    package(executable, destination) {
        return this.findDependentFiles(executable).then(files => {
            return this.tarFiles(files, destination);
        });
    }

    unpack(packageFile, destination) {
        return exec( 'tar', ['-xzvf', packageFile, '-C', destination], []);
    }

    findFile(filename, searchPaths) {
        for (let searchPath of searchPaths) {
            const maybeFile = path.join(searchPath, filename);
            logger.debug(`Looking for ${filename} at ${maybeFile}...`);
            if (fs.existsSync(maybeFile)) {
                logger.debug(`Found ${filename} at ${maybeFile}`);
                return maybeFile;
            }
        }
        throw Error(`Unable to find path for ${filename}`);
    }

    findDependentFiles(executable) {
        // TODO handle configuration of objdumper
        // TODO handle configuration of search path
        const searchPaths = [
            '/usr/lib/gcc/x86_64-linux-gnu/8/',
            '/usr/lib/gcc/x86_64-linux-gnu/8/../../../../x86_64-linux-gnu/lib/x86_64-linux-gnu/8/',
            '/usr/lib/gcc/x86_64-linux-gnu/8/../../../../x86_64-linux-gnu/lib/x86_64-linux-gnu/',
            '/usr/lib/gcc/x86_64-linux-gnu/8/../../../../x86_64-linux-gnu/lib/../lib/',
            '/usr/lib/gcc/x86_64-linux-gnu/8/../../../x86_64-linux-gnu/8/',
            '/usr/lib/gcc/x86_64-linux-gnu/8/../../../x86_64-linux-gnu/',
            '/usr/lib/gcc/x86_64-linux-gnu/8/../../../../lib/',
            '/lib/x86_64-linux-gnu/8/', '/lib/x86_64-linux-gnu/',
            '/lib/../lib/',
            '/usr/lib/x86_64-linux-gnu/8/',
            '/usr/lib/x86_64-linux-gnu/',
            '/usr/lib/../lib/',
            '/usr/lib/gcc/x86_64-linux-gnu/8/../../../../x86_64-linux-gnu/lib/',
            '/usr/lib/gcc/x86_64-linux-gnu/8/../../../',
            '/lib/',
            '/usr/lib/'];
        return exec("objdump", ["-p", executable])
            .then(result => {
                if (result.code !== 0) {
                    return result;
                }
                const NEEDED = /^\s+NEEDED\s+(.*)$/;
                return [executable].concat(
                    utils.splitLines(result.stdout)
                        .map(x => x.match(NEEDED))
                        .filter(x => x)
                        .map(x => this.findFile(x[1], searchPaths)));
            });
    }

    tarFiles(files, destination) {
        return exec("tar", [
            "zcf", destination, // Create the file
            "--dereference", // deref symlinks
            "-P", // Allow leading /
            "--xform", "s:^.*/::" // "flatten" the hierarchy
        ].concat(files))
            .then(result => {
                if (result.code !== 0) {
                    throw Error(`Unable to tar files: ${result.code}`);
                }
                return destination;
            });
    }
}

module.exports = {
    Packager: Packager
};
