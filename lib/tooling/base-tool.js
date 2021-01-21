// Copyright (c) 2018, Compiler Explorer Authors
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

import path from 'path';

import _ from 'underscore';

import * as exec from '../exec';
import { logger } from '../logger';
import * as utils from '../utils';

export class BaseTool {
    constructor(toolInfo, env) {
        this.tool = toolInfo;
        this.env = env;
        this.tool.exclude = this.tool.exclude ? this.tool.exclude.split(':') : [];
    }

    getId() {
        return this.tool.id;
    }

    getType() {
        return this.tool.type || 'independent';
    }

    getUniqueFilePrefix() {
        const timestamp = process.hrtime();
        const timestamp_str = '_' + timestamp[0] * 1000000 + timestamp[1] / 1000;
        return this.tool.id.replace(/[^\da-z]/gi, '_') + timestamp_str + '_';
    }

    isCompilerExcluded(compilerId, compilerProps) {
        if (this.tool.includeKey) {
            // If the includeKey is set, we only support compilers that have a truthy 'includeKey'.
            if (!compilerProps(this.tool.includeKey)) {
                return true;
            }
            // Even if the include key is truthy, we fall back to the exclusion list.
        }
        return this.tool.exclude.find((excl) => compilerId.includes(excl));
    }

    exec(toolExe, args, options) {
        return exec.execute(toolExe, args, options);
    }

    getDefaultExecOptions() {
        return {
            timeoutMs: this.env.ceProps('compileTimeoutMs', 7500),
            maxErrorOutput: this.env.ceProps('max-error-output', 5000),
            wrapper: this.env.compilerProps('compiler-wrapper'),
        };
    }

    createErrorResponse(message) {
        return {
            id: this.tool.id,
            name: this.tool.name,
            code: -1,
            languageId: 'stderr',
            stdout: utils.parseOutput(message),
        };
    }

    // mostly copy&paste from base-compiler.js
    findLibVersion(selectedLib, compiler) {
        const foundLib = _.find(compiler.libs, (o, libId) => libId === selectedLib.id);
        if (!foundLib) return false;

        return _.find(foundLib.versions, (o, versionId) => versionId === selectedLib.version);
    }

    // mostly copy&paste from base-compiler.js
    getIncludeArguments(libraries, compiler) {
        const includeFlag = '-I';

        return _.flatten(_.map(libraries, (selectedLib) => {
            const foundVersion = this.findLibVersion(selectedLib, compiler);
            if (!foundVersion) return false;

            return _.map(foundVersion.path, (path) => includeFlag + path);
        }));
    }

    getLibraryOptions(libraries, compiler) {
        return _.flatten(_.map(libraries, (selectedLib) => {
            const foundVersion = this.findLibVersion(selectedLib, compiler);
            if (!foundVersion) return false;

            return foundVersion.options;
        }));
    }

    async runTool(compilationInfo, inputFilepath, args, stdin) {
        let execOptions = this.getDefaultExecOptions();
        if (inputFilepath) execOptions.customCwd = path.dirname(inputFilepath);
        execOptions.input = stdin;

        args = args ? args : [];
        if (inputFilepath) args.push(inputFilepath);

        const exeDir = path.dirname(this.tool.exe);

        try {
            const result = await this.exec(this.tool.exe, args, execOptions);
            return this.convertResult(result, inputFilepath, exeDir);
        } catch (e) {
            logger.error('Error while running tool: ', e);
            return this.createErrorResponse('Error while running tool');
        }
    }

    convertResult(result, inputFilepath, exeDir) {
        const transformedFilepath = result.filenameTransform(inputFilepath);
        return {
            id: this.tool.id,
            name: this.tool.name,
            code: result.code,
            languageId: this.tool.languageId,
            stderr: utils.parseOutput(result.stderr, transformedFilepath, exeDir),
            stdout: utils.parseOutput(result.stdout, transformedFilepath, exeDir),
        };
    }
}
