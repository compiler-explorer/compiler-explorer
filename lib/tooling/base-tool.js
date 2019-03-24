// Copyright (c) 2018, Compiler Explorer Team
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
    exec = require('../exec'),
    utils = require('../utils'),
    path = require('path');

class BaseTool {
    constructor(toolInfo, env) {
        this.tool = toolInfo;
        this.env = env;
        this.tool.exclude = this.tool.exclude ? this.tool.exclude.split(':') : [];
    }

    getId() {
        return this.tool.id;
    }

    getType() {
        return this.tool.type || "independent";
    }

    isCompilerExcluded(compilerId) {
        return this.tool.exclude.find((excl) => compilerId.includes(excl));
    }

    exec(toolExe, args, options) {
        options = options || {};
        if (toolExe.match(/\.exe$/i) && (process.platform === 'linux')) {
            args.unshift(toolExe);
            toolExe = this.env.ceProps("wine");
            options.env = options.env || {};
            options.env.WINEDLLOVERRIDES = "vcruntime140=b";
            options.env.WINEDEBUG = "-all";
        }
        return exec.execute(toolExe, args, options);
    }

    getDefaultExecOptions() {
        return {
            timeoutMs: this.env.ceProps("compileTimeoutMs", 7500),
            maxErrorOutput: this.env.ceProps("max-error-output", 5000),
            wrapper: this.env.compilerProps("compiler-wrapper")
        };
    }

    createErrorResponse(message) {
        return {
            id: this.tool.id,
            name: this.tool.name,
            code: -1,
            stdout: utils.parseOutput(message)
        };
    }

    runTool(sourcefile, args) {
        let execOptions = this.getDefaultExecOptions();
        execOptions.customCwd = path.dirname(sourcefile);

        args = args ? args : [];
        args = args.filter((arg) => !arg.includes('/'));
        args.push(sourcefile);

        const exeDir = path.dirname(this.tool.exe);

        return this.exec(this.tool.exe, args, execOptions).then(result => {
            return {
                id: this.tool.id,
                name: this.tool.name,
                code: result.code,
                stderr: utils.parseOutput(result.stderr, sourcefile, exeDir),
                stdout: utils.parseOutput(result.stdout, sourcefile, exeDir)
            };
        }).catch(() => {
            return this.createErrorResponse("Error while running tool");
        });
    }
}

module.exports = BaseTool;
