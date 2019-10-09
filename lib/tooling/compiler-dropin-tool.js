// Copyright (c) 2019, Compiler Explorer Authors
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
    _ = require('underscore'),
    BaseTool = require('./base-tool');

class CompilerDropinTool extends BaseTool {
    constructor(toolInfo, env) {
        super(toolInfo, env);
    }

    // mostly copy&paste from base-compiler.js
    // FIXME: remove once #1617 is merged
    findLibVersion(selectedLib, compiler) {
        const foundLib = _.find(compiler.libs, (o, libId) => libId === selectedLib.id);
        if (!foundLib) return false;

        const foundVersion = _.find(foundLib.versions, (o, versionId) => versionId === selectedLib.version);
        return foundVersion;
    }

    // mostly copy&paste from base-compiler.js
    getIncludeArguments(libraries, compiler) {
        const includeFlag = "-I";

        return _.flatten(_.map(libraries, (selectedLib) => {
            const foundVersion = this.findLibVersion(selectedLib, compiler);
            if (!foundVersion) return false;

            return _.map(foundVersion.path, (path) => includeFlag + path);
        }));
    }

    runTool(compilationInfo, inputFilepath, args) {
        const sourcefile = inputFilepath;
        const options = compilationInfo.options;

        const includeflags = this.getIncludeArguments(compilationInfo.libraries, compilationInfo.compiler);

        // order should be:
        //  1) options from the compiler config (compilationInfo.compiler.options)
        //  2) includes from the libraries (includeflags)
        //  3) options from the tool config (this.tool.options)
        //  4) options manually specified in the compiler tab (options)
        //  5) flags from the clang-tidy tab
        const compilerOptions = compilationInfo.compiler.options ? compilationInfo.compiler.options.split(" ") : [];
        let compileFlags = compilerOptions;
        compileFlags = compileFlags.concat(includeflags);

        const manualCompileFlags = options.filter(option => (option !== sourcefile));
        compileFlags = compileFlags.concat(manualCompileFlags);
        const toolOptions = this.tool.options ? this.tool.options.split(" ") : [];
        compileFlags = compileFlags.concat(toolOptions);
        args ? args : [];
        compileFlags = compileFlags.concat(args);

        return super.runTool(compilationInfo, sourcefile, compileFlags);
    }
}

module.exports = CompilerDropinTool;
