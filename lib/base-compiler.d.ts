// Copyright (c) 2022, Compiler Explorer Authors
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

export declare class BaseCompiler {
    constructor(compilerInfo, env);
    public compiler;
    public lang;
    public outputFilebase: string;
    protected mtime;
    protected env;
    protected compileFilename;
    protected asm;
    protected getSharedLibraryPathsAsArguments(libraries: object[], libDownloadPath: string);
    protected getSharedLibraryPathsAsLdLibraryPaths(libraries: object[]);
    protected getCompilerCacheKey(compiler: string, args: string[], options: object);
    protected async execCompilerCached(compiler, args, options);
    protected async newTempDir();
    protected filename(fn: string): string;
    protected optionsForFilter(filters: object, outputFilename: string): string[];
    protected getExtraFilepath(dirPath: string, filename: string): string;
    protected async writeAllFiles(dirPath: string, source: string, files: any[], filters: object);
    protected async writeMultipleFiles(files: any[], dirPath: string): Promise<any[]>;
    public compilerProps: (key: string) => string;
    public getOutputFilename(dirPath: string, outputFilebase: string, key?: object): string;
    public async exec(filepath: string, args: string[], execOptions);
    public parseCompilationOutput(result, filename: string);
    public getDefaultExecOptions();
    public async runCompiler(compiler: string, args: string[], filename: string, execOptions);
    public async getVersion();
    protected getArgumentParser(): class;
}
