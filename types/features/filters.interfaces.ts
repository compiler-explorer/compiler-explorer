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

// These options are used both for the output options and our filtering passes
// applied to the compiler output. They correspond to the "Compiler output
// options" and "Compiler output filters" drop down menu in a compiler pane.

// TODO(jeremy-rifkin): Change name to include "filters"?
export type CompilerOutputOptions = {
    binary: boolean;
    binaryObject: boolean;
    execute: boolean;
    demangle: boolean;
    intel: boolean;
};

export type preProcessLinesFunc = (lines: string[]) => string[];
export type ParseFiltersAndOutputOptions = {
    labels: boolean;
    libraryCode: boolean;
    directives: boolean;
    commentOnly: boolean;
    trim: boolean;
    dontMaskFilenames?: boolean;
    optOutput: boolean;
    preProcessLines?: preProcessLinesFunc;
    preProcessBinaryAsmLines?: preProcessLinesFunc;
} & CompilerOutputOptions;
