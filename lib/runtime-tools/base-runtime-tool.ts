// Copyright (c) 2023, Compiler Explorer Authors
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

import {RuntimeToolOptions, TypicalExecutionFunc} from '../../types/execution/execution.interfaces.js';

export class BaseRuntimeTool {
    protected dirPath: string;
    protected sandboxFunc: TypicalExecutionFunc;
    protected execFunc: TypicalExecutionFunc;
    protected options: RuntimeToolOptions;
    protected sandboxType: string;

    constructor(
        dirPath: string,
        sandboxFunc: TypicalExecutionFunc,
        execFunc: TypicalExecutionFunc,
        options: RuntimeToolOptions,
        sandboxType: string,
    ) {
        this.dirPath = dirPath;
        this.sandboxFunc = sandboxFunc;
        this.execFunc = execFunc;
        this.options = options;
        this.sandboxType = sandboxType;
    }

    protected getOptionValue(name: string): string | undefined {
        const option = this.options.find(opt => opt.name === name);
        if (option) return option.value;
    }
}
