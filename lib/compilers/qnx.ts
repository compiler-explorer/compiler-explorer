// Copyright (c) 2024, Compiler Explorer Authors
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

import {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import {CompilationEnvironment} from '../compilation-env.js';

export class QNXCompiler extends BaseCompiler {
    qnxHost: string;
    qnxTarget: string;
    qnxLicense: string;

    static get key() {
        return 'qnx';
    }

    constructor(info: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(info, env);
        this.qnxHost = this.compilerProps<string>(`compiler.${this.compiler.id}.qnxHost`);
        this.qnxTarget = this.compilerProps<string>(`compiler.${this.compiler.id}.qnxTarget`);
        this.qnxLicense = this.compilerProps<string>(`compiler.${this.compiler.id}.qnxLicense`);
    }

    override getDefaultExecOptions() {
        const execOptions = super.getDefaultExecOptions();
        execOptions.env = {
            ...execOptions.env,
            QNX_HOST: this.qnxHost,
            QNX_TARGET: this.qnxTarget,
            QNX_SHARED_LICENSE_FILE: this.qnxLicense,
        };

        return execOptions;
    }
}
