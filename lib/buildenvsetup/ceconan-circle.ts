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

import {BuildEnvSetupCeConanDirect, ConanBuildProperties} from './ceconan.js';

export class BuildEnvSetupCeConanCircleDirect extends BuildEnvSetupCeConanDirect {
    private linkedCompilerId: string;
    private linkedCompilerType: string;

    static override get key() {
        return 'ceconan-circle';
    }

    constructor(compilerInfo, env) {
        super(compilerInfo, env);

        this.linkedCompilerId = compilerInfo.buildenvsetup.props('linkedCompilerId');
        this.linkedCompilerType = compilerInfo.buildenvsetup.props('linkedCompilerType');
    }

    override async getConanBuildProperties(key): Promise<ConanBuildProperties> {
        const props = await super.getConanBuildProperties(key);
        props['compiler'] = this.linkedCompilerType;
        props['compiler.version'] = this.linkedCompilerId;
        return props;
    }
}
