// Copyright (c) 2026, Compiler Explorer Authors
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

import {describe, expect, it} from 'vitest';

import {cargoBuildSystem, cmakeBuildSystem} from '../lib/build-systems/index.js';
import {getRequestedBuildSystem, type RemoteCompilationRequest} from '../lib/compilation/sqs-compilation-queue.js';

function makeMessage(fields: Partial<RemoteCompilationRequest>): RemoteCompilationRequest {
    return fields as RemoteCompilationRequest;
}

describe('Which build system a queued request asked for', () => {
    it('resolves one it knows, however the request spells it', () => {
        expect(getRequestedBuildSystem(makeMessage({buildSystem: 'cargo'}))).toBe(cargoBuildSystem);
        // The CMake-only boolean, still sent by producers deployed apart from this.
        expect(getRequestedBuildSystem(makeMessage({isCMake: true}))).toBe(cmakeBuildSystem);
    });

    it('is undefined only when nothing was asked for', () => {
        expect(getRequestedBuildSystem(makeMessage({}))).toBeUndefined();
        expect(getRequestedBuildSystem(makeMessage({isCMake: false}))).toBeUndefined();
    });

    it('refuses a build system it does not know rather than reading it as none', () => {
        // Undefined would send the project's manifest through a plain compilation, which answers with syntax errors
        // about the manifest and counts as an ordinary compile. A producer can be ahead of a worker across a deploy.
        expect(() => getRequestedBuildSystem(makeMessage({buildSystem: 'gradle'}))).toThrow(
            /Unknown build system 'gradle'/,
        );
        expect(() => getRequestedBuildSystem(makeMessage({buildSystem: 'toString'}))).toThrow(/Unknown build system/);
    });
});
