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

import {type BuildSystemId, isBuildSystemId} from '../../shared/build-systems.js';
import {splitArguments} from '../../shared/common-utils.js';
import type {BuildSystemDriver} from './build-system.interfaces.js';
import {CargoBuildSystem} from './cargo.js';
import {CMakeBuildSystem} from './cmake.js';
import {MavenBuildSystem} from './maven.js';

export type {
    BuildContext,
    BuildPlan,
    BuildSystemDriver,
    BuildSystemStep,
} from './build-system.interfaces.js';

export const cmakeBuildSystem = new CMakeBuildSystem();
export const cargoBuildSystem = new CargoBuildSystem();
export const mavenBuildSystem = new MavenBuildSystem();

const buildSystems: Record<BuildSystemId, BuildSystemDriver> = {
    cmake: cmakeBuildSystem,
    cargo: cargoBuildSystem,
    maven: mavenBuildSystem,
};

/** Look up a build system driver by id, or undefined if there is no such build system. */
export function getBuildSystem(id: string): BuildSystemDriver | undefined {
    return isBuildSystemId(id) ? buildSystems[id] : undefined;
}

/**
 * The arguments the user gave the build system. `cmakeArgs` is the original name for this and is still what the
 * frontend sends and what shared links contain, so it stays supported.
 */
export function getBuildSystemArgs(backendOptions: Record<string, any>): string[] {
    return splitArguments(backendOptions.buildSystemArgs ?? backendOptions.cmakeArgs);
}
