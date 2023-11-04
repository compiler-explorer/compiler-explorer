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

export enum CompilerOverrideType {
    stdlib = 'stdlib',
    gcclib = 'gcclib',
    toolchain = 'toolchain',
    arch = 'arch',
    env = 'env',
    edition = 'edition',
    stdver = 'stdver',
    action = 'action',
}

export type CompilerOverrideTypes = Set<CompilerOverrideType>;

export type CompilerOverrideOption = {
    name: string;
    value: string;
};

export type CompilerOverrideOptions = Array<CompilerOverrideOption>;

export type CompilerOverrideNameAndOptions = {
    name: CompilerOverrideType;
    display_title: string;
    description: string;
    flags: string[];
    values: CompilerOverrideOptions;
    default?: string;
};

export type AllCompilerOverrideOptions = Array<CompilerOverrideNameAndOptions>;

export type EnvVarOverride = {
    name: string;
    value: string;
};

export type EnvVarOverrides = Array<EnvVarOverride>;

export type ConfiguredOverrideGeneral = {
    name: Exclude<CompilerOverrideType, CompilerOverrideType.env>;
    value: string;
};

export type ConfiguredOverrideEnv = {
    name: CompilerOverrideType.env;
    values: EnvVarOverrides;
};

export type ConfiguredOverride = ConfiguredOverrideGeneral | ConfiguredOverrideEnv;

export type ConfiguredOverrides = Array<ConfiguredOverride>;
