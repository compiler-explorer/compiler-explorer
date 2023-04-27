export enum CompilerOverrideType {
    stdlib = 'stdlib',
    gcclib = 'gcclib',
    arch = 'arch',
    env = 'env',
}

export type CompilerOverrideTypes = Set<CompilerOverrideType>;

export type CompilerOverrideOption = {
    name: string;
    value: string;
};

export type CompilerOverrideOptions = Array<CompilerOverrideOption>;

export type AllCompilerOverrideOptions = Record<CompilerOverrideType, CompilerOverrideOptions>;

export type EnvvarOverride = {
    name: string;
    value: string;
};

export type EnvvarOverrides = Array<EnvvarOverride>;

export type ConfiguredOverride = {
    name: CompilerOverrideType;
    value?: CompilerOverrideOption;
    values?: EnvvarOverrides;
};

export type ConfiguredOverrides = Array<ConfiguredOverride>;
