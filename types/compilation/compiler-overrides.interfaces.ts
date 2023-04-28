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

export type CompilerOverrideNameAndOptions = {
    name: CompilerOverrideType;
    display_title: string;
    description: string;
    flags: string[];
    values: CompilerOverrideOptions;
};

export type AllCompilerOverrideOptions = Array<CompilerOverrideNameAndOptions>;

export type EnvvarOverride = {
    name: string;
    value: string;
};

export type EnvvarOverrides = Array<EnvvarOverride>;

export type ConfiguredOverride = {
    name: CompilerOverrideType;
    value?: string;
    values?: EnvvarOverrides;
};

export type ConfiguredOverrides = Array<ConfiguredOverride>;
