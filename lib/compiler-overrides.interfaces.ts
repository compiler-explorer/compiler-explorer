enum CompilerOverrideType {
    stdlib,
    gcclib,
    arch,
}

type CompilerOverrideTypes = Set<CompilerOverrideType>;

type CompilerOverrideOption = {
    name: string;
    value: string;
};

type CompilerOverrideOptions = Array<CompilerOverrideOption>;

type AllCompilerOverrideOptions = Record<CompilerOverrideType, CompilerOverrideOptions>;
