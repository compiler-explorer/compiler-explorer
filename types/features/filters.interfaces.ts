export type ParseFilters = {
    binary: boolean;
    execute: boolean;
    demangle: boolean;
    intel: boolean;
    commentOnly: boolean;
    directives: boolean;
    labels: boolean;
    optOutput: boolean;
    libraryCode: boolean;
    trim: boolean;
    dontMaskFilenames?: boolean;
};
