export declare class BaseCompiler {
    constructor(compilerInfo, env);
    public compiler;
    public lang;
    public outputFilebase: string;
    public compilerProps: (key: string) => string;
    public getOutputFilename(path: string, filename: string): string;
    public exec(filepath: string, args: string[], execOptions);
    public parseCompilationOutput(result, filename: string);
    public getDefaultExecOptions();
    public runCompiler(compiler: string, args: string[], filename: string, execOptions);
}
