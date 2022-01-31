
export declare class BaseCompiler {
    constructor(compilerInfo: any, env: any);
    public compiler: any;
    public lang: any;
    public outputFilebase: string;
    public compilerProps: (key: string) => string;
    public getOutputFilename(path: string, filename: string): string;
    public exec(filepath: string, args: string[], execOptions: any): any;
    public parseCompilationOutput(result: any, filename: string);
    public getDefaultExecOptions(): any;
    public runCompiler(compiler: string, args: string[], filename: string, execOptions: any);
}
