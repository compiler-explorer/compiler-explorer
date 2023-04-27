import type {LLVMOptPipelineBackendOptions} from '../../types/compilation/llvm-opt-pipeline-output.interfaces.js';
import type {PPOptions} from './pp-view.interfaces.js';
import type {GccDumpViewSelectedPass} from './gccdump-view.interfaces.js';
import type {FiledataPair} from '../../types/compilation/compilation.interfaces.js';
import type {ConfiguredOverrides} from '../compilation/compiler-overrides.interfaces.js';

export type ActiveTools = {
    id: number;
    args: string[];
    stdin: string;
};

export type CompilationRequestOptions = {
    userArguments: string;
    compilerOptions: {
        executorRequest?: boolean;
        skipAsm?: boolean;
        producePp?: PPOptions | null;
        produceAst?: boolean;
        produceGccDump?: {
            opened: boolean;
            pass?: GccDumpViewSelectedPass;
            treeDump?: boolean;
            rtlDump?: boolean;
            ipaDump?: boolean;
            dumpFlags: any;
        };
        produceOptInfo?: boolean;
        produceCfg?: boolean;
        produceGnatDebugTree?: boolean;
        produceGnatDebug?: boolean;
        produceIr?: boolean;
        produceLLVMOptPipeline?: LLVMOptPipelineBackendOptions | null;
        produceDevice?: boolean;
        produceRustMir?: boolean;
        produceRustMacroExp?: boolean;
        produceRustHir?: boolean;
        produceHaskellCore?: boolean;
        produceHaskellStg?: boolean;
        produceHaskellCmm?: boolean;
        cmakeArgs?: string;
        customOutputFilename?: string;
        overrides?: ConfiguredOverrides;
    };
    executeParameters: {
        args: string;
        stdin: string;
    };
    filters: Record<string, boolean>;
    tools: ActiveTools[];
    libraries: CompileChildLibraries[];
};

export type CompilationRequest = {
    source: string;
    compiler: string;
    options: CompilationRequestOptions;
    lang: string | null;
    files: FiledataPair[];
    bypassCache?: boolean;
};

export type LangInfo = {
    compiler: string;
    options: string;
    execArgs: string;
    execStdin: string;
};

export type CompileChildLibraries = {
    id: string;
    version: string;
};
