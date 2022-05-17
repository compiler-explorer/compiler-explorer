import {ResultLine} from '../resultline/resultline.interfaces';

export type CompilationResult = {
    code: number;
    buildResult: unknown;
    asm?: ResultLine[];
    stdout?: ResultLine[];
    stderr?: ResultLine[];
    execResult?: {
        stdout?: ResultLine[];
        stderr?: ResultLine[];
    };
    hasGnatDebugOutput: boolean;
    gnatDebugOutput?: ResultLine[];
    hasGnatDebugTreeOutput: boolean;
    gnatDebugTreeOutput?: ResultLine[];
};
