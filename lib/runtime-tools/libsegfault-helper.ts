import {CompilationEnvironment} from '../compilation-env.js';

export class LibSegFaultHelper {
    public static isSupported(compiler: CompilationEnvironment) {
        return process.platform !== 'win32' && compiler.ceProps('libSegFaultPath', '') !== '';
    }
}
