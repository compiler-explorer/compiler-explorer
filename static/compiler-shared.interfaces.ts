import type {ConfiguredOverrides} from './compilation/compiler-overrides.interfaces.js';
import type {CompilerCurrentState} from './panes/compiler.interfaces.js';

export interface ICompilerShared {
    updateState(state: CompilerCurrentState);
    getOverrides(): ConfiguredOverrides;
}
