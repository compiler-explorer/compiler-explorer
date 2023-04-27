import type {ConfiguredOverrides} from './compilation/compiler-overrides.interfaces.js';
import type {CompilerState} from './panes/compiler.interfaces.js';
import type {ExecutorState} from './panes/executor.interfaces.js';

export interface ICompilerShared {
    updateState(state: CompilerState | ExecutorState);
    getOverrides(): ConfiguredOverrides;
}
