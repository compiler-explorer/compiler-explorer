import $ from 'jquery';
import type {ICompilerShared} from './compiler-shared.interfaces.js';
import {CompilerOverridesWidget} from './widgets/compiler-overrides.js';
import type {CompilerState} from './panes/compiler.interfaces.js';
import type {ConfiguredOverrides} from './compilation/compiler-overrides.interfaces.js';
import type {ExecutorState} from './panes/executor.interfaces.js';

export class CompilerShared implements ICompilerShared {
    private domRoot: JQuery<HTMLElement>;
    private overridesButton: JQuery<HTMLElement>;
    private overridesWidget: CompilerOverridesWidget;
    private onChange: () => void;

    constructor(domRoot: JQuery, onChange: () => void) {
        this.domRoot = domRoot;
        this.onChange = onChange;
        this.initButtons();
        this.initCallbacks();
    }

    public getOverrides(): ConfiguredOverrides {
        return this.overridesWidget.get();
    }

    public updateState(state: CompilerState | ExecutorState) {
        this.overridesWidget.setCompiler(state.compiler, state.lang);

        if (state.overrides) {
            this.overridesWidget.set(state.overrides);
        } else {
            this.overridesWidget.set([]);
        }
    }

    private initButtons() {
        this.overridesButton = this.domRoot.find('.btn.show-overrides');

        this.overridesWidget = new CompilerOverridesWidget(this.domRoot, this.overridesButton, this.onChange);
    }

    private initCallbacks() {
        this.overridesButton.on('click', () => {
            this.overridesWidget.show();
        });
    }
}
