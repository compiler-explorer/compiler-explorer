// Copyright (c) 2023, Compiler Explorer Authors
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright notice,
//       this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

import type {ICompilerShared} from './compiler-shared.interfaces.js';
import {CompilerOverridesWidget} from './widgets/compiler-overrides.js';
import type {CompilerState} from './panes/compiler.interfaces.js';
import type {ConfiguredOverrides} from './compilation/compiler-overrides.interfaces.js';
import type {ExecutorState} from './panes/executor.interfaces.js';

export class CompilerShared implements ICompilerShared {
    private domRoot: JQuery<HTMLElement>;
    private overridesButton: JQuery<HTMLElement>;
    private overridesWidget: CompilerOverridesWidget;

    constructor(domRoot: JQuery, onChange: () => void) {
        this.domRoot = domRoot;
        this.initButtons(onChange);
        this.initCallbacks();
    }

    public getOverrides(): ConfiguredOverrides | undefined {
        return this.overridesWidget.get();
    }

    public updateState(state: CompilerState | ExecutorState) {
        this.overridesWidget.setCompiler(state.compiler, state.lang);

        if (state.overrides) {
            this.overridesWidget.set(state.overrides);
        } else {
            this.overridesWidget.setDefaults();
        }
    }

    private initButtons(onChange: () => void) {
        this.overridesButton = this.domRoot.find('.btn.show-overrides');

        this.overridesWidget = new CompilerOverridesWidget(this.domRoot, this.overridesButton, onChange);
    }

    private initCallbacks() {
        this.overridesButton.on('click', () => {
            this.overridesWidget.show();
        });
    }
}
