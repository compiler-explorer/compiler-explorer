// Copyright (c) 2021, Compiler Explorer Authors
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

import { Container } from 'golden-layout'

export interface PaneCompilerState {
    compilerId: unknown;
    compilerName: string;
    editorId: unknown
}

export interface BasicPane {
    container: Container;
    eventHub: unknown; /* typeof hub.createEventHub() */
    state: unknown; /* typeof state.selection */
    domRoot: JQuery;

    // new(hub: unknown, state: unknown, container: Container);
    initButtons(state: unknown /* typeof state */);
    initCallbacks();

    setTitle();
    getPaneName();
    resize();
    close();

    onCompiler(id: unknown, compiler: unknown, options: unknown, editorId: unknown);
    onCompileResult(id: unknown, compiler: unknown, result: unknown);
    onCompilerClose(id: unknown);
    onDidChangeCursorSelection(event: Event);
    onSettingsChange(settings: unknown);

    getCurrentState();
    updateState();
}
