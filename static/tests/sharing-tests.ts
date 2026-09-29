// Copyright (c) 2026, Compiler Explorer Authors
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

import GoldenLayout from 'golden-layout';
import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';

import {options} from '../options.js';
import {LAYOUT_SETTLE_MS, SharingBase} from '../sharing.js';

// A layout that only does what SharingBase asks of it: hand over a config and let
// something subscribe to `stateChanged`.
class FakeLayout {
    private handler: (() => void) | null = null;
    public config: any;

    constructor(config: any) {
        this.config = config;
    }

    on(event: string, handler: () => void): void {
        if (event === 'stateChanged') this.handler = handler;
    }

    toConfig(): any {
        return this.config;
    }

    /** What GoldenLayout does when a pane writes to its own state. */
    change(config: any): void {
        this.config = config;
        if (this.handler) this.handler();
    }
}

// The state a compiler pane is in before it has finished initialising, and after.
const DURING_INIT = {
    content: [{componentName: 'compiler', componentState: {}, title: ' No compiler set (Tree #undefined)'}],
};
const AFTER_INIT = {
    content: [
        {
            componentName: 'compiler',
            componentState: {filters: {binary: false}, libs: [], options: '', overrides: []},
            title: ' g++ default (Editor #1)',
        },
    ],
};
const USER_EDITED = {
    content: [
        {
            componentName: 'compiler',
            componentState: {filters: {binary: false}, libs: [], options: '-O2', overrides: []},
            title: ' g++ default (Editor #1)',
        },
    ],
};

describe('SharingBase URL freshness', () => {
    let replaceState: ReturnType<typeof vi.fn>;

    let wasEmbedded: unknown;

    beforeEach(() => {
        vi.useFakeTimers();
        replaceState = vi.fn();
        window.history.replaceState = replaceState as any;
        window.httpRoot = '/';
        window.history.pushState(null, '', '/z/abc123');
        wasEmbedded = options.embedded;
        options.embedded = false;
    });

    afterEach(() => {
        vi.useRealTimers();
        options.embedded = wasEmbedded;
    });

    function newSharing(config: any): FakeLayout {
        const layout = new FakeLayout(config);
        new SharingBase(layout as unknown as GoldenLayout);
        return layout;
    }

    it('keeps a shortlink URL through pane initialisation', () => {
        const layout = newSharing(DURING_INIT);

        layout.change(DURING_INIT);
        layout.change(AFTER_INIT);
        vi.advanceTimersByTime(LAYOUT_SETTLE_MS);

        expect(replaceState).not.toHaveBeenCalled();
    });

    it('drops the shortlink URL once the state really changes', () => {
        const layout = newSharing(DURING_INIT);

        layout.change(AFTER_INIT);
        vi.advanceTimersByTime(LAYOUT_SETTLE_MS);
        layout.change(USER_EDITED);

        expect(replaceState).toHaveBeenCalledWith(null, '', '/');
    });

    it('leaves the URL alone when the settled state comes back unchanged', () => {
        const layout = newSharing(DURING_INIT);

        layout.change(AFTER_INIT);
        vi.advanceTimersByTime(LAYOUT_SETTLE_MS);
        layout.change(AFTER_INIT);

        expect(replaceState).not.toHaveBeenCalled();
    });

    it('drops the shortlink URL when the user edits before the layout settles', () => {
        const layout = newSharing(DURING_INIT);

        layout.change(AFTER_INIT);
        // Well inside the window, someone starts typing. What follows is theirs, so the
        // link stops describing the page and has to go.
        vi.advanceTimersByTime(LAYOUT_SETTLE_MS / 4);
        document.body.dispatchEvent(new KeyboardEvent('keydown', {key: 'a', bubbles: true}));
        layout.change(USER_EDITED);

        expect(replaceState).toHaveBeenCalledWith(null, '', '/');
    });

    it('only arms the settle timer where the URL can actually be rewritten', () => {
        newSharing(DURING_INIT);
        expect(vi.getTimerCount()).toBe(1);

        vi.clearAllTimers();
        options.embedded = true;
        newSharing(DURING_INIT);

        expect(vi.getTimerCount()).toBe(0);
    });

    it('does not rewrite a URL that is already the root', () => {
        window.history.pushState(null, '', '/');
        const layout = newSharing(DURING_INIT);

        layout.change(AFTER_INIT);
        vi.advanceTimersByTime(LAYOUT_SETTLE_MS);
        layout.change(USER_EDITED);

        expect(replaceState).not.toHaveBeenCalled();
    });
});
