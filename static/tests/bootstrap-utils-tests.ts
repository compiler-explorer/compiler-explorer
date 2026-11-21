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

import {Popover} from 'bootstrap';
import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';

import {getPopoverInstance, setHtmlPopoverContent} from '../bootstrap-utils.js';

describe('setHtmlPopoverContent', () => {
    let trigger: HTMLElement;

    beforeEach(() => {
        vi.useFakeTimers();
        trigger = document.createElement('div');
        document.body.append(trigger);
    });

    afterEach(() => {
        vi.useRealTimers();
        getPopoverInstance(trigger)?.dispose();
        trigger.remove();
    });

    function tip(): HTMLElement | null {
        return document.querySelector('.popover');
    }

    it('creates an HTML popover on first use', () => {
        const popover = setHtmlPopoverContent(trigger, {body: '<b>hi</b>', className: 'extra'});
        expect(popover).toBeInstanceOf(Popover);
        popover?.show();
        expect(tip()?.querySelector('.popover-body')?.innerHTML).toBe('<b>hi</b>');
        expect(tip()?.classList.contains('extra')).toBe(true);
    });

    it('updates the existing popover instead of replacing it', () => {
        const first = setHtmlPopoverContent(trigger, {body: 'one'});
        const second = setHtmlPopoverContent(trigger, {body: 'two', className: 'later'});
        expect(second).toBe(first);
        second?.show();
        expect(tip()?.querySelector('.popover-body')?.textContent).toBe('two');
        expect(tip()?.classList.contains('later')).toBe(true);
    });

    it('applies a changed title and body to a popover that is already open', () => {
        const popover = setHtmlPopoverContent(trigger, {title: 'old title', body: 'old'});
        popover?.show();
        vi.advanceTimersByTime(1000);
        setHtmlPopoverContent(trigger, {title: 'new title', body: 'new'});
        expect(tip()?.querySelector('.popover-header')?.textContent).toBe('new title');
        expect(tip()?.querySelector('.popover-body')?.textContent).toBe('new');
    });

    it('can drop the title again', () => {
        const popover = setHtmlPopoverContent(trigger, {title: 'a title', body: 'x'});
        setHtmlPopoverContent(trigger, {body: 'y'});
        popover?.show();
        expect(tip()?.querySelector('.popover-header')).toBeNull();
    });

    it('returns null when the element does not exist', () => {
        expect(setHtmlPopoverContent('#does-not-exist', {body: 'x'})).toBeNull();
    });

    it('survives being updated while the popover is fading out', () => {
        const popover = setHtmlPopoverContent(trigger, {body: 'before'});
        popover?.show();
        vi.advanceTimersByTime(1000);
        popover?.hide();
        setHtmlPopoverContent(trigger, {body: 'after'});
        expect(() => vi.advanceTimersByTime(1000)).not.toThrow();
    });
});
