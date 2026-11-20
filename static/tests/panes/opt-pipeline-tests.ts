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

import $ from 'jquery';
import {describe, expect, it, vi} from 'vitest';

import type {Pass} from '../../../types/compilation/opt-pipeline-output.interfaces.js';
import {OptPipeline} from '../../panes/opt-pipeline.js';

function makePipeline(hideUnchanged = false) {
    const passes: Pass[] = [
        {name: 'No-op', before: [], after: [], machine: false, irChanged: false},
        {name: 'Fold <values>', before: [], after: [], machine: false, irChanged: true},
    ];
    const pipeline = Object.create(OptPipeline.prototype) as OptPipeline;
    const displayPass = vi.fn();
    Object.assign(pipeline, {
        state: {selectedIndex: 0},
        results: {group: passes},
        filters: {get: () => ({'filter-inconsequential-passes': hideUnchanged})},
        passesFilter: $('<input>'),
        passesList: $('<div>'),
        displayPass,
    });
    pipeline.selectGroup('group');
    return {passesList: pipeline.passesList, displayPass};
}

describe('LLVM opt pipeline pass list', () => {
    it('labels changed passes in visible text without altering unchanged names', () => {
        const {passesList} = makePipeline();

        expect(passesList.find('.pass').eq(0).text()).toBe('No-op');
        expect(passesList.find('.pass').eq(1).text()).toBe('Fold <values> (changed)');
    });

    it('keeps the pass index when filtering and clicking a labelled pass', () => {
        const {passesList, displayPass} = makePipeline(true);

        expect(passesList.find('.pass')).toHaveLength(1);
        passesList.find('.pass').trigger('click');
        expect(displayPass).toHaveBeenLastCalledWith(1);
    });
});
