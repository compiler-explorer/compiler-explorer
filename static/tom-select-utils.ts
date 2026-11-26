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

import TomSelect from 'tom-select';
import type {TomClearFilter, TomOption} from 'tom-select/dist/esm/types/core.js';

/**
 * Replace a picker's entire option set and repaint the drop-down.
 *
 * `clearOptions()` and `addOption()` only mark the rendered content stale; neither they nor
 * a later `open()` repaint it. Only tom-select's focus and input handlers call
 * `refreshOptions()`, so a drop-down that is open, or that is reopened without first losing
 * focus, keeps showing the previous option list until something forces a re-render.
 *
 * `refreshOptions(false)` is what forces it: tom-select gates both its open and its close
 * branch on that argument, so passing `false` re-renders the list while leaving the
 * drop-down's open/closed state alone.
 *
 * @param picker the tom-select instance to update
 * @param options the new option set
 * @param clearFilter passed through to `clearOptions()`. Return true to keep an option.
 *   Pass `() => false` to drop every option, including those belonging to the current
 *   selection, which tom-select's default filter would otherwise keep.
 */
export function replaceOptions(picker: TomSelect, options: TomOption[], clearFilter?: TomClearFilter): void {
    picker.clearOptions(clearFilter);
    picker.addOptions(options);
    picker.refreshOptions(false);
}
