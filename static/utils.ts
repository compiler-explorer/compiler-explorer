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

export function updateAndCalcTopBarHeight(domRoot: JQuery, topBar: JQuery, hideable: JQuery): number {
    let topBarHeight = 0;
    if (!topBar.hasClass('d-none')) {
        hideable.show();
        const topBarHeightMax = topBar.outerHeight(true) ?? 0;
        hideable.hide();
        const topBarHeightMin = topBar.outerHeight(true) ?? 0;
        topBarHeight = topBarHeightMin;
        if (topBarHeightMin === topBarHeightMax) {
            hideable.show();
        }
    }
    return topBarHeight;
}

/**
 *  Subscribe and unsuscribe the event listener.
 *
 * @param  {JQuery} element
 * @param  {string} eventName
 * @param  {(event:JQuery.Event)=>void} callback
 * @returns void
 */
export function toggleEventListener(element: JQuery, eventName: string, callback: (event: JQuery.Event) => void): void {
    element.on(eventName, (event: JQuery.Event) => {
        callback(event);
        element.off(eventName);
    });
}
