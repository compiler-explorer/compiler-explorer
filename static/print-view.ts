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

import $ from 'jquery';

import * as monaco from 'monaco-editor';

import {Hub} from './hub';
import {Themer, themes} from './themes';
import {unwrap} from './assert';
import {EventHub} from './event-hub';

export class Printerinator {
    printview: JQuery;
    eventHub: EventHub;
    itemsAdded: number;
    css: string | null = null;

    constructor(hub: Hub, private readonly themer: Themer) {
        this.eventHub = hub.createEventHub();
        this.printview = $('#printview');

        window.addEventListener('beforeprint', this.setupPrintView.bind(this));
        this.eventHub.on('printdata', this.addPrintData.bind(this));
    }

    getThemeStyles() {
        if (this.css === null) {
            // Awful hacky stuff to get the theme declarations from monaco
            const container = document.createElement('div');
            const editor = monaco.editor.create(container);
            const theme = this.themer.getCurrentTheme();
            this.themer.setTheme(themes.default);
            const css = (editor as any)._codeEditorService._themeService._themeCSS;
            this.css = css
                .slice(css.indexOf('.mtk1'))
                .trim()
                .split('\n')
                .map(line => '#printview ' + line)
                .join('\n');
            this.themer.setTheme(unwrap(theme));
        }
        return this.css;
    }

    setupPrintView() {
        this.printview.empty();
        this.printview[0].innerHTML = `<style>${this.getThemeStyles()}</style>`;
        this.itemsAdded = 0;
        // It's important that any highlighting is done under the default theme (simply applying the default css
        // to tokens from another theme doesn't quite work)
        const theme = this.themer.getCurrentTheme();
        this.themer.setTheme(themes.default);
        // Request print data from everyone
        this.eventHub.emit('printrequest');
        // Restore theme
        this.themer.setTheme(unwrap(theme));
    }

    addPrintData(data: string) {
        if (this.itemsAdded++ !== 0) {
            this.printview[0].innerHTML += `<div class="pagebreak"></div>`;
        }
        this.printview[0].innerHTML += data;
    }
}
