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

import EventEmitter from 'events';
import $ from 'jquery';
import {editor} from 'monaco-editor';
import {options} from '../options.js';
import {Settings} from '../settings.js';

import IEditor = editor.IEditor;

import {FontScaleState} from './fontscale.interfaces.js';

function getDefaultFontScale() {
    return Settings.getStoredSettings().defaultFontScale ?? options.defaultFontScale;
}

function makeFontSizeDropdown(elem: JQuery, obj: FontScale, buttonDropdown: JQuery) {
    function onClickEvent(this: JQuery) {
        // Toggle off the selection of the others
        $(this).addClass('active').siblings().removeClass('active');
        // Send the data
        obj.applyScale($(this).data('value'));
        obj.emit('change');
    }

    for (let i = 8; i <= 30; i++) {
        const item = $('<button></button>');

        item.attr('data-value', i).addClass('dropdown-item').text(i).appendTo(elem).on('click', onClickEvent);

        if (obj.scale === i) {
            item.addClass('active');
        }
    }

    buttonDropdown.on('wheel', (e: any) => {
        e.preventDefault();
        let selectedId = elem.find('.active').index();
        if (e.originalEvent.deltaY < 0 && selectedId < elem.children().length - 1) {
            selectedId++;
        } else if (e.originalEvent.deltaY >= 0 && selectedId > 0) {
            selectedId--;
        }
        elem.children().eq(selectedId).trigger('click');
    });

    // ctrl+click handler
    buttonDropdown.on('click', e => {
        if (e.ctrlKey) {
            // This is a hack. It prevents bootstrap's click listener from opening the dropdown
            e.stopImmediatePropagation();
            // Set the correct scale as active
            elem.find('.active').removeClass('active');
            elem.find(`[data-value=${getDefaultFontScale()}]`).addClass('active');
            // Set the scale
            obj.applyScale(getDefaultFontScale());
            obj.emit('change');
        }
    });
}

function convertOldScale(oldScale: number): number {
    // New low + ((new max - new low) * (oldScale - old low) / (old max - old low))
    return Math.floor(8 + (22 * (oldScale - 0.3)) / 2.7);
}

export class FontScale extends EventEmitter.EventEmitter {
    private domRoot: JQuery;
    private fontSizeList: JQuery;
    public scale: number;
    private readonly usePxUnits: boolean;
    private fontSelectorOrEditor: JQuery | string | IEditor;
    private isFontOfStr: boolean;

    constructor(domRoot: JQuery, state: FontScaleState & any, fontSelectorOrEditor: JQuery | string | IEditor) {
        super();
        this.domRoot = domRoot;
        // Old scale went from 0.3 to 3. New one uses 8 up to 30, so we can convert the old ones to the new format
        this.scale = state.fontScale || getDefaultFontScale();
        // The check seems pointless, but it ensures a false in case it's undefined
        // FontScale assumes it's an old state if it does not see a fontUsePx in the state, so at first it will use pt.
        // So the second condition is there to make new objects actually use px
        this.usePxUnits = state.fontUsePx === true || !state.fontScale;
        if (this.scale < 8) {
            this.scale = convertOldScale(this.scale);
        }
        this.setTarget(fontSelectorOrEditor);
        this.apply();
        this.fontSizeList = this.domRoot.find('.font-size-list');
        makeFontSizeDropdown(this.fontSizeList, this, this.domRoot.find('.fs-button'));
    }

    apply() {
        if (this.isFontOfStr) {
            this.domRoot
                .find(this.fontSelectorOrEditor as JQuery)
                .css('font-size', this.scale + (this.usePxUnits ? 'px' : 'pt'));
        } else {
            (this.fontSelectorOrEditor as IEditor).updateOptions({
                fontSize: this.scale,
            });
        }
    }

    addState(state: any) {
        state.fontScale = this.scale;
        state.fontUsePx = true;
    }

    applyScale(scale: number) {
        this.scale = scale;
        this.apply();
    }

    setScale(scale: number) {
        this.fontSizeList.find('.active').removeClass('active');
        this.fontSizeList.find(`[data-value=${scale}]`).addClass('active');
        this.applyScale(scale);
    }

    setTarget(target: JQuery | string | IEditor) {
        this.fontSelectorOrEditor = target;
        this.isFontOfStr = typeof this.fontSelectorOrEditor === 'string';
        this.apply();
    }
}
