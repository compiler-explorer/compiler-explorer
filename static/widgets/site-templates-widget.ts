// Copyright (c) 2022, Compiler Explorer Authors
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
import {SiteTemplatesType} from '../../types/features/site-templates.interfaces';
import {Settings} from '../settings';

class SiteTemplatesWidget {
    modal: JQuery;
    img: HTMLImageElement;
    templatesConfig: null | SiteTemplatesType = null;
    populated = false;
    siteTemplateScreenshots: any;
    constructor(siteTemplateScreenshots: any) {
        this.siteTemplateScreenshots = siteTemplateScreenshots;
        this.modal = $('#site-template-loader');
        this.img = document.getElementById('site-template-preview') as HTMLImageElement;
    }
    async getTemplates() {
        if (this.templatesConfig === null) {
            this.templatesConfig = await new Promise((resolve, reject) => {
                $.getJSON(window.location.origin + window.httpRoot + 'api/siteTemplates', resolve);
            });
        }
        return this.templatesConfig as SiteTemplatesType;
    }
    getCurrentTheme() {
        const theme = Settings.getStoredSettings()['theme'];
        if (theme === 'system') {
            if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
                return 'dark';
            } else {
                return 'default';
            }
        } else {
            return theme;
        }
    }
    getAsset(name: string) {
        return this.siteTemplateScreenshots(`./${name}.${this.getCurrentTheme()}.png`);
    }
    async setDefaultPreview() {
        const templatesConfig = await this.getTemplates(); // by the time this is called it will be cached
        const first = Object.entries(templatesConfig.templates)[0][0]; // preview the first entry
        this.img.src = this.getAsset(first.replace(/[^a-z]/gi, ''));
    }
    async populate() {
        const templatesConfig = await this.getTemplates();
        const root = $('#site-templates-list');
        root.empty();
        for (const [name, data] of Object.entries(templatesConfig.templates)) {
            root.append(`<li data-data="${data}" data-name="${name.replace(/[^a-z]/gi, '')}">${name}</li>`);
        }
        for (const li of root.find('li')) {
            const li_copy = li;
            li.addEventListener(
                'mouseover',
                () => {
                    this.img.src = this.getAsset(li_copy.getAttribute('data-name') as string);
                },
                false
            );
            li.addEventListener(
                'click',
                () => {
                    window.location.href =
                        window.location.origin + window.httpRoot + '#' + li_copy.getAttribute('data-data');
                },
                false
            );
        }
        this.populated = true;
    }
    show() {
        this.modal.modal('show');
        if (!this.populated) {
            this.populate();
        }
        this.setDefaultPreview();
    }
}

export function setupSiteTemplateWidgetButton(siteTemplateScreenshots: any) {
    const siteTemplateModal = new SiteTemplatesWidget(siteTemplateScreenshots);
    $('#loadSiteTemplate').on('click', () => {
        siteTemplateModal.show();
    });
}
