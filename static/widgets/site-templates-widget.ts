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

import {SiteTemplatesType, UserSiteTemplate} from '../../types/features/site-templates.interfaces.js';
import {assert, unwrap, unwrapString} from '../assert.js';
import {Settings} from '../settings.js';
import * as local from '../local.js';
import * as url from '../url.js';
import GoldenLayout from 'golden-layout';
import {Alert} from './alert.js';
import {escapeHTML} from '../../shared/common-utils.js';

class SiteTemplatesWidget {
    private readonly modal: JQuery;
    private readonly img: HTMLImageElement;
    private readonly siteTemplateScreenshots: any;
    private readonly alertSystem: Alert;
    private templatesConfig: null | SiteTemplatesType = null;
    private populated = false;
    constructor(siteTemplateScreenshots: any, private readonly layout: GoldenLayout) {
        this.siteTemplateScreenshots = siteTemplateScreenshots;
        this.modal = $('#site-template-loader');
        const siteTemplatePreview = document.getElementById('site-template-preview');
        if (siteTemplatePreview === null) {
            // This can happen in embed mode
            return;
        }
        assert(siteTemplatePreview instanceof HTMLImageElement);
        this.img = siteTemplatePreview;
        this.alertSystem = new Alert();
        this.modal.find('#add-user-template').on('click', this.saveCurrentAsTemplate.bind(this));
    }
    saveCurrentAsTemplate() {
        const config = this.layout.toConfig();
        const data = url.serialiseState(config);
        this.alertSystem.enterSomething('Template Name', '', '', {
            yes: name => {
                const userTemplates: Record<string, UserSiteTemplate> = JSON.parse(
                    local.get('userSiteTemplates', '{}'),
                );
                let timestamp = Date.now();
                while (`t${timestamp}` in userTemplates) timestamp++;
                userTemplates[`t${timestamp}`] = {
                    title: unwrapString(name),
                    data,
                };
                local.set('userSiteTemplates', JSON.stringify(userTemplates));
                this.populateUserTemplates();
            },
        });
    }
    async getTemplates() {
        if (this.templatesConfig === null) {
            this.templatesConfig = await new Promise<SiteTemplatesType>((resolve, reject) => {
                $.getJSON(window.location.origin + window.httpRoot + 'api/siteTemplates', resolve);
            });
        }
        return this.templatesConfig;
    }
    getCurrentTheme() {
        const theme = Settings.getStoredSettings()['theme'];
        if (!theme) {
            // apparently this can happen
            return 'default';
        } else if (theme === 'system') {
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
    populateUserTemplates() {
        const userTemplates: Record<string, UserSiteTemplate> = JSON.parse(local.get('userSiteTemplates', '{}'));
        const userTemplatesList = $('#site-user-templates-list');
        userTemplatesList.empty();
        if (Object.entries(userTemplates).length === 0) {
            userTemplatesList.append(`<span>Nothing here yet</span>`);
        } else {
            for (const [id, {title, data}] of Object.entries(userTemplates)) {
                const li = $(`<li></li>`);
                $(`<div class="title">${escapeHTML(title)}</div>`)
                    .attr('data-data', data)
                    .appendTo(li);
                $(`<div class="delete" data-id="${id}"><i class="fa-solid fa-trash"></i></div>`).appendTo(li);
                li.appendTo(userTemplatesList);
            }
            userTemplatesList.find('li .delete').on('click', e => {
                const userTemplates: Record<string, UserSiteTemplate> = JSON.parse(
                    local.get('userSiteTemplates', '{}'),
                );
                delete userTemplates[unwrap($(e.target).parent('.delete').attr('data-id'))];
                local.set('userSiteTemplates', JSON.stringify(userTemplates));
                this.populate();
            });
        }
    }
    async populateSiteTemplates() {
        const templatesConfig = await this.getTemplates();
        const siteTemplatesList = $('#site-templates-list');
        siteTemplatesList.empty();
        for (const [name, data] of Object.entries(templatesConfig.templates)) {
            // Note: Trusting the server-provided data attribute
            siteTemplatesList.append(
                `<li>` +
                    `<div class="title" data-data="${data}" data-name="${name.replace(/[^a-z]/gi, '')}">${escapeHTML(
                        name,
                    )}</div>` +
                    `</li>`,
            );
        }
        for (const titleDiv of $('#site-user-templates-list li .title, #site-templates-list li .title')) {
            const titleDivCopy = titleDiv;
            titleDiv.addEventListener(
                'mouseover',
                () => {
                    const name = titleDivCopy.getAttribute('data-name');
                    this.img.src = '';
                    if (name) {
                        this.img.src = this.getAsset(name);
                    } else {
                        this.img.src =
                            // eslint-disable-next-line max-len
                            'https://placehold.jp/30/4b4b4b/ffffff/1000x800.png?text=we%27ll+support+screenshot+generation+for+user+templates+some+day';
                    }
                },
                false,
            );
            titleDiv.addEventListener(
                'click',
                () => {
                    window.location.href =
                        window.location.origin + window.httpRoot + '#' + titleDivCopy.getAttribute('data-data');
                },
                false,
            );
        }
    }
    async populate() {
        this.populateUserTemplates();
        await this.populateSiteTemplates();
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

export function setupSiteTemplateWidgetButton(siteTemplateScreenshots: any, layout: GoldenLayout) {
    const siteTemplateModal = new SiteTemplatesWidget(siteTemplateScreenshots, layout);
    $('#loadSiteTemplate').on('click', () => {
        siteTemplateModal.show();
    });
}
