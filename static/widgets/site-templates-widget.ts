import {siteTemplatesType} from '../../types/features/site-templates.interfaces';

class SiteTemplatesWidget {
    modal: JQuery;
    img: HTMLImageElement;
    templatesConfig: null | siteTemplatesType = null;
    populated = false;
    constructor() {
        this.modal = $('#site-template-loader');
        this.img = document.getElementById('site-template-preview') as HTMLImageElement;
    }
    async getTemplates() {
        if (this.templatesConfig === null) {
            this.templatesConfig = await new Promise((resolve, reject) => {
                $.getJSON(window.location.origin + window.httpRoot + 'api/getSiteTemplates', resolve);
            });
        }
        return this.templatesConfig as siteTemplatesType;
    }
    async populate() {
        const templatesConfig = await this.getTemplates();
        const root = $('#site-templates-list');
        root.empty();
        for (const [name, data] of Object.entries(templatesConfig.templates)) {
            root.append(`<li data-data="${data}" data-name="${name.replace(/[^a-z]/gi, '')}">${name}</li>`);
        }
        for (const [k, v] of Object.entries(templatesConfig.meta)) {
            if (k === 'meta.screenshot_dimentions') {
                const [w, h] = v.split('x').map(x => parseInt(x));
                this.img.width = w;
                this.img.height = h;
            }
        }
        for (const li of root.find('li')) {
            const li_copy = li;
            li.addEventListener(
                'mouseover',
                () => {
                    this.img.src =
                        window.location.origin +
                        window.httpRoot +
                        `template_screenshots/${li_copy.getAttribute('data-name')}.png`;
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
    }
}

const siteTemplateModal = new SiteTemplatesWidget();

export function setupSiteTemplateWidgetButton() {
    $('#loadSiteTemplate').on('click', () => {
        siteTemplateModal.show();
    });
}
