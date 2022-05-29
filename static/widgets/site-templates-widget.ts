class SiteTemplatesWidget {
    modal: JQuery;
    iframe: HTMLIFrameElement;
    templates: null | Record<string, string>[] = null;
    populated = false;
    constructor() {
        this.modal = $('#site-template-loader');
        this.iframe = document.getElementById('site-template-preview') as HTMLIFrameElement;
    }
    async getTemplates() {
        if (this.templates === null) {
            this.templates = await new Promise((resolve, reject) => {
                $.getJSON(window.location.origin + window.httpRoot + 'api/getSiteTemplates', resolve);
            });
        }
        return this.templates as Record<string, string>[];
    }
    async populate() {
        const templates = await this.getTemplates();
        const root = $('#site-templates-list');
        root.empty();
        for (const [name, data] of Object.entries(templates)) {
            root.append(`<li data-data="${data}">${name}</li>`);
        }
        for (const li of root.find('li')) {
            const li_copy = li;
            li.addEventListener(
                'mouseover',
                () => {
                    console.log(window.location.origin + window.httpRoot + '#' + li_copy.getAttribute('data-data'));
                    this.iframe.src =
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
