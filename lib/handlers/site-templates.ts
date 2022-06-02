import * as fs from 'fs';

import {siteTemplatesType} from '../../types/features/site-templates.interfaces';

const siteTemplates: siteTemplatesType = {
    meta: {},
    templates: {},
};

function splitProperty(line: string) {
    return [line.substring(0, line.indexOf('=')), line.substring(line.indexOf('=') + 1)];
}

function partition<T>(array: T[], filter: (value: T) => boolean): [T[], T[]] {
    const pass: T[] = [],
        fail: T[] = [];
    for (const item of array) {
        if (filter(item)) {
            pass.push(item);
        } else {
            fail.push(item);
        }
    }
    return [pass, fail];
}

export function loadSiteTemplates(configDir: string) {
    const [meta, templates] = partition(
        fs
            .readFileSync(configDir + '/site-templates.properties', 'utf-8')
            .split('\n')
            .filter(l => l !== '')
            .map(splitProperty),
        ([name, _]) => name.startsWith('meta.'),
    );
    siteTemplates.meta = Object.fromEntries(meta);
    siteTemplates.templates = Object.fromEntries(templates);
}

export function getSiteTemplates() {
    return siteTemplates;
}
