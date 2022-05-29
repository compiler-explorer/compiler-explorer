import * as fs from 'fs';

let siteTemplates: Record<string, string> = {};

function splitProperty(line: string) {
    return [line.substring(0, line.indexOf('=')), line.substring(line.indexOf('=') + 1)];
}

export function loadSiteTemplates(configDir: string) {
    siteTemplates = Object.fromEntries(
        fs
            .readFileSync(configDir + '/site-templates.properties', 'utf-8')
            .split('\n')
            .filter(v => v.length > 0)
            .map(splitProperty),
    );
}

export function getSiteTemplates() {
    return siteTemplates;
}
