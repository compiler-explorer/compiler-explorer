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

// This is a script for generating screenshots for the site templates configured in etc/config/site-templates.yaml
//
// It should be run in the etc/scripts/ directory, because it uses relative paths
// It requires puppeteer which is not installed by default. You can install it locally with:
//   `npm i --no-save puppeteer`
// Run with `node --import=tsx generate_site_template_screenshots.ts`

import * as fsp from 'node:fs/promises';
import * as fss from 'node:fs';

import * as puppeteer from 'puppeteer';
import * as yaml from 'yaml';

const godbolt = 'https://godbolt.org';
const output_dir = '../../views/resources/template_screenshots';
const config = '../config/site-templates.yaml';

// Note: Hardcoded, may need to be updated in the future
// array of pairs [theme, colourScheme]
const themes = [
    ['default', 'rainbow'],
    ['dark', 'gray-shade'],
    ['darkplus', 'gray-shade'],
    ['pink', 'pink'],
    ['real-dark', 'gray-shade'],
    ['onedark', 'gray-shade'],
];

const defaultSettings = {
    showMinimap: true,
    wordWrap: false,
    colourScheme: 'gray-shade',
    theme: 'dark',
    defaultFontScale: 14,
};

// utilities
function sleep(ms: number) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function promisePoolExecutor(jobs: (() => void)[], max_concurrency: number) {
    async function worker(iterator: IterableIterator<[number, () => void]>) {
        for (const [_, job] of iterator) {
            await job();
        }
    }

    const iterator = jobs.entries();
    const workers = new Array(max_concurrency).fill(iterator).map(worker);
    await Promise.allSettled(workers);
}

// end utils

async function generateScreenshot(url: string, output_path: string, settings, width: number, height: number) {
    const browser = await puppeteer.launch({
        dumpio: true,
        defaultViewport: {
            width,
            height,
        },
    });
    const page = await browser.newPage();
    await page.goto(godbolt);
    await page.evaluate(settings => {
        localStorage.setItem('settings', JSON.stringify(settings));
    }, settings);
    await page.goto(url);
    //await sleep(2000);
    //await page.click(".modal.show button.btn.btn-outline-primary[data-dismiss=modal]");
    //await sleep(5000);
    //await page.click("#simplecook .btn.btn-primary.btn-sm.cook-do-consent");
    await page.evaluate(() => {
        for (let element of document.querySelectorAll('.float-link.link')) {
            element.parentNode!.removeChild(element);
        }
    });
    await sleep(10000); // wait for things to settle
    await page.screenshot({path: output_path});

    await browser.close();
}

const {meta, templates} = yaml.parse(await fsp.readFile(config, 'utf-8'));
const names = templates.map((template) => template.name);

// Quickly check there are no name conflicts
if (names.length !== new Set(names).size) {
    console.log('Error: Conflicting cleaned names');
    process.exit(1);
}
const width = parseInt(meta.screenshot_dimensions.width);
const height = parseInt(meta.screenshot_dimensions.height);

if (!fss.existsSync(output_dir)) {
    await fsp.mkdir(output_dir, {recursive: true});
}
const jobs: (() => void)[] = [];
for (const {name, reference} of templates) {
    for (const [theme, colourScheme] of themes) {
        const path = `${output_dir}/${name}.${theme}.png`;
        if (!fss.existsSync(path)) {
            jobs.push(() => generateScreenshot(
                `${godbolt}/e#${reference}`,
                path,
                Object.assign(Object.assign({}, defaultSettings), {theme, colourScheme}),
                width,
                height,
            ));
        }
    }
}
// don't launch too many chrome instances concurrently
await promisePoolExecutor(jobs, 4);
