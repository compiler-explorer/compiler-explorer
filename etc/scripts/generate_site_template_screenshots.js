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

/*
 * Note:
 * - This script should be run in the etc/scripts/ directory, it uses relative paths
 * - This script requires puppeteer which is not installed by default (I install it globally)
 */

const puppeteer = require("puppeteer");
const fs = require("fs");

const godbolt    = "https://godbolt.org";
const output_dir = "../../views/resources/template_screenshots";
const config     = "../config/site-templates.conf";

const defaultViewport = {
    width: 500,
    height: 500
};

// Note: Hardcoded, may need to be updated in the future
// array of pairs [theme, colourScheme]
const themes = [
    ["default",  "rainbow"],
    ["dark",     "gray-shade"],
    ["darkplus", "gray-shade"],
];

const defaultSettings = {
    showMinimap: true,
    wordWrap: false,
    colourScheme: "gray-shade",
    theme: "dark",
    defaultFontScale: 14
};

// utilities
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function splitProperty(line) {
    return [line.substring(0, line.indexOf('=')), line.substring(line.indexOf('=') + 1)];
}

function partition(array, filter) {
    const pass = [], fail = [];
    for(const item of array) {
        if(filter(item)) {
            pass.push(item);
        } else {
            fail.push(item);
        }
    }
    return [pass, fail];
}

async function generate_screenshot(url, output_path, settings) {
    const browser = await puppeteer.launch({
        dumpio: true,
        defaultViewport
    });
    const page = await browser.newPage();
    await page.goto(godbolt);
    await page.evaluate(settings => {
        localStorage.setItem("settings", JSON.stringify(settings));
    }, settings);
    await page.goto(url);
    //await sleep(2000);
    //await page.click(".modal.show button.btn.btn-outline-primary[data-dismiss=modal]");
    //await sleep(5000);
    //await page.click("#simplecook .btn.btn-primary.btn-sm.cook-do-consent");
    await page.evaluate(() => {
        for(let element of document.querySelectorAll(".float-link.link")){
            element.parentNode.removeChild(element);
        }
    });
    await sleep(10000); // wait for things to settle
    await page.screenshot({ path: output_path });

    await browser.close();
}

(async () => {
    const [meta_directives, templates] =
        partition(
            fs
            .readFileSync(config, "utf-8")
            .split("\n")
            .filter(l => l !== "")
            .map(splitProperty)
            .map(([name, data]) => [!name.startsWith("meta.") ? name.replaceAll(/[^a-z]/gi, "") : name, data]),
            ([name, _]) => name.startsWith("meta.")
        );
    if(new Set(templates.map(([line, _]) => line)).size !== templates.length) { // quickly check there are no name conflicts
        console.log("Error: Conflicting cleaned names");
        process.exit(1);
    }
    for(const [k, v] of meta_directives) {
        if(k === "meta.screenshot_dimentions") {
            const [w, h] = v.split("x").map(x => parseInt(x));
            defaultViewport.width = w;
            defaultViewport.height = h;
        }
    }
    if(!fs.existsSync(output_dir)) {
        fs.mkdirSync(output_dir, { recursive: true });
    }
    const promises = [];
    for(const [name, data] of templates) {
        for(const [theme, colourScheme] of themes) {
            const path = `${output_dir}/${name}.${theme}.png`;
            if(!fs.existsSync(path)) {
                promises.push(generate_screenshot(
                    `${godbolt}/e#${data}`,
                    path,
                    Object.assign(Object.assign({}, defaultSettings), {theme, colourScheme})
                ));
            }
        }
    }
    await Promise.all(promises);
})();
