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

const puppeteer = require("puppeteer");
const fs = require("fs");

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

const defaultViewport = {
    width: 500,
    height: 500
};

const defaultSettings = {
    showMinimap: true,
    wordWrap: false,
    colourScheme: "gray-shade",
    theme: "dark",
    defaultFontScale: 14
};

async function generate_screenshot(url, output_path) {
    const browser = await puppeteer.launch({
        dumpio: true,
        defaultViewport
    });
    const page = await browser.newPage();
    await page.goto("https://godbolt.org");
    await page.evaluate(defaultSettings => {
        localStorage.setItem("settings", JSON.stringify(defaultSettings));
    }, defaultSettings);
    await page.goto(url);
    //await sleep(2000);
    //await page.click(".modal.show button.btn.btn-outline-primary[data-dismiss=modal]");
    await sleep(2000);
    await page.click("#simplecook .btn.btn-primary.btn-sm.cook-do-consent");
    await sleep(2000);
    await page.screenshot({ path: output_path });

    await browser.close();
}

(async () => {
    const [meta_directives, templates] =
        partition(
            fs
            .readFileSync("../config/site-templates.properties", "utf-8")
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
    if(!fs.existsSync("../template_screenshots")) {
        fs.mkdirSync("../template_screenshots", { recursive: true });
    }
    const promises = [];
    for(const [name, data] of templates) {
        const path = `../template_screenshots/${name}.png`;
        if(!fs.existsSync(path)) {
            promises.push(generate_screenshot(
                `https://godbolt.org/#${data}`,
                path
            ));
        }
    }
    await Promise.all(promises);
})();
