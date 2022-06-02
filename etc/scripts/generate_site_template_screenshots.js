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
