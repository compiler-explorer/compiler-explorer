# Adding a new site template

Site templates are configured in [`etc/config/site-templates.yaml`](../etc/config/site-templates.yaml).

The configuration is a list of site templates each having a name and a raw configuration string. Use the following steps
to create the configuration string:

1. [Compiler Explorer](https://godbolt.org) and configure the setup you want to export as a template
2. Click the "Share" dropdown in the top-right corner of the page
3. Select the "Copy full link" option
4. Strip the "https://godbolt.org/#" prefix from the link
5. Paste the configuration string into your new entry in the templates list.

After you have created a new site template, you can generate screenshots using Puppeteer.

## Screenshot generation

Below are instructions to generate site templates. Because it's a bit hacky feel free to not run the script, we can do
so when a PR is made.

Site template screenshots are generated with a hacky script located at
[`etc/scripts/generate_site_template_screenshots.ts`](../etc/scripts/generate_site_template_screenshots.ts).

To run the script, `cd` to the `etc/scripts/` directory and run

```bash
npm i puppeteer --no-save && npx node --no-warnings=ExperimentalWarning --import=tsx generate_site_template_screenshots.ts
```

The script uses puppeteer and chrome to generate screenshots. The script will take a little while to run as it generates
multiple screenshots per template and gives pages ample time to load.

Screenshots are located in [`views/resources/template_screenshots/`](../views/resources/template_screenshots/). The
script won't regenerate everything by default, to regenerate delete the screenshot images you want deleted.
