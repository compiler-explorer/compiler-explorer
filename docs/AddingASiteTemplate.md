# Adding a new site template

Site templates are configured in [`etc/config/site-templates.conf`](../etc/config/site-templates.conf).

The configuration format is `Template Name=Godbolt.org full link`.

To create a site template:

- Setup the template on [https://godbolt.org](https://godbolt.org)
- Export with a full link from the "Share" dropdown in the top-right corner of the page
- Add to the config file

## Screenshot generation

Below are instructions to generate site templates. Because it's a bit hacky feel free to not run the script, we can do
so when a PR is made.

Site template screenshots are generated with a hacky script located at
[`etc/scripts/generate_site_template_screenshots.js`](../etc/scripts/generate_site_template_screenshots.js).

To run the script, `cd` to the `etc/scripts/` directory and run

```bash
npm i puppeteer --no-save && node generate_site_template_screenshots.js
```

The script uses puppeteer and chrome to generate screenshots. The script will take a little while to run as it generates
multiple screenshots per template and gives pages ample time to load.

Screenshots are located in [`views/resources/template_screenshots/`](../views/resources/template_screenshots/). The
script won't regenerate everything by default, to regenerate delete the screenshot images you want deleted.
