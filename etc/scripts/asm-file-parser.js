#!/usr/bin/env node
'use strict';

/*
 * Example usage:
 * `etc/scripts/asm-file-parser.js --no-intel < test/filters-cases/bug-1285c.asm`
 */

const parser = new (require('../../lib/asm-parser'))();
const compiler = require('../../lib/base-compiler');
const fs = require('fs');

const argv = require('minimist')(process.argv.slice(2));

// Polyfill for the coalesce operator (`??`)
function coalesce(arg, default_) {
    return arg !== undefined ? arg : default_;
}

const defaults = compiler.getDefaultFilters();
const filters = {
            intel: coalesce(argv.intel, defaults.intel),
            commentOnly: coalesce(argv.commentOnly, defaults.commentOnly),
            directives: coalesce(argv.directives, defaults.directives),
            labels: coalesce(argv.labels, defaults.labels),
            optOutput: coalesce(argv.optOutput, defaults.optOutput),
};

const asm = fs.readFileSync(0, 'utf-8');
for (let line of parser.processAsm(asm, filters).asm) {
    // eslint-disable-next-line no-console
    console.log(line.text);
}
