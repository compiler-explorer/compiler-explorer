#!/usr/bin/env node
'use strict';

const parser = new (require('../../lib/asm-parser'))();
const compiler = require('../../lib/base-compiler');
const fs = require('fs');

const asm = fs.readFileSync(0, 'utf-8');
for (let line of parser.processAsm(asm, compiler.getDefaultFilters()).asm) {
    // eslint-disable-next-line no-console
    console.log(line.text);
}
