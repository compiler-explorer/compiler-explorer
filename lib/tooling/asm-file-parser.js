#!/usr/bin/env node
'use strict';

const getStdin = require('get-stdin');
const parser = new (require('../asm-parser'))();
const compiler = require('../base-compiler');

getStdin(str => {
    for (let line of parser.processAsm(str, compiler.getDefaultFilters())) {
        // eslint-disable-next-line no-console
        console.log(line.text);
    }
}, {});
