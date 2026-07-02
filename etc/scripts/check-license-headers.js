#!/usr/bin/env node
// Check that every source file starts with the BSD-2-Clause license banner.
//
// We used to enforce this via eslint-plugin-header; that check was lost in the
// migration to Biome (which has no equivalent rule). This is a standalone
// replacement, run in CI, on pre-commit (via lint-staged) and by `make pre-commit`.
//
// Usage:
//   node ./etc/scripts/check-license-headers.js            # scan the tracked tree
//   node ./etc/scripts/check-license-headers.js <files...>  # scan just these files (lint-staged)

import {execSync} from 'node:child_process';
import fs from 'node:fs';

// Directories/globs whose source files must carry the banner.
const INCLUDE_DIRS = ['lib', 'static', 'shared', 'types', 'test', 'cypress'];
const INCLUDE_EXTENSIONS = ['.ts', '.js', '.mjs', '.cjs'];

// Files/paths that are exempt (generated, vendored, or third-party).
const EXCLUDE_PATTERNS = [
    /(^|\/)node_modules\//,
    /(^|\/)out\//,
    /^lib\/asm-docs\/generated\//,
    /^etc\/scripts\/docenizer\/vendor\//,
    /\.d\.ts$/,
    // Third-party code carrying its own upstream license (not CE's BSD-2 banner).
    /^static\/ansi-to-html\.ts$/, // MIT, Rob Burns
    /^lib\/node-graceful\.ts$/, // MIT, node-graceful
    /^shared\/rison\.ts$/, // Nanonid/rison port
];

// A file "has an appropriate banner" if, ignoring an optional shebang, it opens
// with a `// Copyright (c) ...` line and contains the BSD-2-Clause disclaimer.
// The year and copyright holder are intentionally not constrained: the tree has
// many holders (Compiler Explorer Authors, Arm Ltd, Microsoft, individuals, ...).
const COPYRIGHT_RE = /^\/\/ Copyright \([cC]\) .+/m;
const DISCLAIMER = 'THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"';

function isIncluded(file) {
    if (!INCLUDE_EXTENSIONS.some(ext => file.endsWith(ext))) return false;
    if (EXCLUDE_PATTERNS.some(re => re.test(file))) return false;
    return INCLUDE_DIRS.some(dir => file === dir || file.startsWith(`${dir}/`));
}

function listTrackedFiles() {
    return execSync('git ls-files', {encoding: 'utf8', maxBuffer: 64 * 1024 * 1024})
        .split('\n')
        .filter(Boolean);
}

function hasBanner(file) {
    // Only the head of the file matters; read enough to cover the banner.
    let head;
    try {
        const buf = Buffer.alloc(4096);
        const fd = fs.openSync(file, 'r');
        const bytesRead = fs.readSync(fd, buf, 0, buf.length, 0);
        fs.closeSync(fd);
        head = buf.toString('utf8', 0, bytesRead);
    } catch {
        return true; // Unreadable/deleted file: not our problem to report here.
    }
    // Skip an optional shebang line so scripts like this one still qualify.
    const body = head.startsWith('#!') ? head.slice(head.indexOf('\n') + 1) : head;
    return COPYRIGHT_RE.test(body) && body.includes(DISCLAIMER);
}

const args = process.argv.slice(2);
const candidates = (args.length > 0 ? args : listTrackedFiles()).filter(isIncluded);
const missing = candidates.filter(file => fs.existsSync(file) && !hasBanner(file));

if (missing.length > 0) {
    console.error(`❌ ${missing.length} file(s) are missing the license header banner:\n`);
    for (const file of missing) console.error(`  ${file}`);
    console.error('\nEvery source file must begin with the BSD-2-Clause banner (see any existing file for the');
    console.error('exact text). The copyright year/holder line is flexible; the disclaimer body is required.');
    process.exit(1);
}

process.exit(0);
