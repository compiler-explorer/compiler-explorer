#!/usr/bin/env node

// Copyright (c) 2025, Compiler Explorer Authors
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

/**
 * Import Path Validator
 *
 * Validates that all import statements in frontend TypeScript files point to existing files.
 * This prevents broken imports from being introduced and catches existing issues.
 */

import {constants} from 'node:fs';
import {access, readFile, readdir} from 'node:fs/promises';
import {dirname, extname, join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const rootDir = resolve(__dirname, '..');

// Extensions to check for when resolving imports
const EXTENSIONS = ['.ts', '.js', '.tsx', '.jsx'];

// Import patterns to match
const IMPORT_PATTERN = /^(?:import|export).*?from\s+['"`]([^'"`]+)['"`]/gm;

/**
 * Recursively find all TypeScript files in a directory
 */
async function findTsFiles(dir) {
    const files = [];
    const entries = await readdir(dir, {withFileTypes: true});

    for (const entry of entries) {
        const fullPath = join(dir, entry.name);

        if (entry.isDirectory()) {
            // Skip node_modules, .git, and other common directories
            if (!['node_modules', '.git', 'out', 'dist', 'build'].includes(entry.name)) {
                files.push(...(await findTsFiles(fullPath)));
            }
        } else if (entry.isFile() && (entry.name.endsWith('.ts') || entry.name.endsWith('.tsx'))) {
            files.push(fullPath);
        }
    }

    return files;
}

/**
 * Extract import paths from TypeScript file content
 */
function extractImports(content) {
    const imports = [];
    let match;

    while ((match = IMPORT_PATTERN.exec(content)) !== null) {
        const importPath = match[1];
        // Skip node_modules imports (don't start with . or /)
        if (importPath.startsWith('.') || importPath.startsWith('/')) {
            imports.push({
                path: importPath,
                line: content.substring(0, match.index).split('\n').length,
            });
        }
    }

    return imports;
}

/**
 * Resolve an import path to an actual file path
 * Handles TypeScript's ".js" import convention where .js imports resolve to .ts files
 */
async function resolveImport(importPath, fromFile) {
    const fromDir = dirname(fromFile);
    const targetPath = resolve(fromDir, importPath);

    // Try the exact path first
    try {
        await access(targetPath, constants.F_OK);
        return targetPath;
    } catch {
        // If exact path doesn't exist, try with extensions
    }

    // Handle TypeScript's ".js" import convention - .js imports should resolve to .ts files
    if (targetPath.endsWith('.js')) {
        const tsPath = targetPath.replace(/\.js$/, '.ts');
        try {
            await access(tsPath, constants.F_OK);
            return tsPath;
        } catch {
            // Try .tsx as well
            const tsxPath = targetPath.replace(/\.js$/, '.tsx');
            try {
                await access(tsxPath, constants.F_OK);
                return tsxPath;
            } catch {
                // Continue with other resolution attempts
            }
        }
    }

    // If the import doesn't have an extension, try adding common extensions
    if (!extname(targetPath)) {
        for (const ext of EXTENSIONS) {
            const pathWithExt = targetPath + ext;
            try {
                await access(pathWithExt, constants.F_OK);
                return pathWithExt;
            } catch {
                // Continue trying other extensions
            }
        }
    }

    // Try index files if the path is a directory
    for (const ext of EXTENSIONS) {
        try {
            const indexPath = join(targetPath, `index${ext}`);
            await access(indexPath, constants.F_OK);
            return indexPath;
        } catch {
            // Continue trying other extensions
        }
    }

    return null; // Could not resolve
}

/**
 * Check imports in a single file
 */
async function checkFileImports(filePath) {
    const content = await readFile(filePath, 'utf-8');
    const imports = extractImports(content);
    const brokenImports = [];

    for (const importInfo of imports) {
        const resolved = await resolveImport(importInfo.path, filePath);
        if (!resolved) {
            brokenImports.push({
                file: filePath,
                line: importInfo.line,
                import: importInfo.path,
                resolvedPath: null,
            });
        }
    }

    return brokenImports;
}

/**
 * Main function to check all imports
 */
async function checkAllImports() {
    const staticDir = join(rootDir, 'static');
    const tsFiles = await findTsFiles(staticDir);

    console.log(`Checking imports in ${tsFiles.length} TypeScript files...`);

    const allBrokenImports = [];
    let checkedFiles = 0;

    for (const file of tsFiles) {
        try {
            const brokenImports = await checkFileImports(file);
            allBrokenImports.push(...brokenImports);
            checkedFiles++;

            if (checkedFiles % 50 === 0) {
                console.log(`Checked ${checkedFiles}/${tsFiles.length} files...`);
            }
        } catch (error) {
            console.error(`Error checking ${file}: ${error.message}`);
        }
    }

    // Report results
    if (allBrokenImports.length === 0) {
        console.log('✅ All imports are valid!');
        return 0;
    }

    console.log(`\n❌ Found ${allBrokenImports.length} broken imports:\n`);

    // Group by file for better readability
    const byFile = new Map();
    for (const broken of allBrokenImports) {
        const relativePath = broken.file.replace(rootDir + '/', '');
        if (!byFile.has(relativePath)) {
            byFile.set(relativePath, []);
        }
        byFile.get(relativePath).push(broken);
    }

    for (const [file, imports] of byFile.entries()) {
        console.log(`📄 ${file}:`);
        for (const imp of imports) {
            console.log(`  Line ${imp.line}: import from '${imp.import}'`);
        }
        console.log();
    }

    console.log('\nTo fix these imports, ensure the target files exist or update the import paths.');
    console.log('Common fixes:');
    console.log("  - './compilation/...' → '../types/compilation/...'");
    console.log("  - './execution/...' → '../types/execution/...'");
    console.log("  - './languages.interfaces.js' → '../types/languages.interfaces.js'");

    return 1; // Exit with error code
}

// Run the check
checkAllImports()
    .then(exitCode => process.exit(exitCode))
    .catch(error => {
        console.error('Fatal error:', error);
        process.exit(1);
    });
