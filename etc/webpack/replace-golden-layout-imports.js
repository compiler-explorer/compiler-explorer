/**
 * Simple webpack loader to replace @import statements for golden-layout with actual CSS content
 * This runs before sass-loader to replace the imports with raw CSS content
 */

import fs from 'fs';
import path from 'path';
import {fileURLToPath} from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const GOLDEN_LAYOUT_PREFIX = '~golden-layout/';

export default function replaceGoldenLayoutImports(source) {
    // Only process if it contains golden-layout imports
    if (!source.includes('@import') || !source.includes('golden-layout')) {
        return source;
    }

    // Find all golden-layout @import statements and replace them
    const importRegex = /@import\s+['"]([^'"]+)['"];?\s*/g;
    
    return source.replace(importRegex, (match, importPath) => {
        if (importPath.startsWith(GOLDEN_LAYOUT_PREFIX)) {
            return replaceGoldenLayoutImport(importPath);
        }
        
        // Return other imports unchanged
        return match;
    });
};

function replaceGoldenLayoutImport(importPath) {
    const goldenLayoutPath = importPath.substring(GOLDEN_LAYOUT_PREFIX.length);
    const cssContent = readGoldenLayoutCSS(goldenLayoutPath);
    const fileName = goldenLayoutPath.split('/').pop();
    const themeName = fileName.replace(/^goldenlayout-/, '').replace(/\.css$/, '');
    
    return `/* Golden Layout ${themeName} - Inlined */\n${cssContent}\n/* End Golden Layout ${themeName} */`;
}

function readGoldenLayoutCSS(relativePath) {
    // Use import.meta.resolve to find the golden-layout package location robustly
    const packageJsonPath = import.meta.resolve('golden-layout/package.json');
    const packageDir = path.dirname(fileURLToPath(packageJsonPath));
    
    // Ensure .css extension if not present
    const finalPath = relativePath.endsWith('.css') ? relativePath : `${relativePath}.css`;
    const cssPath = path.join(packageDir, finalPath);
    
    if (!fs.existsSync(cssPath)) {
        throw new Error(`Golden Layout CSS file not found: ${cssPath}`);
    }
    
    return fs.readFileSync(cssPath, 'utf8');
}
