/**
 * Simple webpack loader to replace @import statements for golden-layout with actual CSS content
 * This runs before sass-loader to replace the imports with raw CSS content
 */

import fs from 'fs';
import path from 'path';
import {fileURLToPath} from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default function replaceGoldenLayoutImports(source) {
    // Only process if it contains golden-layout imports
    if (!source.includes('@import') || !source.includes('golden-layout')) {
        return source;
    }

    // Replace any golden-layout CSS import with actual content
    const goldenLayoutImportRegex = /@import\s+['"]~golden-layout\/src\/css\/([^'"]+)['"];?\s*/g;
    
    return source.replace(goldenLayoutImportRegex, (match, cssFileName) => {
        // Ensure .css extension
        const fileName = cssFileName.endsWith('.css') ? cssFileName : `${cssFileName}.css`;
        const cssContent = readGoldenLayoutCSS(fileName);
        const themeName = fileName.replace(/^goldenlayout-/, '').replace(/\.css$/, '');
        
        return `/* Golden Layout ${themeName} - Inlined */\n${cssContent}\n/* End Golden Layout ${themeName} */`;
    });
};

function readGoldenLayoutCSS(filename) {
    // Use import.meta.resolve to find the golden-layout package location robustly
    const packageJsonPath = import.meta.resolve('golden-layout/package.json');
    const packageDir = path.dirname(fileURLToPath(packageJsonPath));
    const cssPath = path.join(packageDir, 'src', 'css', filename);
    
    if (!fs.existsSync(cssPath)) {
        throw new Error(`Golden Layout CSS file not found: ${cssPath}`);
    }
    
    return fs.readFileSync(cssPath, 'utf8');
}
