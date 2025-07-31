/**
 * Simple webpack loader to replace @import statements for golden-layout with actual CSS content
 * This runs before sass-loader to replace the imports with raw CSS content
 */

const fs = require('fs');
const path = require('path');

module.exports = function replaceGoldenLayoutImports(source) {
    // Only process if it contains golden-layout imports
    if (!source.includes('@import') || !source.includes('golden-layout')) {
        return source;
    }

    // Replace the specific import statements
    let processedSource = source;
    
    // Replace light theme import
    const lightImportRegex = /@import\s+['"]~golden-layout\/src\/css\/goldenlayout-light-theme['"];?\s*/g;
    if (lightImportRegex.test(source)) {
        const lightCSS = readGoldenLayoutCSS('goldenlayout-light-theme.css');
        processedSource = processedSource.replace(lightImportRegex, `/* Golden Layout Light Theme - Inlined */\n${lightCSS}\n/* End Golden Layout Light Theme */`);
    }

    // Replace dark theme import  
    const darkImportRegex = /@import\s+['"]~golden-layout\/src\/css\/goldenlayout-dark-theme['"];?\s*/g;
    if (darkImportRegex.test(source)) {
        const darkCSS = readGoldenLayoutCSS('goldenlayout-dark-theme.css');
        processedSource = processedSource.replace(darkImportRegex, `/* Golden Layout Dark Theme - Inlined */\n${darkCSS}\n/* End Golden Layout Dark Theme */`);
    }

    return processedSource;
};

function readGoldenLayoutCSS(filename) {
    const cssPath = path.resolve(__dirname, '..', '..', 'node_modules', 'golden-layout', 'src', 'css', filename);
    
    if (!fs.existsSync(cssPath)) {
        throw new Error(`Golden Layout CSS file not found: ${cssPath}`);
    }
    
    return fs.readFileSync(cssPath, 'utf8');
}