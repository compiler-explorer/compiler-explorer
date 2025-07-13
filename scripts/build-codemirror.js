#!/usr/bin/env node

// Script to bundle CodeMirror 6 for use in Compiler Explorer
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.join(__dirname, '..');
const staticDir = path.join(projectRoot, 'static', 'codemirror');

console.log('Building CodeMirror 6 bundles...');

// Create the static directory if it doesn't exist
await fs.mkdir(staticDir, { recursive: true });

// Create a simple bundle that re-exports CodeMirror modules
const bundleContent = `
// CodeMirror 6 bundle for Compiler Explorer
export { EditorView, basicSetup } from '../../../node_modules/codemirror/dist/index.js';
`;

const stateBundle = `
// CodeMirror state bundle for Compiler Explorer  
export { EditorState } from '../../../node_modules/@codemirror/state/dist/index.js';
`;

await fs.writeFile(path.join(staticDir, 'codemirror.bundle.js'), bundleContent);
await fs.writeFile(path.join(staticDir, 'state.bundle.js'), stateBundle);

console.log('CodeMirror 6 bundles created successfully!');
console.log('Files created:');
console.log('- static/codemirror/codemirror.bundle.js');
console.log('- static/codemirror/state.bundle.js');