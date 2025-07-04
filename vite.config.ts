import * as fss from 'node:fs';
import * as path from 'node:path';
import * as url from 'node:url';
import {defineConfig} from 'vite';
import {vitePluginHashedPug} from './vite-plugin-hashed-pug.js';

const pwd = url.fileURLToPath(new URL('.', import.meta.url));
const hasGit = fss.existsSync(path.join(pwd, '.git'));

export default defineConfig({
    plugins: [vitePluginHashedPug({useGit: hasGit})],
    build: {
        manifest: true,
        rollupOptions: {
            input: {
                'main.js': url.fileURLToPath(new URL('./static/main.ts', import.meta.url)),
                'noscript.js': url.fileURLToPath(new URL('./static/noscript.ts', import.meta.url)),
            },
        },
        commonjsOptions: {
            defaultIsModuleExports: true,
            include: [
                /golden-layout/,
                /jquery/,
                /clipboard/,
                /lodash\.clonedeep/,
                /big-integer/,
                /events/,
                /buffer/,
                /file-saver/,
                /path-browserify/,
                /jszip/,
                /lz-string/,
                /monaco-vim/,
            ],
        },
    },
});
