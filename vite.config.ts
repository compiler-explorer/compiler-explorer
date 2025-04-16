import {defineConfig} from 'vite';
import {vitePluginPug} from './vite-plugin-pug.js';
import inject from "@rollup/plugin-inject"

export default defineConfig({
    plugins: [vitePluginPug(), inject({
        $: ['jquery', '*'],
        jQuery: 'jquery'
    })],
    build: {
        manifest: true,
        rollupOptions: {
            input: './static/main.ts',
        },
        commonjsOptions: {
            include: [/pug-runtime/],
        },
    },
    optimizeDeps: {
        include: ['pug-runtime']
    },
});
