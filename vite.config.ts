import {defineConfig} from 'vite';
import {vitePluginPug} from './vite-plugin-pug.js';

export default defineConfig({
    plugins: [vitePluginPug()],
    build: {
        manifest: true,
        rollupOptions: {
            input: './static/main.ts',
            output: {
                globals: {
                    'jQuery': 'window.jQuery',
                    'jquery': 'window.jQuery',
                    '$': 'window.jQuery',
                }
            }
        },
        commonjsOptions: {
            include: [/pug-runtime/],
        },
    },
    optimizeDeps: {
        include: ['pug-runtime', 'jquery'],
    },
});
