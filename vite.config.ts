import { defineConfig } from "vite"

export default defineConfig({
    build: {
        manifest: true,
        rollupOptions: {
            input: {
                "main.js": "./static/main.ts",
            }
        }
    },
})
