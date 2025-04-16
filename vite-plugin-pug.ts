import { type Plugin} from 'vite';
import * as fs from 'node:fs/promises';
import { compile } from "pug"

export function vitePluginPug(): Plugin {
    return {
        name: "vite-plugin-pug",
        transform: async (src, id) => {
            if (!id.endsWith(".pug")) {
                return null;
            }
            const sourceCode = await fs.readFile(id, 'utf-8')
            const fun = compile(sourceCode, {
                filename: id,
                compileDebug: false,
                inlineRuntimeFunctions: false,
                pretty: true,
            });
            const code =  `import pug from "pug-runtime"\n\nexport default ${fun.toString()}`
            return {code}
        }
    }
}
