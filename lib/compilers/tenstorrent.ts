/**
 * Compiler Explorer Driver for Tenstorrent SFPI RISC-V Toolchain
 * Patched and production-optimized for tt-metal Issue #46364
 */

import * as fs from 'node:fs';
import * as path from 'node:path';

import {BaseCompiler} from '../base-compiler.js';

export class TenstorrentCompiler extends BaseCompiler {
    private ttMetalHome: string;
    private includePaths: string[];

    constructor(info: any, env: any) {
        super(info, env);

        // Detect tt-metal installation path dynamically
        this.ttMetalHome = process.env.TT_METAL_HOME || '/opt/tt-metal';

        // Build default include paths for Tenstorrent core headers
        this.includePaths = [
            path.join(this.ttMetalHome, 'include'),
            path.join(this.ttMetalHome, 'tt_metal/include'),
            path.join(this.ttMetalHome, 'tt_metal/include/compute_kernel_api'),
            path.join(this.ttMetalHome, 'tt_metal/include/hw'),
        ];
    }

    /**
     * Intercepts compilation filters if needed.
     */
    override optionsForFilter(filters: any, outputFilename: string): string[] {
        return [];
    }

    /**
     * Injects custom Tenstorrent headers and enforces assembly generation layout (-S).
     */
    override async runCompiler(
        compiler: string,
        options: string[],
        inputFilename: string,
        execOptions: any,
        filters?: any,
    ): Promise<any> {
        
        // Ensure options array copy
        const modifiedOptions = [...options];

        // Intercept and inject required structural headers
        for (const includePath of this.includePaths) {
            if (fs.existsSync(includePath)) {
                modifiedOptions.push(`-I${includePath}`);
            }
        }

        // Ensure default 32-bit architecture matching Tensix engine specs
        if (!modifiedOptions.some((o) => o.startsWith('-march'))) {
            modifiedOptions.push('-march=rv32imf');
        }
        if (!modifiedOptions.some((o) => o.startsWith('-O'))) {
            modifiedOptions.push('-O2');
        }

        // Enforce assembly output targets
        if (!modifiedOptions.includes('-S')) modifiedOptions.push('-S');
        if (!modifiedOptions.includes('-c')) modifiedOptions.push('-c');

        // Determine output filename dynamically from the file context configuration
        const outputFilename = this.getOutputFilename(path.dirname(inputFilename), '');

        // Delegate execution to base compiler context
        const result = await super.runCompiler(compiler, modifiedOptions, inputFilename, execOptions, filters);

        // Process assembly file from disk if compilation succeeded
        if (result.code === 0 && fs.existsSync(outputFilename)) {
            const rawAsm = fs.readFileSync(outputFilename, 'utf8');
            const processedLines = this.normalizeAssemblyOutput(rawAsm);
            
            // Write clean assembly back to disk
            fs.writeFileSync(outputFilename, processedLines.map(l => l.text).join('\n'), 'utf8');
            
            // Update structural stdout for the web UI display response
            result.stdout = processedLines;
        }

        return result;
    }

    /**
     * Normalizes assembly, clears out compiler bookkeeping noise (.SFPREPLAY) and annotates vector logic.
     */
    private normalizeAssemblyOutput(rawAssembly: string): any[] {
        const lines = rawAssembly.split('\n');
        const normalized: any[] = [];
        const seenLabels = new Set<string>();

        for (let i = 0; i < lines.length; i++) {
            let line = lines[i];

            // 1. Strip SFPREPLAY directives completely
            if (/^\s*\.SFPREPLAY\s+/.test(line)) {
                continue;
            }

            // 2. Preserve and tag custom vector structural elements
            if (/^\s*(v_if|v_endif|v_else)\b/.test(line)) {
                normalized.push({ text: `${line.trimEnd()} # Tensix vector control flow` });
                continue;
            }

            // 3. Deduplicate duplicate label compiler artifacts
            const labelMatch = line.match(/^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:/);
            if (labelMatch) {
                const labelName = labelMatch[1];
                if (seenLabels.has(labelName)) {
                    continue;
                }
                seenLabels.add(labelName);
            }

            // 4. Content Cleanup & Metadata Annotations
            line = line.replace(/^\s+/, '').replace(/\s+$/, '');
            if (line.length === 0) {
                if (normalized.length > 0 && normalized[normalized.length - 1].text !== '') {
                    normalized.push({ text: '' });
                }
                continue;
            }

            if (/\b(vFloat|vInt|vInt32|vUint32)\b/.test(line)) {
                line = `${line} # Tensix vector type`;
            }

            normalized.push({ text: line });
        }

        return normalized;
    }
}
