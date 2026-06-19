import * as fs from 'node:fs/promises';
import Path from 'node:path';

import _ from 'underscore';

import type {CompilationInfo, CompilationResult} from '../../types/compilation/compilation.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {unwrap} from '../assert.js';
import {BaseCompiler} from '../base-compiler.js';
import {CompilationEnvironment} from '../compilation-env.js';
import {PTXAsmParser} from '../parsers/asm-parser-ptx.js';
import {SassAsmParser} from '../parsers/asm-parser-sass.js';
import {ClangParser} from './argument-parsers.js';

interface ExecResult {
    code: number;
    stdout: string;
    stderr: string;
}

export class ScaleNvccNvidiaCompiler extends BaseCompiler {
    static get key() {
        return 'scale-nvcc-nvidia';
    }

    deviceAsmParser: SassAsmParser;
    ptxParser: PTXAsmParser;

    constructor(info: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(info, env);
        this.compiler.supportsOptOutput = true;
        this.compiler.supportsDeviceAsmView = true;
        this.deviceAsmParser = new SassAsmParser(this.compilerProps);
        this.ptxParser = new PTXAsmParser(this.compilerProps);
    }

    // TEMP: -o commented out because scale can't combine an explicit `-o`
    // with `-Xcompiler=-S`. This means scale (and now nvcc too, while this
    // is shared) falls back to basename-derived default naming instead of
    // a predictable `output.s`. See findHostAsmFile()/extractDeviceCode()
    // below, which auto-detect either naming convention.
    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string, userOptions?: string[]) {
        // const opts = ['-o', this.filename(outputFilename), '-g', '-lineinfo', '--keep-device-functions'];
        const opts = ['-g', '-lineinfo', '--keep-device-functions'];
        if (!filters.execute) {
            opts.push('-keep', '-keep-dir', Path.dirname(outputFilename));
            if (!filters.binary) {
                opts.push('-Xcompiler=-S');
            }
        }
        return opts;
    }

    override getArgumentParserClass() {
        return ClangParser;
    }

    override optOutputRequested(options: string[]) {
        return (
            super.optOutputRequested(options) ||
            options.includes('--optimization-info') ||
            options.includes('-opt-info')
        );
    }

    //Helpers to get de device ptx (.s) and bitcode files
    private static readonly scaleDeviceFileRe = /-cuda-nvptx64-nvidia-cuda-([^./]+)\.s$/;
    private static readonly scaleDeviceBcFileRe = /-cuda-nvptx64-nvidia-cuda-([^./]+)\.bc$/;

    private async findHostAsmFile(dirPath: string): Promise<string | null> {
        try {
            const files = await fs.readdir(dirPath);

            //console.log('all files:', files);

            const hostFiles = files.filter(f => f.endsWith('.s') && !ScaleNvccNvidiaCompiler.scaleDeviceFileRe.test(f));

            //console.log('Host ASM candidates:', hostFiles);

            if (hostFiles.length !== 1) {
                //console.warn(`Expected exactly one host .s file, found ${hostFiles.length}`);
                return null;
            }

            return Path.join(dirPath, hostFiles[0]);
        } catch {
            return null;
        }
    }

    override async postProcess(result, outputFilename: string, filters: ParseFiltersAndOutputOptions) {
        // TEMP (scale support): outputFilename as originally computed may not
        // exist since we no longer pass `-o`. Try to recover the real file.
        if (!filters.binary && result.dirPath) {
            try {
                await fs.stat(outputFilename);
            } catch {
                const hostAsm = await this.findHostAsmFile(result.dirPath);
                if (hostAsm) {
                    try {
                        result.asmSize = (await fs.stat(hostAsm)).size;
                    } catch {
                        // leave asmSize as-is; base behaviour reports "no output" below
                    }
                    outputFilename = hostAsm;
                }
            }
        }

        const maxSize = this.env.ceProps('max-asm-size', 64 * 1024 * 1024);
        const optPromise = result.optPath ? this.processOptOutput(result.optPath) : Promise.resolve([]);
        const postProcess = _.compact(this.compiler.postProcess);
        const asmPromise = (
            filters.binary
                ? this.objdump(outputFilename, {}, maxSize, !!filters.intel, !!filters.demangle, false, false, filters)
                : (async () => {
                      if (result.asmSize === undefined) {
                          result.asm = '<No output file>';
                          return result;
                      }
                      if (result.asmSize >= maxSize) {
                          result.asm =
                              '<No output: generated assembly was too large' +
                              ` (${result.asmSize} > ${maxSize} bytes)>`;
                          return result;
                      }
                      if (postProcess.length > 0) {
                          return await this.execPostProcess(result, postProcess, outputFilename, maxSize);
                      }
                      const contents = await fs.readFile(outputFilename, {encoding: 'utf8'});
                      result.asm = contents.toString();
                      return result;
                  })()
        ).then(asm => {
            result.asm = typeof asm === 'string' ? asm : asm.asm;
            return result;
        });
        return Promise.all([asmPromise, optPromise, []]);
    }

    // Matches the start/end of a GAS inline-assembly block emitted by the host compiler.
    private static readonly appBlockStartRe = /^#APP\b/;
    private static readonly appBlockEndRe = /^#NO_APP\b/;
    // Matches the .nv_fatbin section directive that NVCC injects to hold the fat binary blob.
    private static readonly nvFatBinSectionRe = /^\s*\.section\s+\.nv_fatbin\b/;

    /**
     * Strip `#APP`/`#NO_APP` inline-assembly blocks that contain a `.nv_fatbin`
     * section from the host-side x86 assembly.  These blocks hold the raw CUDA
     * fat binary blob (the `fatbinData` label followed by hundreds of `.quad`
     * hex lines) which is never useful to inspect in the asm view.
     *
     * Only blocks that contain `.nv_fatbin` are removed; any `#APP`/`#NO_APP`
     * blocks originating from genuine user inline-assembly are left intact.
     */
    protected removeNvccFatbinaryBlob(asm: string): string {
        const lines = asm.split('\n');
        const result: string[] = [];
        let inAppBlock = false;
        let hasFatBin = false;
        let appBuffer: string[] = [];

        for (const line of lines) {
            const trimmed = line.trim();
            if (ScaleNvccNvidiaCompiler.appBlockStartRe.test(trimmed)) {
                inAppBlock = true;
                hasFatBin = false;
                appBuffer = [line];
            } else if (ScaleNvccNvidiaCompiler.appBlockEndRe.test(trimmed)) {
                inAppBlock = false;
                if (!hasFatBin) {
                    // Not a fat-binary block — keep it
                    appBuffer.push(line);
                    result.push(...appBuffer);
                }
                appBuffer = [];
            } else if (inAppBlock) {
                if (ScaleNvccNvidiaCompiler.nvFatBinSectionRe.test(line)) {
                    hasFatBin = true;
                }
                appBuffer.push(line);
            } else {
                result.push(line);
            }
        }

        // Handle (malformed) unclosed #APP block: keep it
        if (appBuffer.length > 0) {
            result.push(...appBuffer);
        }

        return result.join('\n');
    }

    override async processAsm(result, filters: ParseFiltersAndOutputOptions, options: string[]) {
        if (filters.labels && typeof result.asm === 'string') {
            result = {...result, asm: this.removeNvccFatbinaryBlob(result.asm)};
        }
        return super.processAsm(result, filters, options);
    }

    // ---- device-code extraction helpers -------------------------------------------------

    /** Pulls the captured arch (e.g. `sm_75`) out of a scale per-target device filename. */
    private static archFromDeviceFileName(name: string, re: RegExp): string {
        return unwrap(name.match(re))[1];
    }

    private async runPtxas(ptxFileName: string, cubinPath: string, arch: string, dirPath: string): Promise<ExecResult> {
        const {ptxas} = this.compiler;
        const args = ['-arch', arch, ptxFileName, '-o', cubinPath];
        //console.log('[runPtxas]: running', ptxas, args.join(' '));
        return this.exec(unwrap(ptxas), args, {customCwd: dirPath});
    }

    private async runNvdisasm(cubinPath: string, dirPath: string): Promise<ExecResult> {
        const {nvdisasm} = this.compiler;
        const args = [cubinPath, '-c', '-g', '-hex'];
        //console.log('[runNvdisasm]: running', nvdisasm, args.join(' '));

        return this.exec(unwrap(nvdisasm), args, {customCwd: dirPath});
    }

    private async runLlvmDis(bcPath: string, dirPath: string): Promise<ExecResult> {
        //console.log('[runLlvmDis]: running', this.compiler.llvmDisassembler, bcPath);

        const result = await this.exec(unwrap(this.compiler.llvmDisassembler), [bcPath], {customCwd: dirPath});

        //console.log('[runLlvmDis]: exit code', result.code);
        //if (result.stderr) console.log('[runLlvmDis]: stderr', result.stderr);

        return result;
    }

    //Tune the display of llvmir here !
    private static llvmIrOptions(demangle: boolean) {
        return {
            demangle,
            filterDebugInfo: false,
            filterIRMetadata: false,
            filterAttributes: false,
            filterComments: false,
        };
    }

    /**
     * Handles a single scale/nvcc device `.s` (PTX) file: registers it under
     * `PTX (<arch>)`, then pipes it through ptxas -> nvdisasm to also
     * register the corresponding `SASS (<arch>)` entry.
     */
    private async processPtxFile(
        name: string,
        dirPath: string,
        filters: ParseFiltersAndOutputOptions,
        demangle: boolean,
        devices: Record<string, CompilationResult>,
    ): Promise<void> {
        const archAndCode = ScaleNvccNvidiaCompiler.archFromDeviceFileName(
            name,
            ScaleNvccNvidiaCompiler.scaleDeviceFileRe,
        );

        const asm = await fs.readFile(Path.join(dirPath, name), 'utf8');

        const ptxNameAndArch = `PTX (${archAndCode.toLowerCase()})`;
        Object.assign(devices, {
            [ptxNameAndArch]: await this.postProcessAsm(
                {
                    okToCache: demangle,
                    ...this.ptxParser.process(asm, {...filters, binary: false}),
                },
                {...filters, binary: false},
            ),
        });

        //
        // PTX -> CUBIN -> SASS
        //
        const cubinPath = Path.join(dirPath, `${Path.basename(name, '.s')}.cubin`);

        let ptxasResult: ExecResult;
        try {
            ptxasResult = await this.runPtxas(name, cubinPath, archAndCode, dirPath);
        } catch (err) {
            //console.error('[processPtxFile]: exception running ptxas for', name, err);
            return;
        }

        if (ptxasResult.code !== 0) {
            //console.warn('[processPtxFile]: ptxas failed for', name, '- skipping SASS');
            return;
        }

        let nvdisasmResult: ExecResult;
        try {
            nvdisasmResult = await this.runNvdisasm(cubinPath, dirPath);
        } catch (err) {
            //console.error('[processPtxFile]: exception running nvdisasm for', cubinPath, err);
            return;
        }

        const sassAsm =
            nvdisasmResult.code === 0
                ? this.postProcessObjdumpOutput(nvdisasmResult.stdout)
                : `<nvdisasm failed with code ${nvdisasmResult.code}>`;

        const sassNameAndArch = `SASS (${archAndCode.toLowerCase()})`;
        Object.assign(devices, {
            [sassNameAndArch]: await this.postProcessAsm(
                {
                    okToCache: demangle,
                    ...this.deviceAsmParser.process(sassAsm, {...filters, binary: true}),
                },
                {...filters, binary: true},
            ),
        });
    }

    /**
     * Handles a single scale/nvcc device `.bc` (LLVM bitcode) file: runs
     * `llvm-dis` to get readable LLVM IR text
     */
    private async processBcDeviceFile(
        name: string,
        dirPath: string,
        filters: ParseFiltersAndOutputOptions,
        demangle: boolean,
        devices: Record<string, CompilationResult>,
    ): Promise<void> {
        const archAndCode = ScaleNvccNvidiaCompiler.archFromDeviceFileName(
            name,
            ScaleNvccNvidiaCompiler.scaleDeviceBcFileRe,
        );

        const irNameAndArch = `Device LLVM IR (${archAndCode.toLowerCase()})`;

        if (!this.compiler.llvmDisassembler) {
            Object.assign(devices, {
                [irNameAndArch]: await this.postProcessAsm(
                    {
                        okToCache: false,
                        ...this.llvmIr.process(
                            '<error: no llvm-dis found to disassemble bitcode>',
                            ScaleNvccNvidiaCompiler.llvmIrOptions(demangle),
                        ),
                    },
                    {...filters, binary: false},
                ),
            });
            return;
        }

        const bcPath = Path.join(dirPath, name);
        const llPath = Path.join(dirPath, `${Path.basename(name, '.bc')}.ll`);

        let irText: string;
        try {
            const disResult = await this.runLlvmDis(bcPath, dirPath);
            irText =
                disResult.code === 0
                    ? await fs.readFile(llPath, 'utf8')
                    : `<llvm-dis failed with code ${disResult.code}>`;
        } catch (err) {
            //console.error('[processBcDeviceFile]: exception running llvm-dis for', name, err);
            irText = `<llvm-dis failed: ${err}>`;
        }

        Object.assign(devices, {
            [irNameAndArch]: await this.postProcessAsm(
                {
                    okToCache: demangle,
                    ...(await this.llvmIr.process(irText, ScaleNvccNvidiaCompiler.llvmIrOptions(demangle))),
                },
                {...filters, binary: false},
            ),
        });
    }

    override async extractDeviceCode(
        result: CompilationResult,
        filters: ParseFiltersAndOutputOptions,
        compilationInfo: CompilationInfo,
    ) {
        const {dirPath} = result;
        const {demangle} = filters;
        const devices: Record<string, CompilationResult> = {...result.devices};

        if (dirPath) {
            const files = await fs.readdir(dirPath);

            //console.log('[extractDeviceCode]: dirPath', dirPath);
            //console.log('[extractDeviceCode]: all files at start', files);

            const ptxFile = files.filter(f => ScaleNvccNvidiaCompiler.scaleDeviceFileRe.test(f)); //Ends with .s, not the host
            const bcDeviceFile = files.filter(f => ScaleNvccNvidiaCompiler.scaleDeviceBcFileRe.test(f)); //ends with .bc, not the host

            await Promise.all([
                ...ptxFile.map(name => this.processPtxFile(name, dirPath, filters, !!demangle, devices)),
                ...bcDeviceFile.map(name => this.processBcDeviceFile(name, dirPath, filters, !!demangle, devices)),
            ]);

            //console.log('[extractDeviceCode]: final devices keys', Object.keys(devices));
            result.devices = devices;
        }

        return result;
    }
}
