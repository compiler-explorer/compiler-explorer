// Copyright (c) 2023, Compiler Explorer Authors
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright notice,
//       this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

import path from 'path';

import fs from 'fs-extra';

import {ToolInfo} from '../../types/tool.interfaces';
import {assert} from '../assert';
import * as exec from '../exec';
import {logger} from '../logger';
import * as utils from '../utils';

import {BaseTool} from './base-tool';
import {ToolEnv} from './base-tool.interface';

export class PvsStudioTool extends BaseTool {
    static get key() {
        return 'pvs-studio-tool';
    }

    plogConverterPath: string;

    constructor(toolInfo: ToolInfo, env: ToolEnv) {
        super(toolInfo, env);
        this.plogConverterPath = this.env.compilerProps('plogConverter', '/usr/bin/plog-converter');

        this.addOptionsToToolArgs = false;
    }

    override async runTool(compilationInfo: Record<any, any>, inputFilepath?: string, args?: string[]) {
        if (compilationInfo.code !== 0) {
            return this.createErrorResponse('Unable to start analysis due to compilation error.');
        }

        assert(inputFilepath);

        const sourceDir = path.dirname(inputFilepath);

        // Collecting the flags of compilation
        let compileFlags = utils.splitArguments(compilationInfo.compiler.options);

        const includeflags = super.getIncludeArguments(compilationInfo.libraries, compilationInfo.compiler);
        compileFlags = compileFlags.concat(includeflags);

        const libOptions = super.getLibraryOptions(compilationInfo.libraries, compilationInfo.compiler);
        compileFlags = compileFlags.concat(libOptions);

        const manualCompileFlags = compilationInfo.options.filter(option => option !== inputFilepath);
        compileFlags = compileFlags.concat(manualCompileFlags);

        compileFlags = compileFlags.filter(function (flag) {
            return flag !== '';
        });

        // Deal with args
        args = [];

        // Necessary arguments
        args.push('analyze', '--source-file', inputFilepath);
        const outputFilePath = path.join(sourceDir, 'pvs-studio-log.log');
        args.push(
            '--output-file',
            outputFilePath,
            // Exclude directories from analysis
            '-e',
            '/opt',
            '-e',
            '/usr',
            '--pvs-studio-path',
            path.dirname(this.tool.exe) + '/pvs-studio',
            // TODO: expand this to switch() for all supported compilers:
            // visualcpp, clang, gcc, bcc, bcc_clang64, iar, keil5, keil5_gnu
            '--preprocessor',
        );
        if (compilationInfo.compiler.group.includes('clang')) args.push('clang');
        else args.push('gcc');

        // Now let's push the path to the compiler
        if (this.tool.compilerLanguage === 'c') {
            args.push('--cc');
        } else {
            args.push('--cxx');
        }
        args.push(compilationInfo.compiler.exe, '--cl-params');
        args = args.concat(compileFlags);

        // If you are to modify this code,
        // don't forget that super.runTool() does args.push(inputFilepath) inside.
        const result = await super.runTool(compilationInfo, inputFilepath, args);
        if (result.code !== 0) {
            return result;
        }

        // Convert log to readable format
        const plogConverterOutputFilePath = path.join(sourceDir, 'pvs-studio-log.err');

        const plogConverterArgs: string[] = [];
        plogConverterArgs.push(
            '-t',
            'errorfile',
            '-a',
            'FAIL:1,2,3;GA:1,2,3',
            '-o',
            plogConverterOutputFilePath,
            'pvs-studio-log.log',
        );

        const plogExecOptions = this.getDefaultExecOptions();
        plogExecOptions.customCwd = sourceDir;

        const plogConverterResult = await exec.execute(this.plogConverterPath, plogConverterArgs, plogExecOptions);
        if (plogConverterResult.code !== 0) {
            logger.warn('plog-converter failed', plogConverterResult);
            return this.convertResult(plogConverterResult, inputFilepath);
        }
        const plogRawOutput = await fs.readFile(plogConverterOutputFilePath, 'utf8');

        // Sometimes if a code fragment can't be analyzed, PVS-Studio makes a warning for a preprocessed file
        // (*.PVS-Studio.i)
        // This name can't be parsed by utils.parseOutput
        // so let's just replace it with a source file name.
        const plogConverterOutput = plogRawOutput
            .toString()
            .replace(sourceDir + '/example.PVS-Studio.i', inputFilepath);

        result.stdout = utils.parseOutput(plogConverterOutput, plogConverterResult.filenameTransform(inputFilepath));

        // Now let's trim the documentation link
        if (result.stdout.length > 0) {
            const idx = result.stdout[0].text.indexOf('The documentation for all analyzer warnings is available here:');
            result.stdout[0].text = result.stdout[0].text.substring(idx).concat('\n\n');
        }

        // The error output is unnecessary for the user
        // If you need to debug PVS-Studio-tool, you can comment out this line
        result.stderr = [];

        return result;
    }

    override getDefaultExecOptions() {
        const execOptions = super.getDefaultExecOptions();
        execOptions.env = {
            ...execOptions.env,
            PATH: process.env.PATH + ':/opt/compiler-explorer/pvs-studio-latest/bin',
        };

        return execOptions;
    }
}
