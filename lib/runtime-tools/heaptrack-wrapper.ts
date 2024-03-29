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

import * as path from 'path';
import {ExecutionOptions} from '../../types/compilation/compilation.interfaces.js';
import {
    RuntimeToolOptions,
    TypicalExecutionFunc,
    UnprocessedExecResult,
} from '../../types/execution/execution.interfaces.js';
import {O_RDWR} from 'constants';
import * as fs from 'fs';
import * as net from 'net';
import {pipeline} from 'stream';
import {unwrap} from '../assert.js';
import {logger} from '../logger.js';
import {executeDirect} from '../exec.js';
import {PropertyGetter} from '../properties.interfaces.js';
import {BaseRuntimeTool} from './base-runtime-tool.js';
import {CompilationEnvironment} from '../compilation-env.js';

const O_NONBLOCK = 2048;

export class HeaptrackWrapper extends BaseRuntimeTool {
    private rawOutput: string;
    private pipe: string;
    private interpretedPath: string;
    private heaptrackPath: string;
    private mkfifoPath: string;
    private preload: string;
    private interpreter: string;
    private printer: string;

    public static FlamegraphFilename = 'heaptrack.flamegraph.txt';

    constructor(
        dirPath: string,
        sandboxFunc: TypicalExecutionFunc,
        execFunc: TypicalExecutionFunc,
        options: RuntimeToolOptions,
        ceProps: PropertyGetter,
        sandboxType: string,
    ) {
        super(dirPath, sandboxFunc, execFunc, options, sandboxType);

        this.mkfifoPath = ceProps('mkfifo', '/usr/bin/mkfifo');

        this.pipe = path.join(this.dirPath, 'heaptrack_fifo');
        this.rawOutput = path.join(this.dirPath, 'heaptrack_raw.txt');
        this.interpretedPath = path.join(this.dirPath, 'heaptrack_interpreted.txt');

        this.heaptrackPath = ceProps('heaptrackPath', '');

        this.preload = path.join(this.heaptrackPath, 'lib/libheaptrack_preload.so');
        this.interpreter = path.join(this.heaptrackPath, 'libexec/heaptrack_interpret');
        this.printer = path.join(this.heaptrackPath, 'bin/heaptrack_print');
    }

    public static isSupported(compiler: CompilationEnvironment) {
        return process.platform !== 'win32' && compiler.ceProps('heaptrackPath', '') !== '';
    }

    private async mkfifo(path: string, rights: number) {
        await executeDirect(this.mkfifoPath, ['-m', rights.toString(8), path], {});
    }

    private async makePipe() {
        await this.mkfifo(this.pipe, 0o666);
    }

    private addToEnv(execOptions: ExecutionOptions) {
        if (!execOptions.env) execOptions.env = {};

        if (execOptions.env.LD_PRELOAD) {
            execOptions.env.LD_PRELOAD = this.preload + ':' + execOptions.env.LD_PRELOAD;
        } else {
            execOptions.env.LD_PRELOAD = this.preload;
        }

        if (this.sandboxType === 'nsjail') {
            execOptions.env.DUMP_HEAPTRACK_OUTPUT = '/app/heaptrack_fifo';
        } else {
            execOptions.env.DUMP_HEAPTRACK_OUTPUT = this.pipe;
        }
    }

    private async interpret(execOptions: ExecutionOptions): Promise<UnprocessedExecResult> {
        return this.execFunc(this.interpreter, [this.rawOutput], execOptions);
    }

    private async finishPipesAndStreams(fd: number, file: fs.WriteStream, socket: net.Socket) {
        socket.push(null);
        await new Promise(resolve => socket.end(() => resolve(true)));

        await new Promise(resolve => file.end(() => resolve(true)));

        file.write(Buffer.from([0]));

        socket.resetAndDestroy();
        socket.unref();

        await new Promise(resolve => {
            file.close(err => {
                if (err) logger.error('Error while closing heaptrack log: ', err);
                resolve(true);
            });
        });

        await new Promise(resolve => fs.close(fd, () => resolve(true)));
    }

    private async interpretAndSave(execOptions: ExecutionOptions, result: UnprocessedExecResult) {
        const dirPath = unwrap(execOptions.appHome);
        execOptions.input = fs.readFileSync(this.rawOutput).toString('utf8');

        const interpretResults = await this.interpret(execOptions);

        if (this.getOptionValue('summary') === 'stderr') {
            result.stderr += interpretResults.stderr;
        }

        fs.writeFileSync(this.interpretedPath, interpretResults.stdout);
    }

    private async saveFlamegraph(execOptions: ExecutionOptions, result: UnprocessedExecResult) {
        const args = [this.interpretedPath];

        if (this.getOptionValue('graph') === 'yes') {
            const flamesFilepath = path.join(this.dirPath, HeaptrackWrapper.FlamegraphFilename);
            args.push('-F', flamesFilepath);
        }

        const printResults = await this.execFunc(this.printer, args, execOptions);
        if (printResults.stderr) result.stderr += printResults.stderr;

        if (this.getOptionValue('details') === 'stderr') {
            result.stderr += printResults.stdout;
        }
    }

    public async exec(filepath: string, args: string[], execOptions: ExecutionOptions): Promise<UnprocessedExecResult> {
        const dirPath = unwrap(execOptions.appHome);

        const runOptions = JSON.parse(JSON.stringify(execOptions));
        const interpretOptions = JSON.parse(JSON.stringify(execOptions));
        interpretOptions.maxOutput = 1024 * 1024 * 1024;
        this.addToEnv(runOptions);

        await this.makePipe();

        const fd = fs.openSync(this.pipe, O_NONBLOCK | O_RDWR);
        const socket = new net.Socket({fd, readable: true, writable: true});

        const file = fs.createWriteStream(this.rawOutput);
        pipeline(socket, file, err => {
            if (err) {
                logger.error('Error during heaptrack pipeline: ', err);
            }
        });

        const result = await this.sandboxFunc(filepath, args, runOptions);

        await this.finishPipesAndStreams(fd, file, socket);

        fs.unlinkSync(this.pipe);

        await this.interpretAndSave(interpretOptions, result);

        await this.saveFlamegraph(execOptions, result);

        return result;
    }
}
