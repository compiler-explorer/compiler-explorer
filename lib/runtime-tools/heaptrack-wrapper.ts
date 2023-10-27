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
import {O_NONBLOCK, O_RDWR} from 'constants';
import * as fs from 'fs';
import * as net from 'net';
import {pipeline} from 'stream';
import {unwrap} from '../assert.js';
import {logger} from '../logger.js';
import {executeDirect} from '../exec.js';
import {PropertyGetter} from '../properties.interfaces.js';
import {BaseRuntimeTool} from './base-runtime-tool.js';
import {CompilationEnvironment} from '../compilation-env.js';

export class HeaptrackWrapper extends BaseRuntimeTool {
    private raw_output: string;
    private pipe: string;
    private heaptrackPath: string;
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
    ) {
        super(dirPath, sandboxFunc, execFunc, options);

        this.pipe = path.join(this.dirPath, 'heaptrack_fifo');
        this.raw_output = path.join(this.dirPath, 'heaptrack.raw');

        this.heaptrackPath = ceProps('heaptrackPath', '/opt/compiler-explorer/heaptrack/1.3.0');

        this.preload = path.join(this.heaptrackPath, 'lib/libheaptrack_preload.so');
        this.interpreter = path.join(this.heaptrackPath, 'libexec/heaptrack_interpret');
        this.printer = path.join(this.heaptrackPath, 'bin/heaptrack_print');
    }

    public static isSupported(compiler: CompilationEnvironment) {
        return compiler.ceProps('heaptrackPath', '') !== '';
    }

    private async mkfifo(path: string, rights: number) {
        await executeDirect('/usr/bin/mkfifo', ['-m', rights.toString(8), path], {});
    }

    private async make_pipe() {
        await this.mkfifo(this.pipe, 0o666);
    }

    private add_to_env(execOptions: ExecutionOptions) {
        if (!execOptions.env) execOptions.env = {};

        if (execOptions.env.LD_PRELOAD) {
            execOptions.env.LD_PRELOAD = this.preload + ':' + execOptions.env.LD_PRELOAD;
        } else {
            execOptions.env.LD_PRELOAD = this.preload;
        }

        execOptions.env.DUMP_HEAPTRACK_OUTPUT = '/app/heaptrack_fifo'; // todo: will not work for local users without nsjail, is that a problem?
    }

    private async interpret(execOptions: ExecutionOptions): Promise<UnprocessedExecResult> {
        return this.execFunc(this.interpreter, [this.raw_output], execOptions);
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

    private async interpretAndSave(execOptions: ExecutionOptions, result: UnprocessedExecResult): Promise<string> {
        const dirPath = unwrap(execOptions.appHome);
        execOptions.input = fs.readFileSync(this.raw_output).toString('utf8');

        const interpretResults = await this.interpret(execOptions);

        if (this.getOptionValue('summary') === 'stderr') {
            result.stderr += interpretResults.stderr;
        }

        const interpretedFilepath = path.join(dirPath, 'heaptrack_interpreted.txt');
        fs.writeFileSync(interpretedFilepath, interpretResults.stdout);

        return interpretedFilepath;
    }

    public async exec(filepath: string, args: string[], execOptions: ExecutionOptions): Promise<UnprocessedExecResult> {
        const dirPath = unwrap(execOptions.appHome);

        const runOptions = JSON.parse(JSON.stringify(execOptions));
        const interpretOptions = JSON.parse(JSON.stringify(execOptions));
        this.add_to_env(runOptions);

        await this.make_pipe();

        const fd = fs.openSync(this.pipe, O_NONBLOCK | O_RDWR);
        const socket = new net.Socket({fd, readable: true, writable: true});

        const file = fs.createWriteStream(this.raw_output);
        pipeline(socket, file, err => {
            if (err) {
                logger.error('Error during heaptrack pipeline: ', err);
            }
        });

        const result = await this.sandboxFunc(filepath, args, runOptions);

        await this.finishPipesAndStreams(fd, file, socket);

        fs.unlinkSync(this.pipe);

        const interpretedFilepath = await this.interpretAndSave(interpretOptions, result);

        const flamesFilepath = path.join(dirPath, HeaptrackWrapper.FlamegraphFilename);
        await this.execFunc(this.printer, [interpretedFilepath, '-F', flamesFilepath], execOptions);

        return result;
    }
}
