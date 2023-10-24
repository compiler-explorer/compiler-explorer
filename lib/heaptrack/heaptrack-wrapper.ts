import * as path from 'path';
import {mkfifoSync} from 'mkfifo';
import {ExecutionOptions} from '../../types/compilation/compilation.interfaces.js';
import {UnprocessedExecResult} from '../../types/execution/execution.interfaces.js';
import {O_NONBLOCK, O_RDWR} from 'constants';
import * as fs from 'fs';
import * as net from 'net';
import {pipeline} from 'stream';
import {unwrap} from '../assert.js';

export class HeaptrackWrapper {
    private dirPath: string;
    private raw_output: string;
    private pipe: string;
    private heaptrackPath: string;
    private preload: string;
    private interpreter: string;
    private printer: string;
    private sandboxFunc: Function;
    private execFunc: Function;

    public static FlamegraphFilename = 'heaptrack.flamegraph.txt';

    constructor(dirPath: string, sandboxFunc: Function, execFunc: Function) {
        this.dirPath = dirPath;
        this.pipe = path.join(this.dirPath, 'heaptrack_fifo');
        this.raw_output = path.join(this.dirPath, 'heaptrack.raw');
        this.heaptrackPath = '/opt/compiler-explorer/heaptrack/1.3.0';
        this.preload = path.join(this.heaptrackPath, 'lib/libheaptrack_preload.so');
        this.interpreter = path.join(this.heaptrackPath, 'libexec/heaptrack_interpret');
        this.printer = path.join(this.heaptrackPath, 'bin/heaptrack_print');
        this.sandboxFunc = sandboxFunc;
        this.execFunc = execFunc;
    }

    private async make_pipe() {
        mkfifoSync(this.pipe, 0o666);
    }

    private add_to_env(execOptions: ExecutionOptions) {
        if (!execOptions.env) execOptions.env = {};

        if (execOptions.env.LD_PRELOAD) {
            execOptions.env.LD_PRELOAD = this.preload + ':' + execOptions.env.LD_PRELOAD;
        } else {
            execOptions.env.LD_PRELOAD = this.preload;
        }

        execOptions.env.DUMP_HEAPTRACK_OUTPUT = '/app/heaptrack_fifo';
    }

    private async interpret(execOptions: ExecutionOptions) {
        return this.execFunc(this.interpreter, [this.raw_output], execOptions);
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
                console.error(err);
            }
        });

        const result = await this.sandboxFunc(filepath, args, runOptions);

        await new Promise(async mainres => {
            socket.push(null);
            await new Promise(res => socket.end(() => res(true)));

            await new Promise(res => file.end(() => res(true)));

            file.write(Buffer.from([0]));

            socket.resetAndDestroy();
            socket.unref();

            await new Promise(res => {
                file.close(err => {
                    if (err) console.error(err);
                    res(true);
                });
            });

            await new Promise(res => fs.close(fd, () => res(true)));

            mainres(true);
        });

        fs.unlinkSync(this.pipe);

        interpretOptions.input = fs.readFileSync(this.raw_output).toString('utf8');

        const interpretResults = await this.interpret(interpretOptions);
        result.stderr += interpretResults.stderr; // summary

        const interpretedFilepath = path.join(dirPath, 'heaptrack_interpreted.txt');
        fs.writeFileSync(interpretedFilepath, interpretResults.stdout);

        const flamesFilepath = path.join(dirPath, HeaptrackWrapper.FlamegraphFilename);
        await this.execFunc(this.printer, [interpretedFilepath, '-F', flamesFilepath], execOptions);

        return result;
    }
}
