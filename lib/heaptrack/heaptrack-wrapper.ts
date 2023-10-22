import * as path from 'path';
import {mkfifoSync} from 'mkfifo';
import {ExecutionOptions} from '../../types/compilation/compilation.interfaces.js';
import {UnprocessedExecResult} from '../../types/execution/execution.interfaces.js';
import {O_NONBLOCK,O_RDWR} from 'constants';
import * as fs from 'fs';
import * as net from 'net';
// import zlib from 'zlib';

export class HeaptrackWrapper {
    private dirPath: string;
    private raw_output: string;
    private pipe: string;
    private heaptrackPath: string;
    private preload: string;
    private interpreter: string;
    private sandboxFunc: Function;
    private execFunc: Function;

    constructor(dirPath: string, sandboxFunc: Function, execFunc: Function) {
        this.dirPath = dirPath;
        this.pipe = path.join(this.dirPath, 'heaptrack_fifo');
        this.raw_output = path.join(this.dirPath, 'heaptrack.raw');
        this.heaptrackPath = "/opt/compiler-explorer/heaptrack/1.3.0";
        this.preload = path.join(this.heaptrackPath, "lib/libheaptrack_preload.so");
        this.interpreter = path.join(this.heaptrackPath, "libexec/heaptrack_interpret");
        this.sandboxFunc = sandboxFunc;
        this.execFunc = execFunc;
    }

    private async make_pipe() {
        mkfifoSync(this.pipe, 0o777);
    }

    private add_to_env(execOptions: ExecutionOptions) {
        if (!execOptions.env) execOptions.env = {};

        if (execOptions.env.LD_PRELOAD) {
            execOptions.env.LD_PRELOAD = this.preload + ':' + execOptions.env.LD_PRELOAD;
        } else {
            execOptions.env.LD_PRELOAD = this.preload;
        }

        execOptions.env.DUMP_HEAPTRACK_OUTPUT = "/app/heaptrack_fifo";
    }

    private async interpret(execOptions: ExecutionOptions) {
        return this.execFunc(this.interpreter, [this.raw_output], execOptions);
    }

    public async exec(filepath: string, args: string[], execOptions: ExecutionOptions): Promise<UnprocessedExecResult> {
        const options = {...execOptions};
        this.add_to_env(options);

        await this.make_pipe();

        const fd = fs.openSync(this.pipe, O_NONBLOCK | O_RDWR);
        const socket = new net.Socket({ fd, readable: true, writable: true });

        const file = fs.createWriteStream(this.raw_output);
        socket.pipe(file);

        // gz.pipe(file);

        const result = await this.sandboxFunc(filepath, args, options);

        console.log('end of execution');

        await new Promise(resolve => {
            socket.end(() => {
                console.log('socket.end()');

                file.end(() => {
                    console.log('file.end()');

                    fs.unlinkSync(this.pipe);
                    console.log('unlinkSync()');

                    resolve(true);
                });
            });
        });


        // gz.flush();
        // gz.close();

        const interpretResults = await this.interpret(execOptions);
        result.stdout += interpretResults.stdout;
        result.stderr += interpretResults.stderr;

        return result;
    }
}
