// Copyright (c) 2016, Compiler Explorer Authors
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

import os from 'os';
import {Writable} from 'stream';

import {LEVEL, MESSAGE} from 'triple-beam';
import winston from 'winston';
import LokiTransport from 'winston-loki';
import {Papertrail} from 'winston-papertrail';
import TransportStream, {TransportStreamOptions} from 'winston-transport';

const consoleTransportInstance = new winston.transports.Console();
export const logger = winston.createLogger({
    format: winston.format.combine(winston.format.colorize(), winston.format.splat(), winston.format.simple()),
    transports: [consoleTransportInstance],
});

// Creates a log stream, suitable to passing to something that writes complete lines of output to a stream, for example
// morgan's http logger. We look for complete text lines and output each as a winston log entry.
export function makeLogStream(level: string, logger_: winston.Logger = logger): {write: (string) => void} {
    let buffer = '';
    return new Writable({
        write: (chunk: string, encoding, callback: () => void) => {
            buffer += chunk;
            while (buffer.length > 0) {
                const eol = buffer.indexOf('\n');
                if (eol < 0) break;
                logger_.log(level, buffer.substring(0, eol));
                buffer = buffer.substring(eol + 1);
            }
            callback();
        },
    });
}

// Our own transport which uses Papertrail under the hood but better adapts it to work in winston 3.0
class MyPapertrailTransport extends TransportStream {
    private readonly hostname: string;
    private readonly program: string;
    public readonly transport: Papertrail;

    constructor(opts: TransportStreamOptions & {host: string; port: number; identifier: string}) {
        super(opts);

        this.hostname = os.hostname();
        this.program = opts.identifier;

        this.transport = new Papertrail({
            host: opts.host,
            port: opts.port,
            logFormat: (level, message) => message,
        });
    }

    override log(info: any, callback: () => void) {
        setImmediate(() => {
            this.emit('logged', info);
        });

        // We can't use callback here as winston-papertrail is a bit lax in calling it back
        this.transport.sendMessage(this.hostname, this.program, info[LEVEL], info[MESSAGE], x => x);
        callback();
    }
}

export function logToLoki(url) {
    const transport = new LokiTransport({
        host: url,
        labels: {job: 'ce'},
        batching: true,
        interval: 3,
        // remove color from log level label - loki really doesn't like it
        format: winston.format.uncolorize({
            message: false,
            raw: false,
        }),
    });

    logger.add(transport);
    logger.info('Configured loki');
}

export function logToPapertrail(host: string, port: number, identifier: string) {
    const transport = new MyPapertrailTransport({
        host: host,
        port: port,
        identifier: identifier,
    });
    transport.transport.on('error', err => {
        logger.error(err);
    });

    transport.transport.on('connect', message => {
        logger.info(message);
    });
    logger.add(transport);
    logger.info('Configured papertrail');
}

class Blackhole extends TransportStream {
    override log(info: any, callback: () => void) {
        callback();
    }
}

export function suppressConsoleLog() {
    logger.remove(consoleTransportInstance);
    logger.add(new Blackhole());
}
