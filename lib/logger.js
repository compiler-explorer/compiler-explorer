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

import { LEVEL, MESSAGE } from 'triple-beam';
import winston from 'winston';
import LokiTransport from 'winston-loki';
import { Papertrail } from 'winston-papertrail';
import TransportStream from 'winston-transport';

const consoleTransportInstance = new winston.transports.Console();
export const logger = winston.createLogger({
    format: winston.format.combine(
        winston.format.colorize(),
        winston.format.splat(),
        winston.format.simple()),
    transports: [consoleTransportInstance],
});

logger.stream = {
    write: message => {
        logger.info(message.trim());
    },
};

logger.warnStream = {
    write: message => {
        logger.warn(message.trim());
    },
};

logger.errStream = {
    write: message => {
        logger.error(message.trim());
    },
};

// Our own transport which uses Papertrail under the hood but better adapts it to work
// in winston 3.0
class MyPapertrailTransport extends TransportStream {
    constructor(opts) {
        super(opts);

        this.hostname = os.hostname();
        this.program = opts.identifier;

        this.transport = new Papertrail(
            {
                host: opts.host,
                port: opts.port,
                logFormat: (level, message) => message,
            },
        );
    }

    log(info, callback) {
        setImmediate(() => {
            this.emit('logged', info);
        });

        // We can't use callback here as winston-papertrail is a bit lax in calling it back
        this.transport.sendMessage(this.hostname, this.program, info[LEVEL], info[MESSAGE], (x) => x);
        callback();
    }
}

export function logToLoki(url) {
    const transport = new LokiTransport({
        host: url,
        labels: { job: 'ce' },
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

export function logToPapertrail(host, port, identifier) {
    const transport = new MyPapertrailTransport({
        host: host, port: port, identifier: identifier,
    });
    transport.transport.on('error', (err) => {
        logger.error(err);
    });

    transport.transport.on('connect', (message) => {
        logger.info(message);
    });
    logger.add(transport);
    logger.info('Configured papertrail');
}

class Blackhole extends TransportStream {
    log(info, callback) {
        callback();
    }
}

export function suppressConsoleLog() {
    logger.remove(consoleTransportInstance);
    logger.add(new Blackhole());
}
