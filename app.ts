// Copyright (c) 2012, Compiler Explorer Authors
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

////
// see https://docs.sentry.io/platforms/javascript/guides/node/install/late-initialization/
import '@sentry/node/preload'; // preload Sentry's "preload" support before any other imports
////
import process from 'node:process';

import {initializeOptionsFromCommandLine} from './lib/app/cli.js';
import {loadConfiguration} from './lib/app/config.js';
import {initializeApplication} from './lib/app/main.js';
import {setBaseDirectory} from './lib/assert.js';
import {logger} from './lib/logger.js';
import * as props from './lib/properties.js';
import * as utils from './lib/utils.js';

// Set base directory for resolving paths
setBaseDirectory(new URL('.', import.meta.url));

// Initialize configuration from command-line arguments
const {appArgs, options: opts} = initializeOptionsFromCommandLine(process.argv);

// Get distribution path for static files
const distPath = utils.resolvePathFromAppRoot('.');

// Load configuration
const config = loadConfiguration({
    appArgs,
    useLocal: opts.local,
    propDebug: opts.propDebug,
});

// Get AWS properties
const awsProps = props.propsFor('aws');

// Set up signal handlers
process.on('uncaughtException', uncaughtHandler);
process.on('SIGINT', signalHandler('SIGINT'));
process.on('SIGTERM', signalHandler('SIGTERM'));
process.on('SIGQUIT', signalHandler('SIGQUIT'));

function signalHandler(name: string) {
    return () => {
        logger.info(`stopping process: ${name}`);
        process.exit(0);
    };
}

function uncaughtHandler(err: Error, origin: NodeJS.UncaughtExceptionOrigin) {
    logger.info(`stopping process: Uncaught exception: ${err}\nException origin: ${origin}`);
    // The app will exit naturally from here, but if we call `process.exit()` we may lose log lines.
    // see https://github.com/winstonjs/winston/issues/1504#issuecomment-1033087411
    process.exitCode = 1;
}

// Initialize and start the application
initializeApplication({
    appArgs,
    options: opts,
    config,
    distPath,
    awsProps,
}).catch(err => {
    logger.error('Top-level error (shutting down):', err);
    // Shut down after a second to hopefully let logs flush.
    setTimeout(() => process.exit(1), 1000);
});
