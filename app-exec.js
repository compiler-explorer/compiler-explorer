#!/usr/bin/env node

// Copyright (c) 2018, Compiler Explorer Authors
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

// Initialise options and properties. Don't load any handlers here; they
// may need an initialised properties library.
const nopt = require('nopt'),
    os = require('os'),
    props = require('./lib/properties'),
    child_process = require('child_process'),
    fs = require('fs-extra'),
    systemdSocket = require('systemd-socket'),
    _ = require('underscore'),
    express = require('express'),
    Raven = require('raven'),
    logger = require('./lib/logger').logger,
    utils = require('./lib/utils');


// Parse arguments from command line 'node ./app.js args...'
const opts = nopt({
    env: [String, Array],
    rootDir: [String],
    host: [String],
    port: [Number],
    propDebug: [Boolean],
    debug: [Boolean]
});

if (opts.debug) logger.level = 'debug';

// Use the canned git_hash if provided
let gitReleaseName = '';
if (opts.static && fs.existsSync(opts.static + "/git_hash")) {
    gitReleaseName = fs.readFileSync(opts.static + "/git_hash").toString().trim();
} else if (fs.existsSync('.git/')) { // Just if we have been cloned and not downloaded (Thanks David!)
    gitReleaseName = child_process.execSync('git rev-parse HEAD').toString().trim();
}

// Set default values for omitted arguments
const defArgs = {
    rootDir: opts.rootDir || './etc',
    env: opts.env || ['dev'],
    hostname: opts.host,
    port: opts.port || 10241,
    gitReleaseName: gitReleaseName
};

const isDevMode = () => process.env.NODE_ENV === "DEV";

const propHierarchy = _.flatten([
    'defaults',
    defArgs.env,
    _.map(defArgs.env, e => e + '.' + process.platform),
    process.platform,
    os.hostname(),
    'local']);
logger.info("properties hierarchy: " + propHierarchy.join(', '));

// Propagate debug mode if need be
if (opts.propDebug) props.setDebug(true);

// *All* files in config dir are parsed
props.initialize(defArgs.rootDir + '/config', propHierarchy);

// Now load up our libraries.
const aws = require('./lib/aws');

const awsProps = props.propsFor("aws");

aws.initConfig(awsProps)
    .then(() => {
        function startListening(server) {
            const ss = systemdSocket();
            let _port;
            if (ss) {
                // ms (5 min default)
                const idleTimeout = process.env.IDLE_TIMEOUT;
                const timeout = (typeof idleTimeout !== 'undefined' ? idleTimeout : 300) * 1000;
                if (idleTimeout) {
                    let exit = () => {
                        logger.info("Inactivity timeout reached, exiting.");
                        process.exit(0);
                    };
                    let idleTimer = setTimeout(exit, timeout);
                    let reset = () => {
                        clearTimeout(idleTimer);
                        idleTimer = setTimeout(exit, timeout);
                    };
                    server.all('*', reset);
                    logger.info(`  IDLE_TIMEOUT: ${idleTimeout}`);
                }
                _port = ss;
            } else {
                _port = defArgs.port;
            }
            logger.info(`  Listening on http://${defArgs.hostname || 'localhost'}:${_port}/`);
            logger.info("=======================================");
            server.listen(_port, defArgs.hostname);
        }

        const ravenPrivateEndpoint = aws.getConfig('ravenPrivateEndpoint');
        if (ravenPrivateEndpoint) {
            Raven.config(ravenPrivateEndpoint, {
                release: gitReleaseName,
                environment: defArgs.env
            }).install();
            logger.info("Configured with raven endpoint", ravenPrivateEndpoint);
        } else {
            Raven.config(false).install();
        }


        const webServer = express(),
            morgan = require('morgan'),
            compression = require('compression');

        logger.info("=======================================");
        if (gitReleaseName) logger.info(`  git release ${gitReleaseName}`);

        const healthCheck = require('./lib/handlers/health-check');

        morgan.token('gdpr_ip', req => utils.anonymizeIp(req.ip));

        // Based on combined format, but: GDPR compliant IP, no timestamp & no unused fields for our usecase
        const morganFormat = isDevMode() ? 'dev' : ':gdpr_ip ":method :url" :status';

        webServer
            .use(Raven.requestHandler())
            .set('trust proxy', true)
            // before morgan so healthchecks aren't logged
            .use('/healthcheck', new healthCheck.HealthCheckHandler().handle)
            .use(morgan(morganFormat, {
                stream: logger.stream,
                // Skip for non errors (2xx, 3xx)
                skip: (req, res) => res.statusCode >= 400
            }))
            .use(morgan(morganFormat, {
                stream: logger.warnStream,
                // Skip for non user errors (4xx)
                skip: (req, res) => res.statusCode < 400 || res.statusCode >= 500
            }))
            .use(morgan(morganFormat, {
                stream: logger.errStream,
                // Skip for non server errors (5xx)
                skip: (req, res) => res.statusCode < 500
            }))
            .use(compression())
            .use((req, res, next) => {
                next({status: 404, message: `page "${req.path}" could not be found`});
            })
            .use((err, req, res, next) => {
                Raven.errorHandler()(err, req, res, next);
            })
            .on('error', err => logger.error('Caught error:', err, "(in web handler; continuing)"));
        startListening(webServer);
    })
    .catch(err => {
        logger.error("AWS Init Promise error", err, "(shutting down)");
        process.exit(1);
    });
