// Copyright (c) 2025, Compiler Explorer Authors
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

import fs from 'node:fs';
import process from 'node:process';

import express from 'express';
import PromClient from 'prom-client';
import systemdSocket from 'systemd-socket';

import type {AppArguments} from '../app.interfaces.js';
import {logger} from '../logger.js';

/**
 * Starts the web server listening for connections
 * @param webServer - Express web server
 * @param appArgs - Application arguments
 */
export function startListening(webServer: express.Express, appArgs: AppArguments): void {
    const ss: {fd: number} | null = systemdSocket();
    if (ss) {
        setupSystemdSocketListening(webServer, ss);
    } else {
        setupStandardHttpListening(webServer, appArgs);
    }

    setupStartupMetrics();
}

/**
 * Set up listening on systemd socket
 * @param webServer - Express web server
 * @param ss - Systemd socket
 */
function setupSystemdSocketListening(webServer: express.Express, ss: {fd: number}): void {
    // ms (5 min default)
    const idleTimeout = process.env.IDLE_TIMEOUT;
    const timeout = (idleTimeout === undefined ? 300 : Number.parseInt(idleTimeout, 10)) * 1000;
    if (idleTimeout) {
        setupIdleTimeout(webServer, timeout);
        logger.info(`  IDLE_TIMEOUT: ${idleTimeout}`);
    }
    logger.info(`  Listening on systemd socket: ${JSON.stringify(ss)}`);
    webServer.listen(ss);
}

/**
 * Set up idle timeout for systemd socket
 * @param webServer - Express web server
 * @param timeout - Timeout in milliseconds
 */
function setupIdleTimeout(webServer: express.Express, timeout: number): void {
    const exit = () => {
        logger.info('Inactivity timeout reached, exiting.');
        process.exit(0);
    };
    let idleTimer = setTimeout(exit, timeout);
    const reset = () => {
        clearTimeout(idleTimer);
        idleTimer = setTimeout(exit, timeout);
    };
    webServer.all('*', reset);
}

/**
 * Set up standard HTTP listening
 * @param webServer - Express web server
 * @param appArgs - Application arguments
 */
function setupStandardHttpListening(webServer: express.Express, appArgs: AppArguments): void {
    logger.info(`  Listening on http://${appArgs.hostname || 'localhost'}:${appArgs.port}/`);
    if (appArgs.hostname) {
        webServer.listen(appArgs.port, appArgs.hostname);
    } else {
        webServer.listen(appArgs.port);
    }
}

function getSystemUptimeSeconds(): number | null {
    try {
        const contents = fs.readFileSync('/proc/uptime', 'utf8');
        const systemUptimeSeconds = parseFloat(contents.split(' ')[0]);
        return Number.isFinite(systemUptimeSeconds) ? systemUptimeSeconds : null;
    } catch {
        return null;
    }
}

function setupStartupMetrics(): void {
    const nodeUptimeSeconds = process.uptime();
    const systemUptimeSeconds = getSystemUptimeSeconds();

    // Total time from OS boot to CE serving requests. On Linux, /proc/uptime gives
    // the system uptime; subtracting Node's own uptime gives boot-to-Node-start time.
    // This captures the full startup chain (OS boot → start.sh → Node → CE ready)
    // which is what matters for health check grace period sizing.
    const totalStartupSeconds = systemUptimeSeconds !== null ? systemUptimeSeconds : nodeUptimeSeconds;

    try {
        const startupGauge = new PromClient.Gauge({
            name: 'ce_startup_seconds',
            help: 'Time from OS boot to CE serving requests (uses /proc/uptime on Linux, falls back to Node process uptime)',
        });
        startupGauge.set(totalStartupSeconds);

        if (systemUptimeSeconds !== null) {
            const nodeOnlyGauge = new PromClient.Gauge({
                name: 'ce_node_startup_seconds',
                help: 'Time from Node process start to CE serving requests',
            });
            nodeOnlyGauge.set(nodeUptimeSeconds);
        }
    } catch (err: unknown) {
        const error = err as Error;
        logger.warn(`Error setting up startup metric: ${error.message}`);
    }

    const nodeStartupDurationMs = Math.floor(nodeUptimeSeconds * 1000);
    const totalStartupDurationMs = Math.floor(totalStartupSeconds * 1000);
    logger.info(`  Startup duration (Node): ${nodeStartupDurationMs}ms`);
    if (systemUptimeSeconds !== null) {
        logger.info(`  Startup duration (total from boot): ${totalStartupDurationMs}ms`);
    }
    logger.info('=======================================');
}
