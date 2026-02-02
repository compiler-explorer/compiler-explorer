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

import os from 'node:os';
import path from 'node:path';
import process from 'node:process';

import PromClient from 'prom-client';
import urljoin from 'url-join';

import type {Language, LanguageKey} from '../../types/languages.interfaces.js';
import {AppArguments} from '../app.interfaces.js';
import {languages as allLanguages} from '../languages.js';
import {logger} from '../logger.js';
import type {PropertyGetter} from '../properties.interfaces.js';
import * as props from '../properties.js';
import type {AppConfiguration} from './config.interfaces.js';

/**
 * Measures event loop lag to monitor server performance.
 * Used to detect when the server is under heavy load or not responding quickly.
 * @param delayMs - The delay in milliseconds to measure against
 * @returns The lag in milliseconds
 */
export function measureEventLoopLag(delayMs: number): Promise<number> {
    return new Promise<number>(resolve => {
        const start = process.hrtime.bigint();
        setTimeout(() => {
            const elapsed = process.hrtime.bigint() - start;
            const delta = elapsed - BigInt(delayMs * 1000000);
            return resolve(Number(delta) / 1000000);
        }, delayMs);
    });
}

/**
 * Creates the property hierarchy for configuration loading
 */
export function createPropertyHierarchy(env: string[], useLocalProps: boolean): string[] {
    const propHierarchy = [
        'defaults',
        env,
        env.map(e => `${e}.${process.platform}`),
        process.platform,
        os.hostname(),
    ].flat();

    if (useLocalProps) {
        propHierarchy.push('local');
    }

    logger.info(`properties hierarchy: ${propHierarchy.join(', ')}`);
    return propHierarchy;
}

/**
 * Filter languages based on wanted languages from configuration
 */
export function filterLanguages(
    wantedLanguages: string[] | undefined,
    existingLanguages: Record<LanguageKey, Language>,
): Record<LanguageKey, Language> {
    if (wantedLanguages) {
        const filteredLangs: Partial<Record<LanguageKey, Language>> = {};
        for (const wantedLang of wantedLanguages) {
            for (const lang of Object.values(existingLanguages)) {
                if (lang.id === wantedLang || lang.name === wantedLang || lang.alias.includes(wantedLang)) {
                    filteredLangs[lang.id] = lang;
                }
            }
        }
        // Always keep cmake for IDE mode, just in case
        filteredLangs[existingLanguages.cmake.id] = existingLanguages.cmake;
        return filteredLangs as Record<LanguageKey, Language>;
    }
    return existingLanguages;
}

/**
 * Configure event loop lag monitoring
 */
export function setupEventLoopLagMonitoring(ceProps: PropertyGetter): void {
    const lagIntervalMs = ceProps('eventLoopMeasureIntervalMs', 0);
    const thresWarn = ceProps('eventLoopLagThresholdWarn', 0);
    const thresErr = ceProps('eventLoopLagThresholdErr', 0);

    let totalLag = 0;
    const ceLagSecondsTotalGauge = new PromClient.Gauge({
        name: 'ce_lag_seconds_total',
        help: 'Total event loop lag since application startup',
    });

    async function eventLoopLagHandler() {
        const lagMs = await measureEventLoopLag(lagIntervalMs);
        totalLag += Math.max(lagMs / 1000, 0);
        ceLagSecondsTotalGauge.set(totalLag);

        if (thresErr && lagMs >= thresErr) {
            logger.error(`Event Loop Lag: ${lagMs} ms`);
        } else if (thresWarn && lagMs >= thresWarn) {
            logger.warn(`Event Loop Lag: ${lagMs} ms`);
        }

        setImmediate(eventLoopLagHandler);
    }

    // Only setup monitoring if interval is set
    if (lagIntervalMs > 0) {
        setImmediate(eventLoopLagHandler);
    }
}

/**
 * Load and initialize application configuration
 */
export function loadConfiguration(appArgs: AppArguments): AppConfiguration {
    // Set up property debugging if needed
    if (appArgs.propDebug) {
        props.setDebug(true);
    }

    // Create property hierarchy based on environment
    const propHierarchy = createPropertyHierarchy(appArgs.env, appArgs.useLocalProps);

    // Initialize properties from config directory
    const configDir = path.join(appArgs.rootDir, 'config');
    props.initialize(configDir, propHierarchy);

    // Get compiler explorer properties
    const ceProps = props.propsFor('compiler-explorer');

    // Check for restricted languages
    const restrictToLanguages = ceProps<string>('restrictToLanguages');
    if (restrictToLanguages) {
        appArgs.wantedLanguages = restrictToLanguages.split(',');
    }

    // Filter languages based on wanted languages
    const languages = filterLanguages(appArgs.wantedLanguages, allLanguages);

    // Set up compiler properties
    const compilerProps = new props.CompilerProps(languages, ceProps);

    // Load environment settings
    const staticMaxAgeSecs = ceProps('staticMaxAgeSecs', 0);
    const maxUploadSize = ceProps('maxUploadSize', '1mb');
    const extraBodyClass = ceProps('extraBodyClass', appArgs.devMode ? 'dev' : '');
    const storageSolution = compilerProps.ceProps('storageSolution', 'local');
    const httpRoot = urljoin(ceProps('httpRoot', '/'), '/');

    const staticUrl = ceProps<string | undefined>('staticUrl');
    const staticRoot = urljoin(staticUrl || httpRoot, '/');

    return {
        ceProps,
        compilerProps,
        languages,
        staticMaxAgeSecs,
        maxUploadSize,
        extraBodyClass,
        storageSolution,
        httpRoot,
        staticRoot,
        staticUrl,
    };
}
