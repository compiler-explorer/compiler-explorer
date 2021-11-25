// Copyright (c) 2021, Compiler Explorer Authors
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

/**
 * @typedef {object} FormatterInfo - describes a formatter for a language
 * @property {string} name - the name (unique across all formatters) for a language
 * @property {string} exe - file path or name of tool executable
 * @property {string[]} styles - list of preset styles for the tool, for example [Google, Microsoft] for Clang-Format
 * @property {string} type - which Formatter class tool execution should be delegated to
 * @property {string} explicitVersion - a hard override for version of the tool
 * @property {string} versionArgument - command line argument for tool to retrieve version information
 * @property {string} versionReExp - regular expression to capture version from version output
 */

import * as exec from './exec';
import { getFormatterTypeByKey } from './formatters';
import { logger } from './logger';

/**
 * Service for querying the properties to locate code formatters for the
 * languages supported by a Compiler Explorer instance
 *
 * The formatter finder has been designed to allow each programming language to
 * have multiple formatters. The Compiler Explorer frontend, however does not
 * have a way to query the different formatters at the moment.
 *
 * Formatters are specified in the .properties files with the following format:
 *
 * ```properties
 * formatters=clangformat:gnuindent
 * formatter.clangformat.name=clang-format
 * formatter.clangformat.exe=/usr/bin/clang-format
 * formatter.clangformat.type=clangformat
 * formatter.clangformat.styles=Google:LLVM:Mozilla:Chromium:WebKit:Microsoft:GNU
 * # Optionally provide
 * formatter.clangformat.explicitVersion=clang-format 13.x.x
 * formatter.versionArgument=--version
 * formatter.versionRegExp=.*
 * ```
 *
 * @property {CompilerProps} compilerProps
 * @property {string[]} languages
 */
export class FormatterFinder {
    /** @param {CompilerProps} compilerProps */
    constructor(compilerProps) {
        this.compilerProps = compilerProps;
        this.languages = Object.keys(compilerProps.languages);
    }
    /**
     * Find all the formatters
     *
     * If no formatters were found, an empty array is returned.
     *
     * @returns {Promise<FormatterInfo[]>}
     */
    async getFormatters() {
        const matrix = await Promise.all(this.languages.map(language => this.getFormatterForLanguage(language)));
        return matrix.flat();
    }

    /**
     * Get the formatters for a programming language
     *
     * If no formatters were found, an empty array is returned
     *
     * @param {string} language
     * @returns {Promise<FormatterInfo[]>}
     */
    async getFormattersForLanguage(language) {
        // If the language is not loaded into CE, we do not want to do anything
        if (!this.languages.includes(language)) return [];
        const props = this.compilerProps.propsByLangId[language];
        const engines = props('formatters', '').split(':');
        const formatters = [];
        // For each of the registered formatters for the language
        for (const engine of engines) {
            const exe = props(`formatter.${engine}.exe`);
            const type = props(`formatter.${engine}.type`);
            if (!exe) {
                logger.warn(`Formatter ${engine} does not have a valid executable. Skipping...`);
                continue;
            }
            if (!type) {
                logger.warn(`Formatter ${engine} does not have a formatter class. Skipping...`);
                continue;
            }
            const versionArg = props(`formatter.${engine}.versionArgument`, '--version');
            const versionRe = props(`formatter.${engine}.versionRegExp`, '.*');
            const hasExplicitVersion = props(`formatter.${engine}.explicitVersion`, '') !== '';
            try {
                const result = await exec.execute(exe, [versionArg], {});
                const match = result.stdout.match(versionRe);
                const formatterClass = getFormatterTypeByKey(type);
                const styleList = props(`formatter.${engine}.styles`);
                const styles = styleList === '' ? [] : styleList.split(':');
                // If there is an explicit version, grab it. Otherwise try to filter the output
                const version = hasExplicitVersion
                    ? props(`formatter.${engine}.explicitVersion`)
                    : (match ? match[0] : result.stdout);
                const instance = new formatterClass({
                    name: props(`formatter.${engine}.name`, exe),
                    exe,
                    version,
                    styles,
                    type,
                });
                formatters.push(instance);
            } catch (err) {
                logger.warn(`Error while fetching tool info for ${exe}:`, {err});
            }
        }
        return formatters;
    }
}
