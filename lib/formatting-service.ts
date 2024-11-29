// Copyright (c) 2024, Compiler Explorer Authors
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

import _ from 'underscore';

import {UnprocessedExecResult} from '../types/execution/execution.interfaces.js';

import * as exec from './exec.js';
import {FormatOptions} from './formatters/base.interfaces.js';
import {BaseFormatter, getFormatterTypeByKey} from './formatters/index.js';
import {logger} from './logger.js';
import {PropertyGetter} from './properties.interfaces.js';

export class FormattingService {
    /** Mapping of the formatter ids to the formatter implementations */
    private registry: Map<string, BaseFormatter> = new Map();

    public getFormatterById(id: string): BaseFormatter | null {
        return this.registry.get(id) || null;
    }

    public getFormatters(): BaseFormatter[] {
        return Array.from(this.registry.values());
    }

    public async format(formatterid: string, source: string, options: FormatOptions): Promise<UnprocessedExecResult> {
        const formatter = this.getFormatterById(formatterid);
        // Ensure the formatter exists
        if (formatter === null) {
            throw new Error(`Formatter ${formatterid} not found`);
        }
        // Ensure the formatting style is valid
        if (!formatter.isValidStyle(options.baseStyle)) {
            throw new Error(`Formatter ${formatterid} does not support style ${options.baseStyle}`);
        }
        return await formatter.format(source, options);
    }

    public async initialize(ceProps: PropertyGetter): Promise<void> {
        const formatters = _.compact(ceProps('formatters', '').split(':'));
        for (const formatter of formatters) {
            logger.info(`Performing discovery of formatter named ${formatter}`);
            const executable = ceProps<string>(`formatter.${formatter}.exe`);
            const type = ceProps<string>(`formatter.${formatter}.type`);
            if (!executable) {
                logger.warn(`Formatter ${formatter} does not have a valid executable. Skipping...`);
                continue;
            }
            if (!type) {
                logger.warn(`Formatter ${formatter} does not have a valid type. Skipping...`);
                continue;
            }
            const versionArgument = ceProps<string>(`formatter.${formatter}.version`, '--version');
            const versionRegExp = ceProps<string>(`formatter.${formatter}.versionRe`, '.*');
            const hasExplicitVersion = ceProps(`formatter.${formatter}.explicitVersion`, '') !== '';
            try {
                const result = await exec.execute(executable, [versionArgument], {});
                const match = result.stdout.match(versionRegExp);
                const formatterClass = getFormatterTypeByKey(type);
                const styleList = ceProps<string>(`formatter.${formatter}.styles`, '');
                const styles = styleList === '' ? [] : styleList.split(':');
                // If there is an explicit version, grab it. Otherwise, try to filter the output
                const version = hasExplicitVersion
                    ? ceProps<string>(`formatter.${formatter}.explicitVersion`)
                    : match
                      ? match[0]
                      : result.stdout;
                const instance = new formatterClass({
                    name: ceProps(`formatter.${formatter}.name`, executable),
                    exe: executable,
                    version,
                    styles,
                    type,
                });
                this.registry.set(formatter, instance);
            } catch (err: unknown) {
                logger.warn(`Error while fetching tool info for ${executable}:`, {err});
            }
        }
    }
}
