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

import express from 'express';
import _ from 'underscore';

import * as exec from '../exec';
import {BaseFormatter, getFormatterTypeByKey} from '../formatters';
import {logger} from '../logger';
import {PropertyGetter} from '../properties.interfaces';

export class FormattingHandler {
    private formatters: Record<string, BaseFormatter> = {};

    public constructor(private ceProps: PropertyGetter) {
        const formatters = _.compact(ceProps('formatters', '').split(':'));
        for (const formatter of formatters) {
            this.populateFormatterInfo(formatter);
        }
    }

    private async populateFormatterInfo(formatterName: string): Promise<void> {
        const exe = this.ceProps<string>(`formatter.${formatterName}.exe`);
        const type = this.ceProps<string>(`formatter.${formatterName}.type`);
        if (!exe) {
            logger.warn(`Formatter ${formatterName} does not have a valid executable. Skipping...`);
            return;
        }
        if (!type) {
            logger.warn(`Formatter ${formatterName} does not have a formatter class. Skipping...`);
            return;
        }
        const versionArgument = this.ceProps<string>(`formatter.${formatterName}.version`, '--version');
        const versionRegExp = this.ceProps<string>(`formatter.${formatterName}.versionRe`, '.*');
        const hasExplicitVersion = this.ceProps(`formatter.${formatterName}.explicitVersion`, '') !== '';
        try {
            const result = await exec.execute(exe, [versionArgument], {});
            const match = result.stdout.match(versionRegExp);
            const formatterClass = getFormatterTypeByKey(type);
            const styleList = this.ceProps<string>(`formatter.${formatterName}.styles`);
            const styles = styleList === '' ? [] : styleList.split(':');
            // If there is an explicit version, grab it. Otherwise try to filter the output
            const version = hasExplicitVersion
                ? this.ceProps<string>(`formatter.${formatterName}.explicitVersion`)
                : match
                ? match[0]
                : result.stdout;
            this.formatters[formatterName] = new formatterClass({
                name: this.ceProps(`formatter.${formatterName}.name`, exe),
                exe,
                version,
                styles,
                type,
            });
        } catch (err: unknown) {
            logger.warn(`Error while fetching tool info for ${exe}:`, {err});
        }
    }

    public getFormatterInfo() {
        return Object.values(this.formatters).map(formatter => formatter.formatterInfo);
    }

    public async handle(req: express.Request, res: express.Response): Promise<void> {
        const name = req.params.tool;
        const formatter = this.formatters[name];
        // Ensure the formatter exists
        if (!formatter) {
            res.status(422).send({
                exit: 2,
                answer: `Unknown format tool '${name}'`,
            });
            return;
        }
        // Ensure there is source code to format
        if (!req.body || !req.body.source) {
            res.send({exit: 0, answer: ''});
            return;
        }
        // Ensure the wanted style is valid for the formatter
        const style = req.body.base;
        if (!formatter.isValidStyle(style)) {
            res.status(422).send({
                exit: 3,
                answer: `Style '${style}' is not supported`,
            });
            return;
        }
        try {
            // Perform the actual formatting
            const result = await formatter.format(req.body.source, {
                useSpaces: req.body.useSpaces === undefined ? true : req.body.useSpaces,
                tabWidth: req.body.tabWidth === undefined ? 4 : req.body.tabWidth,
                baseStyle: req.body.base,
            });
            res.send({
                exit: result.code,
                answer: result.stdout || result.stderr || '',
            });
        } catch (err: unknown) {
            res.status(500).send({
                exit: 1,
                thrown: true,
                answer:
                    (err && Object.hasOwn(err, 'message') && (err as Record<'message', 'string'>).message) ||
                    'Internal server error',
            });
        }
    }

    async internalFormat(formatterName: string, style: string, source: string): Promise<[string, string]> {
        const formatter = this.formatters[formatterName];
        // Ensure the formatter exists
        if (!formatter) {
            throw new Error('Unknown formatter name');
        }
        // Ensure the wanted style is valid for the formatter
        if (!formatter.isValidStyle(style)) {
            throw new Error('Unsupported formatter style');
        }
        // Perform the actual formatting
        const result = await formatter.format(source, {
            useSpaces: true, // hard coded for now, TODO should this be changed?
            tabWidth: 4,
            baseStyle: style,
        });
        return [result.stdout || '', result.stderr];
    }
}
