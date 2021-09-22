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

import _ from 'underscore';

import * as exec from '../exec';
import { getFormatterTypeByKey } from '../formatters';
import { logger } from '../logger';

export class FormattingHandler {
    constructor(ceProps) {
        this.formatters = {};
        this.ceProps = ceProps;
        const formatters = _.compact(ceProps('formatters', '').split(':'));
        _.each(formatters, this.getFormatterInfo.bind(this));
    }

    async getFormatterInfo(formatter) {
        const exe = this.ceProps(`formatter.${formatter}.exe`);
        const type = this.ceProps(`formatter.${formatter}.type`);
        if (!exe) {
            return logger.warn(`Formatter ${formatter} does not have a valid executable. Skipping...`);
        }
        if (!type) {
            return logger.warn(`Formatter ${formatter} does not have a formatter class. Skipping...`);
        }
        const versionArg = this.ceProps(`formatter.${formatter}.version`, '--version');
        const versionRe = this.ceProps(`formatter.${formatter}.versionRe`, '.*');
        try {
            const result = await exec.execute(exe, [versionArg], {});
            const match = result.stdout.match(versionRe);
            const formatterClass = getFormatterTypeByKey(type);
            const styleList = this.ceProps(`formatter.${formatter}.styles`);
            const styles = styleList === '' ? [] : styleList.split(':');
            this.formatters[formatter] = new formatterClass({
                exe: exe,
                version: match ? match[0] : result.stdout,
                name: this.ceProps(`formatter.${formatter}.name`, exe),
                styles,
                type,
            });
        } catch (err) {
            logger.warn(`Error while fetching tool info for ${exe}:`, {err});
        }
    }

    async handle(req, res) {
        const name = req.params.tool;
        const formatter = this.formatters[name];
        // Ensure the formatter exists
        if (!formatter) {
            return res.status(422).send({
                exit: 2,
                answer: `Unknown format tool '${name}'`,
            });
        }
        // Ensure there is source code to format
        if (!req.body || !req.body.source) {
            return res.send({exit: 0, answer: ''});
        }
        // Ensure the wanted style is valid for the formatter
        const style = req.body.base;
        if (!formatter.isValidStyle(style)) {
            return res.status(422).send({
                exit: 3,
                answer: `Style '${style}' is not supported`,
            });
        }
        try {
            // Perform the actual formatting
            const result = await formatter.format(req.body.source, {
                useSpaces: req.body.useSpaces,
                tabWidth: req.body.tabWidth,
                baseStyle: req.body.base,
            });
            res.send({
                exit: result.code,
                answer: result.stdout || '',
            });
        } catch (err) {
            res.status(500).send({
                exit: 1,
                thrown: true,
                answer: err.message || 'Internal server error',
            });
        }
    }
}
