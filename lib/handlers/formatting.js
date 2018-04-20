// Copyright (c) 2018, Compiler Explorer Authors
//
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

const exec = require('../exec'),
    _ = require('underscore'),
    logger = require('../logger').logger;

class Formatter {
    constructor(ceProps) {
        this.tools = {};
        const formatters = _.compact(ceProps('formatters').split(':'));
        _.each(formatters, formatter => {
            const exe = ceProps(`formatter.${formatter}.exe`);
            if (!exe) {
                logger.warn(`Formatter ${formatter} does not have a valid exe. Skipping...`);
                return;
            }
            const versionArg = ceProps(`formatter.${formatter}.version`, '--version');
            const versionRe = ceProps(`formatter.${formatter}.versionRe`, '.*');
            exec.execute(exe, [versionArg], {})
                .then(result => {
                    const match = result.stdout.match(versionRe);
                    this.tools[formatter] = {
                        exe: exe,
                        version: match ? match[0] : result.stdout,
                        name: ceProps(`formatter.${formatter}.name`, exe)
                    };
                });
        });
    }

    formatHandler(req, res) {
        res.setHeader('Content-Type', 'application/json');
        let requestedTool = this.tools[req.params.tool];
        if (!requestedTool) {
            return res.end(JSON.stringify({
                exit: 2,
                answer: "Tool not supported"
            }));
        }
        const args = [];
        // Only clang supported for now
        let style = req.body.base || "Google";  // Default style
        args.push(`-style=${style}`);
        exec.execute(requestedTool.exe, args, {input: req.body.source}).then(result => {
            return res.end(JSON.stringify({
                exit: result.code,
                answer: result.stdout,
            }));
        }).catch(ex => {
            return res.end(JSON.stringify({
                exit: 1,
                thrown: true,
                answer: ex.message
            }));
        });
    }

    listHandler(req, res) {
        res.setHeader('Content-Type', 'application/json');
        const response = [];
        _.each(this.tools, tool => {
            response.push({name: tool.name, version: tool.version});
        });
        return res.end(JSON.stringify(response, null, 4));
    }
}

module.exports = Formatter;
