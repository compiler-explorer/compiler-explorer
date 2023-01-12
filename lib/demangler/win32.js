// Copyright (c) 2018, Microsoft Corporation
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

import {logger} from '../logger';
import * as utils from '../utils';

import {CppDemangler} from './cpp';

export class Win32Demangler extends CppDemangler {
    static get key() {
        return 'win32';
    }

    constructor(demanglerExe, compiler) {
        super(demanglerExe, compiler);

        // 0x28090 stands for:
        //   - 0x00010 : Disable expansion of the declaration language specifier
        //   - 0x00080 : Disable expansion of access specifiers for members
        //   - 0x08000 : Disable enum/class/struct/union prefix
        //   - 0x20000 : Disable expansion of __ptr64 keyword
        this.flags = '0x28090';
        this.allDecoratedLabels = /\?[$?@A-Z_a-z][\w$<>?@]*/g;
        this.allDecoratedLabelsWithQuotes = /"\?[$?@A-Z_a-z][\w$<>?@]*"/;

        // this is true for clang output on windows
        // we set this to true if we see it, and false if we don't
        // null means we haven't set it yet.
        this.hasQuotesAroundDecoratedLabels = null;
    }

    collectLabels() {
        this.win32RawSymbols = [];
        for (const asmLine of this.result.asm) {
            const labels = asmLine.text.match(this.allDecoratedLabels);
            if (labels) {
                if (this.hasQuotesAroundDecoratedLabels === null) {
                    this.hasQuotesAroundDecoratedLabels =
                        asmLine.text.match(this.allDecoratedLabelsWithQuotes) !== null;
                }
                for (const label of labels) {
                    this.win32RawSymbols.push(label);
                }
            }
        }
    }

    processOutput(translations) {
        for (const asmLine of this.result.asm) {
            const labels = this.hasQuotesAroundDecoratedLabels
                ? asmLine.text.match(this.allDecoratedLabelsWithQuotes)
                : asmLine.text.match(this.allDecoratedLabels);
            if (labels) {
                let [, beforeComment, afterComment] = asmLine.text.match(/(.*)(;.*)?/);
                for (const label of labels) {
                    const replacement = translations[label];
                    if (replacement) {
                        beforeComment = beforeComment.replace(label, replacement);
                    } else {
                        logger.warn(`something went wrong: ${label} doesn't have an undecoration.`);
                    }
                }
                asmLine.text = beforeComment + (afterComment || '');
            }
        }

        return this.result;
    }

    async execDemangler() {
        const translations = {};

        const demangleSingleSet = async names => {
            const args = [this.flags, ...names];
            const output = await this.compiler.exec(this.demanglerExe, args, this.compiler.getDefaultExecOptions());
            const outputArray = utils.splitLines(output.stdout);

            for (let i = 0; i < outputArray.length; ++i) {
                let tmp = outputArray[i].match(/^Undecoration of :- "(.*)"/);
                if (tmp) {
                    const decoratedName = tmp[1];
                    ++i;
                    tmp = outputArray[i].match(/^is :- "(.*)"/);
                    if (tmp) {
                        if (this.hasQuotesAroundDecoratedLabels) {
                            translations[`"${decoratedName}"`] = tmp[1];
                        } else {
                            translations[decoratedName] = tmp[1];
                        }
                    } else {
                        logger.error(`Broken undname: ${outputArray[i - 1]}, ${outputArray[i]}`);
                    }
                }
            }
        };

        // give some space for `undname` as well as the flag
        // should probably be done more correctly
        const maxCommandLineLength = 8000;
        const commandLineArray = [];
        this.win32RawSymbols.sort();

        let lastSymbol = null;
        let currentLength = 0;
        let currentArray = [];
        for (const symb of this.win32RawSymbols) {
            if (symb === lastSymbol) {
                continue;
            }
            lastSymbol = symb;
            // note: plus one for the space after an argument
            if (currentLength + symb.length + 1 < maxCommandLineLength) {
                currentArray.push(symb);
                currentLength += symb.length + 1;
            } else {
                commandLineArray.push(currentArray);
                currentLength = symb.length + 1;
                currentArray = [symb];
            }
        }
        if (currentArray.length > 0) {
            commandLineArray.push(currentArray);
        }

        await Promise.all(commandLineArray.map(demangleSingleSet));
        return translations;
    }

    async process(result) {
        if (!this.demanglerExe) {
            logger.error("Attempted to demangle, but there's no demangler set");
            return result;
        }

        this.result = result;

        this.collectLabels();
        return this.processOutput(await this.execDemangler());
    }
}
