// Copyright (c) 2026, Compiler Explorer Authors
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

import path from 'node:path';

import type {ResultLine, ResultLineTag} from '../../types/resultline/resultline.interfaces.js';
import {eachLine, filterEscapeSequences} from '../utils.js';

type Location = {file: string; line: number; column: number};

/**
 * Where a location line sits in a diagnostic, which decides what it marks:
 *
 * - `primary`: the location right after the headline, marked with the headline
 * - `related`: a location after a gutter bar, marked with the label under its snippet
 * - `edit`: a location inside a `= fix:` trailer, marked with the replacement the fix makes there
 */
type Frame = {kind: 'primary' | 'related' | 'edit'; location?: Location};

const severities: Record<string, number> = {error: 3, warning: 2, info: 1, help: 1};

const headlineRe = /^(?<severity>error|warning|info|help): (?<message>.*)$/;
const locationRe = /^\s*--> (?<file>.+):(?<line>\d+):(?<column>\d+)$/;
const barRe = /^\s*\|$/;
// the last underline of a frame, with its label: `  |     --- previous definition here`
const labelRe = /^\s*\| [\t ]*[\^-]+ (?<label>.+)$/;
const trailerRe = /^\s*= (?<tag>note|help|fix): (?<text>.*)$/;
const replacementRe = /^\s*-> replace with `(?<replacement>.*)`$/;

/**
 * Reads the diagnostics `mach build` renders (mach.cli.diagnostic) into result lines, tagging each location a marker
 * belongs at. The layout of one diagnostic is:
 *
 *   <severity>: <message>                       error, warning, info or help
 *    --> <file>:<line>:<col>                    the primary location, if the diagnostic has one
 *     |  <snippet>
 *     |                                         a gutter bar, then each related location with a labelled snippet
 *    --> <file>:<line>:<col>
 *     |  <snippet, last underline labelled>
 *     = note: <text> / = help: <text>           trailers, folded into the headline's marker
 *     = fix: <label>                            a fix, then each of its edits
 *    --> <file>:<line>:<col>
 *       -> replace with `<text>`
 *
 * Files are named by their path from the directory the main source was written to, which is the name a tree pane
 * gives them. A file outside that directory, such as std's, is left unmarked: it belongs to no editor. Paths in the
 * text are shown relative to the project root, the directory above the sources.
 */
export function parseMachDiagnostics(output: string, inputFilename?: string): ResultLine[] {
    const sourceRoot = inputFilename ? path.dirname(inputFilename) : undefined;
    const projectRoot = sourceRoot ? path.dirname(sourceRoot) : undefined;

    const result: ResultLine[] = [];
    let headline: ResultLineTag | undefined;
    let severity = 3;
    let frame: Frame | undefined;
    let fix: string | undefined;
    let afterHeadline = false;
    let afterBar = false;

    const locate = (file: string, line: string, column: string): Location | undefined => {
        const name = sourceRoot
            ? path.relative(sourceRoot, path.resolve(sourceRoot, file)).split(path.sep).join('/')
            : file;
        if (name === '..' || name.startsWith('../') || path.isAbsolute(name)) return undefined;
        return {file: name, line: Number.parseInt(line, 10), column: Number.parseInt(column, 10)};
    };

    eachLine(output, raw => {
        // locations are read from the line as mach wrote it, and shown relative to the project root
        const plain = filterEscapeSequences(raw);
        const lineObj: ResultLine = {text: projectRoot ? raw.split(projectRoot + path.sep).join('') : raw};

        const headlineMatch = plain.match(headlineRe);
        const locationMatch = plain.match(locationRe);
        const trailerMatch = plain.match(trailerRe);

        if (headlineMatch?.groups) {
            severity = severities[headlineMatch.groups.severity];
            headline = undefined;
            frame = undefined;
            fix = undefined;
            result.push(lineObj);
            afterHeadline = true;
            afterBar = false;
            return;
        }

        if (locationMatch?.groups) {
            const {file, line, column} = locationMatch.groups;
            const location = locate(file, line, column);
            if (afterHeadline) {
                frame = {kind: 'primary', location};
                const previous = result[result.length - 1];
                if (location) {
                    headline = {...location, text: filterEscapeSequences(previous.text), severity};
                    previous.tag = headline;
                }
            } else if (fix !== undefined) {
                frame = {kind: 'edit', location};
            } else if (afterBar) {
                frame = {kind: 'related', location};
            } else {
                frame = undefined;
            }
            // the location line itself links to its place, with no text of its own to show
            if (frame && location) lineObj.tag = {...location, text: '', severity};
        } else if (trailerMatch?.groups) {
            const {tag, text: trailer} = trailerMatch.groups;
            frame = undefined;
            if (tag === 'fix') {
                fix = trailer;
            } else if (headline) {
                headline.text += `\n${tag}: ${trailer}`;
            }
        } else if (frame?.kind === 'related' && frame.location) {
            const label = plain.match(labelRe);
            if (label?.groups) lineObj.tag = {...frame.location, text: label.groups.label, severity: 1};
        } else if (frame?.kind === 'edit' && frame.location && fix !== undefined) {
            const replacement = plain.match(replacementRe);
            if (replacement?.groups) {
                const edit = `replace with \`${replacement.groups.replacement}\``;
                // a one-edit fix is usually labelled with its edit already
                const text = fix === edit ? `fix: ${fix}` : `fix: ${fix} (${edit})`;
                lineObj.tag = {...frame.location, text, severity: 1};
                frame = undefined;
            }
        } else if (plain.trim() === '') {
            headline = undefined;
            frame = undefined;
            fix = undefined;
        }

        afterHeadline = false;
        afterBar = barRe.test(plain);
        result.push(lineObj);
    });
    return result;
}
