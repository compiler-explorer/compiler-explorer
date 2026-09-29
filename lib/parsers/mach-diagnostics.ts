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

import type {Fix, ResultLine, ResultLineTag} from '../../types/resultline/resultline.interfaces.js';
import {eachLine} from '../utils.js';

/** A span of mach's diagnostic records (doc/language/diagnostics-json.md): 1-based lines, UTF-8 byte columns. */
type Span = {
    file: string;
    line: number;
    column: number;
    end_line: number;
    end_column: number;
    label?: string | null;
};

type Edit = Span & {replacement: string};

/** A `diagnostic` or `failure` record; a failure has the same members with no notes, help or fixes. */
type Diagnostic = {
    record: 'diagnostic' | 'failure';
    severity: string;
    code: string;
    message: string;
    primary: Span | null;
    related: Span[];
    notes: string[];
    help: string[];
    fixes: {label: string; edits: Edit[]}[];
};

type Summary = {record: 'summary'; errors: number; warnings: number};

/** A tag's location in an editor: the file by its path from `src/`, and a range in the file's own columns. */
type Location = {file: string; line: number; column: number; endline?: number; endcolumn?: number};

/** The text of one line of a source file, by its path from `src/`, or undefined when it cannot be read. */
export type MachSourceLine = (file: string, line: number) => string | undefined;

const severities: Record<string, number> = {error: 3, warning: 2, note: 1};

/** The one schema this reader knows; a record of any other schema is shown as the line mach wrote. */
const schema = 1;

function readRecord(line: string): Diagnostic | Summary | {record: string} | undefined {
    if (!line.startsWith('{')) return undefined;
    try {
        const record = JSON.parse(line);
        return record?.schema === schema && typeof record.record === 'string' ? record : undefined;
    } catch {
        return undefined;
    }
}

function plural(count: number, noun: string) {
    return `${count} ${noun}${count === 1 ? '' : 's'}`;
}

/**
 * Where a span belongs, if in an editor at all. Spans name files from the project root, so a user source is
 * `src/<path>`, which the project tree names `<path>`. Anything else, such as std under `dep/`, the generated
 * `mach.toml` or an absolute path, belongs to no editor. An empty span keeps no end, so the editor marks the token
 * that starts there.
 */
function locate(span: Span | null | undefined): Location | undefined {
    if (!span) return undefined;
    const file = span.file.split(path.sep).join('/');
    if (!file.startsWith('src/') || file.split('/').includes('..')) return undefined;
    const location: Location = {file: file.slice('src/'.length), line: span.line, column: span.column};
    if (span.end_line !== span.line || span.end_column !== span.column) {
        location.endline = span.end_line;
        location.endcolumn = span.end_column;
    }
    return location;
}

/** A fix is offered only where it can be applied whole: every edit in the editor its diagnostic marks. */
function quickFixes(record: Diagnostic, primary: Location): Fix[] {
    const fixes: Fix[] = [];
    for (const fix of record.fixes) {
        const edits = fix.edits.map(edit => ({edit, location: locate(edit)}));
        if (!edits.every(({location}) => location?.file === primary.file)) continue;
        fixes.push({
            title: fix.label,
            edits: edits.map(({edit, location}) => ({
                ...location!,
                endline: edit.end_line,
                endcolumn: edit.end_column,
                text: edit.replacement,
            })),
        });
    }
    return fixes;
}

/**
 * One diagnostic, rendered the way mach's own text rendering heads it, without the source excerpts the editor
 * already shows:
 *
 *   error[name.duplicate]: duplicate definition: ...     marked at the primary span, notes and help in its message
 *    --> src/example.mach:2:5
 *    --> src/example.mach:1:5: previous definition here  each related span, marked with its label
 *     = note: <text> / = help: <text>
 *     = fix: <label>                                     offered as a quick fix on the headline's marker
 */
function renderDiagnostic(record: Diagnostic): ResultLine[] {
    const severity = severities[record.severity] ?? 3;
    const headline = `${record.severity}[${record.code}]: ${record.message}`;
    const lines: ResultLine[] = [{text: headline}];

    const primary = locate(record.primary);
    if (primary) {
        const text = [
            headline,
            ...record.notes.map(note => `note: ${note}`),
            ...record.help.map(help => `help: ${help}`),
        ].join('\n');
        const fixes = quickFixes(record, primary);
        lines[0].tag = {...primary, text, severity, ...(fixes.length > 0 && {fixes})};
    }
    if (record.primary)
        lines.push({text: ` --> ${record.primary.file}:${record.primary.line}:${record.primary.column}`});

    for (const span of record.related) {
        const where = ` --> ${span.file}:${span.line}:${span.column}`;
        const line: ResultLine = {text: span.label ? `${where}: ${span.label}` : where};
        const location = locate(span);
        // a related site shown without a label is marked with what the diagnostic says about it
        if (location) line.tag = {...location, text: span.label ?? headline, severity: 1};
        lines.push(line);
    }

    for (const note of record.notes) lines.push({text: `  = note: ${note}`});
    for (const help of record.help) lines.push({text: `  = help: ${help}`});
    for (const fix of record.fixes) lines.push({text: `  = fix: ${fix.label}`});
    return lines;
}

/**
 * Reads what `mach build --diagnostics=json` writes into result lines. Each diagnostic and failure record becomes a
 * rendered headline marked at its primary span, and the closing summary becomes mach's tally. A line that is not a
 * record, such as a build step's own output, is shown as written, with paths from the project root. Columns are
 * mach's UTF-8 byte columns; `toEditorColumns` turns them into the editor's.
 */
export function parseMachDiagnostics(output: string, inputFilename?: string): ResultLine[] {
    const projectRoot = inputFilename ? path.dirname(path.dirname(inputFilename)) : undefined;
    const result: ResultLine[] = [];
    eachLine(output, line => {
        const record = readRecord(line);
        if (!record) {
            result.push({text: projectRoot ? line.split(projectRoot + path.sep).join('') : line});
        } else if (record.record === 'diagnostic' || record.record === 'failure') {
            result.push(...renderDiagnostic(record as Diagnostic));
        } else if (record.record === 'summary') {
            const {errors, warnings} = record as Summary;
            if (errors + warnings > 0)
                result.push({text: `${plural(errors, 'error')} / ${plural(warnings, 'warning')}`});
        }
        // any other record, such as a test result, is not a diagnostic
    });
    return result;
}

/** The UTF-16 column of a 1-based UTF-8 byte column in a line, which is how the editor counts. */
function utf16Column(text: string | undefined, column: number): number {
    if (text === undefined || !/[^\x00-\x7f]/.test(text)) return column;
    return (
        Buffer.from(text, 'utf8')
            .subarray(0, column - 1)
            .toString('utf8').length + 1
    );
}

function remap<T extends {file?: string; line?: number; column?: number; endline?: number; endcolumn?: number}>(
    location: T,
    sourceLine: MachSourceLine,
): T {
    if (location.file === undefined || location.line === undefined) return location;
    const {file} = location;
    const remapped = {...location, column: utf16Column(sourceLine(file, location.line), location.column ?? 1)};
    if (location.endline !== undefined && location.endcolumn !== undefined)
        remapped.endcolumn = utf16Column(sourceLine(file, location.endline), location.endcolumn);
    return remapped;
}

/**
 * Turns every marker's byte columns into the editor's UTF-16 columns, reading each line from the sources. A quick
 * fix's edits are remapped too, since a wrong column there would edit the wrong characters.
 */
export function toEditorColumns(lines: ResultLine[], sourceLine: MachSourceLine): ResultLine[] {
    return lines.map(line => {
        if (!line.tag) return line;
        const tag: ResultLineTag = remap(line.tag, sourceLine);
        if (tag.fixes) {
            tag.fixes = tag.fixes.map(fix => ({...fix, edits: fix.edits.map(edit => remap(edit, sourceLine))}));
        }
        return {...line, tag};
    });
}
