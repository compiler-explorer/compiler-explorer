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

/** A span of mach's report records (doc/language/report-records.md): 1-based lines, UTF-8 byte columns. */
type Span = {
    file: string;
    line: number;
    column: number;
    end_line: number;
    end_column: number;
    label?: string | null;
};

type Edit = Span & {replacement: string};

type Diagnostic = {
    record: 'diagnostic';
    severity: string;
    code: string;
    message: string;
    origin?: string;
    primary: Span | null;
    related: Span[];
    notes: string[];
    help: string[];
    fixes: {label: string; edits: Edit[]}[];
};

type UnitRecord = {
    record: 'unit';
    verb: string;
    artifact: string;
    target: string;
    profile?: string;
};

type PhaseRecord = {
    record: 'phase';
    name: string;
    parent?: string;
    count?: number;
    unit?: string;
    time_us: number;
};

type Summary = {record: 'summary'; errors: number; warnings: number};

type Location = {file: string; line: number; column: number; endline?: number; endcolumn?: number};

export type MachSourceLine = (file: string, line: number) => string | undefined;

const severities: Record<string, number> = {error: 3, warning: 2, note: 1};

/** Schema version emitted by Mach 7's `--report json` stream. */
const schema = 3;

function readRecord(line: string): Diagnostic | UnitRecord | PhaseRecord | Summary | {record: string} | undefined {
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

function maskSpanFile(rawFile: string, projectRoot?: string, stdPath?: string): string {
    let file = rawFile;
    if (projectRoot && file.startsWith(projectRoot + path.sep)) {
        file = file.slice(projectRoot.length + 1);
    }
    const slashFile = file.split(path.sep).join('/').replace(/^\.\//, '');
    if (stdPath) {
        const slashStd = stdPath.split(path.sep).join('/').replace(/\/+$/, '');
        if (slashFile.startsWith(`${slashStd}/`)) {
            return `std/${slashFile.slice(slashStd.length + 1)}`;
        }
    }
    return slashFile;
}

/**
 * Spans inside `src/` belong to user editors, named by their path beneath `src/` as the tree pane names them.
 * Files outside `src/`, such as `std` or `mach.toml`, belong to no editor.
 */
function locate(span: Span | null | undefined, projectRoot?: string): Location | undefined {
    if (!span) return undefined;
    const file = maskSpanFile(span.file, projectRoot);
    if (!file.startsWith('src/') || file.split('/').includes('..')) return undefined;
    const location: Location = {file: file.slice('src/'.length), line: span.line, column: span.column};
    if (span.end_line !== span.line || span.end_column !== span.column) {
        location.endline = span.end_line;
        location.endcolumn = span.end_column;
    }
    return location;
}

function quickFixes(record: Diagnostic, primary: Location, projectRoot?: string): Fix[] {
    const fixes: Fix[] = [];
    for (const fix of record.fixes) {
        const edits = fix.edits.map(edit => ({edit, location: locate(edit, projectRoot)}));
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

function renderDiagnostic(record: Diagnostic, projectRoot?: string, stdPath?: string): ResultLine[] {
    const severity = severities[record.severity] ?? 3;
    const headline = `${record.severity}[${record.code}]: ${record.message}`;
    const lines: ResultLine[] = [{text: headline}];

    const primary = locate(record.primary, projectRoot);
    if (primary) {
        const text = [
            headline,
            ...record.notes.map(note => `note: ${note}`),
            ...record.help.map(help => `help: ${help}`),
        ].join('\n');
        const fixes = quickFixes(record, primary, projectRoot);
        lines[0].tag = {...primary, text, severity, ...(fixes.length > 0 && {fixes})};
    }
    if (record.primary) {
        const shownFile = maskSpanFile(record.primary.file, projectRoot, stdPath);
        lines.push({text: ` --> ${shownFile}:${record.primary.line}:${record.primary.column}`});
    }

    for (const span of record.related) {
        const shownFile = maskSpanFile(span.file, projectRoot, stdPath);
        const where = ` --> ${shownFile}:${span.line}:${span.column}`;
        const line: ResultLine = {text: span.label ? `${where}: ${span.label}` : where};
        const location = locate(span, projectRoot);
        if (location) line.tag = {...location, text: span.label ?? headline, severity: 1};
        lines.push(line);
    }

    for (const note of record.notes) lines.push({text: `  = note: ${note}`});
    for (const help of record.help) lines.push({text: `  = help: ${help}`});
    for (const fix of record.fixes) lines.push({text: `  = fix: ${fix.label}`});
    return lines;
}

function renderPhase(record: PhaseRecord): ResultLine {
    const indent = record.parent ? '    ' : '  ';
    const ms = (record.time_us / 1000).toFixed(1);
    const count = record.count !== undefined && record.unit ? ` (${record.count} ${record.unit})` : '';
    return {text: `${indent}${record.name}: ${ms}ms${count}`};
}

/**
 * Reads `mach build --report json` (schema 3) into result lines, placing markers and quick fixes in user editors.
 */
export function parseMachDiagnostics(output: string, inputFilename?: string, stdPath?: string): ResultLine[] {
    const projectRoot = inputFilename ? path.dirname(path.dirname(inputFilename)) : undefined;
    const result: ResultLine[] = [];
    eachLine(output, line => {
        const record = readRecord(line);
        if (!record) {
            result.push({text: projectRoot ? line.split(projectRoot + path.sep).join('') : line});
        } else if (record.record === 'diagnostic') {
            result.push(...renderDiagnostic(record as Diagnostic, projectRoot, stdPath));
        } else if (record.record === 'unit') {
            const u = record as UnitRecord;
            const prof = u.profile ? `, ${u.profile}` : '';
            result.push({text: `${u.verb} ${u.artifact} (${u.target}${prof})`});
        } else if (record.record === 'phase') {
            result.push(renderPhase(record as PhaseRecord));
        } else if (record.record === 'summary') {
            const {errors, warnings} = record as Summary;
            if (errors + warnings > 0)
                result.push({text: `${plural(errors, 'error')} / ${plural(warnings, 'warning')}`});
        }
    });
    return result;
}

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
