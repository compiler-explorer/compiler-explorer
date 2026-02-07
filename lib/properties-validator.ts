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

/**
 * Properties file validation module.
 *
 * This module provides validation functions for .properties files used in Compiler Explorer.
 * It operates in two modes:
 * - Raw file validation: Syntax and format checks on unparsed file content
 * - Semantic validation: Checking consistency using loaded CompilerProps
 */

import {parseProperties} from './properties.js';

export interface ValidationIssue {
    line: number;
    text: string;
    id?: string;
}

export interface ParsedProperty {
    key: string;
    value: string;
    line: number;
}

export interface ParsedPropertiesFile {
    filename: string;
    properties: ParsedProperty[];
    disabledIds: Set<string>;
    parseErrors: ValidationIssue[];
}

export interface RawFileValidationResult {
    duplicateKeys: ValidationIssue[];
    emptyListElements: ValidationIssue[];
    typoCompilers: ValidationIssue[];
    invalidPropertyFormat: ValidationIssue[];
    suspiciousPaths: ValidationIssue[];
    duplicatedCompilerRefs: ValidationIssue[];
    duplicatedGroupRefs: ValidationIssue[];
    orphanedCompilerExe: ValidationIssue[];
    orphanedCompilerId: ValidationIssue[];
    orphanedGroups: ValidationIssue[];
    orphanedFormatterExe: ValidationIssue[];
    orphanedFormatterId: ValidationIssue[];
    orphanedToolExe: ValidationIssue[];
    orphanedToolId: ValidationIssue[];
    orphanedLibIds: ValidationIssue[];
    orphanedLibVersions: ValidationIssue[];
    invalidDefaultCompiler: ValidationIssue[];
    noCompilersList: boolean;
}

export interface RawValidatorOptions {
    checkSuspiciousPaths?: boolean;
}

// Regex patterns for property file validation
const PATTERNS = {
    property: /^([^#\s][^=]*)=(.*)$/,
    compilersList: /^compilers=(.*)$/,
    aliasList: /^alias=(.*)$/,
    groupCompilers: /^group\.([^.]+)\.compilers=(.*)$/,
    groupName: /^group\.([^.]+)\./,
    compilerExe: /^compiler\.([^.]+)\.exe=(.*)$/,
    compilerId: /^compiler\.([^.]+)\./,
    typoCompilers: /^compilers\./,
    defaultCompiler: /^defaultCompiler=(.*)$/,
    formattersList: /^formatters=(.*)$/,
    formatterExe: /^formatter\.([^.]+)\.exe=(.*)$/,
    formatterId: /^formatter\.([^.]+)\./,
    libsList: /^libs=(.+)$/,
    libVersionsList: /^libs\.([^.]+)\.versions=(.*)$/,
    libVersion: /^libs\.([^.]+)\.versions\.([^.]+)\.version/,
    toolsList: /^tools=(.+)$/,
    toolExe: /^tools\.([^.]+)\.exe=(.*)$/,
    toolId: /^tools\.([^.]+)\./,
    emptyList: /^.*(compilers|formatters|versions|tools|alias|exclude|libPath)=(.*::.*|:.*|.*:)$/,
    disabled: /^#\s*Disabled?:?\s*(.*)$/i,
};

// File patterns that are allowed to have no compilers list.
// These are non-language config files from the Python propscheck.py implementation.
const ALLOWED_EMPTY_COMPILERS_PATTERNS = [
    'execution.',
    'compiler-explorer.',
    'aws.',
    'asm-docs.',
    'builtin.',
    '.defaults.',
];

// Path prefixes considered valid for production config files.
// Anything outside these paths will be flagged as suspicious.
const VALID_PATH_PREFIXES = ['/opt/compiler-explorer', 'Z:/compilers'];

// Specific system paths that are always allowed. /usr/bin/ldd is used as a tool
// in many language configs to show shared library dependencies of compiled executables.
const ALLOWED_SYSTEM_PATHS = ['/usr/bin/ldd'];

function isSuspiciousPath(path: string): boolean {
    if (ALLOWED_SYSTEM_PATHS.includes(path)) return false;
    return !VALID_PATH_PREFIXES.some(prefix => path.startsWith(prefix));
}

/**
 * Parse a colon-separated list of compiler/group references.
 * Handles group references (&groupname), remote references (id@host), and regular IDs.
 */
export function parseCompilersList(value: string): string[] {
    if (!value || !value.trim()) return [];
    return value.split(':').filter(id => id.trim() !== '');
}

/**
 * Parse a properties file content into a structured format with line numbers.
 * Uses the main properties parser for error detection.
 */
export function parsePropertiesFileRaw(content: string, filename: string): ParsedPropertiesFile {
    const properties: ParsedProperty[] = [];
    const disabledIds = new Set<string>();
    const parseErrors: ValidationIssue[] = [];

    // Use the main parser to collect format errors via callback
    parseProperties(content, filename, {
        onError: error => parseErrors.push({line: error.line, text: error.text}),
    });

    // Also collect our own structured properties with line numbers
    const lines = content.split('\n');
    for (let i = 0; i < lines.length; i++) {
        const lineNumber = i + 1;
        const text = lines[i].trim();

        if (!text) continue;

        // Check for Disabled: comments
        const disabledMatch = text.match(PATTERNS.disabled);
        if (disabledMatch) {
            const ids = disabledMatch[1].split(/\s+/).filter(id => id.trim() !== '');
            for (const id of ids) {
                disabledIds.add(id);
            }
            continue;
        }

        // Skip other comments
        if (text.startsWith('#')) continue;

        // Parse property line
        const match = text.match(PATTERNS.property);
        if (match) {
            properties.push({
                key: match[1].trim(),
                value: match[2].trim(),
                line: lineNumber,
            });
        }
    }

    return {filename, properties, disabledIds, parseErrors};
}

/**
 * Validate a parsed properties file for various issues.
 */
export function validateRawFile(
    parsed: ParsedPropertiesFile,
    options: RawValidatorOptions = {},
): RawFileValidationResult {
    const result: RawFileValidationResult = {
        duplicateKeys: [],
        emptyListElements: [],
        typoCompilers: [],
        invalidPropertyFormat: [...parsed.parseErrors], // Copy parse errors from library
        suspiciousPaths: [],
        duplicatedCompilerRefs: [],
        duplicatedGroupRefs: [],
        orphanedCompilerExe: [],
        orphanedCompilerId: [],
        orphanedGroups: [],
        orphanedFormatterExe: [],
        orphanedFormatterId: [],
        orphanedToolExe: [],
        orphanedToolId: [],
        orphanedLibIds: [],
        orphanedLibVersions: [],
        invalidDefaultCompiler: [],
        noCompilersList: false,
    };

    // Track what we've seen
    const seenKeys = new Set<string>();
    const listedCompilers = new Map<string, number>(); // id -> line number
    const listedGroups = new Map<string, number>(); // group name -> line number
    const seenCompilersExe = new Map<string, number>();
    const seenCompilersId = new Set<string>();
    const seenGroups = new Set<string>();

    const listedFormatters = new Map<string, number>();
    const seenFormattersExe = new Map<string, number>();
    const seenFormattersId = new Set<string>();

    const listedTools = new Map<string, number>();
    const seenToolsExe = new Map<string, number>();
    const seenToolsId = new Set<string>();

    const listedLibsIds = new Map<string, number>();
    const seenLibsIds = new Set<string>();

    const listedLibVersions = new Map<string, number>(); // "libid version" -> line
    const seenLibVersions = new Set<string>(); // "libid version"

    let defaultCompiler: {id: string; line: number} | undefined;

    const checkSuspicious =
        options.checkSuspiciousPaths &&
        !parsed.filename.endsWith('.defaults.properties') &&
        !parsed.filename.endsWith('.local.properties');

    // First pass: collect all data
    for (const prop of parsed.properties) {
        const {key, value, line} = prop;
        const fullLine = `${key}=${value}`;

        // Check for duplicate keys
        if (seenKeys.has(key)) {
            result.duplicateKeys.push({line, text: key, id: key});
        } else {
            seenKeys.add(key);
        }

        // Check for typo: compilers. instead of compiler.
        if (PATTERNS.typoCompilers.test(key)) {
            result.typoCompilers.push({line, text: fullLine, id: key});
        }

        // Check for empty list elements
        if (PATTERNS.emptyList.test(fullLine)) {
            result.emptyListElements.push({line, text: fullLine});
        }

        // Parse compilers= list (top-level or group)
        const compilersListMatch = fullLine.match(PATTERNS.compilersList);
        if (compilersListMatch) {
            const ids = parseCompilersList(compilersListMatch[1]);
            const seenInThisList = new Set<string>();
            for (const id of ids) {
                if (id.startsWith('&')) {
                    const groupName = id.slice(1);
                    if (listedGroups.has(groupName)) {
                        result.duplicatedGroupRefs.push({line, text: groupName, id: groupName});
                    } else if (seenInThisList.has(id)) {
                        result.duplicatedGroupRefs.push({line, text: groupName, id: groupName});
                    }
                    listedGroups.set(groupName, line);
                    seenInThisList.add(id);
                } else if (!id.includes('@')) {
                    // Not a remote reference
                    if (listedCompilers.has(id)) {
                        result.duplicatedCompilerRefs.push({line, text: id, id});
                    } else if (seenInThisList.has(id)) {
                        result.duplicatedCompilerRefs.push({line, text: id, id});
                    }
                    listedCompilers.set(id, line);
                    seenInThisList.add(id);
                }
            }
        }

        // Parse group.X.compilers= list
        const groupCompilersMatch = fullLine.match(PATTERNS.groupCompilers);
        if (groupCompilersMatch) {
            const groupName = groupCompilersMatch[1];
            const ids = parseCompilersList(groupCompilersMatch[2]);
            seenGroups.add(groupName);
            const seenInThisList = new Set<string>();
            for (const id of ids) {
                if (id.startsWith('&')) {
                    const subGroupName = id.slice(1);
                    if (listedGroups.has(subGroupName)) {
                        result.duplicatedGroupRefs.push({line, text: subGroupName, id: subGroupName});
                    } else if (seenInThisList.has(id)) {
                        result.duplicatedGroupRefs.push({line, text: subGroupName, id: subGroupName});
                    }
                    listedGroups.set(subGroupName, line);
                    seenInThisList.add(id);
                } else if (!id.includes('@')) {
                    if (listedCompilers.has(id)) {
                        result.duplicatedCompilerRefs.push({line, text: id, id});
                    } else if (seenInThisList.has(id)) {
                        result.duplicatedCompilerRefs.push({line, text: id, id});
                    }
                    listedCompilers.set(id, line);
                    seenInThisList.add(id);
                }
            }
        }

        // Parse group.X.* (marks group as seen/defined)
        const groupMatch = key.match(PATTERNS.groupName);
        if (groupMatch) {
            seenGroups.add(groupMatch[1]);
        }

        // Parse compiler.X.exe=
        const compilerExeMatch = fullLine.match(PATTERNS.compilerExe);
        if (compilerExeMatch) {
            seenCompilersExe.set(compilerExeMatch[1], line);

            // Check suspicious path
            if (checkSuspicious) {
                const path = compilerExeMatch[2];
                if (isSuspiciousPath(path)) {
                    result.suspiciousPaths.push({line, text: path, id: compilerExeMatch[1]});
                }
            }
        }

        // Parse compiler.X.*
        const compilerIdMatch = key.match(PATTERNS.compilerId);
        if (compilerIdMatch) {
            seenCompilersId.add(compilerIdMatch[1]);
        }

        // Parse alias= (adds to seen compilers)
        const aliasMatch = fullLine.match(PATTERNS.aliasList);
        if (aliasMatch) {
            const ids = parseCompilersList(aliasMatch[1]);
            for (const id of ids) {
                seenCompilersExe.set(id, line);
            }
        }

        // Parse defaultCompiler=
        const defaultMatch = fullLine.match(PATTERNS.defaultCompiler);
        if (defaultMatch) {
            defaultCompiler = {id: defaultMatch[1].trim(), line};
        }

        // Parse formatters=
        const formattersMatch = fullLine.match(PATTERNS.formattersList);
        if (formattersMatch) {
            const ids = parseCompilersList(formattersMatch[1]);
            for (const id of ids) {
                listedFormatters.set(id, line);
            }
        }

        // Parse formatter.X.exe=
        const formatterExeMatch = fullLine.match(PATTERNS.formatterExe);
        if (formatterExeMatch) {
            seenFormattersExe.set(formatterExeMatch[1], line);

            if (checkSuspicious) {
                const path = formatterExeMatch[2];
                if (isSuspiciousPath(path)) {
                    result.suspiciousPaths.push({line, text: path, id: formatterExeMatch[1]});
                }
            }
        }

        // Parse formatter.X.*
        const formatterIdMatch = key.match(PATTERNS.formatterId);
        if (formatterIdMatch) {
            seenFormattersId.add(formatterIdMatch[1]);
        }

        // Parse tools=
        const toolsMatch = fullLine.match(PATTERNS.toolsList);
        if (toolsMatch) {
            const ids = parseCompilersList(toolsMatch[1]);
            for (const id of ids) {
                listedTools.set(id, line);
            }
        }

        // Parse tools.X.exe=
        const toolExeMatch = fullLine.match(PATTERNS.toolExe);
        if (toolExeMatch) {
            seenToolsExe.set(toolExeMatch[1], line);

            if (checkSuspicious) {
                const path = toolExeMatch[2];
                if (isSuspiciousPath(path)) {
                    result.suspiciousPaths.push({line, text: path, id: toolExeMatch[1]});
                }
            }
        }

        // Parse tools.X.*
        const toolIdMatch = key.match(PATTERNS.toolId);
        if (toolIdMatch) {
            seenToolsId.add(toolIdMatch[1]);
        }

        // Parse libs=
        const libsMatch = fullLine.match(PATTERNS.libsList);
        if (libsMatch) {
            const ids = parseCompilersList(libsMatch[1]);
            for (const id of ids) {
                listedLibsIds.set(id, line);
            }
        }

        // Parse libs.X.versions=
        const libVersionsMatch = fullLine.match(PATTERNS.libVersionsList);
        if (libVersionsMatch) {
            const libId = libVersionsMatch[1];
            seenLibsIds.add(libId);
            const versions = parseCompilersList(libVersionsMatch[2]);
            for (const version of versions) {
                listedLibVersions.set(`${libId} ${version}`, line);
            }
        }

        // Parse libs.X.versions.Y.version
        const libVersionMatch = fullLine.match(PATTERNS.libVersion);
        if (libVersionMatch) {
            const libId = libVersionMatch[1];
            const version = libVersionMatch[2];
            seenLibVersions.add(`${libId} ${version}`);
        }
    }

    // Second pass: compute symmetric differences (orphans)

    // Orphaned compilers (listed but no .exe, or .exe but not listed)
    if (seenCompilersExe.size > 0) {
        for (const [id, line] of listedCompilers) {
            if (!seenCompilersExe.has(id)) {
                result.orphanedCompilerExe.push({line, text: id, id});
            }
        }
        for (const [id, line] of seenCompilersExe) {
            if (!listedCompilers.has(id)) {
                result.orphanedCompilerExe.push({line, text: id, id});
            }
        }
    }

    if (seenCompilersId.size > 0) {
        for (const [id, line] of listedCompilers) {
            if (!seenCompilersId.has(id)) {
                result.orphanedCompilerId.push({line, text: id, id});
            }
        }
        for (const id of seenCompilersId) {
            if (!listedCompilers.has(id)) {
                const line = parsed.properties.find(p => p.key.startsWith(`compiler.${id}.`))?.line ?? 0;
                result.orphanedCompilerId.push({line, text: id, id});
            }
        }
    }

    // Orphaned groups
    for (const [groupName, line] of listedGroups) {
        if (!seenGroups.has(groupName)) {
            result.orphanedGroups.push({line, text: groupName, id: groupName});
        }
    }
    for (const groupName of seenGroups) {
        if (!listedGroups.has(groupName)) {
            const line = parsed.properties.find(p => p.key.startsWith(`group.${groupName}.`))?.line ?? 0;
            result.orphanedGroups.push({line, text: groupName, id: groupName});
        }
    }

    // Orphaned formatters
    for (const [id, line] of listedFormatters) {
        if (!seenFormattersExe.has(id)) {
            result.orphanedFormatterExe.push({line, text: id, id});
        }
    }
    for (const [id, line] of seenFormattersExe) {
        if (!listedFormatters.has(id)) {
            result.orphanedFormatterExe.push({line, text: id, id});
        }
    }

    for (const [id, line] of listedFormatters) {
        if (!seenFormattersId.has(id)) {
            result.orphanedFormatterId.push({line, text: id, id});
        }
    }
    for (const id of seenFormattersId) {
        if (!listedFormatters.has(id)) {
            const line = parsed.properties.find(p => p.key.startsWith(`formatter.${id}.`))?.line ?? 0;
            result.orphanedFormatterId.push({line, text: id, id});
        }
    }

    // Orphaned tools
    for (const [id, line] of listedTools) {
        if (!seenToolsExe.has(id)) {
            result.orphanedToolExe.push({line, text: id, id});
        }
    }
    for (const [id, line] of seenToolsExe) {
        if (!listedTools.has(id)) {
            result.orphanedToolExe.push({line, text: id, id});
        }
    }

    for (const [id, line] of listedTools) {
        if (!seenToolsId.has(id)) {
            result.orphanedToolId.push({line, text: id, id});
        }
    }
    for (const id of seenToolsId) {
        if (!listedTools.has(id)) {
            const line = parsed.properties.find(p => p.key.startsWith(`tools.${id}.`))?.line ?? 0;
            result.orphanedToolId.push({line, text: id, id});
        }
    }

    // Orphaned libs
    for (const [id, line] of listedLibsIds) {
        if (!seenLibsIds.has(id)) {
            result.orphanedLibIds.push({line, text: id, id});
        }
    }
    for (const id of seenLibsIds) {
        if (!listedLibsIds.has(id)) {
            const line = parsed.properties.find(p => p.key.startsWith(`libs.${id}.`))?.line ?? 0;
            result.orphanedLibIds.push({line, text: id, id});
        }
    }

    // Orphaned lib versions
    for (const [key, line] of listedLibVersions) {
        if (!seenLibVersions.has(key)) {
            result.orphanedLibVersions.push({line, text: key, id: key});
        }
    }
    for (const key of seenLibVersions) {
        if (!listedLibVersions.has(key)) {
            const [libId, version] = key.split(' ');
            const line = parsed.properties.find(p => p.key.startsWith(`libs.${libId}.versions.${version}.`))?.line ?? 0;
            result.orphanedLibVersions.push({line, text: key, id: key});
        }
    }

    // Invalid default compiler (default not in listed)
    // If defaultCompiler is set but doesn't appear in the compilers list, it's invalid
    // This matches Python behaviour where bad_default = default_compiler - listed_compilers
    // Skip if defaultCompiler is empty (some files have defaultCompiler= with no value)
    if (defaultCompiler?.id && !listedCompilers.has(defaultCompiler.id)) {
        result.invalidDefaultCompiler.push({
            line: defaultCompiler.line,
            text: defaultCompiler.id,
            id: defaultCompiler.id,
        });
    }

    // Check for missing compilers= in language files
    // A language file (e.g., c++.amazon.properties) should have either compilers= or group definitions
    const isAllowedEmpty = ALLOWED_EMPTY_COMPILERS_PATTERNS.some(pattern => parsed.filename.includes(pattern));
    const isLanguageFile = !isAllowedEmpty && parsed.filename.endsWith('.properties');

    if (isLanguageFile) {
        const hasCompilersList = parsed.properties.some(p => p.key === 'compilers');
        const hasGroups = seenGroups.size > 0 || listedGroups.size > 0;
        const hasCompilerDefinitions = seenCompilersExe.size > 0 || seenCompilersId.size > 0;

        // If the file defines compilers but has no compilers= or groups, flag it
        if (hasCompilerDefinitions && !hasCompilersList && !hasGroups) {
            result.noCompilersList = true;
        }
    }

    return result;
}

/**
 * Filter out disabled IDs from validation issues.
 */
export function filterDisabled(result: RawFileValidationResult, disabledIds: Set<string>): RawFileValidationResult {
    const filterIssues = (issues: ValidationIssue[]): ValidationIssue[] => {
        return issues.filter(issue => !issue.id || !disabledIds.has(issue.id));
    };

    return {
        duplicateKeys: filterIssues(result.duplicateKeys),
        emptyListElements: filterIssues(result.emptyListElements),
        typoCompilers: filterIssues(result.typoCompilers),
        invalidPropertyFormat: filterIssues(result.invalidPropertyFormat),
        suspiciousPaths: filterIssues(result.suspiciousPaths),
        duplicatedCompilerRefs: filterIssues(result.duplicatedCompilerRefs),
        duplicatedGroupRefs: filterIssues(result.duplicatedGroupRefs),
        orphanedCompilerExe: filterIssues(result.orphanedCompilerExe),
        orphanedCompilerId: filterIssues(result.orphanedCompilerId),
        orphanedGroups: filterIssues(result.orphanedGroups),
        orphanedFormatterExe: filterIssues(result.orphanedFormatterExe),
        orphanedFormatterId: filterIssues(result.orphanedFormatterId),
        orphanedToolExe: filterIssues(result.orphanedToolExe),
        orphanedToolId: filterIssues(result.orphanedToolId),
        orphanedLibIds: filterIssues(result.orphanedLibIds),
        orphanedLibVersions: filterIssues(result.orphanedLibVersions),
        invalidDefaultCompiler: filterIssues(result.invalidDefaultCompiler),
        noCompilersList: result.noCompilersList, // boolean, not filterable
    };
}

/**
 * Check if a validation result has any issues.
 */
export function hasIssues(result: RawFileValidationResult): boolean {
    return (
        result.duplicateKeys.length > 0 ||
        result.emptyListElements.length > 0 ||
        result.typoCompilers.length > 0 ||
        result.invalidPropertyFormat.length > 0 ||
        result.suspiciousPaths.length > 0 ||
        result.duplicatedCompilerRefs.length > 0 ||
        result.duplicatedGroupRefs.length > 0 ||
        result.orphanedCompilerExe.length > 0 ||
        result.orphanedCompilerId.length > 0 ||
        result.orphanedGroups.length > 0 ||
        result.orphanedFormatterExe.length > 0 ||
        result.orphanedFormatterId.length > 0 ||
        result.orphanedToolExe.length > 0 ||
        result.orphanedToolId.length > 0 ||
        result.orphanedLibIds.length > 0 ||
        result.orphanedLibVersions.length > 0 ||
        result.invalidDefaultCompiler.length > 0 ||
        result.noCompilersList
    );
}

/**
 * Format validation results for display.
 */
export function formatValidationResult(filename: string, result: RawFileValidationResult): string {
    const lines: string[] = [];

    const formatIssues = (name: string, issues: ValidationIssue[]) => {
        if (issues.length === 0) return;
        lines.push(`${name}:`);
        for (const issue of issues.sort((a, b) => a.line - b.line)) {
            lines.push(`  Line ${issue.line}: ${issue.text}`);
        }
    };

    lines.push(`## ${filename}`);
    formatIssues('Duplicate keys', result.duplicateKeys);
    formatIssues('Empty list elements', result.emptyListElements);
    formatIssues('Typo compilers', result.typoCompilers);
    formatIssues('Invalid property format', result.invalidPropertyFormat);
    formatIssues('Suspicious paths', result.suspiciousPaths);
    formatIssues('Duplicated compiler refs', result.duplicatedCompilerRefs);
    formatIssues('Duplicated group refs', result.duplicatedGroupRefs);
    formatIssues('Orphaned compiler .exe', result.orphanedCompilerExe);
    formatIssues('Orphaned compiler ID', result.orphanedCompilerId);
    formatIssues('Orphaned groups', result.orphanedGroups);
    formatIssues('Orphaned formatter .exe', result.orphanedFormatterExe);
    formatIssues('Orphaned formatter ID', result.orphanedFormatterId);
    formatIssues('Orphaned tool .exe', result.orphanedToolExe);
    formatIssues('Orphaned tool ID', result.orphanedToolId);
    formatIssues('Orphaned lib IDs', result.orphanedLibIds);
    formatIssues('Orphaned lib versions', result.orphanedLibVersions);
    formatIssues('Invalid default compiler', result.invalidDefaultCompiler);
    if (result.noCompilersList) {
        lines.push('No compilers list: File defines compilers but has no compilers= or group definitions');
    }

    return lines.join('\n');
}

export interface CrossFileValidationResult {
    duplicateCompilerIds: Map<string, Array<{filename: string; line: number}>>;
}

/**
 * Check for compiler IDs defined in multiple files.
 * Takes parsed files rather than filenames to allow testing.
 */
export function validateCrossFileCompilerIds(
    files: Array<{filename: string; parsed: ParsedPropertiesFile}>,
): CrossFileValidationResult {
    const compilerIdLocations = new Map<string, Array<{filename: string; line: number}>>();

    for (const {filename, parsed} of files) {
        const seenInFile = new Set<string>();

        for (const prop of parsed.properties) {
            const match = prop.key.match(PATTERNS.compilerId);
            if (match) {
                const compilerId = match[1];
                if (!seenInFile.has(compilerId)) {
                    seenInFile.add(compilerId);
                    const locations = compilerIdLocations.get(compilerId) ?? [];
                    locations.push({filename, line: prop.line});
                    compilerIdLocations.set(compilerId, locations);
                }
            }
        }
    }

    // Filter to only those with duplicates
    const duplicateCompilerIds = new Map<string, Array<{filename: string; line: number}>>();
    for (const [id, locations] of compilerIdLocations) {
        if (locations.length > 1) {
            duplicateCompilerIds.set(id, locations);
        }
    }

    return {duplicateCompilerIds};
}
