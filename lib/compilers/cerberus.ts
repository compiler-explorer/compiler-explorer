// Copyright (c) 2023, Compiler Explorer Authors
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

import Parser from 'tree-sitter';
import coreLanguage from 'tree-sitter-core';
import {isNull} from 'underscore';

import {AsmResultSource, ParsedAsmResultLine} from '../../types/asmresult/asmresult.interfaces.js';
import {BypassCache, CacheKey, ExecutionOptions} from '../../types/compilation/compilation.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import {ExecutableExecutionOptions} from '../../types/execution/execution.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {assert} from '../assert.js';
import {BaseCompiler} from '../base-compiler.js';
import {CompilationEnvironment} from '../compilation-env.js';
import {logger} from '../logger.js';
import * as utils from '../utils.js';

type Range = {
    start: Parser.Point;
    end: Parser.Point;
};

type LocationRange = {
    location: Range;
    cursor: Range | Parser.Point | null;
};

type ASTwithLocations = {
    root: Parser.SyntaxNode;
    locations: Map<number, LocationRange>;
};

type SyntaxNodeWithLocation = {
    node: Parser.SyntaxNode;
    loc: LocationRange | undefined;
};

export class CerberusCompiler extends BaseCompiler {
    static get key() {
        return 'cerberus';
    }

    constructor(compilerInfo: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(
            {
                // Default is to disable all "cosmetic" filters
                disabledFilters: ['labels', 'directives', 'commentOnly', 'trim', 'debugCalls'],
                ...compilerInfo,
            },
            env,
        );
    }

    override getSharedLibraryPathsAsArguments() {
        return [];
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFileName: string) {
        filters.binary = true;
        return ['-c', '-o', outputFileName];
    }

    override getOutputFilename(dirPath: string) {
        return path.join(dirPath, `${path.basename(this.compileFilename, this.lang.extensions[0])}.co`);
    }

    override async objdump(outputFilename: string, result: any, maxSize: number) {
        if (!(await utils.fileExists(outputFilename))) {
            result.asm = '<No output file ' + outputFilename + '>';
            return result;
        }

        const execOptions: ExecutionOptions = {
            maxOutput: maxSize,
            customCwd: (result.dirPath as string) || path.dirname(outputFilename),
        };

        const args = ['--pp_flags=loc', '--pp=core', outputFilename];

        const objResult = await this.exec(this.compiler.objdumper, args, execOptions);
        if (objResult.code === 0) {
            result.objdumpTime = objResult.execTime;
            result.asm = this.postProcessObjdumpOutput(objResult.stdout);
        } else {
            logger.error(`Error executing objdump ${this.compiler.objdumper}`, objResult);
            result.asm = `<No output: objdump returned ${objResult.code}>`;
        }

        return result;
    }

    override async handleInterpreting(key: CacheKey, executeParameters: ExecutableExecutionOptions) {
        const executionPackageHash = this.env.getExecutableHash(key);
        const compileResult = await this.getOrBuildExecutable(key, BypassCache.None, executionPackageHash);
        assert(compileResult.dirPath !== undefined);
        if (compileResult.code === 0) {
            executeParameters.args = [
                '--exec',
                this.getOutputFilename(compileResult.dirPath),
                '--',
                ...executeParameters.args,
            ];
            const result = await this.runExecutable(this.compiler.exe, executeParameters, compileResult.dirPath);
            return {
                ...result,
                didExecute: true,
                buildResult: compileResult,
            };
        }
        return {
            stdout: compileResult.stdout,
            stderr: compileResult.stderr,
            code: compileResult.code,
            didExecute: false,
            buildResult: compileResult,
            timedOut: false,
        };
    }

    private parse_position(node: Parser.SyntaxNode): Parser.Point | null {
        if (node.type !== 'position') return null;

        const filenameNode = node.childForFieldName('filename');
        if (!filenameNode || filenameNode.text !== 'example.c') return null;

        const rowNode = node.childForFieldName('line');
        const columnNode = node.childForFieldName('column');
        if (!rowNode || !columnNode) return null;

        const row = Number.parseInt(rowNode.text, 10);
        const column = Number.parseInt(columnNode.text, 10);
        if (Number.isNaN(row) || Number.isNaN(column)) return null;

        return {row: row, column: column};
    }

    /* Attempt to parse 'location' node.
       Returns null if the node does not correspond to location range.        
     */
    private parse_location(node: Parser.SyntaxNode): LocationRange | null {
        if (node.firstNamedChild === null || node.firstNamedChild.type !== 'location_range') return null;

        const startNode = node.firstNamedChild.childForFieldName('start');
        const endNode = node.firstNamedChild.childForFieldName('end');
        if (!startNode || !endNode) return null;

        const start = this.parse_position(startNode);
        const end = this.parse_position(endNode);
        if (isNull(start) || isNull(end)) return null;

        const startCursorNode = node.firstNamedChild.childForFieldName('start_cursor');
        if (!startCursorNode) return {location: {start: start, end: end}, cursor: null};
        const start_cursor = this.parse_position(startCursorNode);
        if (!start_cursor) return {location: {start: start, end: end}, cursor: null};

        const endCursorNode = node.firstNamedChild.childForFieldName('end_cursor');
        if (!endCursorNode) return {location: {start: start, end: end}, cursor: start_cursor};
        const end_cursor = this.parse_position(endCursorNode);
        if (end_cursor) {
            return {location: {start: start, end: end}, cursor: {start: start_cursor, end: end_cursor}};
        }
        return {location: {start: start, end: end}, cursor: start_cursor};
    }

    private annotate_ast(node: Parser.SyntaxNode, locmap: Map<number, LocationRange>): void {
        let loc: LocationRange | null = null;
        for (const n of node.children) {
            if (n.type === 'location') {
                loc = this.parse_location(n);
            } else {
                if (loc !== null && n.isNamed) {
                    //console.log(`ANNOTATING ${n.id} ${n.type} at ${point_to_string(n.startPosition)}-${point_to_string(n.endPosition)} with location ${location_range_to_string(loc)}`);
                    locmap[n.id] = loc;
                    loc = null;
                }
                this.annotate_ast(n, locmap);
            }
        }
    }

    private findNodesByRange(ast: ASTwithLocations, range: Range): SyntaxNodeWithLocation[] {
        const matchingNodes: SyntaxNodeWithLocation[] = [];

        function search(n: Parser.SyntaxNode): void {
            if (
                (n.startPosition.row < range.start.row ||
                    (n.startPosition.row === range.start.row && n.startPosition.column <= range.start.column)) &&
                (n.endPosition.row > range.end.row ||
                    (n.endPosition.row === range.end.row && n.endPosition.column >= range.end.column))
            ) {
                const loc = ast.locations[n.id];
                matchingNodes.push({node: n, loc: loc});
            }

            for (const child of n.children) {
                search(child);
            }
        }

        search(ast.root);
        return matchingNodes.reverse();
    }

    override async processAsm(result) {
        // Handle "error" documents.
        if (!result.asm.includes('\n') && result.asm[0] === '<') {
            return {asm: [{text: result.asm, source: null}]};
        }

        const core = result.asm.replaceAll(/\n{3,}/g, '\n\n');
        const parser = new Parser();
        parser.setLanguage(coreLanguage);

        const tree = parser.parse(core);
        const ast: ASTwithLocations = {root: tree.rootNode, locations: new Map<number, LocationRange>()};
        this.annotate_ast(ast.root, ast.locations);

        const lines = core.split('\n');
        const plines: ParsedAsmResultLine[] = lines.map((l: string, n: number) => {
            const ltrimmed = l.replace(/^\s*{-# .+ #-}\s*/, '');
            const start_col = l.length - ltrimmed.length;
            const rtrimmed = ltrimmed.trimEnd();
            const r: Range = {start: {row: n, column: start_col}, end: {row: n, column: start_col + rtrimmed.length}};
            const matchingNodes = this.findNodesByRange(ast, r);

            const coreNode = matchingNodes.find(x => x.loc !== undefined);
            // the second disjunct is redundant and added to make typechecker happy
            if (coreNode === undefined || coreNode.loc === undefined) {
                //console.log(`No node with location for ${range_to_string(r)}`);
                return {text: l};
            }

            const loc = coreNode.loc;
            //console.log(`Found ${coreNode.node.id} ${coreNode.node.type} at ${point_to_string(coreNode.node.startPosition)}-${point_to_string(coreNode.node.endPosition)} for ${range_to_string(r)} with location ${location_range_to_string(coreNode.loc)}`);
            const src: AsmResultSource = {
                file: null,
                line: loc.location.start.row,
                column: loc.location.start.column,
            };
            return {text: l, source: src};
        });
        return {
            asm: plines,
            languageId: 'core',
        };
    }
}
