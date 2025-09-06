// Copyright (c) 2025, Compiler Explorer Authors
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

import {splitArguments} from '../../shared/common-utils.js';
import {
    ActiveTool,
    BypassCache,
    ExecutionParams,
    LegacyCompatibleActiveTool,
    UnparsedExecutionParams,
} from '../../types/compilation/compilation.interfaces.js';
import {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {SelectedLibraryVersion} from '../../types/libraries/libraries.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';

/**
 * Core compilation request data that can come from various sources (JSON requests, SQS messages, etc.)
 */
export type CompilationRequestData = {
    source: string;
    userArguments?: string | string[];
    compilerOptions?: Record<string, any>;
    executeParameters?: UnparsedExecutionParams;
    filters?: Record<string, boolean>;
    tools?: LegacyCompatibleActiveTool[];
    libraries?: SelectedLibraryVersion[];
    bypassCache?: BypassCache;
};

/**
 * Shared utility functions for parsing compilation parameters
 * These can be used by both web handlers and SQS workers for consistency
 */

/**
 * Parse user arguments from various formats into a string array
 */
export function parseUserArguments(userArgs: string | string[] | undefined): string[] {
    if (Array.isArray(userArgs)) {
        return userArgs;
    }
    if (typeof userArgs === 'string') {
        return splitArguments(userArgs);
    }
    return [];
}

/**
 * Parse execution parameters with proper defaults
 */
export function parseExecutionParameters(execParams: UnparsedExecutionParams = {}): ExecutionParams {
    return {
        args: Array.isArray(execParams.args) ? execParams.args : splitArguments(execParams.args || ''),
        stdin: execParams.stdin || '',
        runtimeTools: execParams.runtimeTools || [],
    };
}

/**
 * Parse tools array and ensure args are properly split
 */
export function parseTools(tools: LegacyCompatibleActiveTool[] = []): ActiveTool[] {
    return tools.map(tool => {
        if (typeof tool.args === 'string') {
            return {...tool, args: splitArguments(tool.args)};
        }
        return tool as ActiveTool;
    });
}

/**
 * Merge compiler default filters with request filters
 */
export function parseFilters(
    compiler: BaseCompiler,
    requestFilters: Record<string, boolean> = {},
): ParseFiltersAndOutputOptions {
    return {
        ...compiler.getDefaultFilters(),
        ...requestFilters,
    };
}
