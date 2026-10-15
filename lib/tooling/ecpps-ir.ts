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

import fs from "node:fs/promises";
import path from "node:path";

import { splitArguments } from "../../shared/common-utils.js";
import { CompilationInfo } from "../../types/compilation/compilation.interfaces.js";
import { ToolResult } from "../../types/tool.interfaces.js";
import { logger } from "../logger.js";
import { OptionsHandlerLibrary } from "../options-handler.js";
import { BaseTool } from "./base-tool.js";

const STAGES: Record<string, string> = {
    ir: "ir",
    "virtual instructions": "vinst",
    ast: "ast",
};
const DEFAULT_STAGE = "ir";

const OWN_OPTION = /^\/(verbose(:.*)?|asm|out:.*|output:.*)$/i;

export class EcppsIrTool extends BaseTool {
    static get key() {
        return "ecpps-ir-tool";
    }

    static stageFor(args?: string[]): string {
        const wanted = (args ?? []).join(" ").trim().toLowerCase();
        return STAGES[wanted] ?? DEFAULT_STAGE;
    }

    override async runTool(
        compilationInfo: CompilationInfo,
        inputFilepath?: string,
        args?: string[],
        stdin?: string,
        supportedLibraries?: Record<string, OptionsHandlerLibrary>,
    ): Promise<ToolResult> {
        if (!inputFilepath) return this.createErrorResponse("No input file");

        const stage = EcppsIrTool.stageFor(args);
        const dirPath = compilationInfo.dirPath ?? path.dirname(inputFilepath);
        const outputFilepath = path.join(dirPath, `ecpps-${stage}.txt`);

        const compilerOptions = compilationInfo.compiler.options
            ? splitArguments(compilationInfo.compiler.options)
            : [];
        const userOptions = (compilationInfo.options ?? []).filter(
            (option) => option !== inputFilepath && !OWN_OPTION.test(option),
        );

        const toolArgs = [
            `/verbose:${stage}`,
            "/asm",
            `/out:${outputFilepath}`,
            ...compilerOptions.filter((option) => !OWN_OPTION.test(option)),
            ...userOptions,
            inputFilepath,
        ];

        const execOptions = this.getDefaultExecOptions();
        execOptions.customCwd = dirPath;

        try {
            const result = await this.exec(
                compilationInfo.compiler.exe,
                toolArgs,
                execOptions,
            );

            let dump = "";
            try {
                dump = await fs.readFile(outputFilepath, "utf8");
            } catch {}

            const transformedInput = result.filenameTransform(inputFilepath);
            const stdout = dump
                ? dump
                      .replace(/\r?\n$/, "")
                      .split(/\r?\n/)
                      .map((text) => ({ text }))
                : [];
            return {
                id: this.tool.id,
                name: this.tool.name,
                code: result.code,
                languageId: this.tool.languageId,
                stdout:
                    stdout.length > 0
                        ? stdout
                        : this.parseOutput(result.stdout, transformedInput),
                stderr: this.parseOutput(
                    result.stderr,
                    transformedInput,
                    path.dirname(compilationInfo.compiler.exe),
                ),
            };
        } catch (e) {
            logger.error("Error while running ECPPS IR tool: ", e);
            return this.createErrorResponse("Error while running tool");
        }
    }
}
