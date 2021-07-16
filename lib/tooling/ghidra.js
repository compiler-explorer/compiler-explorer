// Copyright (c) 2020, Compiler Explorer Authors
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

// import {spawn} from "child_process"
import fs from "fs/promises"
import path from "path"
import {BaseTool} from "./base-tool"

const START_INDICATOR = "[[[[START"
const END_INDICATOR = "END]]]]"

const _getFunctions = (compilationInfo) => {
    const {asm} = compilationInfo.asmParser.process(compilationInfo.asm, {
        ...compilationInfo.filters,
        demangle: false,
    })

    return asm
        .filter((line) => !("opcode" in line) && line.text.endsWith(":"))
        .map((line) => line.text.substring(0, line.text.length - 1))
}

// const _demange = async (buffer) =>
//     await new Promise((resolve, reject) => {
//         const child = spawn("c++filt", ["--no-params"])
//         child.on("error", (error) => reject(error))

//         const bufs = []
//         child.stdout.on("data", (chunk) => bufs.push(chunk))
//         child.stdout.on("end", () => resolve(Buffer.concat(bufs)))

//         child.stdin.write(buffer)
//         child.stdin.end()
//     })

export class GhidraTool extends BaseTool {
    static get key() {
        return "ghidra-tool"
    }

    getDefaultExecOptions() {
        // TODO: we need to make sure SLEIGHHOME environment variable is set so the decompiler can load the spec files
        return {
            ...super.getDefaultExecOptions(),
            // env: {
            //     SLEIGHHOME: "/workspaces/ghidra/",
            // },
        }
    }

    async runTool(compilationInfo, inputFilename, args, stdin) {
        if (
            !compilationInfo.filters.binary ||
            compilationInfo.filters.demangle
        ) {
            return this.createErrorResponse(
                "Ghidra requires a binary output. Demangle must be off",
            )
        }

        const inPath = path.join(
            compilationInfo.dirPath,
            "ghidra-decompiler-input.in",
        )

        const fnCommands = []
        for (const fn of _getFunctions(compilationInfo)) {
            if (fn.indexOf("@") !== -1) {
                continue
            }
            fnCommands.push(`load function ${fn}`)
            fnCommands.push(`decompile`)
            fnCommands.push(`echo ${START_INDICATOR}`)
            fnCommands.push(`print C`)
            fnCommands.push(`echo ${END_INDICATOR}`)
        }

        const additionalDirectives = stdin
            .split("\n")
            .map((line) => line.trim())
            .filter((line) => line)
            .map((line) => `parse line ${line}`)

        const commands = [
            `load file ${compilationInfo.executableFilename}`,
            ...additionalDirectives,
            `read symbols`,
            `echo ${START_INDICATOR}`,
            `print C globals`,
            `echo ${END_INDICATOR}`,
            ...fnCommands,
            `quit`,
        ]

        await fs.writeFile(inPath, commands.join("\n"))

        let result = await super.runTool(compilationInfo, undefined, [
            "-i",
            inPath,
        ])

        result.stdout = result.stdout.filter((line) => {
            const trimmed = line.text.trim()
            return trimmed && !trimmed.startsWith("init> ")
        })

        const relevantParts = []
        let startIndex = undefined
        for (let i = 0; i < result.stdout.length; ++i) {
            const trimmed = result.stdout[i].text.trim()
            if (startIndex === undefined && trimmed === START_INDICATOR) {
                startIndex = i + 1
            }
            if (startIndex !== undefined && trimmed === END_INDICATOR) {
                relevantParts.push(result.stdout.slice(startIndex, i))
                startIndex = undefined
            }
        }
        result.stdout = relevantParts.flatMap((lines) => [...lines, {text: ""}])

        // const startLineIndex = result.stdout.findIndex(
        //     (line) => line.text.trim() === START_INDICATOR,
        // )
        // const endLineIndex = result.stdout.findIndex(
        //     (line) => line.text.trim() === END_INDICATOR,
        // )
        // if (startLineIndex !== -1 && endLineIndex !== -1) {
        //     result.stdout = result.stdout.slice(
        //         startLineIndex + 1,
        //         endLineIndex,
        //     )
        // }

        // const exeDir = path.dirname(this.tool.exe)
        // const output = await _demange(
        //     result.stdout.map((line) => line.text).join("\n"),
        // )
        // result.stdout = utils.parseOutput(
        //     output.toString("utf-8"),
        //     compilationInfo.executableFilename,
        //     exeDir,
        // )

        return result
    }
}
