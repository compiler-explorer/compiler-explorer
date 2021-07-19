// Copyright (c) 2021, Compiler Explorer Authors
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

import fs from "fs/promises"
import path from "path"
import {BaseTool} from "./base-tool"

const START_INDICATOR = "[[[[START"
const END_INDICATOR = "END]]]]"

// this function creates a regex which globally matches the input string and returns a list of all matches.
// we use this to the String.prototype.replaceAll functionality (i'm not sure if String.prototype.replaceAll is supported on enough browsers to use).
function escapedRegex(string) {
    return new RegExp(
        // regex special characters must be escaped
        string.replace(/[.*+\-?^${}()|[\]\\]/g, "\\$&"),
        "g",
    )
}

export class GhidraTool extends BaseTool {
    static get key() {
        return "ghidra-tool"
    }

    getDefaultExecOptions() {
        return {
            ...super.getDefaultExecOptions(),
            env: {
                // SLEIGH refers to the processor specification files for the Ghidra decompiler.
                // the installer should have placed them in /usr/local/share/ghidra/
                SLEIGHHOME: "/usr/local/share/ghidra/",
            },
        }
    }

    // this is supposed to be used to give Ghidra hints about the input program by giving it function and struct declarations.
    // i have not taken the time to fully figure out how this works, so it's disabled for now.
    // here is a good start on the feature in Ghidra: https://ghidra.re/courses/GhidraClass/Advanced/improvingDisassemblyAndDecompilation.pdf
    getStdInCommands(_stdin) {
        return []
        // return stdin
        //     .split("\n")
        //     .map((line) => line.trim())
        //     .filter((line) => line)
        //     .map((line) => `parse line ${line}`)
    }

    getLabels(compilationInfo, mangled) {
        // if mangled is true, then we get the mangled labels of the asm from compilationInfo.mangledAsm.
        // otherwise, we get the demangled labels of the asm from compilationInfo.asm.
        let asm = compilationInfo.asm
        if (mangled && compilationInfo.mangledAsm) {
            asm = compilationInfo.mangledAsm
        }

        const processed = compilationInfo.asmParser.process(asm, {
            ...compilationInfo.filters,
            demangle: !mangled,
        })
        return Object.keys(processed.labelDefinitions)
    }

    demangle(compilationInfo, mangledLabels, result) {
        // we re-run the asmParser, but in this case, we pass compilationInfo.asm instead of compilationInfo.mangledAsm
        const demangledLabels = this.getLabels(compilationInfo, false)

        // zip the mangledLabels and demangledLabels together
        const labelPairs = mangledLabels.map((mangled, i) => [
            mangled,
            demangledLabels[i],
        ])
        const replacements = labelPairs.map(([mangled, demangled]) => [
            // for each mangled name, we need to create a precompiled regex object that will match all instances the mangled name
            // we do this to perform the "replace all" operation for the string replacement
            // String.prototype.replaceAll already exists, but I'm not sure if it's supported on enough browsers to use in this context
            escapedRegex(mangled),
            // because the demangled name includes all kinds of information such as parameter types, we wrap it in `...` to improve clarity.
            demangled ? `\`${demangled}\`` : mangled,
        ])
        result.stdout = result.stdout.map((line) => ({
            ...line,
            text:
                // essentially, this does a String.prototype.replaceAll on each stdout line, using the precompiled regexes we created above.
                replacements.reduce(
                    (acc, [mangledRegex, demangledString]) =>
                        acc.replace(mangledRegex, demangledString),
                    line.text,
                ),
        }))
    }

    async runTool(compilationInfo, inputFilename, args, stdin) {
        if (!compilationInfo.filters.binary) {
            return this.createErrorResponse("Ghidra requires a binary output.")
        }

        // get the mangled labels of the asm
        const mangledLabels = this.getLabels(compilationInfo, true)
        // for each function in the asm, we send the following commands:
        // load function, decompile, print C
        // we also print start and end indicators in the decompilation output to make it easier to parse.
        const fnCommands = mangledLabels.map((fn) => [
            `load function ${fn}`, // loads the function with the given label
            `decompile`,
            `echo ${START_INDICATOR}`,
            `print C`, // prints the decompiled C code for the function with the given label
            `echo ${END_INDICATOR}`,
        ])

        const commands = [
            `load file ${compilationInfo.executableFilename}`, // loads the executable
            ...this.getStdInCommands(stdin), // the stdin commands are used for adding C struct and function declarations to help the decompiler. These are not needed for decompiling.
            `read symbols`, // reads the symbols from the binary
            `echo ${START_INDICATOR}`,
            `print C globals`,
            `echo ${END_INDICATOR}`,
            ...fnCommands.flat(), // we send the commands to decompile all functions
            `quit`,
        ]

        // we write all our commands to the ghira-decompiler-input.in file, which is fed into ghidra as stdin.
        // this is much easier than trying to pipe the commands to ghidra's stdin.
        const inPath = path.join(
            compilationInfo.dirPath,
            "ghidra-decompiler-input.in",
        )
        await fs.writeFile(inPath, commands.join("\n"))

        let result = await super.runTool(compilationInfo, undefined, [
            "-i", // tells ghidra to read commands from the input file, rather than stdin
            inPath,
        ])

        // the init> prompt is added by ghidra to indicate that the tool is ready to receive commands.
        // It is not needed in the decompilation output that we want to send back to the user.
        result.stdout = result.stdout.filter((line) => {
            const trimmed = line.text.trim()
            return trimmed && !trimmed.startsWith("init> ")
        })

        // we use the start and end indicators to split the decompiled output into relevant sections that we are interested in showing to the user.
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

        // remove the first line if it is empty
        if (result.stdout.length > 0 && !result.stdout[0].text.trim()) {
            result.stdout.shift()
        }

        // to demangle, we create a dictionary that maps from all mangled names to their demangled names
        // we then iterate over all lines in result.stdout and perform a replacement based on the entries of the dictionary
        if (compilationInfo.filters.demangle) {
            this.demangle(compilationInfo, mangledLabels, result)
        }

        return result
    }
}
