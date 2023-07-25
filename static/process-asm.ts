import {ParseFiltersAndOutputOptions} from '../types/features/filters.interfaces.js';
import {
    AsmResultLabel,
    AsmResultSource,
    ParsedAsmResult,
    ParsedAsmResultLine,
} from '../types/asmresult/asmresult.interfaces.js';
import _ from 'underscore';

const assignmentDef = /^\s*([$.A-Z_a-z][\w$.]*)\s*=/;
const blockComments = /^[\t ]*\/\*(\*(?!\/)|[^*])*\*\/\s*/gm;
const commentOnly = /^\s*(((#|@|\/\/).*)|(\/\*.*\*\/)|(;\s*)|(;[^;].*)|(;;\s*[^\s#].*))$/;
const commentOnlyNvcc = /^\s*(((#|;|\/\/).*)|(\/\*.*\*\/))$/;
const commentRe = /[#;]/;
const cudaBeginDef = /\.(entry|func)\s+(?:\([^)]*\)\s*)?([$.A-Z_a-z][\w$.]*)\($/;
const cudaEndDef = /^\s*\)\s*$/;
const dataDefn = /^\s*\.(string|asciz|ascii|[1248]?byte|short|half|[dhx]?word|long|quad|octa|value|zero)/;
const definesFunctionRegEx = /^\s*\.(type.*,\s*[#%@]function|proc\s+[.A-Z_a-z][\w$.]*:.*)$/;
const definesGlobal = /^\s*\.(?:globa?l|GLB|export)\s*([.A-Z_a-z][\w$.]*)/;
const definesWeak = /^\s*\.(?:weakext|weak)\s*([.A-Z_a-z][\w$.]*)/;
const directive = /^\s*\..*$/;
const endAppBlock = /\s*#NO_APP.*/;
const endAsmNesting = /\s*# End ASM.*/;
const endBlock = /\.cfi_endproc/;
const fileFind = /^\s*\.(?:cv_)?file\s+(\d+)\s+"([^"]+)"(\s+"([^"]+)")?.*/;
const findQuotes = /(.*?)("(?:[^"\\]|\\.)*")(.*)/;
const hasNvccOpcodeRe = /^\s*[@A-Za-z|]/;
const hasOpcodeRe = /^\s*(%[$.A-Z_a-z][\w$.]*\s*=\s*)?[A-Za-z]/;
const identifierFindRe = /[$.@A-Z_a-z]\w*/g;
const indentedLabelDef = /^\s*([$.A-Z_a-z][\w$.]*):/;
const instOpcodeRe = /(\.inst\.?\w?)\s*(.*)/;
const instructionRe = /^\s*[A-Za-z]+/;
const labelDef = /^(?:.proc\s+)?([\w$.@]+):/i;
const labelDefRegEx = /^(?:.proc\s+)?([\w$.@]+):/i;
const labelFindMips = /[$.A-Z_a-z][\w$.]*/g;
const labelFindNonMips = /[.A-Z_a-z][\w$.]*/g;
const lineRe = /\r?\n/;
const mipsLabelDefinition = /^\$[\w$.]+:/;
const source6502Dbg = /^\s*\.dbg\s+line,\s*"([^"]+)",\s*(\d+)/;
const source6502DbgEnd = /^\s*\.dbg\s+line[^,]/;
const sourceCVTag = /^\s*\.cv_loc\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+).*/;
const sourceD2Tag = /^\s*\.d2line\s+(\d+),?\s*(\d*).*/;
const sourceStab = /^\s*\.stabn\s+(\d+),0,(\d+),.*/;
const sourceTag = /^\s*\.loc\s+(\d+)\s+(\d+)\s+(.*)/;
const startAppBlock = /\s*#APP.*/;
const startAsmNesting = /\s*# Begin ASM.*/;
const startBlock = /\.cfi_startproc/;
const stdInLooking = /<stdin>|^-$|example\.[^/]+$|<source>/;
const tabsRe = /\t/g;

function splitLines(text: string): string[] {
    if (!text) return [];
    const result = text.split(lineRe);
    if (result.length > 0 && result[result.length - 1] === '') return result.slice(0, -1);
    return result;
}

function labelFindFor(asmLines) {
    const isMips = _.any(asmLines, line => !!mipsLabelDefinition.test(line));
    return isMips ? labelFindMips : labelFindNonMips;
}

function fixLabelIndentation(line) {
    const match = line.match(indentedLabelDef);
    if (match) {
        return line.replace(/^\s+/, '');
    } else {
        return line;
    }
}

function hasOpcode(line, inNvccCode) {
    // Remove any leading label definition...
    const match = line.match(labelDef);
    if (match) {
        line = line.substr(match[0].length);
    }
    // Strip any comments
    line = line.split(commentRe, 1)[0];
    // .inst generates an opcode, so also counts
    if (instOpcodeRe.test(line)) return true;
    // Detect assignment, that's not an opcode...
    if (assignmentDef.test(line)) return false;
    if (inNvccCode) {
        return !!hasNvccOpcodeRe.test(line);
    }
    return !!hasOpcodeRe.test(line);
}

function findUsedLabels(asmLines, filterDirectives) {
    const labelsUsed = {};
    const weakUsages = {};
    const labelFind = labelFindFor(asmLines);
    // The current label set is the set of labels all pointing at the current code, so:
    // foo:
    // bar:
    //    add r0, r0, #1
    // in this case [foo, bar] would be the label set for the add instruction.
    let currentLabelSet: string[] = [];
    let inLabelGroup = false;
    let inCustomAssembly = 0;
    let inFunction = false;
    let inNvccCode = false;

    // Scan through looking for definite label usages (ones used by opcodes),
    // and ones that are weakly used: that is, their use is conditional on another label.
    // For example:
    // .foo: .string "moo"
    // .baz: .quad .foo
    //       mov eax, .baz
    // In this case, the '.baz' is used by an opcode, and so is strongly used.
    // The '.foo' is weakly used by .baz.
    // Also, if we have random data definitions within a block of a function (between
    // cfi_startproc and cfi_endproc), we assume they are strong usages. This covers things
    // like jump tables embedded in ARM code.
    // See https://github.com/compiler-explorer/compiler-explorer/issues/2788
    for (let line of asmLines) {
        if (startAppBlock.test(line) || startAsmNesting.test(line)) {
            inCustomAssembly++;
        } else if (endAppBlock.test(line) || endAsmNesting.test(line)) {
            inCustomAssembly--;
        } else if (startBlock.test(line)) {
            inFunction = true;
        } else if (endBlock.test(line)) {
            inFunction = false;
        } else if (cudaBeginDef.test(line)) {
            inNvccCode = true;
        }

        if (inCustomAssembly > 0) line = fixLabelIndentation(line);

        let match = line.match(labelDef);
        if (match) {
            if (inLabelGroup) currentLabelSet.push(match[1]);
            else currentLabelSet = [match[1]];
            inLabelGroup = true;
        } else {
            inLabelGroup = false;
        }
        match = line.match(definesGlobal);
        if (!match) match = line.match(definesWeak);
        if (!match) match = line.match(cudaBeginDef);
        if (match) {
            labelsUsed[match[1]] = true;
        }

        const definesFunction = line.match(definesFunctionRegEx);
        if (!definesFunction && (!line || line[0] === '.')) continue;

        match = line.match(labelFind);
        if (!match) continue;

        if (!filterDirectives || hasOpcode(line, inNvccCode) || definesFunction) {
            // Only count a label as used if it's used by an opcode, or else we're not filtering directives.
            for (const label of match) labelsUsed[label] = true;
        } else {
            // If we have a current label, then any subsequent opcode or data definition's labels are referred to
            // weakly by that label.
            const isDataDefinition = !!dataDefn.test(line);
            const isOpcode = hasOpcode(line, inNvccCode);
            if (isDataDefinition || isOpcode) {
                for (const currentLabel of currentLabelSet) {
                    if (inFunction && isDataDefinition) {
                        // Data definitions in the middle of code should be treated as if they were used strongly.
                        for (const label of match) labelsUsed[label] = true;
                    } else {
                        if (!weakUsages[currentLabel]) weakUsages[currentLabel] = [];
                        for (const label of match) weakUsages[currentLabel].push(label);
                    }
                }
            }
        }
    }

    // Now follow the chains of used labels, marking any weak references they refer
    // to as also used. We iteratively do this until either no new labels are found,
    // or we hit a limit (only here to prevent a pathological case from hanging).
    function markUsed(label) {
        labelsUsed[label] = true;
    }

    const MaxLabelIterations = 10;
    for (let iter = 0; iter < MaxLabelIterations; ++iter) {
        const toAdd: string[] = [];
        _.each(labelsUsed, (t, label) => {
            // jshint ignore:line
            _.each(weakUsages[label], (nowused: string) => {
                if (labelsUsed[nowused]) return;
                toAdd.push(nowused);
            });
        });
        // if (!toAdd) break;
        _.each(toAdd, markUsed);
    }
    return labelsUsed;
}

function parseFiles(asmLines) {
    const files = {};
    for (const line of asmLines) {
        const match = line.match(fileFind);
        if (match) {
            const lineNum = parseInt(match[1]);
            if (match[4] && !line.includes('.cv_file')) {
                // Clang-style file directive '.file X "dir" "filename"'
                files[lineNum] = match[2] + '/' + match[4];
            } else {
                files[lineNum] = match[2];
            }
        }
    }
    return files;
}

function maskRootdir(filepath: string): string {
    // if (filepath) {
    //     // todo: make this compatible with local installations etc
    //     if (process.platform === 'win32') {
    //         return filepath
    //             .replace(/^C:\/Users\/[\w\d-.]*\/AppData\/Local\/Temp\/compiler-explorer-compiler[\w\d-.]*\//, '/app/')
    //             .replace(/^\/app\//, '');
    //     } else {
    //         return filepath.replace(/^\/tmp\/compiler-explorer-compiler[\w\d-.]*\//, '/app/').replace(/^\/app\//, '');
    //     }
    // } else {
    //     return filepath;
    // }
    return filepath;
}

function expandTabs(line: string): string {
    let extraChars = 0;
    return line.replace(tabsRe, (match, offset) => {
        const total = offset + extraChars;
        const spacesNeeded = (total + 8) & 7;
        extraChars += spacesNeeded - 1;
        return '        '.substr(spacesNeeded);
    });
}

function squashHorizontalWhitespace(line: string, atStart: boolean): string {
    const quotes = line.match(findQuotes);
    if (quotes) {
        return (
            squashHorizontalWhitespace(quotes[1], atStart) + quotes[2] + squashHorizontalWhitespace(quotes[3], false)
        );
    }
    return squashHorizontalWhitespace(line, atStart);
}

function filterAsmLine(line: string, filters: ParseFiltersAndOutputOptions): string {
    if (!filters.trim) return line;
    return squashHorizontalWhitespace(line, true);
}

// Get labels which are used in the given line.
function getUsedLabelsInLine(line) {
    const labelsInLine: AsmResultLabel[] = [];

    // Strip any comments
    const instruction = line.split(commentRe, 1)[0];

    // Remove the instruction.
    const params = instruction.replace(instructionRe, '');

    const removedCol = instruction.length - params.length + 1;
    params.replace(identifierFindRe, (label, index) => {
        const startCol = removedCol + index;
        labelsInLine.push({
            name: label,
            range: {
                startCol: startCol,
                endCol: startCol + label.length,
            },
        });
    });

    return labelsInLine;
}

// Remove labels which do not have a definition.
function removeLabelsWithoutDefinition(asm, labelDefinitions) {
    for (const obj of asm) {
        if (obj.labels) {
            obj.labels = obj.labels.filter(label => labelDefinitions[label.name]);
        }
    }
}

// eslint-disable-next-line max-statements
export function processAsm(asmResult: string, filters: ParseFiltersAndOutputOptions): ParsedAsmResult {
    // const startTime = process.hrtime.bigint();

    if (filters.commentOnly) {
        // Remove any block comments that start and end on a line if we're removing comment-only lines.
        asmResult = asmResult.replace(blockComments, '');
    }

    const asm: ParsedAsmResultLine[] = [];
    const labelDefinitions: Record<string, number> = {};

    let asmLines = splitLines(asmResult);
    const startingLineCount = asmLines.length;
    if (filters.preProcessLines !== undefined) {
        asmLines = filters.preProcessLines(asmLines);
    }

    const labelsUsed = findUsedLabels(asmLines, filters.directives);
    const files = parseFiles(asmLines);
    let prevLabel = '';

    let source: AsmResultSource | undefined | null;
    let mayRemovePreviousLabel = true;
    let keepInlineCode = false;

    let lastOwnSource: AsmResultSource | undefined | null;
    const dontMaskFilenames = filters.dontMaskFilenames;

    function maybeAddBlank() {
        const lastBlank = asm.length === 0 || asm[asm.length - 1].text === '';
        if (!lastBlank) asm.push({text: '', source: null, labels: []});
    }

    const handleSource = line => {
        let match = line.match(sourceTag);
        if (match) {
            const file = maskRootdir(files[parseInt(match[1])]);
            const sourceLine = parseInt(match[2]);
            if (file) {
                if (dontMaskFilenames) {
                    source = {
                        file: file,
                        line: sourceLine,
                        mainsource: !!stdInLooking.test(file),
                    };
                } else {
                    source = {
                        file: stdInLooking.test(file) ? null : file,
                        line: sourceLine,
                    };
                }
                const sourceCol = parseInt(match[3]);
                if (!isNaN(sourceCol) && sourceCol !== 0) {
                    source.column = sourceCol;
                }
            } else {
                source = null;
            }
        } else {
            match = line.match(sourceD2Tag);
            if (match) {
                const sourceLine = parseInt(match[1]);
                source = {
                    file: null,
                    line: sourceLine,
                };
            } else {
                match = line.match(sourceCVTag);
                if (match) {
                    // cv_loc reports: function file line column
                    const sourceLine = parseInt(match[3]);
                    const file = maskRootdir(files[parseInt(match[2])]);
                    if (dontMaskFilenames) {
                        source = {
                            file: file,
                            line: sourceLine,
                            mainsource: !!stdInLooking.test(file),
                        };
                    } else {
                        source = {
                            file: stdInLooking.test(file) ? null : file,
                            line: sourceLine,
                        };
                    }
                    const sourceCol = parseInt(match[4]);
                    if (!isNaN(sourceCol) && sourceCol !== 0) {
                        source.column = sourceCol;
                    }
                }
            }
        }
    };

    const handleStabs = line => {
        const match = line.match(sourceStab);
        if (!match) return;
        // cf http://www.math.utah.edu/docs/info/stabs_11.html#SEC48
        switch (parseInt(match[1])) {
            case 68: {
                source = {file: null, line: parseInt(match[2])};
                break;
            }
            case 132:
            case 100: {
                source = null;
                prevLabel = '';
                break;
            }
        }
    };

    const handle6502 = line => {
        const match = line.match(source6502Dbg);
        if (match) {
            const file = maskRootdir(match[1]);
            const sourceLine = parseInt(match[2]);
            if (dontMaskFilenames) {
                source = {
                    file: file,
                    line: sourceLine,
                    mainsource: !!stdInLooking.test(file),
                };
            } else {
                source = {
                    file: stdInLooking.test(file) ? null : file,
                    line: sourceLine,
                };
            }
        } else if (source6502DbgEnd.test(line)) {
            source = null;
        }
    };

    let inNvccDef = false;
    let inNvccCode = false;

    let inCustomAssembly = 0;

    // TODO: Make this function smaller
    // eslint-disable-next-line max-statements
    for (let line of asmLines) {
        if (line.trim() === '') {
            maybeAddBlank();
            continue;
        }

        if (startAppBlock.test(line) || startAsmNesting.test(line)) {
            inCustomAssembly++;
        } else if (endAppBlock.test(line) || endAsmNesting.test(line)) {
            inCustomAssembly--;
        }

        handleSource(line);
        handleStabs(line);
        handle6502(line);

        if (source && (source.file === null || source.mainsource)) {
            lastOwnSource = source;
        }

        // eslint-disable-next-line @typescript-eslint/prefer-includes
        if (endBlock.test(line) || (inNvccCode && line.includes('}'))) {
            source = null;
            prevLabel = '';
            lastOwnSource = null;
        }

        if (filters.libraryCode && !lastOwnSource && source && source.file !== null && !source.mainsource) {
            if (mayRemovePreviousLabel && asm.length > 0) {
                const lastLine = asm[asm.length - 1];

                const labelDef = lastLine.text ? lastLine.text.match(labelDefRegEx) : null;

                if (labelDef) {
                    asm.pop();
                    keepInlineCode = false;
                    delete labelDefinitions[labelDef[1]];
                } else {
                    keepInlineCode = true;
                }
                mayRemovePreviousLabel = false;
            }

            if (!keepInlineCode) {
                continue;
            }
        } else {
            mayRemovePreviousLabel = true;
        }

        if (
            filters.commentOnly &&
            ((commentOnly.test(line) && !inNvccCode) || (commentOnlyNvcc.test(line) && inNvccCode))
        ) {
            continue;
        }

        if (inCustomAssembly > 0) line = fixLabelIndentation(line);

        let match = line.match(labelDef);
        if (!match) match = line.match(assignmentDef);
        if (!match) {
            match = line.match(cudaBeginDef);
            if (match) {
                inNvccDef = true;
                inNvccCode = true;
            }
        }
        if (match) {
            // It's a label definition.
            if (labelsUsed[match[1]] === undefined) {
                // It's an unused label.
                if (filters.labels) {
                    continue;
                }
            } else {
                // A used label.
                prevLabel = match[1];
                labelDefinitions[match[1]] = asm.length + 1;
            }
        }
        if (inNvccDef) {
            if (cudaEndDef.test(line)) inNvccDef = false;
        } else if (!match && filters.directives) {
            // Check for directives only if it wasn't a label; the regexp would
            // otherwise misinterpret labels as directives.
            if (dataDefn.test(line) && prevLabel) {
                // We're defining data that's being used somewhere.
            } else {
                // .inst generates an opcode, so does not count as a directive
                if (directive.test(line) && !instOpcodeRe.test(line)) {
                    continue;
                }
            }
        }

        line = expandTabs(line);
        const text = filterAsmLine(line, filters);

        const labelsInLine = match ? [] : getUsedLabelsInLine(text);

        asm.push({
            text: text,
            source: hasOpcode(line, inNvccCode) ? source || null : null,
            labels: labelsInLine,
        });
    }

    removeLabelsWithoutDefinition(asm, labelDefinitions);

    // const endTime = process.hrtime.bigint();
    return {
        asm: asm,
        labelDefinitions: labelDefinitions,
        // parsingTime: ((endTime - startTime) / BigInt(1000000)).toString(),
        filteredCount: startingLineCount - asm.length,
    };
}
