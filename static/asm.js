function processAsm(asm, filters) {
    var result = [];
    var asmLines = asm.split("\n");
    var labelsUsed = {};
    var labelFind = /[.a-zA-Z0-9_][a-zA-Z0-9$_.]*/g;
    var files = {};
    var prevLabel = "";
    var dataDefn = /\.(string|asciz|ascii|[1248]?byte|short|word|long|quad|value|zero)/;
    var fileFind = /^\s*\.file\s+(\d+)\s+"([^"]+)"$/;
    var hasOpcode = /^\s*([a-zA-Z0-9$_][a-zA-Z0-9$_.]*:\s*)?[a-zA-Z].*/;
    asmLines.forEach(function(line) {
        if (line == "" || line[0] == ".") return;
        var match = line.match(labelFind);
        if (match && (!filters.directives || line.match(hasOpcode))) {
            // Only count a label as used if it's used by an opcode, or else we're not filtering directives.
            match.forEach(function(label) { labelsUsed[label] = true; });
        }
        match = line.match(fileFind);
        if (match) {
            files[parseInt(match[1])] = match[2];
        }
    });

    var directive = /^\s*\..*$/;
    var labelDefinition = /^([a-zA-Z0-9$_.]+):/;
    var commentOnly = /^\s*(#|@|\/\/).*/;
    var sourceTag = /^\s*\.loc\s+(\d+)\s+(\d+).*/;
    var stdInLooking = /.*<stdin>|-/;
    var endBlock = /\.(cfi_endproc|data|text|section)/;
    var source = null;
    asmLines.forEach(function(line) {
        var match;
        if (line.trim() == "") {
            result.push({text:"", source:null});
            return;
        }
        if (match = line.match(sourceTag)) {
            source = null;
            var file = files[parseInt(match[1])];
            if (file && file.match(stdInLooking)) {
                source = parseInt(match[2]);
            }
        }
        if (line.match(endBlock)) {
            source = null;
            prevLabel = null;
        }

        if (filters.commentOnly && line.match(commentOnly)) return;

        match = line.match(labelDefinition);
        if (match) {
            // It's a label definition.
            if (labelsUsed[match[1]] == undefined) {
                // It's an unused label.
                if (filters.labels) return;
            } else {
                // A used label.
                prevLabel = match;
            }
        }
        if (!match && filters.directives) {
            // Check for directives only if it wasn't a label; the regexp would
            // otherwise misinterpret labels as directives.
            if (line.match(dataDefn) && prevLabel) {
                // We're defining data that's being used somewhere.
            } else {
                if (line.match(directive)) return;
            }
        }

        var hasOpcodeMatch = line.match(hasOpcode);
        result.push({text: line, source: hasOpcodeMatch ? source : null});
    });
    return result;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        processAsm: processAsm
    };
}
