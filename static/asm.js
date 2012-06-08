
CodeMirror.defineMode("asm", function() {
  function tokenString(quote) {
    return function(stream) {
      var escaped = false, next, end = false;
      while ((next = stream.next()) != null) {
        if (next == quote && !escaped) {end = true; break;}
        escaped = !escaped && next == "\\";
      }
      return "string";
    };
  }

  return {
    token: function(stream) {
        if (stream.match(/^.+:$/)) {
            return "variable-2";
        }
        if (stream.sol() && stream.match(/^\s*\.\w+/)) {
            return "header";
        }
        if (stream.sol() && stream.match(/^\s\w+/)) {
            return "keyword";
        }
        if (stream.eatSpace()) return null;
        var ch = stream.next();
        if (ch == '"' || ch == "'") {
            return tokenString(ch)(stream);
        }
        if (/[\[\]{}\(\),;\:]/.test(ch)) return null;
        if (/[\d$]/.test(ch) || (ch == '-' && stream.peek().match(/[0-9]/))) {
            stream.eatWhile(/[\w\.]/);
            return "number";
        }
        if (ch == '%') {
            stream.eatWhile(/\w+/);
            return "variable-3";
        }
        if (ch == '#') {
            stream.eatWhile(/.*/);
            return "comment";
        }
        stream.eatWhile(/[\w\$_]/);
        return "word";
    }
  };
});

CodeMirror.defineMIME("text/x-asm", "asm");

function processAsm(asm, filters) {
    var result = [];
    var asmLines = asm.split("\n");
    var labelsUsed = {};
    var labelFind = /\.[a-zA-Z0-9$_.]+/g;
    var files = {};
    var fileFind = /^\s*\.file\s+(\d+)\s+"([^"]+)"$/;
    var hasOpcode = /^\s*([a-zA-Z0-9$_][a-zA-Z0-9$_.]*:\s*)?[a-zA-Z].*/;
    $.each(asmLines, function(_, line) {
        if (line == "" || line[0] == ".") return;
        var match = line.match(labelFind);
        if (match && (!filters.directives || line.match(hasOpcode))) {
            // Only count a label as used if it's used by an opcode, or else we're not filtering directives.
            $.each(match, function(_, label) { labelsUsed[label] = true; });
        }
        match = line.match(fileFind);
        if (match) {
            files[parseInt(match[1])] = match[2];
        }
    });

    var directive = /^\s*\..*$/;
    var labelDefinition = /^(\.[a-zA-Z0-9$_.]+):/;
    var commentOnly = /^\s*(#|@|\/\/).*/;
    var sourceTag = /^\s*\.loc\s+(\d+)\s+(\d+).*/;
    var stdInLooking = /.*<stdin>|-/;
    var source = null;
    $.each(asmLines, function(_, line) {
        var match;
        if (line.trim() == "") return;
        if (match = line.match(sourceTag)) {
            source = null;
            var file = files[parseInt(match[1])];
            if (file && file.match(stdInLooking)) {
                source = parseInt(match[2]);
            }
        }

        if (filters.commentOnly && line.match(commentOnly)) return;
        match = line.match(labelDefinition);
        if (match && labelsUsed[match[1]] == undefined) {
            if (filters.labels) return;
        }
        if (!match && filters.directives) {
            // Check for directives only if it wasn't a label; the regexp would
            // otherwise misinterpret labels as directives.
            match = line.match(directive);
            if (match) return;
        }

        var hasOpcodeMatch = line.match(hasOpcode);
        result.push({text: line, source: hasOpcodeMatch ? source : null});
    });
    return result;
}

