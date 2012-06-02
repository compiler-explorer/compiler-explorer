
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
        if (/[\d$]/.test(ch)) {
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

function filterAsm(asm, filters) {
    var result = [];
    var asmLines = asm.split("\n");
    var labelsUsed = {};
    var labelFind = /\.[a-zA-Z0-9$_.]+/g;
    $.each(asmLines, function(_, line) {
        if (line == "" || line[0] == ".") return;
        var match = line.match(labelFind);
        if (match) $.each(match, function(_, label) { labelsUsed[label] = true; });
    });
    var directive = /^\s*\..*$/;
    var labelDefinition = /^(\.[a-zA-Z0-9$_.]+):/;
    var commentOnly = /^\s*#.*/;
    $.each(asmLines, function(_, line) {
        if (line.trim() == "") return;
        if (filters.commentOnly && line.match(commentOnly)) return;
        var match = line.match(labelDefinition);
        if (match && labelsUsed[match[1]] == undefined) {
            if (filters.labels) return;
        }
        if (!match && filters.directives) {
            // Check for directives only if it wasn't a label; the regexp would
            // otherwise misinterpret labels as directives.
            match = line.match(directive);
            if (match) return;
        }
        result.push(line);
    });
    return result.join("\n");
}

