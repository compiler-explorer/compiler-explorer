define(function (require) {
    "use strict";

    var NumRainbowColours = 12;

    function clearBackground(editor) {
        for (var i = 0; i < editor.lineCount(); ++i) {
            editor.removeLineClass(i, "background", null);
        }
    }

    function applyColours(editor, colours) {
        editor.operation(function () {
            clearBackground(editor);
            _.each(colours, function (ordinal, line) {
                editor.addLineClass(parseInt(line),
                    "background", "rainbow-" + (ordinal % NumRainbowColours));
            });
        });
    }

    return {
        applyColours: applyColours,
        clearBackground: clearBackground
    };
});
