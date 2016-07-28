var fs = require('fs');
var child_process = require('child_process');

function cleanAndGetIndexes(text) {
    var addRules = {
        name: "add",
        openTag: "{+",
        replaceOpenTag: "",
        closeTag: "+}",
        replaceCloseTag: "",
    }
    var delRules = {
        name: "del",
        openTag: "[-",
        replaceOpenTag: "",
        closeTag: "-]",
        replaceCloseTag: "",
    }
    var rules = [addRules, delRules];

    TagTypeEnum = {
        OPENING : 1,
        CLOSING : 2
    }

    function tagLookup(rules, text, pos) {
        var seen = false;
        var type = null;
        var rule = null;
        for (var i = 0; i<rules.length; i++) {
            var candidateTag = text.slice(pos, pos+rules[i].openTag.length);
            // console.log("Comparing "+candidateTag+" with "+rules[i].openTag);
            if (rules[i].openTag == candidateTag) {
                seen = true;
                type = TagTypeEnum.OPENING;
                rule = i;
                break;
            }
            var candidateTag = text.slice(pos, pos+rules[i].closeTag.length);
            // console.log("Comparing "+candidateTag+" with "+rules[i].closeTag);
            if (rules[i].closeTag == candidateTag) {
                seen = true;
                type = TagTypeEnum.CLOSING;
                rule = i;
                break;
            }
        }

        return {seen: seen,
            rule: rule,
            type: type}
    }

    var finalText = "";
    var posInFinalText = 0; // character that is going to be treated
    // The position in the original text: 
    var posInText = 0; // character that is going to be treated
    StateEnum = {
        OUTSIDE_TAG : 1,
        INSIDE_TAG : 2
    }
    var state = StateEnum.OUTSIDE_TAG;
    var zones = [[],[]];
    var currentTextBeginPos = 0;
    var currentTextEndPos = null;
    var currentTagBeginPos = null;
    var currentTagEndPos = null;
    var currentTag = null;

    function forward() {
        posInFinalText = posInFinalText + 1;
        posInText = posInText + 1;
    }

    function seenOpeningTag() {
        memorizeText();
        currentTagBeginPos = posInFinalText;
        finalText = finalText.concat(rules[currentTag].replaceOpenTag);
        posInFinalText = posInFinalText + rules[currentTag].replaceOpenTag.length;
        posInText = posInText + rules[currentTag].openTag.length;
        currentTextBeginPos = posInText;
    }

    function seenClosingTag() {
        memorizeText();
        finalText = finalText.concat(rules[currentTag].replaceCloseTag);
        posInFinalText = posInFinalText + rules[currentTag].replaceCloseTag.length;
        posInText = posInText + rules[currentTag].closeTag.length;
        currentTagEndPos = posInFinalText - 1;
        zones[currentTag].push({begin: currentTagBeginPos, end: currentTagEndPos});
        currentTextBeginPos = posInText;
    }

    function memorizeText() {
        currentTextEndPos = posInText - 1;
        if (currentTextEndPos >= currentTextBeginPos) {
            finalText = finalText.concat(text.slice(currentTextBeginPos,currentTextEndPos+1));
        }
    }

    function end() {
        memorizeText();
    }

    function log_error(string) {
        console.log(string);
    }

    while (posInText < text.length) {
        var tag = tagLookup(rules, text, posInText);
        if(tag.seen && tag.type == TagTypeEnum.OPENING) {
            if (state != StateEnum.OUTSIDE_TAG) {
                log_error("Opening tag while not outside tag (tags cannot be nested)");
                return null;
            }
            currentTag = tag.rule;
            seenOpeningTag();
            state = StateEnum.INSIDE_TAG;
        } else if(tag.seen && tag.type == TagTypeEnum.CLOSING) {
            if (state != StateEnum.INSIDE_TAG) {
                log_error("Closing tag while not inside tag.");
                return null;
            }
            if (currentTag != tag.rule) {
                log_error("Closing tag, but not of the same type as previously opened.");
                return null;
            }
            seenClosingTag();
            state = StateEnum.OUTSIDE_TAG;
        } else {
            forward();
        }
    }
    end();

    // console.log(JSON.stringify(finalText)+JSON.stringify(zones));
    return {text: finalText, zones: zones};
}

function diffHandler(req, res, next) {
    // console.log("req: "+JSON.stringify(JSON.decycle(req)));
    // console.log("");
    // console.log("res: "+JSON.stringify(JSON.decycle(res)));
    // console.log("");
    // console.log("next: "+JSON.stringify(JSON.decycle(next)));
    var before = req.body.before;
    var after = req.body.after;
    if (before === undefined) {
        console.log("Warning : Bad request : wrong \"before\"");
        //return next(new Error("Bad request : wrong \"before\""));
    }
    if (after === undefined) {
        console.log("Warning : Bad request : wrong \"after\"");
        //return next(new Error("Bad request : wrong \"after\""));
    }
    //console.log("Before: ");
    //console.log(before);
    //console.log("After: ");
    //console.log(after);
    // TODO : make async the two creation of temp files + call to wdiff
    var before_temp_file = "/tmp/gcc-explorer-before"
    fs.writeFileSync(before_temp_file, before);

    var after_temp_file = "/tmp/gcc-explorer-after"
    fs.writeFileSync(after_temp_file, after);

    var wdiff_exe = "/work1/gdevillers/compiler-explorer/external/wdiff-1.2.2/src/wdiff";
    var maxSize = 100000;
    var wdiffResult = child_process.spawnSync(
        "/work1/gdevillers/compiler-explorer/external/wdiff-1.2.2/src/wdiff", 
        ["/tmp/gcc-explorer-before", "/tmp/gcc-explorer-after"],
        {maxBuffer: 100000});

    res.set('Content-Type', 'application/json');
    var cleaned = cleanAndGetIndexes(wdiffResult.stdout.toString());
    if (cleaned == null) {
        res.end(JSON.stringify({
            computedDiff: "Failed to clean the diff",
            zones: null
        }));
    } else {
    res.end(JSON.stringify({
        computedDiff: cleaned.text,
        //computedDiff: cleaned.text+
        //    "\n//// for reference: ////\n"+ // for debug only
        //    wdiffResult.stdout.toString(),
        zones: cleaned.zones
        //computedDiff: "aaa\nbbb[-ccc-]\n[-ddd-]eee\n[-fff-]\nsafe"
        //computedDiff: "[-aaa-]"
        //computedDiff: "aaa"
        //computedDiff: "aa[--]a"
        //computedDiff: "aa[-b-]"
        }));
    }
}
module.exports = {
    diffHandler: diffHandler,
};
