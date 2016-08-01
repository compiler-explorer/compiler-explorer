var fs = require('fs');
var child_process = require('child_process');

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
    //res.end(JSON.stringify(result));
    res.end(JSON.stringify({
        computedDiff: wdiffResult.stdout.toString()
        //computedDiff: "aaa\nbbb[-ccc-]\n[-ddd-]eee\n[-fff-]\nsafe"
        }));
}
module.exports = {
    diffHandler: diffHandler,
};
