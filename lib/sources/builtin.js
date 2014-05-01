(function() {

var props = require('../properties.js'),
    path = require('path'),
    fs = require('fs');

var sourcePath = props.get('builtin', 'sourcepath', './examples/c++');
var sourceMatch = new RegExp(props.get('builtin', 'extensionRe', '.*\.cpp$'));
var examples = fs.readdirSync(sourcePath)
    .filter(function(file) { return file.match(sourceMatch); })
    .map(function(file) {
        var nicename = file.replace(/\.cpp$/, '');
        return { urlpart: nicename, name: nicename.replace(/_/g, ' '), path: path.join(sourcePath, file) };
    }).sort(function(x,y) { return y.name < x.name; });
var byUrlpart = {};
examples.forEach(function(e) { byUrlpart[e.urlpart] = e.path });

function load(filename, callback) {
    var path = byUrlpart[filename];
    if (!path) { callback("No such path"); return; }
    fs.readFile(path, 'utf-8', function(err, res) {
        if (err) { callback(err); return; }
        callback(null, {file: res});
    });
}

function list(callback) {
    callback(null, examples.map(function(example) {
        return {urlpart: example.urlpart, name: example.name};
    }));
}

exports.load = load;
exports.save = null;
exports.list = list;
exports.name = "Examples";
exports.urlpart = "builtin";

}).call(this);
