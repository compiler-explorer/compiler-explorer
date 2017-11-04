var Promise = require('es6-promise').Promise;
var _loaded = false
var _loadPromise = null;

// Returns promise that will be fulfilled when monaco is available
var waitForMonaco = function() {
    if (_loadPromise) {
        return _loadPromise;
    }

    _loadPromise = new Promise(function(resolve, reject) {
        if (typeof(window.monaco) === 'object') {
            resolve(window.monaco);
            return window.monaco;
        }

        window.require(['vs/editor/editor.main'], function() {
            _loaded = true;
            resolve(window.monaco);
        });
    });
    return _loadPromise;
};

module.exports = waitForMonaco;