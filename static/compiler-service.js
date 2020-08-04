// Copyright (c) 2017, Matt Godbolt
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

'use strict';
var Sentry = require('@sentry/browser');
var $ = require('jquery');
var _ = require('underscore');
var LruCache = require('lru-cache');
var options = require('./options');
var Promise = require('es6-promise').Promise;

function CompilerService(eventHub) {
    this.base = window.httpRoot;
    this.allowStoreCodeDebug = true;
    this.cache = new LruCache({
        max: 200 * 1024,
        length: function (n) {
            return JSON.stringify(n).length;
        },
    });
    this.compilersByLang = {};
    _.each(options.compilers, function (compiler) {
        if (!this.compilersByLang[compiler.lang]) this.compilersByLang[compiler.lang] = {};
        this.compilersByLang[compiler.lang][compiler.id] = compiler;
    }, this);
    // settingsChange is triggered on page load
    eventHub.on('settingsChange', function (newSettings) {
        this.allowStoreCodeDebug = newSettings.allowStoreCodeDebug;
    }, this);
}

CompilerService.prototype.getDefaultCompilerForLang = function (langId) {
    return options.defaultCompiler[langId];
};

CompilerService.prototype.processFromLangAndCompiler = function (languageId, compilerId) {
    var langId = languageId;
    var foundCompiler = null;
    try {
        if (langId) {
            if (!compilerId) {
                compilerId = this.getDefaultCompilerForLang(langId);
            }

            foundCompiler = this.findCompiler(langId, compilerId);
            if (!foundCompiler) {
                var compilers = this.getCompilersForLang(langId);
                foundCompiler = compilers[_.first(_.keys(compilers))];
            }
        } else if (compilerId) {
            var matchingCompilers =_.map(options.languages, function (lang) {
                var compiler = this.findCompiler(lang.id, compilerId);
                if (compiler) {
                    return {
                        langId: lang.id,
                        compiler: compiler,
                    };
                }
                return null;
            }, this);

            return _.find(matchingCompilers, function (match) {
                return (match !== null);
            });
        } else {
            var firstLang = _.first(_.values(options.languages));
            if (firstLang) {
                return this.processFromLangAndCompiler(firstLang.id, compilerId);
            }
        }
    } catch (e) {
        Sentry.captureException(e);
    }

    return {
        langId: langId,
        compiler: foundCompiler,
    };
};

CompilerService.prototype.getGroupsInUse = function (langId) {
    return _.chain(this.getCompilersForLang(langId))
        .map()
        .uniq(false, function (compiler) {
            return compiler.group;
        })
        .map(function (compiler) {
            return {value: compiler.group, label: compiler.groupName || compiler.group};
        })
        .sortBy('label')
        .value();
};

CompilerService.prototype.getCompilersForLang = function (langId) {
    return this.compilersByLang[langId] || {};
};

CompilerService.prototype.findCompilerInList = function (compilers, compilerId) {
    if (compilers && compilers[compilerId]) {
        return compilers[compilerId];
    }
    return _.find(compilers, function (compiler) {
        return compiler.alias.includes(compilerId);
    });
};

CompilerService.prototype.findCompiler = function (langId, compilerId) {
    if (!compilerId) return null;
    var compilers = this.getCompilersForLang(langId);
    return this.findCompilerInList(compilers, compilerId);
};

function handleRequestError(request, reject, xhr, textStatus, errorThrown) {
    var error = errorThrown;
    if (!error) {
        switch (textStatus) {
            case 'timeout':
                error = 'Request timed out';
                break;
            case 'abort':
                error = 'Request was aborted';
                break;
            case 'error':
                switch (xhr.status) {
                    case 500:
                        error = 'Request failed: internal server error';
                        break;
                    case 504:
                        error = 'Request failed: gateway timeout';
                        break;
                    default:
                        error = 'Request failed: HTTP error code ' + xhr.status;
                        break;
                }
                break;
            default:
                error = 'Error sending request';
                break;
        }
    }
    reject({
        request: request,
        error: error,
    });
}

CompilerService.prototype.submit = function (request) {
    request.allowStoreCodeDebug = this.allowStoreCodeDebug;
    var jsonRequest = JSON.stringify(request);
    if (options.doCache) {
        var cachedResult = this.cache.get(jsonRequest);
        if (cachedResult) {
            return Promise.resolve({
                request: request,
                result: cachedResult,
                localCacheHit: true,
            });
        }
    }
    return new Promise(_.bind(function (resolve, reject) {
        var bindHandler = _.partial(handleRequestError, request, reject);
        var compilerId = encodeURIComponent(request.compiler);
        $.ajax({
            type: 'POST',
            url: window.location.origin + this.base + 'api/compiler/' + compilerId + '/compile',
            dataType: 'json',
            contentType: 'application/json',
            data: jsonRequest,
            success: _.bind(function (result) {
                if (result && result.okToCache && options.doCache) {
                    this.cache.set(jsonRequest, result);
                }
                resolve({
                    request: request,
                    result: result,
                    localCacheHit: false,
                });
            }, this),
            error: bindHandler,
        });
    }, this));
};

CompilerService.prototype.requestPopularArguments = function (compilerId, options) {
    return new Promise(_.bind(function (resolve, reject) {
        var bindHandler = _.partial(handleRequestError, compilerId, reject);
        $.ajax({
            type: 'POST',
            url: window.location.origin + this.base + 'api/popularArguments/' + compilerId,
            dataType: 'json',
            data: JSON.stringify({
                usedOptions: options,
                presplit: false,
            }),
            success: _.bind(function (result) {
                resolve({
                    request: compilerId,
                    result: result,
                    localCacheHit: false,
                });
            }, this),
            error: bindHandler,
        });
    }, this));
};

CompilerService.prototype.expand = function (source) {
    var includeFind = /^\s*#\s*include\s*["<](https?:\/\/[^>"]+)[>"]/;
    var lines = source.split('\n');
    var promises = [];
    _.each(lines, function (line, lineNumZeroBased) {
        var match = line.match(includeFind);
        if (match) {
            promises.push(new Promise(function (resolve) {
                var req = $.get(match[1], function (data) {
                    data = '#line 1 "' + match[1] + '"\n' + data + '\n\n#line ' +
                        (lineNumZeroBased + 1) + ' "<stdin>"\n';

                    lines[lineNumZeroBased] = data;
                    resolve();
                });
                req.fail(function () {
                    resolve();
                });
            }));
        }
    });
    return Promise.all(promises).then(function () {
        return lines.join('\n');
    });
};

CompilerService.prototype.getSelectizerOrder = function () {
    return [
        {field: '$order'},
        {field: '$score'},
        {field: 'name'},
    ];
};

module.exports = CompilerService;
