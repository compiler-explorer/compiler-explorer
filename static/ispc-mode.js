'use strict';

var jquery = require('jquery');
var cpp = require('./vs/basic-languages/src/cpp');

function definition() {
    var ispc = jquery.extend(true, {}, cpp.language); // deep copy

    ispc.tokenPostfix = '.ispc';

    ispc.keywords.push(
        'cbreak',
        'ccontinue',
        'cdo',
        'cfor',
        'cif',
        'creturn',
        'cwhile',
        'delete',
        'export',
        'foreach',
        'foreach_active',
        'foreach_tiled',
        'foreach_unique',
        'int16',
        'int32',
        'int64',
        'int8',
        'launch',
        'new',
        'operator',
        'programCount',
        'programIndex',
        'reference',
        'soa',
        'sync',
        'task',
        'taskCount',
        'taskCount0',
        'taskCount1',
        'taskCount2',
        'taskIndex',
        'taskIndex0',
        'taskIndex1',
        'taskIndex2',
        'threadCount',
        'threadIndex',
        'uniform',
        'unmasked',
        'varying'
    );
}

monaco.languages.register({id: 'ispc'});
monaco.languages.setLanguageConfiguration('ispc', cpp.conf);
monaco.languages.setMonarchTokensProvider('ispc', definition());

