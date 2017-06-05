define(function (require) {
    'use strict';
    var jquery = require('jquery');
    var monaco = require('monaco');
    var cpp = require('vs/basic-languages/src/cpp');

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
            'print',
            'programCount',
            'programIndex',
            'reference',
            'soa',
            'sync',
            'task',
            'taskCount',
            'taskCount0',
            'taskCount1',
            'taskCount3',
            'taskIndex',
            'taskIndex0',
            'taskIndex1',
            'taskIndex2',
            'uniform',
            'varying'
        );

        return ispc;
    }

    monaco.languages.register({id: 'ispc'});
    monaco.languages.setMonarchTokensProvider('ispc', definition());
});
