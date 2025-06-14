import fs from 'node:fs/promises';
import path from 'node:path';
import {describe, expect, it} from 'vitest';

import {GCCCompiler} from '../../lib/compilers/index.js';
import {makeCompilationEnvironment} from '../utils.js';

describe('gcc tests', () => {
    const languages = {'c++': {id: 'c++'}};
    const info = {
        exe: 'foobar',
        remote: true,
        lang: 'c++',
        ldPath: [],
    };
    const gcc = new GCCCompiler(info as any, makeCompilationEnvironment({languages}));

    it('should process gcc opt output', async () => {
        const test = `<source>:5:9: optimized: loop with 1 iterations completely unrolled (header execution count 78082503)
<source>:3:6: note: ***** Analysis failed with vector mode V4SI
<source>:11:6: missed: splitting region at control altering definition _44 = std::basic_filebuf<char>::open (&fs._M_filebuf, "myfile", 16);
/opt/compiler-explorer/gcc-14.1.0/include/c++/14.1.0/bits/basic_ios.h:466:59: missed: statement clobbers memory: std::ios_base::ios_base (&MEM[(struct basic_ios *)&fs + 248B].D.46591);
`;

        const dirPath = await gcc.newTempDir();
        const optPath = path.join(dirPath, 'temp.out');
        await fs.writeFile(optPath, test);
        const dummyCompilationResult = {optPath: optPath, code: 0, stdout: [], stderr: [], timedOut: false};
        gcc.compiler.optArg = '-fopt-info-all=temp.out';
        expect(await gcc.processOptOutput(dummyCompilationResult)).toEqual([
            {
                Args: [],
                DebugLoc: {File: '<source>', Line: 5, Column: 9},
                Function: '',
                Name: '',
                Pass: '',
                displayString: 'loop with 1 iterations completely unrolled (header execution count 78082503)',
                optType: 'Passed',
            },
            {
                Args: [],
                DebugLoc: {File: '<source>', Line: 3, Column: 6},
                Function: '',
                Name: '',
                Pass: '',
                displayString: '***** Analysis failed with vector mode V4SI',
                optType: 'Analysis',
            },
            {
                Args: [],
                DebugLoc: {File: '<source>', Line: 11, Column: 6},
                Function: '',
                Name: '',
                Pass: '',
                displayString:
                    'splitting region at control altering definition _44 = std::basic_filebuf<char>::open (&fs._M_filebuf, "myfile", 16);',
                optType: 'Missed',
            },
        ]);
    });
});
