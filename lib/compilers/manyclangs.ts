// Copyright (c) 2022, Compiler Explorer Authors
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

import path from 'path';

import {CompilerInfo, ICompilerRepository, QueryRevisionsResult} from '../../types/compiler.interfaces';
import {BaseCompiler} from '../base-compiler';

/**
 * The manyclangs cmpiler library.
 */
export class ManyClangsCompiler extends BaseCompiler implements ICompilerRepository {
    static get key() {
        return 'manyclangs';
    }

    constructor(compilerInfo: CompilerInfo & Record<string, any>, env: any) {
        super(compilerInfo, env);
        this.compiler.isCompilerRepository = true;
    }

    /**
     * Returns the available clang revisions.
     */
    queryRevisions(q: string, limit: number, offset: number): Promise<QueryRevisionsResult> {
        // TODO: Query the manyclangs wrapper we've specified in the properties.
        // I suspect caching the result of elfshaker list in a file and reading it using the readline module, relying
        // on the internal node bufferring would be fairly performant.
        const data = [
            '20220330-03207T152227-0bda12b5bcead78',
            '20220330-03208T152340-e8e32e5714e4855',
            '20220330-03209T151522-ee51aefba043dbb',
            '20220330-03210T152824-f1cb816f9085aaf',
            '20220330-03211T153456-4d5bf24e3df21c9',
            '20220330-03214T154140-4d4ec37037f5f96',
            '20220331-03207T152227-0bda12b5bcea578',
            '20220331-03208T152340-e8e32e5714e4855',
            '20220331-03209T151522-ee51aefba043bbb',
            '20220331-03210T152824-f1cb816f90859af',
            '20220331-03211T153456-4d5b824e3df21c9',
            '20220331-03214T154140-4dfec37037f5f96',
            '20220331-03218T160353-2267549296dabfe',
            '20220331-03219T161634-db17ebd593f67a9',
            '20220331-03220T161634-898d5776ec3ab8a',
            '20220331-03221T163401-19054163e11a663',
            '20220331-03223T170551-fac17299243b8bf',
            '20220331-03224T170552-535211c3ebf057b',
            '20220331-03225T172535-1a6aa8b1952b289',
            '20220331-03227T173821-46774df30715944',
            '20220331-03228T174539-0e890904ea342ab',
            '20220331-03230T175034-c7639f896c27642',
            '20220331-03231T180048-14e3650f01d158f',
            '20220331-03233T182141-1ae449f9a33b9c8',
            '20220331-03236T190630-2e55bc9f3c23367',
            '20220331-03238T192251-e6e5e3e025ec1cc',
            '20220331-03239T192543-1c5663458bbbee0',
            '20220331-03243T194012-33e197112a21b24',
            '20220331-03244T201246-585c85abe545a42',
            '20220331-03246T202912-596af141b24c988',
            '20220331-03247T203001-ae8d35b8ee760b8',
            '20220331-03248T203001-395f8ccfc9742dc',
            '20220331-03249T203001-cc2e2b80a1f36a2',
            '20220331-03250T204623-0fb6856afffb3cf',
            '20220331-03251T204912-fe528e72163371e',
            '20220331-03254T214517-83bde93cef369f8',
            '20220331-03255T215107-de4bcdc2baccc0c',
            '20220331-03256T215441-fc7573f29c79a4b',
            '20220331-03258T220413-6d481adb35655da',
            '20220331-03259T224922-f942cde61a96e67',
            '20220331-03260T224922-4a8665e23ed875c',
            '20220331-03261T224923-4d72acf9913dc53',
            '20220331-03262T224923-f635be30144d890',
            '20220331-03263T224923-14744622edac9ea',
        ];
        // Return in descending order
        data.sort().reverse();

        const matches = data.filter(x => x.includes(q));
        return Promise.resolve({
            revisions: matches.slice(offset, offset + limit),
            total: matches.length,
        });
    }
}
