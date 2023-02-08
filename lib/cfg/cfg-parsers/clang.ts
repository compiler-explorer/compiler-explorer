// Copyright (c) 2023, Compiler Explorer Authors
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

import _ from 'underscore';

import {ResultLine} from '../../../types/resultline/resultline.interfaces';
import * as utils from '../../utils';

import {BaseCFGParser} from './base';

export class ClangCFGParser extends BaseCFGParser {
    static override get key() {
        return 'clang';
    }

    override filterData(assembly: ResultLine[]) {
        const jmpLabelRegex = /\.LBB\d+_\d+:/;
        const isCode = x => x && x.text && (x.source !== null || jmpLabelRegex.test(x.text) || this.isFunctionName(x));

        const removeComments = (x: ResultLine) => {
            const pos_x86 = x.text.indexOf('# ');
            const pos_arm = x.text.indexOf('// ');
            if (pos_x86 !== -1) x.text = utils.trimRight(x.text.substring(0, pos_x86));
            if (pos_arm !== -1) x.text = utils.trimRight(x.text.substring(0, pos_arm));
            return x;
        };

        return this.filterTextSection(assembly).map(_.clone).filter(isCode).map(removeComments);
    }

    override extractNodeName(inst: string) {
        return inst.match(/\.LBB\d+_\d+/) + ':';
    }
}
