import {Transform, TransformCallback} from 'stream';

import * as R from 'ramda';
import * as YAML from 'yamljs';

type Path = string;
type OptType = 'Missed' | 'Passed' | 'Analysis';

interface OptInfo {
    optType: OptType;
    displayString: string;
}

interface LLVMOptInfo extends OptInfo {
    Pass: string;
    Name: string;
    DebugLoc: DebugLoc;
    Function: string;
    Args: Array<object>;
}

interface DebugLoc {
    File: Path;
    Line: number;
    Column: number;
}

function DisplayOptInfo(optInfo: LLVMOptInfo) {
    return optInfo.Args.reduce((acc, x) => {
        return (
            acc + R.pipe(R.partial(R.pickBy, [(v: any, k: string) => k !== 'DebugLoc']), R.toPairs, R.head, R.last)(x)
        );
    }, '');
}

const optTypeMatcher = /---\s(.*)\r?\n/;
const docStart = '---';
const docEnd = '\n...';
const IsDocumentStart = (x: string) => x.substring(0, 3) === docStart;
const FindDocumentEnd = (x: string) => {
    const index = x.indexOf(docEnd);
    return {found: index > -1, endpos: index + docEnd.length};
};

export class LLVMOptTransformer extends Transform {
    _buffer: string;
    constructor(options: object) {
        super(R.merge(options || {}, {objectMode: true}));
        this._buffer = '';
    }
    override _flush(done: TransformCallback) {
        this.processBuffer();
        done();
    }
    override _transform(chunk: any, encoding: string, done: TransformCallback) {
        this._buffer += chunk.toString();
        //buffer until we have a start and and end
        //if at any time i care about improving performance stash the offset
        this.processBuffer();
        done();
    }
    processBuffer() {
        while (IsDocumentStart(this._buffer)) {
            const {found, endpos} = FindDocumentEnd(this._buffer);
            if (found) {
                const [head, tail] = R.splitAt(endpos, this._buffer);
                const optTypeMatch = head.match(optTypeMatcher);
                const opt = YAML.parse(head);
                if (!optTypeMatch) {
                    console.warn('missing optimization type');
                } else {
                    opt.optType = optTypeMatch[1].replace('!', '');
                }
                opt.displayString = DisplayOptInfo(opt);
                this.push(opt as LLVMOptInfo);
                this._buffer = tail.replace(/^\n/, '');
            } else {
                break;
            }
        }
    }
}
