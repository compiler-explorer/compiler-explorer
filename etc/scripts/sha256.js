import {getHashDigest} from 'loader-utils';

export default function (source) {
    return `export default "${getHashDigest(source, 'sha256', 'hex', 16)}";`;
}
