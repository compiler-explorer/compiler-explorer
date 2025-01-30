// Copyright (c) 2021, Compiler Explorer Authors
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

import bigInt from 'big-integer';
import {addDigitSeparator} from '../shared/common-utils.js';

export function updateAndCalcTopBarHeight(domRoot: JQuery, topBar: JQuery, hideable: JQuery): number {
    let topBarHeight = 0;
    if (!topBar.hasClass('d-none')) {
        hideable.show();
        const topBarHeightMax = topBar.outerHeight(true) ?? 0;
        hideable.hide();
        const topBarHeightMin = topBar.outerHeight(true) ?? 0;
        topBarHeight = topBarHeightMin;
        if (topBarHeightMin === topBarHeightMax) {
            hideable.show();
        }
    }
    return topBarHeight;
}

export function formatDateTimeWithSpaces(d: Date) {
    const t = (x: string) => x.slice(-2);
    // Hopefully some day we can use the temporal api to make this less of a pain
    return (
        `${d.getFullYear()} ${t('0' + (d.getMonth() + 1))} ${t('0' + d.getDate())}` +
        `${t('0' + d.getHours())} ${t('0' + d.getMinutes())} ${t('0' + d.getSeconds())}`
    );
}

export function formatISODate(dt: Date, full = false) {
    const month = '' + (dt.getUTCMonth() + 1);
    const day = '' + dt.getUTCDate();
    const hrs = '' + dt.getUTCHours();
    const min = '' + dt.getUTCMinutes();
    const today = new Date(Date.now());
    if (full || dt.toDateString() === today.toDateString()) {
        return (
            dt.getUTCFullYear() +
            '-' +
            month.padStart(2, '0') +
            '-' +
            day.padStart(2, '0') +
            ' ' +
            hrs.padStart(2, '0') +
            ':' +
            min.padStart(2, '0')
        );
    } else {
        return dt.getUTCFullYear() + '-' + month.padStart(2, '0') + '-' + day.padStart(2, '0');
    }
}

const hexLike = /^(#?[$]|0x)([0-9a-fA-F]+)$/;
const hexLike2 = /^(#?)([0-9a-fA-F]+)H$/;
const decimalLike = /^(#?)(-?[0-9]+)$/;
const ptxFloat32 = /^0[fF]([0-9a-fA-F]{8})$/;
const ptxFloat64 = /^0[dD]([0-9a-fA-F]{16})$/;

function parseNumericValue(value: string): bigInt.BigInteger | null {
    const hexMatch = hexLike.exec(value) || hexLike2.exec(value);
    if (hexMatch) return bigInt(hexMatch[2], 16);

    const hexMatchPTX = ptxFloat32.exec(value) ?? ptxFloat64.exec(value);
    if (hexMatchPTX) return bigInt(hexMatchPTX[1], 16);

    const decMatch = decimalLike.exec(value);
    if (decMatch) return bigInt(decMatch[2]);

    return null;
}

export function getNumericToolTip(value: string, digitSeparator?: string): string | null {
    const formatNumber = (num: bigInt.BigInteger, base: number, chunkSize: number) => {
        const numberString = num.toString(base).toUpperCase();
        if (digitSeparator !== undefined) {
            return addDigitSeparator(numberString, digitSeparator, chunkSize);
        } else {
            return numberString;
        }
    };
    const numericValue = parseNumericValue(value);
    if (numericValue === null) return null;

    // PTX floats
    const view = new DataView(new ArrayBuffer(8));
    view.setBigUint64(0, BigInt(numericValue.toString()), true);
    if (ptxFloat32.test(value)) return view.getFloat32(0, true).toPrecision(9) + 'f';
    if (ptxFloat64.test(value)) return view.getFloat64(0, true).toPrecision(17);

    // Decimal representation.
    let result = formatNumber(numericValue, 10, 3);

    // Hexadecimal representation.
    if (numericValue.isNegative()) {
        const masked = bigInt('ffffffffffffffff', 16).and(numericValue);
        result += ' = 0x' + formatNumber(masked, 16, 4);
    } else {
        result += ' = 0x' + formatNumber(numericValue, 16, 4);
    }

    // Float32/64 representation.
    view.setBigUint64(0, BigInt(numericValue.toString()), true);
    if (numericValue.bitLength().lesserOrEquals(32)) result += ' = ' + view.getFloat32(0, true).toPrecision(9) + 'f';
    // only subnormal doubles and zero may have upper 32 bits all 0, assume unlikely to be double
    else result += ' = ' + view.getFloat64(0, true).toPrecision(17);

    // Printable UTF-8 characters.
    const bytes = numericValue.isNegative()
        ? // bytes of negative number without sign extension
          numericValue
              .add(1)
              .toArray(256)
              .value.map(byte => byte ^ 0xff)
        : numericValue.toArray(256).value;
    // This assumes that `numericValue` is encoded as little-endian.
    bytes.reverse();
    const decoder = new TextDecoder('utf-8', {fatal: true});
    try {
        result += ' = ' + JSON.stringify(decoder.decode(Uint8Array.from(bytes)));
    } catch (e) {
        // ignore `TypeError` when the number is not valid UTF-8
    }

    return result;
}
