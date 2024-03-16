import {describe, expect, it} from 'vitest';

import * as utils from '../../static/utils.js';

describe('numeric tooltip', () => {
    it('handles ASCII characters', () => {
        expect(utils.getNumericToolTip('42')).toEqual('42 = 0x2A = 5.88545355e-44f = "*"');
        expect(utils.getNumericToolTip('97')).toEqual('97 = 0x61 = 1.35925951e-43f = "a"');
        expect(utils.getNumericToolTip('10')).toEqual('10 = 0xA = 1.40129846e-44f = "\\n"');
        expect(utils.getNumericToolTip('92')).toEqual('92 = 0x5C = 1.28919459e-43f = "\\\\"');
        expect(utils.getNumericToolTip('1')).toEqual('1 = 0x1 = 1.40129846e-45f = "\\u0001"');
    });
    it('handles ASCII strings', () => {
        expect(utils.getNumericToolTip('0x61626364')).toEqual('1633837924 = 0x61626364 = 2.61007876e+20f = "dcba"');
        expect(utils.getNumericToolTip('8583909746840200552')).toEqual(
            '8583909746840200552 = 0x77202C6F6C6C6568 = 6.5188685003648344e+265 = "hello, w"',
        );
    });
    it('handles UTF-8 strings', () => {
        expect(utils.getNumericToolTip('-6934491452449512253')).toEqual(
            '-6934491452449512253 = 0x9FC3BCC3B6C3A4C3 = -1.1500622354593239e-155 = "äöüß"',
        );
        expect(utils.getNumericToolTip('-1530699166')).toEqual(
            '-1530699166 = 0xFFFFFFFFA4C36262 = -8.47344364e-17f = "bbä"',
        );
        expect(utils.getNumericToolTip('-23357')).toEqual('-23357 = 0xFFFFFFFFFFFFA4C3 = NaNf = "ä"');
    });
    it('inserts digit separators', () => {
        expect(utils.getNumericToolTip('1234', '_')).toEqual('1_234 = 0x4D2 = 1.72920230e-42f');
        expect(utils.getNumericToolTip('1234567', "'")).toEqual("1'234'567 = 0x12'D687 = 1.72999684e-39f");
        expect(utils.getNumericToolTip('-6934491452449512253', '_')).toEqual(
            '-6_934_491_452_449_512_253 = 0x9FC3_BCC3_B6C3_A4C3 = -1.1500622354593239e-155 = "äöüß"',
        );
    });
    it('does nothing for strings that are not numbers', () => {
        expect(utils.getNumericToolTip('word')).toEqual(null);
        expect(utils.getNumericToolTip('NaN')).toEqual(null);
        expect(utils.getNumericToolTip('1.5')).toEqual(null);
    });
});
