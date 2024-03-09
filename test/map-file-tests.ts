// Copyright (c) 2017, Compiler Explorer Authors
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

import {describe, expect, it} from 'vitest';

import {unwrap} from '../lib/assert.js';
import {MapFileReaderDelphi} from '../lib/mapfiles/map-file-delphi.js';
import {MapFileReaderVS} from '../lib/mapfiles/map-file-vs.js';

describe('Map setup', () => {
    it('VS-map preferred load address', () => {
        const reader = new MapFileReaderVS('');
        expect(reader.preferredLoadAddress).toEqual(0x400000);

        reader.tryReadingPreferredAddress(' Preferred load address is 00400000');
        expect(reader.preferredLoadAddress).toEqual(0x400000);

        reader.tryReadingPreferredAddress(' Preferred load address is 00410000');
        expect(reader.preferredLoadAddress).toEqual(0x410000);
    });
});

describe('Code Segments', () => {
    it('One normal Delphi-Map segment', () => {
        const reader = new MapFileReaderDelphi('');
        reader.tryReadingCodeSegmentInfo(' 0001:00002838 00000080 C=CODE     S=.text    G=(none)   M=output   ACBP=A9');
        expect(reader.segments.length).toEqual(1);

        let info = reader.getSegmentInfoByStartingAddress('0001', 0x2838);
        expect(unwrap(info).unitName).toBe('output.pas');

        info = reader.getSegmentInfoByStartingAddress(undefined, reader.getSegmentOffset('0001') + 0x2838);
        expect(unwrap(info).unitName).toBe('output.pas');

        info = reader.getSegmentInfoByStartingAddress('0001', 0x1234);
        expect(info, 'Address should not be a Start for any segment').to.be.undefined;

        info = reader.getSegmentInfoAddressIsIn('0001', 0x2838 + 0x10);
        expect(unwrap(info).unitName).toBe('output.pas');

        info = reader.getSegmentInfoAddressIsIn(undefined, reader.getSegmentOffset('0001') + 0x2838 + 0x10);
        expect(unwrap(info).unitName).toBe('output.pas');

        info = reader.getSegmentInfoAddressIsIn('0001', reader.getSegmentOffset('0001') + 0x2838 + 0x80 + 1);
        expect(info, 'Address should not be in any segment').to.be.undefined;

        info = reader.getSegmentInfoByUnitName('output.pas');
        expect(unwrap(info).unitName).toBe('output.pas');
        expect(unwrap(info).addressInt).toEqual(reader.getSegmentOffset('0001') + 0x2838);
    });

    it('Not include this segment', () => {
        const reader = new MapFileReaderDelphi('');
        reader.tryReadingCodeSegmentInfo(' 0002:000000B0 00000023 C=ICODE    S=.itext   G=(none)   M=output   ACBP=A9');
        expect(reader.segments.length).toEqual(0);
    });

    it('ICode/IText segments', () => {
        const reader = new MapFileReaderDelphi('');
        reader.tryReadingCodeSegmentInfo(' 0002:000000B0 00000023 C=ICODE    S=.itext   G=(none)   M=output   ACBP=A9');
        expect(reader.isegments.length).toEqual(1);
    });

    it('One normal VS-Map segment', () => {
        const reader = new MapFileReaderVS('');
        reader.tryReadingCodeSegmentInfo(' 0001:00002838 00000080H .text$mn                CODE');
        expect(reader.segments.length).toEqual(1);

        let info = reader.getSegmentInfoByStartingAddress('0001', 0x2838);
        expect(unwrap(info).addressInt).toEqual(reader.getSegmentOffset('0001') + 0x2838);

        info = reader.getSegmentInfoByStartingAddress(undefined, 0x403838);
        expect(unwrap(info).addressInt).toEqual(reader.getSegmentOffset('0001') + 0x2838);

        info = reader.getSegmentInfoAddressIsIn(undefined, reader.getSegmentOffset('0001') + 0x2838 + 0x10);
        expect(unwrap(info).addressInt).toEqual(reader.getSegmentOffset('0001') + 0x2838);

        info = reader.getSegmentInfoAddressIsIn('0001', reader.getSegmentOffset('0001') + 0x2837);
        expect(info).to.be.undefined;
    });

    it('Repair VS-Map code segment info', () => {
        const reader = new MapFileReaderVS('');
        reader.tryReadingCodeSegmentInfo(' 0002:00000000 00004c73H .text$mn                CODE');
        reader.tryReadingNamedAddress(
            ' 0002:000007f0       _main                      004117f0 f   ConsoleApplication1.obj',
        );

        let info = reader.getSegmentInfoByStartingAddress('0002', 0);
        expect(unwrap(info).unitName).toBe('ConsoleApplication1.obj');

        expect(reader.getSegmentOffset('0002')).toEqual(0x411000);

        info = reader.getSegmentInfoByStartingAddress(undefined, 0x411000);
        expect(unwrap(info).unitName).toBe('ConsoleApplication1.obj');
    });
});

describe('Symbol info', () => {
    it('Delphi-Map symbol test', () => {
        const reader = new MapFileReaderDelphi('');
        reader.tryReadingNamedAddress(' 0001:00002838       Square');
        expect(reader.namedAddresses.length).toEqual(1);

        let info = reader.getSymbolAt('0001', 0x2838);
        expect(info).not.toBe(undefined);
        expect(unwrap(info).displayName).toBe('Square');

        info = reader.getSymbolAt(undefined, reader.getSegmentOffset('0001') + 0x2838);
        expect(info).not.toBe(undefined);
        expect(unwrap(info).displayName).toBe('Square');
    });

    it('Delphi-Map D2009 symbol test', () => {
        const reader = new MapFileReaderDelphi('');
        reader.tryReadingNamedAddress(' 0001:00002C4C       output.MaxArray');
        expect(reader.namedAddresses.length).toEqual(1);

        let info = reader.getSymbolAt('0001', 0x2c4c);
        expect(info).not.toBe(undefined);
        expect(unwrap(info).displayName).toBe('output.MaxArray');

        //todo should not be undefined
        info = reader.getSymbolAt(undefined, reader.getSegmentOffset('0001') + 0x2c4c);
        expect(info).not.toBe(undefined);
        expect(unwrap(info).displayName).toBe('output.MaxArray');
    });

    it('VS-Map symbol test', () => {
        const reader = new MapFileReaderVS('');
        reader.tryReadingNamedAddress(
            ' 0002:000006b0       ??$__vcrt_va_start_verify_argument_type@QBD@@YAXXZ 004116b0 f i ConsoleApplication1.obj',
        );
        expect(reader.namedAddresses.length).toEqual(1);

        let info = reader.getSymbolAt('0002', 0x6b0);
        expect(info).not.toBe(undefined);
        expect(unwrap(info).displayName).toBe('??$__vcrt_va_start_verify_argument_type@QBD@@YAXXZ');

        info = reader.getSymbolAt(undefined, 0x4116b0);
        expect(info).not.toBe(undefined);
        expect(unwrap(info).displayName).toBe('??$__vcrt_va_start_verify_argument_type@QBD@@YAXXZ');
    });

    it('Delphi-Map Duplication prevention', () => {
        const reader = new MapFileReaderDelphi('');
        reader.tryReadingNamedAddress(' 0001:00002838       Square');
        expect(reader.namedAddresses.length).toEqual(1);

        reader.tryReadingNamedAddress(' 0001:00002838       Square');
        expect(reader.namedAddresses.length).toEqual(1);
    });
});

describe('Delphi-Map Line number info', () => {
    it('No line', () => {
        const reader = new MapFileReaderDelphi('');
        expect(reader.tryReadingLineNumbers('')).toEqual(false);
    });

    it('One line', () => {
        const reader = new MapFileReaderDelphi('');
        expect(reader.tryReadingLineNumbers('    17 0001:000028A4')).toEqual(true);

        let lineInfo = reader.getLineInfoByAddress('0001', 0x28a4);
        expect(unwrap(lineInfo).lineNumber).toBe(17);

        lineInfo = reader.getLineInfoByAddress(undefined, reader.getSegmentOffset('0001') + 0x28a4);
        expect(unwrap(lineInfo).lineNumber).toBe(17);
    });

    it('Multiple lines', () => {
        const reader = new MapFileReaderDelphi('');
        expect(
            reader.tryReadingLineNumbers(
                '    12 0001:00002838    13 0001:0000283B    14 0001:00002854    15 0001:00002858',
            ),
        ).toEqual(true);

        let lineInfo = reader.getLineInfoByAddress('0001', 0x2838);
        expect(unwrap(lineInfo).lineNumber).toBe(12);

        lineInfo = reader.getLineInfoByAddress('0001', 0x2858);
        expect(unwrap(lineInfo).lineNumber).toBe(15);

        lineInfo = reader.getLineInfoByAddress('0001', 0x2854);
        expect(unwrap(lineInfo).lineNumber).toBe(14);

        lineInfo = reader.getLineInfoByAddress('0001', 0x283b);
        expect(unwrap(lineInfo).lineNumber).toBe(13);
    });
});

describe('Delphi-Map load test', () => {
    it('Minimal map', () => {
        const reader = new MapFileReaderDelphi('test/maps/minimal-delphi.map');
        reader.run();

        expect(reader.segments.length).toEqual(4);
        expect(reader.lineNumbers.length).toEqual(7);
        expect(reader.namedAddresses.length).toEqual(11);

        let info = reader.getSegmentInfoByUnitName('output.pas');
        expect(unwrap(info).addressInt).toEqual(reader.getSegmentOffset('0001') + 0x2c4c);

        info = reader.getICodeSegmentInfoByUnitName('output.pas');
        expect(unwrap(info).segment).toEqual('0002');
        expect(unwrap(info).addressWithoutOffset).toEqual(0xb0);
        expect(unwrap(info).addressInt).toEqual(0x4040b0);
    });
});

describe('VS-Map load test', () => {
    it('Minimal map', () => {
        const reader = new MapFileReaderVS('test/maps/minimal-vs15.map');
        reader.run();

        expect(reader.segments.length).toEqual(1);
        expect(unwrap(reader.getSegmentInfoByUnitName('ConsoleApplication1.obj')).addressInt).toEqual(0x411000);

        expect(reader.getSegmentOffset('0001')).toEqual(0x401000);
        expect(reader.getSegmentOffset('0002')).toEqual(0x411000);
        expect(reader.getSegmentOffset('0003')).toEqual(0x416000);
        expect(reader.getSegmentOffset('0004')).toEqual(0x419000);
        expect(reader.getSegmentOffset('0005')).toEqual(0x41a000);
        expect(reader.getSegmentOffset('0007')).toEqual(0x41c000);
    });
});

describe('VS-Map address checking', () => {
    it('Normal defined spaces', () => {
        const reader = new MapFileReaderVS('');

        const mainAddresses = [
            {startAddress: 1, startAddressHex: '00000001', endAddress: 10, endAddressHex: '0000000A'},
            {startAddress: 16, startAddressHex: '00000010', endAddress: 255, endAddressHex: '000000FF'},
        ];

        expect(reader.isWithinAddressSpace(mainAddresses, 3, 5)).toEqual(true);
        expect(reader.isWithinAddressSpace(mainAddresses, 10, 5)).toEqual(false);
        expect(reader.isWithinAddressSpace(mainAddresses, 11, 4)).toEqual(false);
        expect(reader.isWithinAddressSpace(mainAddresses, 16, 10)).toEqual(true);
        expect(reader.isWithinAddressSpace(mainAddresses, 32, 10)).toEqual(true);
    });

    it('Overlapping regions', () => {
        const reader = new MapFileReaderVS('');

        const mainAddresses = [
            {startAddress: 1, startAddressHex: '00000001', endAddress: 10, endAddressHex: '0000000A'},
            {startAddress: 16, startAddressHex: '00000010', endAddress: 255, endAddressHex: '000000FF'},
        ];

        expect(reader.isWithinAddressSpace(mainAddresses, 0, 5)).toEqual(true);
        expect(reader.isWithinAddressSpace(mainAddresses, 11, 5)).toEqual(true);
        expect(reader.isWithinAddressSpace(mainAddresses, 11, 6)).toEqual(true);
        expect(reader.isWithinAddressSpace(mainAddresses, 11, 258)).toEqual(true);
    });
});
