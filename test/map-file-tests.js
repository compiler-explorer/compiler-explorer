// Copyright (c) 2017, Patrick Quist
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

var chai = require('chai');
var should = chai.should();
var assert = chai.assert;
var MapFileReader = require('../lib/map-file').MapFileReader;
var logger = require('../lib/logger').logger;

describe('Setup', function () {
    it('VS-map preferred load address', function () {
        var reader = new MapFileReader();
        reader.preferredLoadAddress.should.equal(0x400000, "default load address");

        reader.TryReadingPreferredAddress(" Preferred load address is 00400000");
        reader.preferredLoadAddress.should.equal(0x400000);
        
        reader.TryReadingPreferredAddress(" Preferred load address is 00410000");
        reader.preferredLoadAddress.should.equal(0x410000);
    });
});

describe('Code Segments', function () {
    it('One normal Delphi-Map segment', function () {
        var reader = new MapFileReader();
        reader.TryReadingCodeSegmentInfo(" 0001:00002838 00000080 C=CODE     S=.text    G=(none)   M=output   ACBP=A9");
        reader.segments.length.should.equal(1);

        var info = reader.GetSegmentInfoByStartingAddress("0001", 0x2838);
        info.unitName.should.equal("output");

        info = reader.GetSegmentInfoByStartingAddress(false, reader.GetSegmentOffset("0001") + 0x2838);
        info.unitName.should.equal("output");

        info = reader.GetSegmentInfoByStartingAddress("0001", "2838");
        assert(info === false, "Address should not be a Start for any segment");

        info = reader.GetSegmentInfoAddressIsIn("0001", 0x2838 + 0x10);
        info.unitName.should.equal("output");

        info = reader.GetSegmentInfoAddressIsIn(false, reader.GetSegmentOffset("0001") + 0x2838 + 0x10);
        info.unitName.should.equal("output");

        info = reader.GetSegmentInfoAddressIsIn("0001", reader.GetSegmentOffset("0001") + 0x2838 + 0x80 + 1);
        assert(info === false, "Address should not be in any segment");

        info = reader.GetSegmentInfoByUnitName("output");
        info.unitName.should.equal("output");
        info.addressInt.should.equal(reader.GetSegmentOffset("0001") + 0x2838);
    });

    it('Not include this segment', function () {
        var reader = new MapFileReader();
        reader.TryReadingCodeSegmentInfo(" 0002:000000B0 00000023 C=ICODE    S=.itext   G=(none)   M=output   ACBP=A9");
        reader.segments.length.should.equal(0);
    });

    it('ICode/IText segments', function () {
        var reader = new MapFileReader();
        reader.TryReadingCodeSegmentInfo(" 0002:000000B0 00000023 C=ICODE    S=.itext   G=(none)   M=output   ACBP=A9");
        reader.isegments.length.should.equal(1);
    });

    it('One normal VS-Map segment', function () {
        var reader = new MapFileReader();
        reader.TryReadingCodeSegmentInfo(" 0001:00002838 00000080H .text$mn                CODE");
        reader.segments.length.should.equal(1);

        var info = reader.GetSegmentInfoByStartingAddress("0001", 0x2838);
        info.addressInt.should.equal(reader.GetSegmentOffset("0001") + 0x2838);

        info = reader.GetSegmentInfoByStartingAddress(false, 0x403838);
        info.addressInt.should.equal(reader.GetSegmentOffset("0001") + 0x2838);

        info = reader.GetSegmentInfoAddressIsIn(false, reader.GetSegmentOffset("0001") + 0x2838 + 0x10);
        info.addressInt.should.equal(reader.GetSegmentOffset("0001") + 0x2838);
        
        info = reader.GetSegmentInfoAddressIsIn("0001", reader.GetSegmentOffset("0001") + 0x2837);
        assert(info === false);
    });

    it('Repair VS-Map code segment info', function () {
        var reader = new MapFileReader();
        reader.TryReadingCodeSegmentInfo(" 0002:00000000 00004c73H .text$mn                CODE");
        reader.TryReadingNamedAddress(" 0002:000007f0       _main                      004117f0 f   ConsoleApplication1.obj");

        var info = reader.GetSegmentInfoByStartingAddress("0002", 0);
        info.unitName.should.equal("ConsoleApplication1.obj");

        reader.GetSegmentOffset("0002").should.equal(0x411000);

        info = reader.GetSegmentInfoByStartingAddress(false, 0x411000);
        info.unitName.should.equal("ConsoleApplication1.obj");
    });
});

describe('Symbol info', function () {
    it('Delphi-Map symbol test', function () {
        var reader = new MapFileReader();
        reader.TryReadingNamedAddress(" 0001:00002838       Square");
        reader.namedAddresses.length.should.equal(1);
        
        var info = reader.GetSymbolAt("0001", 0x2838);
        assert(info !== false, "Symbol Square should have been returned 1");
        info.displayName.should.equal("Square");

        info = reader.GetSymbolAt(false, reader.GetSegmentOffset("0001") + 0x2838);
        assert(info !== false, "Symbol Square should have been returned 2");
        info.displayName.should.equal("Square");
    });

    it('Delphi-Map D2009 symbol test', function () {
        var reader = new MapFileReader();
        reader.TryReadingNamedAddress(" 0001:00002C4C       output.MaxArray");
        reader.namedAddresses.length.should.equal(1);

        var info = reader.GetSymbolAt("0001", 0x2C4C);
        assert(info !== false, "Symbol MaxArray should have been returned");
        info.displayName.should.equal("output.MaxArray");

        info = reader.GetSymbolAt(false, reader.GetSegmentOffset("0001") + 0x2C4C);
        assert(info !== false, "Symbol MaxArray should have been returned");
        info.displayName.should.equal("output.MaxArray");
    });

    it('VS-Map symbol test', function () {
        var reader = new MapFileReader();
        reader.TryReadingNamedAddress(" 0002:000006b0       ??$__vcrt_va_start_verify_argument_type@QBD@@YAXXZ 004116b0 f i ConsoleApplication1.obj");
        reader.namedAddresses.length.should.equal(1);

        var info = reader.GetSymbolAt("0002", 0x6b0);
        assert(info !== false, "Symbol start_verify_argument should have been returned 1");
        info.displayName.should.equal("??$__vcrt_va_start_verify_argument_type@QBD@@YAXXZ");
        
        info = reader.GetSymbolAt(false, 0x4116b0);
        assert(info !== false, "Symbol start_verify_argument should have been returned 2");
        info.displayName.should.equal("??$__vcrt_va_start_verify_argument_type@QBD@@YAXXZ");
    });

    it('Delphi-Map Duplication prevention', function () {
        var reader = new MapFileReader();
        reader.TryReadingNamedAddress(" 0001:00002838       Square");
        reader.namedAddresses.length.should.equal(1);
        
        reader.TryReadingNamedAddress(" 0001:00002838       Square");
        reader.namedAddresses.length.should.equal(1);
    });
});

describe('Delphi-Map Line number info', function () {
    it('No line', function () {
        var reader = new MapFileReader();
        assert(reader.TryReadingLineNumbers("") === false);
    });

    it('One line', function () {
        var reader = new MapFileReader();
        assert(reader.TryReadingLineNumbers("    17 0001:000028A4") === true);

        var lineInfo = reader.GetLineInfoByAddress("0001", 0x28A4);
        lineInfo.lineNumber.should.equal(17);

        lineInfo = reader.GetLineInfoByAddress(false, reader.GetSegmentOffset("0001") + 0x28A4);
        lineInfo.lineNumber.should.equal(17);
    });

    it('Multiple lines', function () {
        var reader = new MapFileReader();
        assert(reader.TryReadingLineNumbers("    12 0001:00002838    13 0001:0000283B    14 0001:00002854    15 0001:00002858") === true);

        var lineInfo = reader.GetLineInfoByAddress("0001", 0x2838);
        lineInfo.lineNumber.should.equal(12);

        lineInfo = reader.GetLineInfoByAddress("0001", 0x2858);
        lineInfo.lineNumber.should.equal(15);

        lineInfo = reader.GetLineInfoByAddress("0001", 0x2854);
        lineInfo.lineNumber.should.equal(14);

        lineInfo = reader.GetLineInfoByAddress("0001", 0x283B);
        lineInfo.lineNumber.should.equal(13);
    });
});

describe('Delphi-Map load test', function () {
    it('Minimal map', function () {
        var reader = new MapFileReader("test/maps/minimal-delphi.map");
        reader.Run();

        reader.segments.length.should.equal(4);
        reader.lineNumbers.length.should.equal(7);
        reader.namedAddresses.length.should.equal(11);

        var info = reader.GetSegmentInfoByUnitName("output");
        info.addressInt.should.equal(reader.GetSegmentOffset("0001") + 0x2C4C);

        info = reader.GetICodeSegmentInfoByUnitName("output");
        info.segment.should.equal("0002");
        info.addressWithoutOffset.should.equal(0xB0);
        info.addressInt.should.equal(0x4040B0);
    });
});

describe('VS-Map load test', function () {
    it('Minimal map', function () {
        var reader = new MapFileReader("test/maps/minimal-vs15.map");
        reader.Run();

        reader.segments.length.should.equal(1);
        reader.GetSegmentInfoByUnitName("ConsoleApplication1.obj").addressInt.should.equal(0x411000);

        reader.GetSegmentOffset("0001").should.equal(0x401000, "offset 1");
        reader.GetSegmentOffset("0002").should.equal(0x411000, "offset 2");
        reader.GetSegmentOffset("0003").should.equal(0x416000, "offset 3");
        reader.GetSegmentOffset("0004").should.equal(0x419000, "offset 4");
        reader.GetSegmentOffset("0005").should.equal(0x41a000, "offset 5");
        //reader.GetSegmentOffset("0006").should.equal(0x41a000);
        reader.GetSegmentOffset("0007").should.equal(0x41c000, "offset 7");
    });
});
