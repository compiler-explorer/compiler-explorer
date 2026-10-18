// Copyright (c) 2026, Compiler Explorer Authors
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
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS AND CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

import {Buffer} from 'buffer';

import {describe, expect, it} from 'vitest';

import {DotNetAsmParser} from '../lib/parsers/asm-parser-dotnet.js';
import type {DotNetMethodKey, DotNetSourceMapping, DotNetTypeSignature} from '../lib/parsers/pdb-parser-dotnet.js';
import {DotNetPdbParser, getCanonicalTypeSignature} from '../lib/parsers/pdb-parser-dotnet.js';
import type {AsmResultSource, ParsedAsmResult} from '../types/asmresult/asmresult.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../types/features/filters.interfaces.js';

const filters = (overrides: ParseFiltersAndOutputOptions = {}): ParseFiltersAndOutputOptions => ({
    commentOnly: false,
    labels: false,
    ...overrides,
});

const valueType = (name: string): DotNetTypeSignature => ({
    name,
    arguments: [],
    suffix: '',
    typeKind: 'value',
});

const referenceType = (name: string, args: DotNetTypeSignature[] = []): DotNetTypeSignature => ({
    name,
    arguments: args,
    suffix: '',
    typeKind: 'reference',
});

const genericParameter = (genericParameter: string): DotNetTypeSignature => ({
    name: genericParameter,
    arguments: [],
    suffix: '',
    typeKind: 'generic',
    genericParameter,
});

const arrayType = (element: DotNetTypeSignature, suffix = '[]'): DotNetTypeSignature => ({
    name: '',
    arguments: [element],
    suffix,
    typeKind: 'reference',
});

function bytes(...values: number[]) {
    return Buffer.from(values);
}

function u16(value: number) {
    const buffer = Buffer.alloc(2);
    buffer.writeUInt16LE(value, 0);
    return buffer;
}

function u32(value: number) {
    const buffer = Buffer.alloc(4);
    buffer.writeUInt32LE(value, 0);
    return buffer;
}

function u64(value: bigint) {
    const buffer = Buffer.alloc(8);
    buffer.writeUInt32LE(Number(value & 0xffff_ffffn), 0);
    buffer.writeUInt32LE(Number((value >> 32n) & 0xffff_ffffn), 4);
    return buffer;
}

function compressedUInt(value: number): Buffer {
    if (value < 0x80) {
        return bytes(value);
    }

    if (value < 0x4000) {
        return bytes(0x80 | (value >> 8), value & 0xff);
    }

    return bytes(0xc0 | (value >> 24), (value >> 16) & 0xff, (value >> 8) & 0xff, value & 0xff);
}

function compressedSignedInt(value: number): Buffer {
    const encoded = value >= 0 ? value << 1 : (((1 << 6) + value) << 1) | 1;
    return compressedUInt(encoded);
}

function align4(value: number) {
    return (value + 3) & ~3;
}

class StringHeapBuilder {
    private readonly indexes = new Map<string, number>();
    private readonly chunks = [Buffer.from([0])];
    private length = 1;

    index(value: string) {
        const existing = this.indexes.get(value);
        if (existing !== undefined) {
            return existing;
        }

        const index = this.length;
        const chunk = Buffer.from(`${value}\0`, 'utf8');
        this.chunks.push(chunk);
        this.length += chunk.length;
        this.indexes.set(value, index);
        return index;
    }

    toBuffer() {
        return Buffer.concat(this.chunks);
    }
}

class BlobHeapBuilder {
    private readonly chunks = [Buffer.from([0])];
    private length = 1;

    index(value: Buffer) {
        const index = this.length;
        const chunk = Buffer.concat([compressedUInt(value.length), value]);
        this.chunks.push(chunk);
        this.length += chunk.length;
        return index;
    }

    toBuffer() {
        return Buffer.concat(this.chunks);
    }
}

function metadataRoot(streams: Array<{name: string; data: Buffer}>) {
    // Emit a standalone ECMA-335 metadata root, starting at the BSJB signature.
    const version = Buffer.from('v4.0.30319\0\0', 'utf8');
    const streamHeadersSize = streams.reduce((size, stream) => size + 8 + align4(stream.name.length + 1), 0);
    let streamDataOffset = align4(4 + 2 + 2 + 4 + 4 + version.length + 2 + 2 + streamHeadersSize);
    const streamHeaders: Buffer[] = [];

    // Stream headers contain offsets relative to the metadata root, followed by the byte size and
    // a null-terminated, 4-byte-aligned stream name.
    for (const stream of streams) {
        streamDataOffset = align4(streamDataOffset);
        streamHeaders.push(
            Buffer.concat([
                u32(streamDataOffset),
                u32(stream.data.length),
                Buffer.concat([
                    Buffer.from(`${stream.name}\0`, 'utf8'),
                    Buffer.alloc(align4(stream.name.length + 1) - (stream.name.length + 1)),
                ]),
            ]),
        );
        streamDataOffset += stream.data.length;
    }

    const header = Buffer.concat([
        u32(0x424a5342), // BSJB metadata signature
        u16(1), // MajorVersion
        u16(1), // MinorVersion
        u32(0), // Reserved
        u32(version.length),
        version,
        u16(0), // Flags
        u16(streams.length),
        ...streamHeaders,
    ]);
    const dataStartPadding = Buffer.alloc(align4(header.length) - header.length);
    let offset = align4(header.length);
    const dataChunks: Buffer[] = [];
    for (const stream of streams) {
        const alignedOffset = align4(offset);
        dataChunks.push(Buffer.alloc(alignedOffset - offset));
        dataChunks.push(stream.data);
        offset = alignedOffset + stream.data.length;
    }

    return Buffer.concat([header, dataStartPadding, ...dataChunks]);
}

function tableStream(rows: Record<number, Buffer[]>) {
    // Emit a minimal #~ metadata table stream.
    const tableIds = Object.keys(rows)
        .map(Number)
        .sort((left, right) => left - right);
    const validTables = tableIds.reduce((mask, tableId) => mask | (1n << BigInt(tableId)), 0n);

    return Buffer.concat([
        u32(0), // Reserved
        bytes(2, 0, 0, 1), // MajorVersion, MinorVersion, HeapSizes, Reserved
        u64(validTables),
        u64(0n), // Sorted table mask; the parser only needs the valid table mask and row order.
        ...tableIds.map(tableId => u32(rows[tableId].length)),
        ...tableIds.flatMap(tableId => rows[tableId]),
    ]);
}

function methodSignature(returnType: Buffer, parameters: Buffer[], genericParameterCount = 0) {
    return Buffer.concat([
        bytes(genericParameterCount === 0 ? 0 : 0x10),
        ...(genericParameterCount === 0 ? [] : [compressedUInt(genericParameterCount)]),
        compressedUInt(parameters.length),
        returnType,
        ...parameters,
    ]);
}

function typeDefOrRef(table: 'TypeDef' | 'TypeRef' | 'TypeSpec', row: number) {
    return compressedUInt((row << 2) | (table === 'TypeDef' ? 0 : table === 'TypeRef' ? 1 : 2));
}

function typeOrMethodDef(table: 'TypeDef' | 'MethodDef', row: number) {
    return (row << 1) | (table === 'TypeDef' ? 0 : 1);
}

function genericInstance(kind: 'class' | 'valuetype', type: Buffer, args: Buffer[]) {
    return Buffer.concat([bytes(0x15, kind === 'class' ? 0x12 : 0x11), type, compressedUInt(args.length), ...args]);
}

function szArray(element: Buffer) {
    return Buffer.concat([bytes(0x1d), element]);
}

function mdArray(element: Buffer, rank: number, sizes: number[] = [], lowerBounds: number[] = []) {
    return Buffer.concat([
        bytes(0x14),
        element,
        compressedUInt(rank),
        compressedUInt(sizes.length),
        ...sizes.map(compressedUInt),
        compressedUInt(lowerBounds.length),
        ...lowerBounds.map(compressedSignedInt),
    ]);
}

function byRef(element: Buffer) {
    return Buffer.concat([bytes(0x10), element]);
}

function pointer(element: Buffer) {
    return Buffer.concat([bytes(0x0f), element]);
}

function functionPointer(returnType: Buffer, parameters: Buffer[]) {
    return Buffer.concat([bytes(0x1b), methodSignature(returnType, parameters)]);
}

function fieldSignature(type: Buffer) {
    return Buffer.concat([bytes(0x06), type]);
}

function sequencePoints(points: Array<{offset: number; line: number | null; column?: number}>) {
    // Emit a Portable PDB sequence-points blob.
    const chunks = [compressedUInt(0)]; // The local-signature token
    let previousOffset = 0;
    let previousLine = 0;
    let previousColumn = 0;
    let hasPrevious = false;

    for (const point of points) {
        chunks.push(compressedUInt(hasPrevious ? point.offset - previousOffset : point.offset));
        if (point.line === null) {
            // Hidden sequence point: delta lines and delta columns are both zero, and no source span follows.
            chunks.push(compressedUInt(0), compressedUInt(0));
        } else {
            // Visible points in this fixture are one-column spans on a single source line.
            chunks.push(compressedUInt(0), compressedUInt(1));
            if (hasPrevious) {
                chunks.push(compressedSignedInt(point.line - previousLine));
                chunks.push(compressedSignedInt((point.column ?? 1) - previousColumn));
            } else {
                chunks.push(compressedUInt(point.line), compressedUInt(point.column ?? 1));
            }
            previousLine = point.line;
            previousColumn = point.column ?? 1;
        }

        previousOffset = point.offset;
        hasPrevious = true;
    }

    return Buffer.concat(chunks);
}

function ecma335Fixture() {
    // Build a tiny ECMA-335 metadata root plus matching portable-PDB metadata.
    //
    // The modeled user types are:
    //   Examples.Program
    //   Examples.Outer<TOuter>
    //   Examples.Outer<TOuter>.Inner<TInner>
    //   Examples.Point
    //   Examples.TinyEnum : byte
    //
    // The method rows cover primitive element types, enum/value-type resolution, TypeSpec resolution,
    // required/optional custom modifiers, sentinels, pinned types, nested generics, arrays, byrefs,
    // pointers, function pointers, and method/type generic parameters.
    const strings = new StringHeapBuilder();
    const blobs = new BlobHeapBuilder();

    // TypeSpec row 1: System.Collections.Generic.Dictionary<System.String, int>.
    const dictionaryOfStringAndIntType = genericInstance('class', typeDefOrRef('TypeRef', 1), [
        bytes(0x0e),
        bytes(0x08),
    ]);

    // Shared Mix<TMethod> signature fragment:
    // Dictionary<!0, Examples.Outer+Inner<!0, List<!!0[]>>>.
    const dictionaryType = genericInstance('class', typeDefOrRef('TypeRef', 1), [
        bytes(0x13, 0),
        genericInstance('class', typeDefOrRef('TypeDef', 3), [
            bytes(0x13, 0),
            genericInstance('class', typeDefOrRef('TypeRef', 2), [szArray(bytes(0x1e, 0))]),
        ]),
    ]);

    // Examples.Outer+Inner<!0, !1>[].
    const innerArrayType = szArray(
        genericInstance('class', typeDefOrRef('TypeDef', 3), [bytes(0x13, 0), bytes(0x13, 1)]),
    );

    // !!0[,] with explicit size/lower-bound payloads, so the parser must consume and ignore the
    // array-shape metadata while preserving the rank.
    const methodArrayType = mdArray(bytes(0x1e, 0), 2, [3, 4], [0, -1]);

    // Value-type encodings used to verify TypeDef and enum-underlying-type resolution.
    const pointType = Buffer.concat([bytes(0x11), typeDefOrRef('TypeDef', 4)]);
    const tinyEnumType = Buffer.concat([bytes(0x11), typeDefOrRef('TypeDef', 5)]);

    // TypeRef row 3 is System.Enum; this exercises class TypeDefOrRef decoding separately from the
    // TinyEnum value-type path above.
    const enumClassType = Buffer.concat([bytes(0x12), typeDefOrRef('TypeRef', 3)]);
    const valueTupleType = genericInstance('valuetype', typeDefOrRef('TypeRef', 4), [bytes(0x0e), bytes(0x08)]);
    const typeSpecDictionary = Buffer.concat([bytes(0x12), typeDefOrRef('TypeSpec', 1)]);

    // void PrimitiveKitchenSink(bool, char, sbyte, byte, short, ushort, int, uint, long, ulong,
    //                           float, double, string, nint, nuint, object, typedref)
    const primitiveSignature = blobs.index(
        methodSignature(bytes(0x01), [
            bytes(0x02),
            bytes(0x03),
            bytes(0x04),
            bytes(0x05),
            bytes(0x06),
            bytes(0x07),
            bytes(0x08),
            bytes(0x09),
            bytes(0x0a),
            bytes(0x0b),
            bytes(0x0c),
            bytes(0x0d),
            bytes(0x0e),
            bytes(0x18),
            bytes(0x19),
            bytes(0x1c),
            bytes(0x16),
        ]),
    );

    // Examples.Point EnumAndValueTypes(Examples.Point, Examples.TinyEnum), where TinyEnum resolves to byte.
    const enumAndValueTypesSignature = blobs.index(methodSignature(pointType, [pointType, tinyEnumType]));

    // Dictionary<string, int> TypeSpecAndModifiers(
    //     modreq(System.Enum) long,
    //     modopt(System.Enum) int,
    //     sentinel string,
    //     pinned object,
    //     System.Enum,
    //     System.ValueTuple<string, int>,
    //     Dictionary<string, int>)
    const typeSpecAndModifiersSignature = blobs.index(
        methodSignature(typeSpecDictionary, [
            Buffer.concat([bytes(0x1f), typeDefOrRef('TypeRef', 3), bytes(0x0a)]),
            Buffer.concat([bytes(0x20), typeDefOrRef('TypeRef', 3), bytes(0x08)]),
            Buffer.concat([bytes(0x41), bytes(0x0e)]),
            Buffer.concat([bytes(0x45), bytes(0x1c)]),
            enumClassType,
            valueTupleType,
            typeSpecDictionary,
        ]),
    );

    // Dictionary<!0, Inner<!0, List<!!0[]>>> Mix<TMethod>(
    //     Dictionary<!0, Inner<!0, List<!!0[]>>>,
    //     Inner<!0, !1>[],
    //     !!0[,],
    //     byref !!0,
    //     byref !0,
    //     pointer int,
    //     function pointer Dictionary<!0, Inner<!0, List<!!0[]>>>(byref !1, pointer !!0[], function pointer void()))
    const mixSignature = blobs.index(
        methodSignature(
            dictionaryType,
            [
                dictionaryType,
                innerArrayType,
                methodArrayType,
                byRef(bytes(0x1e, 0)),
                byRef(bytes(0x13, 0)),
                pointer(bytes(0x08)),
                functionPointer(dictionaryType, [
                    byRef(bytes(0x13, 1)),
                    pointer(szArray(bytes(0x1e, 0))),
                    functionPointer(bytes(0x01), []),
                ]),
            ],
            1,
        ),
    );
    const addSignature = blobs.index(methodSignature(bytes(0x08), [bytes(0x08), bytes(0x08)]));
    const enumFieldSignature = blobs.index(fieldSignature(bytes(0x05)));
    const typeSpecSignature = blobs.index(dictionaryOfStringAndIntType);
    const assemblyTables = tableStream({
        // TypeRef rows:
        //   1. System.Collections.Generic.Dictionary`2
        //   2. System.Collections.Generic.List`1
        //   3. System.Enum
        //   4. System.ValueTuple`2
        1: [
            Buffer.concat([
                u16(0),
                u16(strings.index('Dictionary`2')),
                u16(strings.index('System.Collections.Generic')),
            ]),
            Buffer.concat([u16(0), u16(strings.index('List`1')), u16(strings.index('System.Collections.Generic'))]),
            Buffer.concat([u16(0), u16(strings.index('Enum')), u16(strings.index('System'))]),
            Buffer.concat([u16(0), u16(strings.index('ValueTuple`2')), u16(strings.index('System'))]),
        ],
        // TypeDef rows. MethodList ranges assign MethodDef rows 1-4 to Program and row 5 to Inner.
        // FieldList ranges assign the value__ field to TinyEnum.
        2: [
            Buffer.concat([
                u32(0),
                u16(strings.index('Program')),
                u16(strings.index('Examples')),
                u16(0),
                u16(1),
                u16(1),
            ]),
            Buffer.concat([
                u32(0),
                u16(strings.index('Outer`1')),
                u16(strings.index('Examples')),
                u16(0),
                u16(1),
                u16(5),
            ]),
            Buffer.concat([u32(0), u16(strings.index('Inner`1')), u16(0), u16(0), u16(1), u16(5)]),
            Buffer.concat([
                u32(0),
                u16(strings.index('Point')),
                u16(strings.index('Examples')),
                u16(0),
                u16(1),
                u16(6),
            ]),
            Buffer.concat([
                u32(0),
                u16(strings.index('TinyEnum')),
                u16(strings.index('Examples')),
                u16((3 << 2) | 1),
                u16(1),
                u16(6),
            ]),
        ],
        // Field row 1: TinyEnum.value__ : byte. This lets enum value types collapse to their
        // underlying primitive signature in parser output.
        4: [Buffer.concat([u16(0), u16(strings.index('value__')), u16(enumFieldSignature)])],
        // MethodDef rows:
        //   1. Program.Add
        //   2. Program.PrimitiveKitchenSink
        //   3. Program.EnumAndValueTypes
        //   4. Program.TypeSpecAndModifiers
        //   5. Outer.Inner.Mix
        6: [
            Buffer.concat([u32(0), u16(0), u16(0), u16(strings.index('Add')), u16(addSignature), u16(1)]),
            Buffer.concat([
                u32(0),
                u16(0),
                u16(0),
                u16(strings.index('PrimitiveKitchenSink')),
                u16(primitiveSignature),
                u16(1),
            ]),
            Buffer.concat([
                u32(0),
                u16(0),
                u16(0),
                u16(strings.index('EnumAndValueTypes')),
                u16(enumAndValueTypesSignature),
                u16(1),
            ]),
            Buffer.concat([
                u32(0),
                u16(0),
                u16(0),
                u16(strings.index('TypeSpecAndModifiers')),
                u16(typeSpecAndModifiersSignature),
                u16(1),
            ]),
            Buffer.concat([u32(0), u16(0), u16(0), u16(strings.index('Mix')), u16(mixSignature), u16(1)]),
        ],
        // TypeSpec row 1 stores Dictionary<string, int>.
        27: [u16(typeSpecSignature)],
        // NestedClass row: TypeDef row 3 (Inner) is nested inside TypeDef row 2 (Outer).
        41: [Buffer.concat([u16(3), u16(2)])],
        // GenericParam rows:
        //   Outer<TOuter>
        //   Inner has !0/TOuter and !1/TInner in scope for nested generic signatures
        //   Mix<TMethod>
        42: [
            Buffer.concat([u16(0), u16(0), u16(typeOrMethodDef('TypeDef', 2)), u16(strings.index('TOuter'))]),
            Buffer.concat([u16(0), u16(0), u16(typeOrMethodDef('TypeDef', 3)), u16(strings.index('TOuter'))]),
            Buffer.concat([u16(1), u16(0), u16(typeOrMethodDef('TypeDef', 3)), u16(strings.index('TInner'))]),
            Buffer.concat([u16(0), u16(0), u16(typeOrMethodDef('MethodDef', 5)), u16(strings.index('TMethod'))]),
        ],
    });
    const assembly = metadataRoot([
        {name: '#~', data: assemblyTables},
        {name: '#Strings', data: strings.toBuffer()},
        {name: '#Blob', data: blobs.toBuffer()},
    ]);

    const pdbBlobs = new BlobHeapBuilder();
    // Portable PDB sequence points keyed by MethodDebugInformation row number, which matches the
    // MethodDef row id. Add also includes a hidden sequence point at IL offset 3.
    const addSequencePoints = pdbBlobs.index(
        sequencePoints([
            {offset: 0, line: 46, column: 13},
            {offset: 3, line: null},
        ]),
    );
    const mixSequencePoints = pdbBlobs.index(
        sequencePoints([
            {offset: 0, line: 116, column: 17},
            {offset: 32, line: 117, column: 17},
            {offset: 40, line: 118, column: 17},
            {offset: 54, line: 119, column: 17},
        ]),
    );
    const primitiveSequencePoints = pdbBlobs.index(sequencePoints([{offset: 0, line: 80, column: 9}]));
    const enumAndValueTypesSequencePoints = pdbBlobs.index(sequencePoints([{offset: 0, line: 90, column: 9}]));
    const typeSpecAndModifiersSequencePoints = pdbBlobs.index(sequencePoints([{offset: 0, line: 100, column: 9}]));
    const pdbTables = tableStream({
        // Document row 1 is intentionally empty because the parser currently only preserves line/column.
        48: [Buffer.concat([u16(0), u16(0), u16(0), u16(0)])],
        // MethodDebugInformation rows 1-5 correspond to MethodDef rows 1-5 above.
        49: [
            Buffer.concat([u16(1), u16(addSequencePoints)]),
            Buffer.concat([u16(1), u16(primitiveSequencePoints)]),
            Buffer.concat([u16(1), u16(enumAndValueTypesSequencePoints)]),
            Buffer.concat([u16(1), u16(typeSpecAndModifiersSequencePoints)]),
            Buffer.concat([u16(1), u16(mixSequencePoints)]),
        ],
    });
    const pdb = metadataRoot([
        // The type-system table mask is empty; this fixture keeps all type-system rows in the assembly metadata.
        {name: '#Pdb', data: Buffer.concat([Buffer.alloc(24), u64(0n)])},
        {name: '#~', data: pdbTables},
        {name: '#Blob', data: pdbBlobs.toBuffer()},
    ]);

    return {assembly, pdb};
}

const methodArrayType = arrayType(genericParameter('!!0'));
const nestedInnerWithListType = referenceType('Examples.Outer+Inner', [
    genericParameter('!0'),
    referenceType('System.Collections.Generic.List', [methodArrayType]),
]);
const complexDictionaryType = referenceType('System.Collections.Generic.Dictionary', [
    genericParameter('!0'),
    nestedInnerWithListType,
]);
const asyncRunResultType = referenceType('Result', [
    referenceType('System.ValueTuple', [valueType('int'), referenceType('System.String')]),
]);

const source = (line: number, column: number): AsmResultSource => ({file: null, line, column});

function findLine(result: ParsedAsmResult, text: string) {
    return result.asm.find(line => line.text.trim() === text);
}

function requestedMethod(typeArguments: string[] = [], methodArguments: string[] = []): DotNetMethodKey {
    return {
        typeName: 'Example.Box',
        typeArguments,
        methodName: 'Project',
        methodArguments,
        parameters: [],
        returnType: '',
    };
}

const sourceMapping: DotNetSourceMapping = [
    {
        method: {
            typeName: 'Examples.Program',
            typeArguments: [],
            methodName: 'Add',
            methodArguments: [],
            parameters: ['int', 'int'],
            parameterTypes: [valueType('int'), valueType('int')],
            returnType: 'int',
            returnTypeSignature: valueType('int'),
        },
        offsets: {
            0: source(46, 13),
        },
    },
    {
        method: {
            typeName: 'Examples.Box',
            typeArguments: ['!0'],
            methodName: 'Identity',
            methodArguments: ['!!0'],
            parameters: ['!!0'],
            parameterTypes: [genericParameter('!!0')],
            returnType: '!!0',
            returnTypeSignature: genericParameter('!!0'),
        },
        offsets: {
            0: source(62, 13),
            16: source(63, 13),
        },
    },
    {
        method: {
            typeName: 'Examples.Counter',
            typeArguments: [],
            methodName: 'Increment',
            methodArguments: [],
            parameters: [],
            parameterTypes: [],
            returnType: '',
            returnTypeSignature: valueType('void'),
        },
        offsets: {
            0: source(74, 13),
            14: source(75, 9),
        },
    },
    {
        method: {
            typeName: 'Examples.Outer+Inner',
            typeArguments: ['!0', '!1'],
            methodName: 'Mix',
            methodArguments: ['!!0'],
            parameters: [
                'System.Collections.Generic.Dictionary[!0,Examples.Outer+Inner[!0,System.Collections.Generic.List[!!0[]]]]',
                'Examples.Outer+Inner[!0,!1][]',
                '!!0[,]',
                'byref',
                'byref',
                'ptr',
                'ptr',
            ],
            parameterTypes: [
                complexDictionaryType,
                arrayType(referenceType('Examples.Outer+Inner', [genericParameter('!0'), genericParameter('!1')])),
                arrayType(genericParameter('!!0'), '[,]'),
                valueType('byref'),
                valueType('byref'),
                valueType('ptr'),
                valueType('ptr'),
            ],
            returnType:
                'System.Collections.Generic.Dictionary[!0,Examples.Outer+Inner[!0,System.Collections.Generic.List[!!0[]]]]',
            returnTypeSignature: complexDictionaryType,
        },
        offsets: {
            0: source(116, 17),
            32: source(117, 17),
            40: source(118, 17),
            54: source(119, 17),
        },
    },
    {
        method: {
            typeName: 'MappingSample',
            typeArguments: ['!0'],
            methodName: 'RunAsync',
            methodArguments: ['!!0', '!!1'],
            parameters: ['System.Collections.Generic.IEnumerable[System.String]', 'System.Threading.CancellationToken'],
            parameterTypes: [
                referenceType('System.Collections.Generic.IEnumerable', [referenceType('System.String')]),
                valueType('System.Threading.CancellationToken'),
            ],
            returnType: 'System.Threading.Tasks.ValueTask[Result[System.ValueTuple[int,System.String]]]',
            returnTypeSignature: referenceType('System.Threading.Tasks.ValueTask', [asyncRunResultType]),
        },
        offsets: {
            0: source(214, 9),
            16: source(216, 9),
        },
    },
    {
        method: {
            typeName: 'MappingSample+<RunAsync>d__8',
            typeArguments: ['!0', '!1', '!2'],
            methodName: 'MoveNext',
            methodArguments: [],
            parameters: [],
            parameterTypes: [],
            returnType: '',
            returnTypeSignature: valueType('void'),
        },
        offsets: {
            24: source(104, 9),
            54: source(106, 9),
        },
    },
    {
        method: {
            typeName: 'MappingSample+<NormalizeAsync>d__10',
            typeArguments: ['!0'],
            methodName: 'MoveNext',
            methodArguments: [],
            parameters: [],
            parameterTypes: [],
            returnType: '',
            returnTypeSignature: valueType('void'),
        },
        offsets: {
            58: source(187, 29),
            109: source(190, 13),
        },
    },
    {
        method: {
            typeName: 'MappingSample+<Enumerate>d__9',
            typeArguments: ['!0', '!1'],
            methodName: 'MoveNext',
            methodArguments: [],
            parameters: [],
            parameterTypes: [],
            returnType: 'bool',
            returnTypeSignature: valueType('bool'),
        },
        offsets: {
            26: source(165, 14),
            56: source(167, 13),
        },
    },
];

const addDisasm = [
    '; Assembly listing for method Examples.Program:Add(int,int):int (FullOpts)',
    'G_M34084_IG01:  ;; offset=0x0000',
    'G_M34084_IG02:  ;; offset=0x0000',
    '                            ; INLRT @ 0x000[E--]',
    '       lea      eax, [rcx+rdx]',
    'G_M34084_IG03:  ;; offset=0x0003',
    '       ret',
].join('\n');

const identityDisasm = [
    '; Assembly listing for method Examples.Box`1[System.__Canon]:Identity[int](int):int:this (FullOpts)',
    'G_M20789_IG01:  ;; offset=0x0000',
    'G_M20789_IG02:  ;; offset=0x0000',
    '                            ; INLRT @ 0x000[E--]',
    '       mov      rax, gword ptr [rcx+0x08]',
    '                            ; INLRT @ 0x010[E--]',
    '       mov      eax, r8d',
    'G_M20789_IG03:  ;; offset=0x0007',
    '       ret',
].join('\n');

const inlineChainDisasm = [
    '; Assembly listing for method Examples.Box`1[System.__Canon]:Identity[int](int):int:this (FullOpts)',
    'G_M29733_IG01:  ;; offset=0x0000',
    '                            ; INL01 @ 0x000[E--] <- INLRT @ 0x010[E--]',
    '       mov      eax, r8d',
    'G_M29733_IG02:  ;; offset=0x0003',
    '       ret',
].join('\n');

const inlineChainUnknownRootDisasm = [
    '; Assembly listing for method Examples.Box`1[System.__Canon]:Identity[int](int):int:this (FullOpts)',
    'G_M53476_IG01:  ;; offset=0x0000',
    '                            ; INLRT @ 0x000[E--]',
    '       mov      rax, gword ptr [rcx+0x08]',
    '                            ; INL01 @ 0x000[E--] <- INLRT @ ???',
    '       mov      eax, r8d',
    'G_M53476_IG02:  ;; offset=0x0007',
    '       ret',
].join('\n');

const inlineChainUnknownChildDisasm = [
    '; Assembly listing for method MappingSample`1+<Enumerate>d__9`1[System.__Canon,int]:MoveNext():bool:this (FullOpts)',
    'G_M49354_IG01:  ;; offset=0x0000',
    '                            ; INL58 @ 0x009[E--] <- INL57 @ 0x020[E--] <- INL46 @ ??? <- INLRT @ 0x038[E--]',
    '       mov      rcx, gword ptr [rbx+0x08]',
    'G_M49354_IG02:  ;; offset=0x0007',
    '       ret',
].join('\n');

const incrementDisasm = [
    '; Assembly listing for method Examples.Counter:Increment():this (FullOpts)',
    'G_M8093_IG01:  ;; offset=0x0000',
    'G_M8093_IG02:  ;; offset=0x0000',
    '                            ; INLRT @ 0x000[E--]',
    '       inc      dword ptr [rcx+0x08]',
    'G_M8093_IG03:  ;; offset=0x0003',
    '                            ; INLRT @ 0x00E[E--]',
    '       ret',
].join('\n');

const complexSignatureDisasm = [
    '; Assembly listing for method Examples.Outer`1+Inner`1[System.__Canon,int]:Mix[System.__Canon](System.Collections.Generic.Dictionary`2[System.__Canon,System.__Canon],System.__Canon[],System.__Canon[,],byref,byref,ptr,ptr):System.Collections.Generic.Dictionary`2[System.__Canon,System.__Canon] (FullOpts)',
    'G_M48429_IG01:  ;; offset=0x0000',
    '       push     r14',
    'G_M48429_IG02:  ;; offset=0x0018',
    '                            ; INLRT @ 0x000[E--]',
    '       mov      ebp, dword ptr [rdi]',
    '       mov      r14d, dword ptr [rsi+0x38]',
    '       mov      rax, qword ptr [rsp+0x88]',
    '       call     rax',
    '                            ; INLRT @ 0x020[E--]',
    '       xor      rcx, rcx',
    '       mov      rdi, bword ptr [rsp+0x78]',
    '       mov      gword ptr [rdi], rcx',
    'G_M48429_IG03:  ;; offset=0x004A',
    '                            ; INLRT @ 0x028[E--]',
    '       sub      ecx, dword ptr [rbx+0x18]',
    '       mov      rdx, gword ptr [rbx+8*rcx+0x20]',
    '       mov      rcx, bword ptr [rsp+0x70]',
    '       call     CORINFO_HELP_CHECKED_ASSIGN_REF',
    '                            ; INLRT @ 0x036[E--]',
    '       mov      rax, rsi',
    'G_M48429_IG04:  ;; offset=0x0075',
    '       ret',
].join('\n');

const iteratorStateMachineDisasm = [
    '; Assembly listing for method MappingSample`1+<Enumerate>d__9`1[System.__Canon,int]:MoveNext():bool:this (FullOpts)',
    'G_M49354_IG01:  ;; offset=0x0000',
    '                            ; INLRT @ 0x01A[E--]',
    '       xor      ecx, ecx',
    '                            ; INLRT @ 0x038[E--]',
    '       mov      rcx, gword ptr [rbx+0x08]',
    'G_M49354_IG02:  ;; offset=0x0008',
    '       ret',
].join('\n');

const asyncIteratorStateMachineDisasm = [
    '; Assembly listing for method MappingSample`1+<NormalizeAsync>d__10[System.__Canon]:MoveNext():this (FullOpts)',
    'G_M1647_IG01:  ;; offset=0x0000',
    '                            ; INLRT @ 0x03A[E--]',
    '       mov      rcx, gword ptr [rcx+0x10]',
    '                            ; INLRT @ 0x06D[E--]',
    '       mov      byte  ptr [rbp-0x28], 0',
    'G_M1647_IG02:  ;; offset=0x0008',
    '       ret',
].join('\n');

const asyncMethodStateMachineDisasm = [
    '; Assembly listing for method MappingSample`1+<RunAsync>d__8`2[System.__Canon,IntParser,int]:MoveNext():this (FullOpts)',
    'G_M33439_IG01:  ;; offset=0x0000',
    '                            ; INLRT @ 0x018[E--]',
    '       cmp      gword ptr [rsi], 0',
    '                            ; INLRT @ 0x036[E--]',
    '       xor      ecx, ecx',
    'G_M33439_IG02:  ;; offset=0x0008',
    '       ret',
].join('\n');

const runtimeAsyncMethodDisasm = [
    '; Assembly listing for method MappingSample`1[System.__Canon]:RunAsync[IntParser,int](System.Collections.Generic.IEnumerable`1[System.String],System.Threading.CancellationToken):Result`1[System.ValueTuple`2[int,System.String]]:this (FullOpts)',
    'G_M40702_IG01:  ;; offset=0x0000',
    '                            ; INLRT @ 0x000[E--]',
    '       call     CORINFO_HELP_GETSHARED_GCSTATIC_BASE',
    '                            ; INLRT @ 0x010[E--]',
    '       mov      rcx, gword ptr [rax+0x08]',
    'G_M40702_IG02:  ;; offset=0x0008',
    '       ret',
].join('\n');

describe('DotNetAsmParser', () => {
    it('maps source lines from debug-JIT INLRT offsets', () => {
        const result = new DotNetAsmParser(sourceMapping).process(addDisasm, filters());

        expect(result.labelDefinitions).toMatchObject({
            'Examples.Program:Add(int,int):int': 1,
            G_M34084_IG01: 2,
            G_M34084_IG02: 3,
            G_M34084_IG03: 6,
        });
        expect(findLine(result, 'lea      eax, [rcx+rdx]')?.source).toEqual(source(46, 13));
        expect(findLine(result, 'ret')?.source).toBeNull();
    });

    it('keeps source mapping attached when comment-only filtering removes INLRT lines', () => {
        const result = new DotNetAsmParser(sourceMapping).process(identityDisasm, filters({commentOnly: true}));

        expect(result.asm.some(line => line.text.includes('INLRT'))).toBe(false);
        expect(findLine(result, 'mov      rax, gword ptr [rcx+0x08]')?.source).toEqual(source(62, 13));
        expect(findLine(result, 'mov      eax, r8d')?.source).toEqual(source(63, 13));
    });

    it('matches shared-generic JIT method names against metadata generic parameters', () => {
        const result = new DotNetAsmParser(sourceMapping).process(identityDisasm, filters());

        expect(findLine(result, 'mov      rax, gword ptr [rcx+0x08]')?.source).toEqual(source(62, 13));
        expect(findLine(result, 'mov      eax, r8d')?.source).toEqual(source(63, 13));
    });

    it('uses the root offset from recursive inline debug-info chains', () => {
        const result = new DotNetAsmParser(sourceMapping).process(inlineChainDisasm, filters());

        expect(findLine(result, 'mov      eax, r8d')?.source).toEqual(source(63, 13));
    });

    it('does not leak source mapping through inline debug-info chains with unknown root offsets', () => {
        const result = new DotNetAsmParser(sourceMapping).process(inlineChainUnknownRootDisasm, filters());
        const filteredResult = new DotNetAsmParser(sourceMapping).process(
            inlineChainUnknownRootDisasm,
            filters({commentOnly: true}),
        );

        expect(findLine(result, 'mov      rax, gword ptr [rcx+0x08]')?.source).toEqual(source(62, 13));
        expect(findLine(result, 'mov      eax, r8d')?.source).toBeNull();
        expect(findLine(filteredResult, 'mov      eax, r8d')?.source).toBeNull();
    });

    it('uses the root offset when only an intermediate inline frame has an unknown offset', () => {
        const result = new DotNetAsmParser(sourceMapping).process(inlineChainUnknownChildDisasm, filters());

        expect(findLine(result, 'mov      rcx, gword ptr [rbx+0x08]')?.source).toEqual(source(167, 13));
    });

    it('matches instance void methods that end with this in JIT disassembly', () => {
        const result = new DotNetAsmParser(sourceMapping).process(incrementDisasm, filters());

        expect(findLine(result, 'inc      dword ptr [rcx+0x08]')?.source).toEqual(source(74, 13));
        expect(findLine(result, 'ret')?.source).toEqual(source(75, 9));
    });

    it('matches nested generic, byref, pointer, and function-pointer signatures', () => {
        const result = new DotNetAsmParser(sourceMapping).process(complexSignatureDisasm, filters());

        expect(findLine(result, 'call     rax')?.source).toEqual(source(116, 17));
        expect(findLine(result, 'mov      gword ptr [rdi], rcx')?.source).toEqual(source(117, 17));
        expect(findLine(result, 'sub      ecx, dword ptr [rbx+0x18]')?.source).toEqual(source(118, 17));
        expect(findLine(result, 'mov      rax, rsi')?.source).toEqual(source(119, 17));
    });

    it('matches compiler-generated iterator and async state-machine MoveNext methods', () => {
        const iteratorResult = new DotNetAsmParser(sourceMapping).process(iteratorStateMachineDisasm, filters());
        const asyncIteratorResult = new DotNetAsmParser(sourceMapping).process(
            asyncIteratorStateMachineDisasm,
            filters(),
        );
        const asyncMethodResult = new DotNetAsmParser(sourceMapping).process(asyncMethodStateMachineDisasm, filters());

        expect(findLine(iteratorResult, 'xor      ecx, ecx')?.source).toEqual(source(165, 14));
        expect(findLine(iteratorResult, 'mov      rcx, gword ptr [rbx+0x08]')?.source).toEqual(source(167, 13));
        expect(findLine(asyncIteratorResult, 'mov      rcx, gword ptr [rcx+0x10]')?.source).toEqual(source(187, 29));
        expect(findLine(asyncIteratorResult, 'mov      byte  ptr [rbp-0x28], 0')?.source).toEqual(source(190, 13));
        expect(findLine(asyncMethodResult, 'cmp      gword ptr [rsi], 0')?.source).toEqual(source(104, 9));
        expect(findLine(asyncMethodResult, 'xor      ecx, ecx')?.source).toEqual(source(106, 9));
    });

    it('should not take the return type into account when matching methods', () => {
        const result = new DotNetAsmParser(sourceMapping).process(runtimeAsyncMethodDisasm, filters());

        expect(findLine(result, 'call     CORINFO_HELP_GETSHARED_GCSTATIC_BASE')?.source).toEqual(source(214, 9));
        expect(findLine(result, 'mov      rcx, gword ptr [rax+0x08]')?.source).toEqual(source(216, 9));
    });
});

describe('DotNetPdbParser', () => {
    it('parses ECMA-335 method signatures and portable-PDB sequence points', () => {
        const fixture = ecma335Fixture();
        const parsed = new DotNetPdbParser(fixture.assembly, fixture.pdb).parse();
        const method = (name: string) => parsed.find(entry => entry.method.methodName === name)!;

        expect(method('Add')).toMatchObject({
            method: {
                typeName: 'Examples.Program',
                methodName: 'Add',
                parameters: ['int', 'int'],
                returnType: 'int',
            },
            offsets: {
                0: source(46, 13),
                3: null,
            },
        });

        expect(method('PrimitiveKitchenSink').method).toMatchObject({
            parameters: [
                'bool',
                'char',
                'sbyte',
                'byte',
                'short',
                'ushort',
                'int',
                'uint',
                'long',
                'ulong',
                'float',
                'double',
                'System.String',
                'nint',
                'nuint',
                'System.Object',
                'System.TypedReference',
            ],
            returnType: '',
        });
        expect(method('PrimitiveKitchenSink').offsets).toEqual({0: source(80, 9)});

        expect(method('EnumAndValueTypes').method).toMatchObject({
            parameters: ['Examples.Point', 'byte'],
            returnType: 'Examples.Point',
        });
        expect(method('EnumAndValueTypes').method.parameterTypes).toMatchObject([
            {name: 'Examples.Point', typeKind: 'value'},
            {name: 'byte', typeKind: 'value'},
        ]);

        expect(method('TypeSpecAndModifiers').method).toMatchObject({
            parameters: [
                'long',
                'int',
                'System.String',
                'System.Object',
                'System.Enum',
                'System.ValueTuple[System.String,int]',
                'System.Collections.Generic.Dictionary[System.String,int]',
            ],
            returnType: 'System.Collections.Generic.Dictionary[System.String,int]',
        });

        expect(method('Mix')).toMatchObject({
            method: {
                typeName: 'Examples.Outer+Inner',
                typeArguments: ['!0', '!1'],
                methodName: 'Mix',
                methodArguments: ['!!0'],
                parameters: [
                    'System.Collections.Generic.Dictionary[!0,Examples.Outer+Inner[!0,System.Collections.Generic.List[!!0[]]]]',
                    'Examples.Outer+Inner[!0,!1][]',
                    '!!0[,]',
                    'byref',
                    'byref',
                    'ptr',
                    'ptr',
                ],
                returnType:
                    'System.Collections.Generic.Dictionary[!0,Examples.Outer+Inner[!0,System.Collections.Generic.List[!!0[]]]]',
            },
            offsets: {
                0: source(116, 17),
                32: source(117, 17),
                40: source(118, 17),
                54: source(119, 17),
            },
        });
    });
});

describe('DotNetPdbParser helpers', () => {
    it('canonicalizes generic signatures using the requested JIT method arguments', () => {
        const signature = referenceType('System.Collections.Generic.Dictionary', [
            genericParameter('!0'),
            valueType('int'),
        ]);

        expect(getCanonicalTypeSignature(signature, requestedMethod(['System.String'], ['int'])).text).toBe(
            'System.Collections.Generic.Dictionary[System.String,int]',
        );
    });

    it('canonicalizes deeply nested type and method generic parameters', () => {
        const signature = referenceType('System.Collections.Generic.Dictionary', [
            referenceType('Examples.Outer+Inner', [
                genericParameter('!0'),
                referenceType('System.Collections.Generic.List', [arrayType(genericParameter('!!0'))]),
            ]),
            arrayType(referenceType('System.ValueTuple', [genericParameter('!1'), genericParameter('!!1')]), '[,]'),
        ]);

        const result = getCanonicalTypeSignature(
            signature,
            requestedMethod(['System.String', 'nint'], ['int', 'System.Object']),
        );

        expect(result).toEqual({
            text: 'System.Collections.Generic.Dictionary[Examples.Outer+Inner[System.String,System.Collections.Generic.List[int[]]],System.ValueTuple[nint,System.Object][,]]',
            containsSharedGenericReference: false,
        });
    });

    it('preserves array suffixes around canonical generic elements', () => {
        const signature = arrayType(
            referenceType('Examples.Outer+Inner', [genericParameter('!0'), genericParameter('!!0')]),
            '[,,]',
        );

        expect(getCanonicalTypeSignature(signature, requestedMethod(['System.String'], ['int'])).text).toBe(
            'Examples.Outer+Inner[System.String,int][,,]',
        );
    });

    it('leaves unmatched generic parameters in canonical text', () => {
        const signature = referenceType('Examples.Missing', [genericParameter('!2'), genericParameter('!!3')]);

        expect(getCanonicalTypeSignature(signature, requestedMethod(['System.String'], ['int'])).text).toBe(
            'Examples.Missing[!2,!!3]',
        );
    });

    it('collapses shared generic references inside generic reference-type arguments', () => {
        const signature = referenceType('System.Collections.Generic.Dictionary', [
            valueType('int'),
            referenceType('Examples.Box', [arrayType(genericParameter('!0'))]),
        ]);

        expect(getCanonicalTypeSignature(signature, requestedMethod(['System.__Canon'])).text).toBe(
            'System.Collections.Generic.Dictionary[int,System.__Canon]',
        );
    });

    it('can preserve shared generic reference structure when collapse is disabled', () => {
        const signature = referenceType('System.Collections.Generic.Dictionary', [
            valueType('int'),
            referenceType('Examples.Box', [arrayType(genericParameter('!0'))]),
        ]);

        expect(getCanonicalTypeSignature(signature, requestedMethod(['System.__Canon']), false, false)).toEqual({
            text: 'System.Collections.Generic.Dictionary[int,Examples.Box[System.__Canon[]]]',
            containsSharedGenericReference: true,
        });
    });
});
