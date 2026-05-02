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
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

import type {AsmResultSource} from '../../types/asmresult/asmresult.interfaces.js';

export type OffsetSourceMap = Record<string, AsmResultSource | null>;
export type DotNetMethodKey = {
    typeName: string;
    typeArguments: string[];
    methodName: string;
    methodArguments: string[];
    parameters: string[];
    returnType: string;
};
type DotNetTypeKey = {
    name: string;
    genericParameterCount: number;
};
type DotNetTypeRef = {
    name: string;
    namespaceName: string;
    resolutionScope: number;
};
type DotNetTypeDef = {
    name: string;
    namespaceName: string;
    extends: number;
    fieldList: number;
    methodList: number;
};
export type DotNetMethodSourceMapping = {
    method: DotNetMethodKey;
    offsets: OffsetSourceMap;
};

export type DotNetSourceMapping = DotNetMethodSourceMapping[];

const hiddenSequencePoint = 0xfeefee;

type MetadataStreams = Record<string, {offset: number; size: number}>;
type RowCounts = Record<number, number>;
type MetadataTables = {
    buffer: Buffer;
    streams: MetadataStreams;
    rowCounts: RowCounts;
    tableOffsets: Record<number, number>;
    heapSizes: number;
    typeSystemTableRows?: RowCounts;
};

class BufferCursor {
    constructor(
        private readonly buffer: Buffer,
        public offset: number,
    ) {}

    readUInt8() {
        const value = this.buffer.readUInt8(this.offset);
        this.offset += 1;
        return value;
    }

    readUInt16() {
        const value = this.buffer.readUInt16LE(this.offset);
        this.offset += 2;
        return value;
    }

    readUInt32() {
        const value = this.buffer.readUInt32LE(this.offset);
        this.offset += 4;
        return value;
    }

    readBigUInt64() {
        const value = this.buffer.readBigUInt64LE(this.offset);
        this.offset += 8;
        return value;
    }
}

const codedIndexes: Record<string, {tagBits: number; tables: number[]}> = {
    CustomAttributeType: {tagBits: 3, tables: [0x06, 0x0a]},
    HasConstant: {tagBits: 2, tables: [0x04, 0x08, 0x17]},
    HasCustomAttribute: {
        tagBits: 5,
        tables: [
            0x06, 0x04, 0x01, 0x02, 0x08, 0x09, 0x0a, 0x00, 0x0e, 0x17, 0x14, 0x11, 0x1a, 0x1b, 0x20, 0x23, 0x26, 0x27,
            0x28, 0x2a, 0x2c, 0x2b,
        ],
    },
    HasDeclSecurity: {tagBits: 2, tables: [0x02, 0x06, 0x20]},
    HasFieldMarshal: {tagBits: 1, tables: [0x04, 0x08]},
    HasSemantics: {tagBits: 1, tables: [0x14, 0x17]},
    Implementation: {tagBits: 2, tables: [0x26, 0x23, 0x27]},
    MemberForwarded: {tagBits: 1, tables: [0x04, 0x06]},
    MemberRefParent: {tagBits: 3, tables: [0x02, 0x01, 0x1a, 0x06, 0x1b]},
    MethodDefOrRef: {tagBits: 1, tables: [0x06, 0x0a]},
    ResolutionScope: {tagBits: 2, tables: [0x00, 0x1a, 0x23, 0x01]},
    TypeDefOrRef: {tagBits: 2, tables: [0x02, 0x01, 0x1b]},
    TypeOrMethodDef: {tagBits: 1, tables: [0x02, 0x06]},
};

// .NET metadata and PDB parsing based on ECMA-335 and the Portable PDB specification.
// The parser only implements the features necessary to extract source mappings for JIT-compiled code.
export class DotNetPdbParser {
    constructor(
        private readonly assembly: Buffer,
        private readonly pdb: Buffer,
    ) {}

    private readCompressedUInt(buffer: Buffer, offsetRef: {offset: number}) {
        const firstByte = buffer.readUInt8(offsetRef.offset);
        offsetRef.offset += 1;

        if ((firstByte & 0x80) === 0) {
            return {value: firstByte, bytes: 1};
        }

        if ((firstByte & 0xc0) === 0x80) {
            const secondByte = buffer.readUInt8(offsetRef.offset);
            offsetRef.offset += 1;
            return {value: ((firstByte & 0x3f) << 8) | secondByte, bytes: 2};
        }

        const secondByte = buffer.readUInt8(offsetRef.offset);
        const thirdByte = buffer.readUInt8(offsetRef.offset + 1);
        const fourthByte = buffer.readUInt8(offsetRef.offset + 2);
        offsetRef.offset += 3;
        return {value: ((firstByte & 0x1f) << 24) | (secondByte << 16) | (thirdByte << 8) | fourthByte, bytes: 4};
    }

    private readCompressedSignedInt(buffer: Buffer, offsetRef: {offset: number}) {
        const compressed = this.readCompressedUInt(buffer, offsetRef);
        const value = compressed.value >> 1;
        if ((compressed.value & 1) === 0) {
            return value;
        }

        const signedBitCount = compressed.bytes === 1 ? 6 : compressed.bytes === 2 ? 13 : 28;
        return value - (1 << signedBitCount);
    }

    private getMetadataRootOffset(buffer: Buffer) {
        const metadataSignature = 0x424a5342;
        if (buffer.length >= 4 && buffer.readUInt32LE(0) === metadataSignature) {
            return 0; // Portable PDBs start directly at the metadata root.
        }

        if (buffer.length < 0x40 || buffer.readUInt16LE(0) !== 0x5a4d) {
            throw new Error('Metadata root not found');
        }

        const peHeaderOffset = buffer.readUInt32LE(0x3c);
        if (peHeaderOffset + 24 > buffer.length || buffer.readUInt32LE(peHeaderOffset) !== 0x00004550) {
            throw new Error('Invalid PE header');
        }

        const sectionCount = buffer.readUInt16LE(peHeaderOffset + 6);
        const optionalHeaderSize = buffer.readUInt16LE(peHeaderOffset + 20);
        const optionalHeaderOffset = peHeaderOffset + 24;
        const sectionTableOffset = optionalHeaderOffset + optionalHeaderSize;
        if (sectionTableOffset + sectionCount * 40 > buffer.length) {
            throw new Error('Invalid PE optional header');
        }

        const optionalHeaderMagic = buffer.readUInt16LE(optionalHeaderOffset);
        const dataDirectoryOffset = optionalHeaderMagic === 0x20b ? 112 : optionalHeaderMagic === 0x10b ? 96 : -1;
        const cliHeaderDirectoryOffset = optionalHeaderOffset + dataDirectoryOffset + 14 * 8;
        if (dataDirectoryOffset === -1 || cliHeaderDirectoryOffset + 8 > sectionTableOffset) {
            throw new Error('Invalid PE optional header');
        }

        const rvaToFileOffset = (rva: number) => {
            for (let section = 0; section < sectionCount; section++) {
                const sectionOffset = sectionTableOffset + section * 40;
                const virtualSize = buffer.readUInt32LE(sectionOffset + 8);
                const virtualAddress = buffer.readUInt32LE(sectionOffset + 12);
                const rawDataSize = buffer.readUInt32LE(sectionOffset + 16);
                const rawDataPointer = buffer.readUInt32LE(sectionOffset + 20);
                const sectionSize = Math.max(virtualSize, rawDataSize);
                if (rva >= virtualAddress && rva < virtualAddress + sectionSize) {
                    return rawDataPointer + (rva - virtualAddress);
                }
            }

            throw new Error('RVA not found in PE sections');
        };

        const cliHeaderRva = buffer.readUInt32LE(cliHeaderDirectoryOffset);
        if (cliHeaderRva === 0) {
            throw new Error('CLI header not found');
        }

        const cliHeaderOffset = rvaToFileOffset(cliHeaderRva);
        if (cliHeaderOffset + 12 > buffer.length) {
            throw new Error('Invalid CLI header');
        }

        const metadataRva = buffer.readUInt32LE(cliHeaderOffset + 8);
        const metadataRoot = rvaToFileOffset(metadataRva);

        if (metadataRoot + 4 > buffer.length || buffer.readUInt32LE(metadataRoot) !== metadataSignature) {
            throw new Error('Invalid metadata signature');
        }

        return metadataRoot;
    }

    private parseMetadataStreams(buffer: Buffer): MetadataStreams {
        const metadataRoot = this.getMetadataRootOffset(buffer);
        const cursor = new BufferCursor(buffer, metadataRoot);

        if (cursor.readUInt32() !== 0x424a5342) {
            throw new Error('Invalid metadata signature');
        }

        cursor.readUInt16();
        cursor.readUInt16();
        cursor.readUInt32();

        const versionLength = cursor.readUInt32();
        cursor.offset = (cursor.offset + versionLength + 3) & ~3;

        cursor.readUInt16();
        const streamCount = cursor.readUInt16();
        const streams: MetadataStreams = {};

        for (let i = 0; i < streamCount; i++) {
            const offset = cursor.readUInt32();
            const size = cursor.readUInt32();
            const nameStart = cursor.offset;

            while (buffer.readUInt8(cursor.offset) !== 0) {
                cursor.offset++;
            }

            const name = buffer.toString('utf8', nameStart, cursor.offset);
            cursor.offset = (cursor.offset + 4) & ~3;
            streams[name] = {offset: metadataRoot + offset, size};
        }

        return streams;
    }

    private heapIndexSize(heapSizes: number, bit: number) {
        return (heapSizes & bit) === 0 ? 2 : 4;
    }

    private tableIndexSize(rowCounts: RowCounts, typeSystemTableRows: RowCounts | undefined, tableId: number) {
        return (rowCounts[tableId] ?? typeSystemTableRows?.[tableId] ?? 0) < 0x10000 ? 2 : 4;
    }

    parse(): DotNetSourceMapping {
        const assemblyMethods = this.parseAssemblyMethods();
        const sequencePoints = this.parsePdbSequencePoints();
        const sourceMapping: DotNetSourceMapping = [];

        for (const methodRowId of Object.keys(sequencePoints).map(Number)) {
            sourceMapping.push({method: assemblyMethods[methodRowId], offsets: sequencePoints[methodRowId]});
        }

        return sourceMapping;
    }

    private genericParameterArguments(count: number, prefix: '!' | '!!') {
        return Array.from({length: count}, (_, index) => `${prefix}${index}`);
    }

    private fallbackMetadataNameGenericArity(name: string) {
        const arity = name.match(/`(?<arity>\d+)$/);
        return arity?.groups ? Number.parseInt(arity.groups.arity, 10) : 0;
    }

    private stripMetadataGenericArity(name: string, genericParameterCount: number) {
        if (genericParameterCount === 0) {
            return name;
        }

        const arity = `\`${genericParameterCount}`;
        return name.endsWith(arity) ? name.substring(0, name.length - arity.length) : name;
    }

    private codedIndexSize(
        codedIndex: keyof typeof codedIndexes,
        rowCounts: RowCounts,
        typeSystemTableRows?: RowCounts,
    ) {
        const definition = codedIndexes[codedIndex];
        const largestRowCount = Math.max(
            ...definition.tables.map(table => rowCounts[table] ?? typeSystemTableRows?.[table] ?? 0),
        );
        return largestRowCount < 1 << (16 - definition.tagBits) ? 2 : 4;
    }

    private tableRowSize(tableId: number, tables: MetadataTables) {
        const strings = this.heapIndexSize(tables.heapSizes, 0x01);
        const guid = this.heapIndexSize(tables.heapSizes, 0x02);
        const blob = this.heapIndexSize(tables.heapSizes, 0x04);
        const table = (id: number) => this.tableIndexSize(tables.rowCounts, tables.typeSystemTableRows, id);
        const coded = (name: keyof typeof codedIndexes) =>
            this.codedIndexSize(name, tables.rowCounts, tables.typeSystemTableRows);

        switch (tableId) {
            case 0x00: // Module
                return 2 + strings + guid + guid + guid;
            case 0x01: // TypeRef
                return coded('ResolutionScope') + strings + strings;
            case 0x02: // TypeDef
                return 4 + strings + strings + coded('TypeDefOrRef') + table(0x04) + table(0x06);
            case 0x03: // FieldPtr
                return table(0x04);
            case 0x04: // Field
                return 2 + strings + blob;
            case 0x05: // MethodPtr
                return table(0x06);
            case 0x06: // MethodDef
                return 4 + 2 + 2 + strings + blob + table(0x08);
            case 0x07: // ParamPtr
                return table(0x08);
            case 0x08: // Param
                return 2 + 2 + strings;
            case 0x09: // InterfaceImpl
                return table(0x02) + coded('TypeDefOrRef');
            case 0x0a: // MemberRef
                return coded('MemberRefParent') + strings + blob;
            case 0x0b: // Constant
                return 2 + coded('HasConstant') + blob;
            case 0x0c: // CustomAttribute
                return coded('HasCustomAttribute') + coded('CustomAttributeType') + blob;
            case 0x0d: // FieldMarshal
                return coded('HasFieldMarshal') + blob;
            case 0x0e: // DeclSecurity
                return 2 + coded('HasDeclSecurity') + blob;
            case 0x0f: // ClassLayout
                return 2 + 4 + table(0x02);
            case 0x10: // FieldLayout
                return 4 + table(0x04);
            case 0x11: // StandAloneSig
                return blob;
            case 0x12: // EventMap
                return table(0x02) + table(0x14);
            case 0x13: // EventPtr
                return table(0x14);
            case 0x14: // Event
                return 2 + strings + coded('TypeDefOrRef');
            case 0x15: // PropertyMap
                return table(0x02) + table(0x17);
            case 0x16: // PropertyPtr
                return table(0x17);
            case 0x17: // Property
                return 2 + strings + blob;
            case 0x18: // MethodSemantics
                return 2 + table(0x06) + coded('HasSemantics');
            case 0x19: // MethodImpl
                return table(0x02) + coded('MethodDefOrRef') + coded('MethodDefOrRef');
            case 0x1a: // ModuleRef
                return strings;
            case 0x1b: // TypeSpec
                return blob;
            case 0x1c: // ImplMap
                return 2 + coded('MemberForwarded') + strings + table(0x1a);
            case 0x1d: // FieldRVA
                return 4 + table(0x04);
            case 0x1e: // EncLog
                return 4 + 4;
            case 0x1f: // EncMap
                return 4;
            case 0x20: // Assembly
                return 4 + 2 + 2 + 2 + 2 + 4 + blob + strings + strings;
            case 0x21: // AssemblyProcessor
                return 4;
            case 0x22: // AssemblyOS
                return 4 + 4 + 4;
            case 0x23: // AssemblyRef
                return 2 + 2 + 2 + 2 + 4 + blob + strings + strings + blob;
            case 0x24: // AssemblyRefProcessor
                return 4 + table(0x23);
            case 0x25: // AssemblyRefOS
                return 4 + 4 + 4 + table(0x23);
            case 0x26: // File
                return 4 + strings + blob;
            case 0x27: // ExportedType
                return 4 + 4 + strings + strings + coded('Implementation');
            case 0x28: // ManifestResource
                return 4 + 4 + strings + coded('Implementation');
            case 0x29: // NestedClass
                return table(0x02) + table(0x02);
            case 0x2a: // GenericParam
                return 2 + 2 + coded('TypeOrMethodDef') + strings;
            case 0x2b: // MethodSpec
                return coded('MethodDefOrRef') + blob;
            case 0x2c: // GenericParamConstraint
                return table(0x2a) + coded('TypeDefOrRef');
            case 0x30: // Document
                return blob + guid + blob + guid;
            case 0x31: // MethodDebugInformation
                return table(0x30) + blob;
            case 0x32: // LocalScope
                return table(0x06) + table(0x35) + table(0x33) + table(0x34) + 4 + 4;
            case 0x33: // LocalVariable
                return 2 + 2 + strings;
            case 0x34: // LocalConstant
                return strings + blob;
            case 0x35: // ImportScope
                return table(0x35) + blob;
            case 0x36: // StateMachineMethod
                return table(0x06) + table(0x06);
            case 0x37: // CustomDebugInformation
                return coded('HasCustomAttribute') + guid + blob;
            default:
                throw new Error(`Unsupported metadata table 0x${tableId.toString(16)}`);
        }
    }

    private parsePdbTypeSystemTableRows(buffer: Buffer, streams: MetadataStreams): RowCounts {
        const pdbStream = streams['#Pdb']!;

        const cursor = new BufferCursor(buffer, pdbStream.offset + 24); // 20-byte PDB id, then entry point token.
        const referencedTables = cursor.readBigUInt64();
        const rowCounts: RowCounts = {};

        for (let tableId = 0; tableId < 64; tableId++) {
            if (((referencedTables >> BigInt(tableId)) & 1n) !== 0n) {
                rowCounts[tableId] = cursor.readUInt32();
            }
        }

        return rowCounts;
    }

    private parseMetadataTables(buffer: Buffer, streams: MetadataStreams, typeSystemTableRows?: RowCounts) {
        const tableStream = (streams['#~'] ?? streams['#-'])!;

        const cursor = new BufferCursor(buffer, tableStream.offset);
        cursor.readUInt32();
        cursor.readUInt8();
        cursor.readUInt8();
        const heapSizes = cursor.readUInt8();
        cursor.readUInt8();
        const validTables = cursor.readBigUInt64();
        cursor.readBigUInt64(); // Sorted table mask; row counts are keyed only by the valid table mask above.

        const rowCounts: RowCounts = {};
        for (let tableId = 0; tableId < 64; tableId++) {
            if (((validTables >> BigInt(tableId)) & 1n) !== 0n) {
                rowCounts[tableId] = cursor.readUInt32();
            }
        }

        const tables: MetadataTables = {
            buffer,
            heapSizes,
            rowCounts,
            streams,
            tableOffsets: {},
            typeSystemTableRows,
        };

        let tableOffset = cursor.offset;
        for (let tableId = 0; tableId < 64; tableId++) {
            const rowCount = rowCounts[tableId] ?? 0;
            if (rowCount === 0) {
                continue;
            }

            tables.tableOffsets[tableId] = tableOffset;
            tableOffset += rowCount * this.tableRowSize(tableId, tables);
        }

        return tables;
    }

    private readIndex(buffer: Buffer, offset: number, size: number) {
        return size === 2 ? buffer.readUInt16LE(offset) : buffer.readUInt32LE(offset);
    }

    private getString(tables: MetadataTables, index: number) {
        if (index === 0) {
            return '';
        }

        const strings = tables.streams['#Strings']!;

        const start = strings.offset + index;
        let end = start;
        while (end < strings.offset + strings.size && tables.buffer.readUInt8(end) !== 0) {
            end++;
        }

        return tables.buffer.toString('utf8', start, end);
    }

    private getBlob(tables: MetadataTables, index: number) {
        if (index === 0) {
            return Buffer.alloc(0);
        }

        const blobs = tables.streams['#Blob']!;

        const offsetRef = {offset: blobs.offset + index};
        const length = this.readCompressedUInt(tables.buffer, offsetRef).value;
        return tables.buffer.subarray(offsetRef.offset, offsetRef.offset + length);
    }

    private rowOffset(tables: MetadataTables, tableId: number, rowId: number) {
        const offset = tables.tableOffsets[tableId];
        return offset + (rowId - 1) * this.tableRowSize(tableId, tables);
    }

    private parseElementType(
        blob: Buffer,
        offsetRef: {offset: number},
        resolveType: (encoded: number) => string,
    ): string {
        while (blob[offsetRef.offset] === 0x1f || blob[offsetRef.offset] === 0x20) {
            offsetRef.offset++;
            this.readCompressedUInt(blob, offsetRef);
        }

        const type = blob.readUInt8(offsetRef.offset);
        offsetRef.offset++;

        switch (type) {
            case 0x01: // Void
                return 'void';
            case 0x02: // Boolean
                return 'bool';
            case 0x03: // Char
                return 'char';
            case 0x04: // SByte
                return 'sbyte';
            case 0x05: // Byte
                return 'byte';
            case 0x06: // Int16
                return 'short';
            case 0x07: // UInt16
                return 'ushort';
            case 0x08: // Int32
                return 'int';
            case 0x09: // UInt32
                return 'uint';
            case 0x0a: // Int64
                return 'long';
            case 0x0b: // UInt64
                return 'ulong';
            case 0x0c: // Single
                return 'float';
            case 0x0d: // Double
                return 'double';
            case 0x0e: // String
                return 'System.String';
            case 0x0f: // Pointer
                return `${this.parseElementType(blob, offsetRef, resolveType)}*`;
            case 0x10: // ByReference
                return `${this.parseElementType(blob, offsetRef, resolveType)}&`;
            case 0x11: // ValueType
            case 0x12: // Class
                return resolveType(this.readCompressedUInt(blob, offsetRef).value);
            case 0x13: // GenericTypeParameter
                return `!${this.readCompressedUInt(blob, offsetRef).value}`;
            case 0x14: // Array
                {
                    const elementType = this.parseElementType(blob, offsetRef, resolveType);
                    const rank = this.readCompressedUInt(blob, offsetRef).value;
                    const sizeCount = this.readCompressedUInt(blob, offsetRef).value;
                    for (let i = 0; i < sizeCount; i++) {
                        this.readCompressedUInt(blob, offsetRef);
                    }
                    const lowerBoundCount = this.readCompressedUInt(blob, offsetRef).value;
                    for (let i = 0; i < lowerBoundCount; i++) {
                        this.readCompressedSignedInt(blob, offsetRef);
                    }
                    return `${elementType}[${','.repeat(Math.max(rank - 1, 0))}]`;
                }
            case 0x15: // GenericTypeInstance
                {
                    const genericType = this.parseElementType(blob, offsetRef, resolveType);
                    const argumentCount = this.readCompressedUInt(blob, offsetRef).value;
                    const argumentsText: string[] = [];
                    for (let i = 0; i < argumentCount; i++) {
                        argumentsText.push(this.parseElementType(blob, offsetRef, resolveType));
                    }
                    return `${this.stripMetadataGenericArity(genericType, argumentCount)}[${argumentsText.join(',')}]`;
                }
            case 0x16: // TypedByReference
                return 'System.TypedReference';
            case 0x18: // IntPtr
                return 'nint';
            case 0x19: // UIntPtr
                return 'nuint';
            case 0x1c: // Object
                return 'System.Object';
            case 0x1b: // FunctionPointer
                this.parseMethodSignature(blob, resolveType, offsetRef); // Function pointer signatures are omitted in JIT disasm, ignoring the parsed signature here
                return 'ptr';
            case 0x1d: // SZArray
                return `${this.parseElementType(blob, offsetRef, resolveType)}[]`;
            case 0x1e: // GenericMethodParameter
                return `!!${this.readCompressedUInt(blob, offsetRef).value}`;
            case 0x41: // Sentinel
                return this.parseElementType(blob, offsetRef, resolveType);
            case 0x45: // Pinned
                return this.parseElementType(blob, offsetRef, resolveType);
            default:
                throw new Error(`Unsupported element type 0x${type.toString(16)}`);
        }
    }

    private parseMethodSignature(
        blob: Buffer,
        resolveType: (encoded: number) => string,
        offsetRef: {offset: number} = {offset: 0},
    ) {
        const callingConvention = blob.readUInt8(offsetRef.offset);
        offsetRef.offset++;

        const genericParameterCount =
            (callingConvention & 0x10) !== 0 ? this.readCompressedUInt(blob, offsetRef).value : 0;

        const parameterCount = this.readCompressedUInt(blob, offsetRef).value;
        const returnType = this.parseElementType(blob, offsetRef, resolveType);

        const parameters: string[] = [];
        for (let i = 0; i < parameterCount; i++) {
            parameters.push(this.parseElementType(blob, offsetRef, resolveType));
        }

        return {genericParameterCount, parameters, returnType};
    }

    private parseAssemblyMethods(): Record<number, DotNetMethodKey> {
        const assembly = this.assembly;
        const streams = this.parseMetadataStreams(assembly);
        const tables = this.parseMetadataTables(assembly, streams);
        const strings = this.heapIndexSize(tables.heapSizes, 0x01);
        const blob = this.heapIndexSize(tables.heapSizes, 0x04);
        const typeDefIndex = this.tableIndexSize(tables.rowCounts, tables.typeSystemTableRows, 0x02);
        const methodDefIndex = this.tableIndexSize(tables.rowCounts, tables.typeSystemTableRows, 0x06);
        const fieldIndex = this.tableIndexSize(tables.rowCounts, tables.typeSystemTableRows, 0x04);
        const typeDefOrRef = this.codedIndexSize('TypeDefOrRef', tables.rowCounts, tables.typeSystemTableRows);
        const resolutionScope = this.codedIndexSize('ResolutionScope', tables.rowCounts, tables.typeSystemTableRows);
        const typeOrMethodDef = this.codedIndexSize('TypeOrMethodDef', tables.rowCounts, tables.typeSystemTableRows);
        const typeRefs: Record<number, DotNetTypeRef> = {};
        const typeDefs: Record<number, DotNetTypeDef> = {};
        const fields: Record<number, {name: string; signature: number}> = {};
        const nestedTypes: Record<number, number> = {};
        const typeGenericParameterCounts: Record<number, number> = {};
        const methodGenericParameterCounts: Record<number, number> = {};

        for (let rowId = 1; rowId <= (tables.rowCounts[0x01] ?? 0); rowId++) {
            const offset = this.rowOffset(tables, 0x01, rowId);
            const resolutionScopeValue = this.readIndex(assembly, offset, resolutionScope);
            const nameOffset = offset + resolutionScope;
            const namespaceOffset = nameOffset + strings;
            typeRefs[rowId] = {
                name: this.getString(tables, this.readIndex(assembly, nameOffset, strings)),
                namespaceName: this.getString(tables, this.readIndex(assembly, namespaceOffset, strings)),
                resolutionScope: resolutionScopeValue,
            };
        }

        for (let rowId = 1; rowId <= (tables.rowCounts[0x02] ?? 0); rowId++) {
            const offset = this.rowOffset(tables, 0x02, rowId);
            const nameOffset = offset + 4;
            const namespaceOffset = nameOffset + strings;
            const extendsOffset = namespaceOffset + strings;
            const fieldListOffset = extendsOffset + typeDefOrRef;
            const methodListOffset = fieldListOffset + fieldIndex;
            typeDefs[rowId] = {
                extends: this.readIndex(assembly, extendsOffset, typeDefOrRef),
                fieldList: this.readIndex(assembly, fieldListOffset, fieldIndex),
                methodList: this.readIndex(assembly, methodListOffset, methodDefIndex),
                name: this.getString(tables, this.readIndex(assembly, nameOffset, strings)),
                namespaceName: this.getString(tables, this.readIndex(assembly, namespaceOffset, strings)),
            };
        }

        for (let rowId = 1; rowId <= (tables.rowCounts[0x04] ?? 0); rowId++) {
            const offset = this.rowOffset(tables, 0x04, rowId);
            const nameOffset = offset + 2;
            const signatureOffset = nameOffset + strings;
            fields[rowId] = {
                name: this.getString(tables, this.readIndex(assembly, nameOffset, strings)),
                signature: this.readIndex(assembly, signatureOffset, blob),
            };
        }

        for (let rowId = 1; rowId <= (tables.rowCounts[0x29] ?? 0); rowId++) {
            const offset = this.rowOffset(tables, 0x29, rowId);
            const nested = this.readIndex(assembly, offset, typeDefIndex);
            const enclosing = this.readIndex(assembly, offset + typeDefIndex, typeDefIndex);
            nestedTypes[nested] = enclosing;
        }

        for (let rowId = 1; rowId <= (tables.rowCounts[0x2a] ?? 0); rowId++) {
            const offset = this.rowOffset(tables, 0x2a, rowId);
            const parameterNumber = assembly.readUInt16LE(offset);
            const owner = this.readIndex(assembly, offset + 4, typeOrMethodDef);
            const ownerRowId = owner >> 1;
            const ownerTag = owner & 0x01;
            const ownerCounts = ownerTag === 0 ? typeGenericParameterCounts : methodGenericParameterCounts;
            ownerCounts[ownerRowId] = Math.max(ownerCounts[ownerRowId] ?? 0, parameterNumber + 1);
        }

        const resolvedTypeDefs: Record<number, DotNetTypeKey> = {};
        const resolveTypeDef = (rowId: number): DotNetTypeKey => {
            if (resolvedTypeDefs[rowId]) {
                return resolvedTypeDefs[rowId];
            }

            const typeDef = typeDefs[rowId];
            const enclosingType = nestedTypes[rowId]
                ? resolveTypeDef(nestedTypes[rowId])
                : {genericParameterCount: 0, name: ''};
            const ownGenericParameterCount =
                typeGenericParameterCounts[rowId] ?? this.fallbackMetadataNameGenericArity(typeDef.name);
            const typeNameWithoutArity = this.stripMetadataGenericArity(typeDef.name, ownGenericParameterCount);
            const typeName = enclosingType.name
                ? `${enclosingType.name}+${typeNameWithoutArity}`
                : typeDef.namespaceName
                  ? `${typeDef.namespaceName}.${typeNameWithoutArity}`
                  : typeNameWithoutArity;
            resolvedTypeDefs[rowId] = {
                genericParameterCount: enclosingType.genericParameterCount + ownGenericParameterCount,
                name: typeName,
            };
            return resolvedTypeDefs[rowId];
        };

        const resolvedTypeDefsByRow: Record<number, DotNetTypeKey> = {};
        for (const rowId of Object.keys(typeDefs).map(Number)) {
            resolvedTypeDefsByRow[rowId] = resolveTypeDef(rowId);
        }

        const methodDeclaringTypes: Record<number, DotNetTypeKey> = {};
        const sortedTypeDefs = Object.keys(typeDefs)
            .map(Number)
            .sort((a, b) => a - b);
        const methodCount = tables.rowCounts[0x06] ?? 0;
        for (let i = 0; i < sortedTypeDefs.length; i++) {
            const typeDefRowId = sortedTypeDefs[i];
            const start = typeDefs[typeDefRowId].methodList;
            const end = i + 1 < sortedTypeDefs.length ? typeDefs[sortedTypeDefs[i + 1]].methodList : methodCount + 1;
            for (let methodRowId = start; methodRowId < end; methodRowId++) {
                methodDeclaringTypes[methodRowId] = resolvedTypeDefsByRow[typeDefRowId];
            }
        }

        const methodInfos: Record<number, DotNetMethodKey> = {};
        const resolvedTypeRefs: Record<number, string> = {};
        const resolveTypeRef = (rowId: number): string => {
            if (resolvedTypeRefs[rowId]) {
                return resolvedTypeRefs[rowId];
            }

            const typeRef = typeRefs[rowId];
            resolvedTypeRefs[rowId] =
                (typeRef.resolutionScope & 0x03) === 3
                    ? `${resolveTypeRef(typeRef.resolutionScope >> 2)}+${typeRef.name}`
                    : typeRef.namespaceName
                      ? `${typeRef.namespaceName}.${typeRef.name}`
                      : typeRef.name;
            return resolvedTypeRefs[rowId];
        };
        const enumUnderlyingTypes: Record<number, string> = {};
        const resolvedTypeSpecs: Record<number, string> = {};
        const resolveType = (encoded: number) => {
            const tag = encoded & 0x03;
            const rowId = encoded >> 2;
            if (tag === 0) {
                return enumUnderlyingTypes[rowId] ?? resolvedTypeDefsByRow[rowId].name;
            }
            if (tag === 1) {
                return resolveTypeRef(rowId);
            }
            if (tag === 2) {
                resolvedTypeSpecs[rowId] ??= this.parseElementType(
                    this.getBlob(tables, this.readIndex(assembly, this.rowOffset(tables, 0x1b, rowId), blob)),
                    {offset: 0},
                    resolveType,
                );
                return resolvedTypeSpecs[rowId];
            }
            throw new Error(`Unsupported TypeDefOrRef tag ${tag}`);
        };
        for (let i = 0; i < sortedTypeDefs.length; i++) {
            const typeDefRowId = sortedTypeDefs[i];
            if (typeDefs[typeDefRowId].extends === 0 || resolveType(typeDefs[typeDefRowId].extends) !== 'System.Enum') {
                continue;
            }

            const start = typeDefs[typeDefRowId].fieldList;
            const end =
                i + 1 < sortedTypeDefs.length
                    ? typeDefs[sortedTypeDefs[i + 1]].fieldList
                    : (tables.rowCounts[0x04] ?? 0) + 1;
            for (let fieldRowId = start; fieldRowId < end; fieldRowId++) {
                if (fields[fieldRowId].name === 'value__') {
                    enumUnderlyingTypes[typeDefRowId] = this.parseElementType(
                        this.getBlob(tables, fields[fieldRowId].signature),
                        {offset: 1},
                        resolveType,
                    );
                    break;
                }
            }
        }
        for (let rowId = 1; rowId <= methodCount; rowId++) {
            const offset = this.rowOffset(tables, 0x06, rowId);
            const type = methodDeclaringTypes[rowId];
            const nameOffset = offset + 4 + 2 + 2;
            const signatureOffset = nameOffset + strings;
            const methodName = this.getString(tables, this.readIndex(assembly, nameOffset, strings));
            const {
                genericParameterCount: signatureGenericParameterCount,
                parameters,
                returnType,
            } = this.parseMethodSignature(
                this.getBlob(tables, this.readIndex(assembly, signatureOffset, blob)),
                resolveType,
            );

            methodInfos[rowId] = {
                methodArguments: this.genericParameterArguments(
                    methodGenericParameterCounts[rowId] ?? signatureGenericParameterCount,
                    '!!',
                ),
                methodName,
                parameters,
                returnType: returnType === 'void' ? '' : returnType,
                typeArguments: this.genericParameterArguments(type.genericParameterCount, '!'),
                typeName: type.name,
            };
        }

        return methodInfos;
    }

    private decodeSequencePoints(blob: Buffer, initialDocument: number) {
        const offsets: OffsetSourceMap = {};
        if (blob.length === 0) {
            return offsets;
        }

        const offsetRef = {offset: 0};
        this.readCompressedUInt(blob, offsetRef);

        if (initialDocument === 0) {
            this.readCompressedUInt(blob, offsetRef); // The first sequence point carries the document row when needed.
        }

        let previousIlOffset = 0;
        let hasPreviousIlOffset = false;
        let previousNonHiddenLine = 0;
        let previousNonHiddenColumn = 0;
        let hasPreviousNonHiddenSource = false;

        while (offsetRef.offset < blob.length) {
            const deltaIlOffset = this.readCompressedUInt(blob, offsetRef).value;
            if (deltaIlOffset === 0 && hasPreviousIlOffset) {
                this.readCompressedUInt(blob, offsetRef); // Document switch record.
                continue;
            }

            const ilOffset = hasPreviousIlOffset ? previousIlOffset + deltaIlOffset : deltaIlOffset;
            const deltaLines = this.readCompressedUInt(blob, offsetRef).value;
            const deltaColumns =
                deltaLines === 0
                    ? this.readCompressedUInt(blob, offsetRef).value
                    : this.readCompressedSignedInt(blob, offsetRef);

            if (deltaLines === 0 && deltaColumns === 0) {
                offsets[ilOffset] = null;
            } else {
                const startLine = hasPreviousNonHiddenSource
                    ? previousNonHiddenLine + this.readCompressedSignedInt(blob, offsetRef)
                    : this.readCompressedUInt(blob, offsetRef).value;
                const startColumn = hasPreviousNonHiddenSource
                    ? previousNonHiddenColumn + this.readCompressedSignedInt(blob, offsetRef)
                    : this.readCompressedUInt(blob, offsetRef).value;

                if (startLine === hiddenSequencePoint) {
                    offsets[ilOffset] = null;
                } else {
                    offsets[ilOffset] = {file: null, line: startLine, column: startColumn};
                    previousNonHiddenLine = startLine;
                    previousNonHiddenColumn = startColumn;
                    hasPreviousNonHiddenSource = true;
                }
            }

            previousIlOffset = ilOffset;
            hasPreviousIlOffset = true;
        }

        return offsets;
    }

    private parsePdbSequencePoints(): Record<number, OffsetSourceMap> {
        const pdb = this.pdb;
        const streams = this.parseMetadataStreams(pdb);
        const tables = this.parseMetadataTables(pdb, streams, this.parsePdbTypeSystemTableRows(pdb, streams));
        const documentIndex = this.tableIndexSize(tables.rowCounts, tables.typeSystemTableRows, 0x30);
        const blob = this.heapIndexSize(tables.heapSizes, 0x04);
        const methodSequencePoints: Record<number, OffsetSourceMap> = {};

        for (let methodRowId = 1; methodRowId <= (tables.rowCounts[0x31] ?? 0); methodRowId++) {
            const offset = this.rowOffset(tables, 0x31, methodRowId);
            const initialDocument = this.readIndex(pdb, offset, documentIndex);
            const sequencePointsBlob = this.getBlob(tables, this.readIndex(pdb, offset + documentIndex, blob));
            methodSequencePoints[methodRowId] = this.decodeSequencePoints(sequencePointsBlob, initialDocument);
        }

        return methodSequencePoints;
    }
}
