import {describe, expect, it} from 'vitest';

import {BinaryInfoLinux} from '../lib/binaries/binary-utils.js';
import {BaseExecutionTriple, ExecutionSpecialty} from '../lib/execution/base-execution-triple.js';
import {getExecutionTriplesForCurrentHost, matchesCurrentHost} from '../lib/execution/execution-triple.js';

describe('Execution triple utils', () => {
    it('default always matches', () => {
        const triples: BaseExecutionTriple[] = getExecutionTriplesForCurrentHost();
        expect(matchesCurrentHost(triples[0])).toEqual(true);
    });
    it('can parse a thing', () => {
        const triple = new BaseExecutionTriple();
        triple.parse('aarch64-linux-cpu');
        expect(triple.instructionSet).toEqual('aarch64');
        expect(triple.os).toEqual('linux');
        expect(triple.specialty).toEqual(ExecutionSpecialty.cpu);
    });
    it('would parse nvgpu', () => {
        const triple = new BaseExecutionTriple();
        triple.parse('amd64-linux-nvgpu');
        expect(triple.instructionSet).toEqual('amd64');
        expect(triple.os).toEqual('linux');
        expect(triple.specialty).toEqual(ExecutionSpecialty.nvgpu);
    });
    it('recognizes aarch64', () => {
        const info = BinaryInfoLinux.parseFileInfo(
            'ELF 64-bit LSB pie executable, ARM aarch64, version 1 (SYSV), dynamically linked, interpreter /lib/ld-linux-aarch64.so.1, for GNU/Linux 5.17.0, with debug_info, not stripped',
        );
        expect(info?.instructionSet).toEqual('aarch64');
        expect(info?.os).toEqual('linux');
    });
    it('recognizes arm32', () => {
        const info = BinaryInfoLinux.parseFileInfo(
            'ELF 32-bit LSB executable, ARM, EABI5 version 1 (GNU/Linux), dynamically linked, interpreter /lib/ld-linux-armhf.so.3, for GNU/Linux 4.4.255, with debug_info, not stripped',
        );
        expect(info?.instructionSet).toEqual('arm32');
        expect(info?.os).toEqual('linux');
    });
    it('recognizes lin64', () => {
        const info = BinaryInfoLinux.parseFileInfo(
            'ELF 64-bit LSB executable, x86-64, version 1 (GNU/Linux), dynamically linked, interpreter /lib64/ld-linux-x86-64.so.2, BuildID[sha1]=9def7a0f68cf33177d0d8566b152af24d9ec1ac4, for GNU/Linux 3.2.0, stripped',
        );
        expect(info?.instructionSet).toEqual('amd64');
        expect(info?.os).toEqual('linux');
    });
    it('recognizes win64', () => {
        const info = BinaryInfoLinux.parseFileInfo(
            'PE32+ executable (DLL) (console) x86-64 (stripped to external PDB), for MS Windows, 12 sections',
        );
        expect(info?.instructionSet).toEqual('amd64');
        expect(info?.os).toEqual('win32');
    });
    it('recognizes win alternatives', () => {
        const filteredLine = BinaryInfoLinux.removeComments(
            'PE32+ executable (DLL) (console) Aarch64 (stripped to external PDB)',
        );
        expect(filteredLine).toEqual('PE32+ executable   Aarch64');

        const info = BinaryInfoLinux.parseFileInfo(
            'PE32+ executable (DLL) (console) Aarch64 (stripped to external PDB), for MS Windows, 12 sections',
        );
        expect(info?.instructionSet).toEqual('aarch64');
        expect(info?.os).toEqual('win32');
    });
    it('recognizes win32', () => {
        const info = BinaryInfoLinux.parseFileInfo('PE32 executable (GUI) Intel 80386, for MS Windows, 4 sections');
        expect(info?.instructionSet).toEqual('x86');
        expect(info?.os).toEqual('win32');
    });
    it('recognizes avr', () => {
        const info = BinaryInfoLinux.parseFileInfo(
            'ELF 32-bit LSB executable, Atmel AVR 8-bit, version 1 (SYSV), statically linked, with debug_info, not stripped',
        );
        expect(info?.instructionSet).toEqual('avr');
        expect(info?.os).toEqual('linux');
    });
});
