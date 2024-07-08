import fs from 'fs';
import {fileURLToPath} from 'url';

import {assert} from '../../lib/assert';
import {ElfReader, SecHeader, SH_TYPE} from '../../lib/tooling/readers/elf-reader';

describe('test ElfReader', () => {
    const reader = new ElfReader();
    before(() => {
        const file = fileURLToPath(new URL('..\\tasking\\cpp_demo.cpp.o', import.meta.url));
        reader.readElf(fs.readFileSync(file));
    });

    it('sections', () => {
        const file1 = fileURLToPath(new URL('..\\tasking\\section-name', import.meta.url));
        const secs = fs.readFileSync(file1).toString().split('\n');
        const sec_names: string[] = [];
        for (const sec of secs) {
            const name = sec.substring('section name: '.length);
            sec_names.push(name);
        }
        let i = 0;
        const elfgroups = reader.getGroups();
        for (const group of elfgroups) {
            const sections = reader.getSecsOf(group, (a: number) => {
                return true;
            });
            for (const section of sections) {
                reader.readSecName(section).should.equal(sec_names[i]);
                i++;
            }
        }
    });
    it('section .debug_line', () => {
        //read file
        const file1 = fileURLToPath(new URL('..\\tasking\\debug-line', import.meta.url));
        const secs = fs.readFileSync(file1).toString().split('\n');
        const sec_info: {offs: bigint; size: number}[] = [];
        for (const sec of secs) {
            sec_info.push({
                offs: BigInt(sec.split(' ')[0]),
                size: Number(sec.split(' ')[1]),
            });
        }

        const groups = reader.getGroups();
        groups.length.should.equal(sec_info.length);
        for (const [i, element] of sec_info.entries()) {
            const dbg_secs = reader.getDbgLineSecsOf(groups[i]);
            dbg_secs.length.should.equal(1);
            reader.readSecName(dbg_secs[0]).startsWith('.debug_line').should.equal(true);
            dbg_secs[0].sh_offset.should.equal(element.offs);
            dbg_secs[0].sh_size.should.equal(element.size);
        }
    });
});
