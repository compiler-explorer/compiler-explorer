"""Docenizer for AMD GPU ISA XML specifications.

Parses AMD's machine-readable ISA XML files (from https://gpuopen.com/download/machine-readable-isa/latest/)
and generates TypeScript asm-docs files for Compiler Explorer.

Run by hand, not from the Makefile: the specs are downloaded rather than fetched, and the
generated .ts files are committed, so this is only needed when AMD publishes new specs.
Regenerate the affected architectures, verify, and commit the result.

    python docenizer-amdgpu.py -i amdgpu_isa_rdna4.xml \\
        -o ../../../lib/asm-docs/generated/asm-docs-amd_rdna4.ts -a amd_rdna4
    python docenizer-amdgpu.py -i amdgpu_isa_rdna4.xml \\
        -o ../../../lib/asm-docs/generated/asm-docs-amd_rdna4.ts -a amd_rdna4 --verify

Architectures: amd_cdna1..5, amd_rdna1, amd_rdna2, amd_rdna3, amd_rdna3_5, amd_rdna4,
each from the matching amdgpu_isa_<arch>.xml.

Unlike the other docenizers this emits data rather than a switch of pre-rendered HTML,
because the same encoding, format and operand descriptions recur across thousands of
instructions. lib/asm-docs/amdgpu-render.ts turns the pooled data back into markup.

The XML schema is documented at:
    https://github.com/GPUOpen-Tools/isa_spec_manager/blob/main/documentation/spec_documentation.md
"""

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ISA_DOCS_BASE_URL = 'https://gpuopen.com/amd-gpu-architecture-programming-documentation/'

# Map architecture names to their ISA reference guide URLs.
ARCH_DOC_URLS = {
    'amd_cdna1': ISA_DOCS_BASE_URL,
    'amd_cdna2': ISA_DOCS_BASE_URL,
    'amd_cdna3': ISA_DOCS_BASE_URL,
    'amd_cdna4': ISA_DOCS_BASE_URL,
    'amd_cdna5': ISA_DOCS_BASE_URL,
    'amd_rdna1': ISA_DOCS_BASE_URL,
    'amd_rdna2': ISA_DOCS_BASE_URL,
    'amd_rdna3': ISA_DOCS_BASE_URL,
    'amd_rdna3_5': ISA_DOCS_BASE_URL,
    'amd_rdna4': ISA_DOCS_BASE_URL,
}

# Display names for functional groups and subgroups, matching FunctionalGroupNames and
# FunctionalSubgroupNames in AMD's isa_decoder.h so tooltips read the way RGA labels them.
FUNCTIONAL_GROUP_NAMES = {
    'SALU': 'Scalar ALU',
    'SMEM': 'Scalar Memory',
    'VALU': 'Vector ALU',
    'VMEM': 'Vector Memory',
    'EXPORT': 'Export',
    'BRANCH': 'Branch',
    'MESSAGE': 'Message',
    'WAVE_CONTROL': 'Wave Control',
    'TRAP': 'Trap',
}

FUNCTIONAL_SUBGROUP_NAMES = {
    'FLOATING_POINT': 'Floating Point',
    'BUFFER': 'Buffer',
    'TEXTURE': 'Texture',
    'LOAD': 'Load',
    'STORE': 'Store',
    'SAMPLE': 'Sample',
    'BVH': 'BVH',
    'ATOMIC': 'Atomic',
    'FLAT': 'Flat',
    'DATA_SHARE': 'Data Share',
    'STATIC': 'Static',
    'MFMA': 'Matrix-Fused Multiply-Add',
    'WMMA': 'Wave Matrix Multiply Accumulate',
    'TRANSCENDENTAL': 'Transcendental',
}

# Bit positions in the packed operand flags, mirrored by OperandFlag in amdgpu-render.ts.
OPERAND_FLAG_INPUT = 1
OPERAND_FLAG_OUTPUT = 2
OPERAND_FLAG_IMPLICIT = 4
OPERAND_FLAG_BMR = 8

# Bit positions in the packed instruction flags, mirrored by INSTRUCTION_FLAG_LABELS.
INSTRUCTION_FLAG_NAMES = [
    'IsBranch',
    'IsConditionalBranch',
    'IsIndirectBranch',
    'IsProgramTerminator',
    'IsImmediatelyExecuted',
]

parser = argparse.ArgumentParser(
    description='Generate Compiler Explorer asm-docs from AMD GPU ISA XML specifications'
)
parser.add_argument('-i', '--input', type=str, required=True,
                    help='Path to the AMD ISA XML specification file')
parser.add_argument('-o', '--output', type=str, required=True,
                    help='Output path for the generated .ts file')
parser.add_argument('-a', '--arch', type=str, required=True,
                    help='Architecture name (e.g., amd_rdna3, amd_rdna3_5, amd_rdna4)')
parser.add_argument('--verify', action='store_true',
                    help='Check the existing output against the XML instead of regenerating it. '
                         'The XML is not committed, so this is how regeneration is proved faithful.')


def _text(elem, child_name: str) -> str:
    """Text of a child element, or empty string when absent."""
    if elem is None:
        return ''
    child = elem.find(child_name)
    return child.text.strip() if child is not None and child.text else ''


def _int(elem, child_name: str) -> int:
    value = _text(elem, child_name)
    return int(value) if value.lstrip('-').isdigit() else 0


def _is_true(elem, child_name: str) -> bool:
    return _text(elem, child_name).upper() == 'TRUE'


def parse_encodings(root) -> dict:
    """<Encodings>: description, width and bitfield layout, keyed by encoding name."""
    encodings = {}
    section = root.find('.//Encodings')
    if section is None:
        return encodings
    for enc in section.findall('Encoding'):
        fields = []
        for fld in enc.findall('./MicrocodeFormat/BitMap/Field'):
            bit_range = fld.find('./BitLayout/Range')
            if bit_range is None:
                continue
            fields.append([
                _text(fld, 'FieldName'),
                _text(fld, 'Description'),
                _int(bit_range, 'BitOffset'),
                _int(bit_range, 'BitCount'),
            ])
        encodings[_text(enc, 'EncodingName')] = [_text(enc, 'Description'), _int(enc, 'BitCount'), fields]
    return encodings


def build_suffix_encodings(encodings: dict) -> dict:
    """Map the encoding suffix LLVM appends to a mnemonic to the encodings it selects.

    llvm-objdump writes v_add_f32_e32 for the 32-bit VOP form and _e64 for the 64-bit VOP3
    form, so the suffix identifies which of an instruction's encodings is the live one.
    """
    suffixes = {'e32': [], 'e64': [], 'dpp8': [], 'dpp16': [], 'sdwa': []}
    for name, (_description, bit_count, _fields) in encodings.items():
        if name.endswith('VOP_DPP8'):
            suffixes['dpp8'].append(name)
        elif name.endswith('VOP_DPP16'):
            suffixes['dpp16'].append(name)
        elif 'SDWA' in name:
            suffixes['sdwa'].append(name)
        elif name in ('ENC_VOP1', 'ENC_VOP2', 'ENC_VOPC') and bit_count == 32:
            suffixes['e32'].append(name)
        elif name in ('ENC_VOP3', 'ENC_VOP3P', 'VOP3_SDST_ENC'):
            suffixes['e64'].append(name)
    return {suffix: names for suffix, names in suffixes.items() if names}


def parse_data_formats(root) -> dict:
    """<DataFormats>: what the FMT_* identifiers on operands actually mean."""
    section = root.find('.//DataFormats')
    if section is None:
        return {}
    return {
        _text(fmt, 'DataFormatName'): [_text(fmt, 'Description'), _text(fmt, 'DataType')]
        for fmt in section.findall('DataFormat')
    }


def parse_operand_types(root) -> dict:
    """<OperandTypes>: what the OPR_* identifiers on operands actually mean."""
    section = root.find('.//OperandTypes')
    if section is None:
        return {}
    return {
        _text(op, 'OperandTypeName'): _text(op, 'Description')
        for op in section.findall('OperandType')
    }


def parse_functional_groups(root) -> dict:
    """<FunctionalGroups>: display name plus the prose describing each group."""
    section = root.find('.//FunctionalGroups')
    if section is None:
        return {}
    groups = {}
    for grp in section.findall('FunctionalGroup'):
        name = _text(grp, 'Name')
        groups[name] = [FUNCTIONAL_GROUP_NAMES.get(name, name), _text(grp, 'Description')]
    return groups


class Pool:
    """Interns values by equality so repeated rows are stored once and referenced by index."""

    def __init__(self):
        self.items = []
        self._index = {}

    def intern(self, key, value) -> int:
        if key not in self._index:
            self._index[key] = len(self.items)
            self.items.append(value)
        return self._index[key]


def parse_instructions(root, operand_pool: Pool, shape_pool: Pool) -> list:
    """<Instructions>: one record per instruction, referencing pooled operands and shapes."""
    section = root.find('.//Instructions')
    if section is None:
        print('Error: no <Instructions> element found', file=sys.stderr)
        sys.exit(1)

    instructions = []
    for instr in section.findall('Instruction'):
        name = _text(instr, 'InstructionName')
        if not name:
            continue

        encodings = []
        for enc in instr.findall('./InstructionEncodings/InstructionEncoding'):
            operand_indices = []
            for op in enc.findall('./Operands/Operand'):
                flags = 0
                if op.get('Input', '').upper() == 'TRUE':
                    flags |= OPERAND_FLAG_INPUT
                if op.get('Output', '').upper() == 'TRUE':
                    flags |= OPERAND_FLAG_OUTPUT
                if op.get('IsImplicit', '').upper() == 'TRUE':
                    flags |= OPERAND_FLAG_IMPLICIT
                if op.get('IsBinaryMicrocodeRequired', '').upper() == 'TRUE':
                    flags |= OPERAND_FLAG_BMR
                row = [
                    _text(op, 'FieldName'),
                    _text(op, 'OperandType'),
                    _text(op, 'DataFormatName'),
                    _int(op, 'OperandSize'),
                    flags,
                ]
                # Operands are emitted in <Order>, so position in this list is the order.
                operand_indices.append(operand_pool.intern(tuple(row), row))

            cond_elem = enc.find('EncodingCondition')
            shape = [
                _text(enc, 'EncodingName'),
                cond_elem.text.strip() if cond_elem is not None and cond_elem.text else '',
                cond_elem.get('Id', '') if cond_elem is not None else '',
                operand_indices,
            ]
            shape_index = shape_pool.intern((shape[0], shape[1], shape[2], tuple(operand_indices)), shape)
            encodings.append([shape_index, _int(enc, 'Opcode')])

        flags = 0
        flags_elem = instr.find('InstructionFlags')
        for bit, flag_name in enumerate(INSTRUCTION_FLAG_NAMES):
            if _is_true(flags_elem, flag_name):
                flags |= 1 << bit

        subgroups = []
        for sub in instr.findall('./FunctionalGroup/FunctionalSubgroups/Subgroup'):
            if sub.text:
                text = sub.text.strip()
                # NOT_ASSIGNED means the instruction has no subgroup; the renderer shows
                # N/A for an empty list rather than surfacing the XML's jargon.
                if text != 'NOT_ASSIGNED':
                    subgroups.append(FUNCTIONAL_SUBGROUP_NAMES.get(text, text))

        aliases = [a.text.strip() for a in instr.findall('./AliasedInstructionNames/InstructionName') if a.text]

        instructions.append([
            name,
            _text(instr, 'Description'),
            aliases,
            _text(instr.find('FunctionalGroup'), 'Name') if instr.find('FunctionalGroup') is not None else '',
            subgroups,
            flags,
            encodings,
        ])

    return instructions


def build_index(instructions: list) -> dict:
    """Lowercased names and aliases mapped to their instruction index.

    Aliases are the legacy GCN spellings AMD disassembly still emits, e.g. S_LOAD_DWORD
    for S_LOAD_B32.
    """
    index = {}
    for position, instr in enumerate(instructions):
        name, _, aliases = instr[0], instr[1], instr[2]
        index.setdefault(name.lower(), position)
        for alias in aliases:
            index.setdefault(alias.lower(), position)
    return index


def format_spec(spec: dict) -> str:
    """Serialise the spec one record per line.

    Fully minified costs a single 570KB line, which no editor enjoys and which makes every
    regeneration a whole-file diff. Breaking at record boundaries costs under 1% and lets
    git show which instructions actually changed.
    """
    compact = lambda value: json.dumps(value, separators=(',', ':'))
    parts = []
    for key, value in spec.items():
        if isinstance(value, list):
            rows = ',\n'.join(compact(row) for row in value)
            parts.append(f'{compact(key)}:[\n{rows}\n]')
        elif isinstance(value, dict):
            rows = ',\n'.join(f'{compact(k)}:{compact(v)}' for k, v in value.items())
            parts.append(f'{compact(key)}:{{\n{rows}\n}}')
        else:
            parts.append(f'{compact(key)}:{compact(value)}')
    return '{\n' + ',\n'.join(parts) + '\n}'


def write_ts_file(spec: dict, output_path: str):
    payload = format_spec(spec)
    with open(output_path, 'w', newline='\n', encoding='utf8') as f:
        f.write(
            '// Generated by etc/scripts/docenizers/docenizer-amdgpu.py -- do not edit.\n'
            "import {type AmdIsaSpec, buildAsmDocs} from '../amdgpu-render.js';\n"
            '\n'
            f'export const SPEC = {payload} as unknown as AmdIsaSpec;\n'
            '\n'
            'export function getAsmOpcode(opcode: string | undefined) {\n'
            '    return buildAsmDocs(SPEC, opcode);\n'
            '}\n'
        )


def read_generated_spec(path: str) -> dict:
    """Recover the SPEC literal from a generated .ts file. Only used by --verify."""
    src = Path(path).read_text(encoding='utf8')
    match = re.search(r'^export const SPEC = (\{.*\}) as unknown as AmdIsaSpec;$', src, re.M | re.S)
    if match is None:
        raise ValueError(f'{path} does not contain a SPEC literal')
    return json.loads(match.group(1))


def verify(spec: dict, root, output_path: str) -> list:
    """Compare a generated spec field-by-field against the XML it came from.

    A bad operand or shape index still produces well-formed output, just describing the
    wrong instruction, so every divergence is collected rather than failing on the first.
    """
    problems = []
    xml_instructions = [i for i in root.findall('.//Instructions/Instruction') if _text(i, 'InstructionName')]

    if len(xml_instructions) != len(spec['instructions']):
        problems.append(f"instruction count: xml {len(xml_instructions)} != spec {len(spec['instructions'])}")
        return problems

    for xml_instr, row in zip(xml_instructions, spec['instructions']):
        name = _text(xml_instr, 'InstructionName')
        if name != row[0]:
            problems.append(f'name: xml {name} != spec {row[0]}')
            continue
        if _text(xml_instr, 'Description') != row[1]:
            problems.append(f'{name}: description differs')

        xml_encodings = xml_instr.findall('./InstructionEncodings/InstructionEncoding')
        if len(xml_encodings) != len(row[6]):
            problems.append(f'{name}: encoding count {len(xml_encodings)} != {len(row[6])}')
            continue

        for xml_enc, (shape_index, opcode) in zip(xml_encodings, row[6]):
            if not 0 <= shape_index < len(spec['shapes']):
                problems.append(f'{name}: shape index {shape_index} out of range')
                continue
            shape = spec['shapes'][shape_index]
            if shape[0] != _text(xml_enc, 'EncodingName'):
                problems.append(f"{name}: encoding name {_text(xml_enc, 'EncodingName')} != {shape[0]}")
            if _int(xml_enc, 'Opcode') != opcode:
                problems.append(f"{name}/{shape[0]}: opcode {_int(xml_enc, 'Opcode')} != {opcode}")

            cond = xml_enc.find('EncodingCondition')
            cond_text = cond.text.strip() if cond is not None and cond.text else ''
            cond_id = cond.get('Id', '') if cond is not None else ''
            if cond_text != shape[1]:
                problems.append(f'{name}/{shape[0]}: condition {cond_text!r} != {shape[1]!r}')
            if cond_id != shape[2]:
                problems.append(f'{name}/{shape[0]}: condition id {cond_id!r} != {shape[2]!r}')

            xml_operands = xml_enc.findall('./Operands/Operand')
            if len(xml_operands) != len(shape[3]):
                problems.append(f'{name}/{shape[0]}: operand count {len(xml_operands)} != {len(shape[3])}')
                continue
            for xml_op, operand_index in zip(xml_operands, shape[3]):
                if not 0 <= operand_index < len(spec['operands']):
                    problems.append(f'{name}/{shape[0]}: operand index {operand_index} out of range')
                    continue
                flags = 0
                if xml_op.get('Input', '').upper() == 'TRUE':
                    flags |= OPERAND_FLAG_INPUT
                if xml_op.get('Output', '').upper() == 'TRUE':
                    flags |= OPERAND_FLAG_OUTPUT
                if xml_op.get('IsImplicit', '').upper() == 'TRUE':
                    flags |= OPERAND_FLAG_IMPLICIT
                if xml_op.get('IsBinaryMicrocodeRequired', '').upper() == 'TRUE':
                    flags |= OPERAND_FLAG_BMR
                expected = [
                    _text(xml_op, 'FieldName'),
                    _text(xml_op, 'OperandType'),
                    _text(xml_op, 'DataFormatName'),
                    _int(xml_op, 'OperandSize'),
                    flags,
                ]
                if expected != spec['operands'][operand_index]:
                    problems.append(
                        f'{name}/{shape[0]}: operand {expected} != {spec["operands"][operand_index]}'
                    )

    return problems


def main():
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f'Error: Input file not found: {input_path}', file=sys.stderr)
        sys.exit(1)

    root = ET.parse(str(input_path)).getroot()

    if args.verify:
        problems = verify(read_generated_spec(args.output), root, args.output)
        if problems:
            print(f'{args.arch}: FAIL - {len(problems)} divergence(s) from {input_path.name}', file=sys.stderr)
            for problem in problems[:10]:
                print(f'    {problem}', file=sys.stderr)
            if len(problems) > 10:
                print(f'    ... and {len(problems) - 10} more', file=sys.stderr)
            sys.exit(1)
        print(f'{args.arch}: OK - generated spec matches {input_path.name}')
        return

    operand_pool = Pool()
    shape_pool = Pool()
    instructions = parse_instructions(root, operand_pool, shape_pool)
    if not instructions:
        print(f'Error: No instructions parsed from {input_path}', file=sys.stderr)
        sys.exit(1)

    encodings = parse_encodings(root)
    spec = {
        'url': ARCH_DOC_URLS.get(args.arch, ISA_DOCS_BASE_URL),
        'encodings': encodings,
        'suffixEncodings': build_suffix_encodings(encodings),
        'formats': parse_data_formats(root),
        'operandTypes': parse_operand_types(root),
        'groups': parse_functional_groups(root),
        'operands': operand_pool.items,
        'shapes': shape_pool.items,
        'instructions': instructions,
        'index': build_index(instructions),
    }

    write_ts_file(spec, args.output)

    alias_keys = len(spec['index']) - len(instructions)
    print(
        f'{len(instructions)} instructions (+{alias_keys} alias keys), '
        f'{len(operand_pool.items)} pooled operands, {len(shape_pool.items)} pooled encodings '
        f'-> {Path(args.output).stat().st_size / 1024:.0f} KB'
    )


if __name__ == '__main__':
    main()
