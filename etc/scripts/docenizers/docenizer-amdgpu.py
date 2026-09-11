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

--verify re-derives every field from the XML and compares it to the committed .ts, so a
divergence means the file no longer matches the spec it claims to come from. Generation
refuses to write a file containing a number it could not read.

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

# The architectures we ship docs for, each from the matching amdgpu_isa_<arch>.xml.
ARCHITECTURES = (
    'amd_cdna1',
    'amd_cdna2',
    'amd_cdna3',
    'amd_cdna4',
    'amd_cdna5',
    'amd_rdna1',
    'amd_rdna2',
    'amd_rdna3',
    'amd_rdna3_5',
    'amd_rdna4',
)

# Per-architecture overrides for the "More information" link in a tooltip. AMD publishes a
# separate ISA reference PDF per architecture, but those URLs are versioned and go stale as
# revisions land, so every architecture currently points at the index page listing them all.
# Add an entry here to deep-link one architecture.
ARCH_DOC_URLS: dict[str, str] = {}

# Display names for functional groups and subgroups, following FunctionalGroupNames and
# FunctionalSubgroupNames in include/amdisa/isa_decoder.h of
# https://github.com/GPUOpen-Tools/isa_spec_manager, so tooltips read the way RGA labels them.
# MFMA and WMMA are the exception: the header abbreviates them, and we spell them out.
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
parser.add_argument('-a', '--arch', type=str, required=True, choices=ARCHITECTURES,
                    help='Architecture name (e.g., amd_rdna3, amd_rdna3_5, amd_rdna4)')
parser.add_argument('--verify', action='store_true',
                    help='Check the existing output against the XML instead of regenerating it. '
                         'The XML is not committed, so this is how regeneration is proved faithful.')

# Values that would not parse as integers, recorded rather than silently becoming 0. Keyed by
# (element name, raw text) with an occurrence count, so a systematic problem reports once.
_INT_PROBLEMS: dict[tuple[str, str], int] = {}


def _text(elem, child_name: str) -> str:
    """Text of a child element, or empty string when absent."""
    if elem is None:
        return ''
    child = elem.find(child_name)
    return child.text.strip() if child is not None and child.text else ''


def _parse_int(value: str) -> int | None:
    """Decimal first, then 0x/0o/0b-prefixed. None when neither applies.

    Decimal has to win: base 0 would read a zero-padded '010' as octal, and bare hex without
    a prefix is genuinely ambiguous, so it is left to fail rather than guessed at.
    """
    try:
        return int(value, 10)
    except ValueError:
        pass
    try:
        return int(value, 0)
    except ValueError:
        return None


def _int(elem, child_name: str) -> int:
    """Integer text of a child element, or 0 when the element is absent or empty.

    The XML omits these elements rather than writing zero, so absent really is 0. Anything
    else that will not parse is a schema change we must not paper over: an Opcode silently
    becoming 0 collides with the real opcode 0 and quietly mislabels a tooltip. Those are
    recorded and reported before anything is written.
    """
    value = _text(elem, child_name)
    if not value:
        return 0
    parsed = _parse_int(value)
    if parsed is None:
        key = (child_name, value)
        _INT_PROBLEMS[key] = _INT_PROBLEMS.get(key, 0) + 1
        return 0
    return parsed


def int_problems() -> list:
    """One line per distinct value that would not parse, with how often it occurred."""
    return [
        f'{child_name}: cannot parse {value!r} as an integer, treated as 0 ({count} occurrence(s))'
        for (child_name, value), count in sorted(_INT_PROBLEMS.items())
    ]


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


# Deriving a spec field from the XML is written once and used by both parse_instructions and
# verify, so the two cannot drift into disagreeing about what the XML says.


def operand_row(op) -> list:
    """One <Operand> as the pooled row the renderer expects."""
    flags = 0
    if op.get('Input', '').upper() == 'TRUE':
        flags |= OPERAND_FLAG_INPUT
    if op.get('Output', '').upper() == 'TRUE':
        flags |= OPERAND_FLAG_OUTPUT
    if op.get('IsImplicit', '').upper() == 'TRUE':
        flags |= OPERAND_FLAG_IMPLICIT
    if op.get('IsBinaryMicrocodeRequired', '').upper() == 'TRUE':
        flags |= OPERAND_FLAG_BMR
    return [
        _text(op, 'FieldName'),
        _text(op, 'OperandType'),
        _text(op, 'DataFormatName'),
        _int(op, 'OperandSize'),
        flags,
    ]


def encoding_condition(enc) -> tuple:
    """The <EncodingCondition> text and Id, both empty when the element is absent."""
    cond = enc.find('EncodingCondition')
    if cond is None:
        return '', ''
    return (cond.text.strip() if cond.text else ''), cond.get('Id', '')


def instruction_flags(instr) -> int:
    """<InstructionFlags> packed into the bitfield mirrored by INSTRUCTION_FLAG_LABELS."""
    flags_elem = instr.find('InstructionFlags')
    flags = 0
    for bit, flag_name in enumerate(INSTRUCTION_FLAG_NAMES):
        if _is_true(flags_elem, flag_name):
            flags |= 1 << bit
    return flags


def instruction_subgroups(instr) -> list:
    """Display names of an instruction's functional subgroups."""
    subgroups = []
    for sub in instr.findall('./FunctionalGroup/FunctionalSubgroups/Subgroup'):
        if sub.text:
            text = sub.text.strip()
            # NOT_ASSIGNED means the instruction has no subgroup. Drop it rather than
            # surfacing the XML's jargon; the renderer omits the field when the list is empty.
            if text != 'NOT_ASSIGNED':
                subgroups.append(FUNCTIONAL_SUBGROUP_NAMES.get(text, text))
    return subgroups


def instruction_aliases(instr) -> list:
    """Legacy GCN spellings this instruction also answers to."""
    return [a.text.strip() for a in instr.findall('./AliasedInstructionNames/InstructionName') if a.text]


def instruction_group_name(instr) -> str:
    """Raw <FunctionalGroup><Name>, which indexes into the groups section."""
    group = instr.find('FunctionalGroup')
    return _text(group, 'Name') if group is not None else ''


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
            # Operands are emitted in <Order>, so position in this list is the order.
            operand_indices = [
                operand_pool.intern(tuple(row), row)
                for row in (operand_row(op) for op in enc.findall('./Operands/Operand'))
            ]

            cond_text, cond_id = encoding_condition(enc)
            shape = [_text(enc, 'EncodingName'), cond_text, cond_id, operand_indices]
            shape_index = shape_pool.intern((shape[0], shape[1], shape[2], tuple(operand_indices)), shape)
            encodings.append([shape_index, _int(enc, 'Opcode')])

        instructions.append([
            name,
            _text(instr, 'Description'),
            instruction_aliases(instr),
            instruction_group_name(instr),
            instruction_subgroups(instr),
            instruction_flags(instr),
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


def _brief(value) -> str:
    """A value short enough to sit on one line of a failure report."""
    text = json.dumps(value, separators=(',', ':'))
    return text if len(text) <= 120 else f'{text[:117]}...'


def _compare_section(problems: list, section: str, expected: dict, actual, expected_from: str = 'xml') -> None:
    """Key-by-key dict comparison, naming the entry rather than dumping both sides."""
    if not isinstance(actual, dict):
        problems.append(f'{section}: missing or malformed in generated spec')
        return
    for key in sorted(expected.keys() - actual.keys()):
        problems.append(f'{section}/{key}: in {expected_from} but not in generated spec')
    for key in sorted(actual.keys() - expected.keys()):
        problems.append(f'{section}/{key}: in generated spec but not in {expected_from}')
    for key in sorted(expected.keys() & actual.keys()):
        if expected[key] != actual[key]:
            problems.append(f'{section}/{key}: {_brief(expected[key])} != {_brief(actual[key])}')


def verify(spec: dict, root, arch: str) -> list:
    """Compare every field of a generated spec against the XML it came from.

    Covers the documentation url, the pooled sections (encodings, data formats, operand types,
    functional groups), the two derived maps (suffix encodings and the lookup index), and every
    field of every instruction row down to the pooled operands each encoding references.

    A bad operand or shape index still produces well-formed output, just describing the wrong
    instruction, so every divergence is collected rather than failing on the first.
    """
    problems = []

    expected_url = ARCH_DOC_URLS.get(arch, ISA_DOCS_BASE_URL)
    if spec.get('url') != expected_url:
        problems.append(f'url: {expected_url!r} != {spec.get("url")!r}')

    xml_encodings = parse_encodings(root)
    _compare_section(problems, 'encodings', xml_encodings, spec.get('encodings'))
    _compare_section(problems, 'suffixEncodings', build_suffix_encodings(xml_encodings), spec.get('suffixEncodings'))
    _compare_section(problems, 'formats', parse_data_formats(root), spec.get('formats'))
    _compare_section(problems, 'operandTypes', parse_operand_types(root), spec.get('operandTypes'))
    _compare_section(problems, 'groups', parse_functional_groups(root), spec.get('groups'))

    # The index is derived from the instruction list rather than read from the XML, so checking it
    # against a rebuild catches a stale or hand-edited index no instruction check would notice.
    _compare_section(
        problems, 'index', build_index(spec['instructions']), spec.get('index'), 'the instruction list'
    )

    xml_instructions = [i for i in root.findall('.//Instructions/Instruction') if _text(i, 'InstructionName')]
    if len(xml_instructions) != len(spec['instructions']):
        problems.append(f"instruction count: xml {len(xml_instructions)} != spec {len(spec['instructions'])}")
        return problems

    for xml_instr, row in zip(xml_instructions, spec['instructions']):
        name = _text(xml_instr, 'InstructionName')
        if len(row) != 7:
            problems.append(f'{name}: instruction row has {len(row)} fields, expected 7')
            continue
        spec_name, description, aliases, group_name, subgroups, flags, encodings = row

        if name != spec_name:
            problems.append(f'name: xml {name} != spec {spec_name}')
            continue
        if _text(xml_instr, 'Description') != description:
            problems.append(f'{name}: description differs')
        if instruction_aliases(xml_instr) != aliases:
            problems.append(f'{name}: aliases {_brief(instruction_aliases(xml_instr))} != {_brief(aliases)}')
        if instruction_group_name(xml_instr) != group_name:
            problems.append(f'{name}: functional group {instruction_group_name(xml_instr)!r} != {group_name!r}')
        if instruction_subgroups(xml_instr) != subgroups:
            problems.append(f'{name}: subgroups {_brief(instruction_subgroups(xml_instr))} != {_brief(subgroups)}')
        if instruction_flags(xml_instr) != flags:
            problems.append(f'{name}: flags {instruction_flags(xml_instr)} != {flags}')

        xml_instr_encodings = xml_instr.findall('./InstructionEncodings/InstructionEncoding')
        if len(xml_instr_encodings) != len(encodings):
            problems.append(f'{name}: encoding count {len(xml_instr_encodings)} != {len(encodings)}')
            continue

        for xml_enc, (shape_index, opcode) in zip(xml_instr_encodings, encodings):
            if not 0 <= shape_index < len(spec['shapes']):
                problems.append(f'{name}: shape index {shape_index} out of range')
                continue
            shape = spec['shapes'][shape_index]
            if shape[0] != _text(xml_enc, 'EncodingName'):
                problems.append(f"{name}: encoding name {_text(xml_enc, 'EncodingName')} != {shape[0]}")
            if _int(xml_enc, 'Opcode') != opcode:
                problems.append(f"{name}/{shape[0]}: opcode {_int(xml_enc, 'Opcode')} != {opcode}")

            cond_text, cond_id = encoding_condition(xml_enc)
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
                expected = operand_row(xml_op)
                if expected != spec['operands'][operand_index]:
                    problems.append(
                        f'{name}/{shape[0]}: operand {_brief(expected)} != {_brief(spec["operands"][operand_index])}'
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
        problems = verify(read_generated_spec(args.output), root, args.arch) + int_problems()
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

    # Nothing gets written if a number would not parse: a zero standing in for an unread Opcode
    # is indistinguishable from a real opcode 0 once it is in the committed file.
    problems = int_problems()
    if problems:
        print(f'Error: {input_path.name} has values this script cannot read', file=sys.stderr)
        for problem in problems:
            print(f'    {problem}', file=sys.stderr)
        sys.exit(1)

    write_ts_file(spec, args.output)

    alias_keys = len(spec['index']) - len(instructions)
    print(
        f'{len(instructions)} instructions (+{alias_keys} alias keys), '
        f'{len(operand_pool.items)} pooled operands, {len(shape_pool.items)} pooled encodings '
        f'-> {Path(args.output).stat().st_size / 1024:.0f} KB'
    )


if __name__ == '__main__':
    main()
