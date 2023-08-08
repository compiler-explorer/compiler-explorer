import {AssemblyInstructionInfo} from '../base.js';

export function getAsmOpcode(opcode: string | undefined): AssemblyInstructionInfo | undefined {
    if (!opcode) return;
    switch (opcode.toUpperCase()) {
        case "STOP":
            return {
                "html": "Halts execution\nInput: -\nOutput: -",
                "tooltip": "Halts execution",
                "url": "https://www.evm.codes/#00"
            };

        case "ADD":
            return {
                "html": "Addition operation\nInput: <code>a | b</code>\nOutput: <code>a + b</code>",
                "tooltip": "Addition operation",
                "url": "https://www.evm.codes/#01"
            };

        case "MUL":
            return {
                "html": "Multiplication operation\nInput: <code>a | b</code>\nOutput: <code>a * b</code>",
                "tooltip": "Multiplication operation",
                "url": "https://www.evm.codes/#02"
            };

        case "SUB":
            return {
                "html": "Subtraction operation\nInput: <code>a | b</code>\nOutput: <code>a - b</code>",
                "tooltip": "Subtraction operation",
                "url": "https://www.evm.codes/#03"
            };

        case "DIV":
            return {
                "html": "Integer division operation\nInput: <code>a | b</code>\nOutput: <code>a // b</code>",
                "tooltip": "Integer division operation",
                "url": "https://www.evm.codes/#04"
            };

        case "SDIV":
            return {
                "html": "Signed integer division operation (truncated)\nInput: <code>a | b</code>\nOutput: <code>a // b</code>",
                "tooltip": "Signed integer division operation (truncated)",
                "url": "https://www.evm.codes/#05"
            };

        case "MOD":
            return {
                "html": "Modulo remainder operation\nInput: <code>a | b</code>\nOutput: <code>a % b</code>",
                "tooltip": "Modulo remainder operation",
                "url": "https://www.evm.codes/#06"
            };

        case "SMOD":
            return {
                "html": "Signed modulo remainder operation\nInput: <code>a | b</code>\nOutput: <code>a % b</code>",
                "tooltip": "Signed modulo remainder operation",
                "url": "https://www.evm.codes/#07"
            };

        case "ADDMOD":
            return {
                "html": "Modulo addition operation\nInput: <code>a | b | N</code>\nOutput: <code>(a + b) % N</code>",
                "tooltip": "Modulo addition operation",
                "url": "https://www.evm.codes/#08"
            };

        case "MULMOD":
            return {
                "html": "Modulo multiplication operation\nInput: <code>a | b | N</code>\nOutput: <code>(a * b) % N</code>",
                "tooltip": "Modulo multiplication operation",
                "url": "https://www.evm.codes/#09"
            };

        case "EXP":
            return {
                "html": "Exponential operation\nInput: <code>a | exponent</code>\nOutput: <code>a ** exponent</code>",
                "tooltip": "Exponential operation",
                "url": "https://www.evm.codes/#0a"
            };

        case "SIGNEXTEND":
            return {
                "html": "Extend length of two\u2019s complement signed integer\nInput: <code>b | x</code>\nOutput: <code>y</code>",
                "tooltip": "Extend length of two\u2019s complement signed integer",
                "url": "https://www.evm.codes/#0b"
            };

        case "LT":
            return {
                "html": "Less-than comparison\nInput: <code>a | b</code>\nOutput: <code>a < b</code>",
                "tooltip": "Less-than comparison",
                "url": "https://www.evm.codes/#10"
            };

        case "GT":
            return {
                "html": "Greater-than comparison\nInput: <code>a | b</code>\nOutput: <code>a > b</code>",
                "tooltip": "Greater-than comparison",
                "url": "https://www.evm.codes/#11"
            };

        case "SLT":
            return {
                "html": "Signed less-than comparison\nInput: <code>a | b</code>\nOutput: <code>a < b</code>",
                "tooltip": "Signed less-than comparison",
                "url": "https://www.evm.codes/#12"
            };

        case "SGT":
            return {
                "html": "Signed greater-than comparison\nInput: <code>a | b</code>\nOutput: <code>a > b</code>",
                "tooltip": "Signed greater-than comparison",
                "url": "https://www.evm.codes/#13"
            };

        case "EQ":
            return {
                "html": "Equality comparison\nInput: <code>a | b</code>\nOutput: <code>a == b</code>",
                "tooltip": "Equality comparison",
                "url": "https://www.evm.codes/#14"
            };

        case "ISZERO":
            return {
                "html": "Simple not operator\nInput: <code>a</code>\nOutput: <code>a == 0</code>",
                "tooltip": "Simple not operator",
                "url": "https://www.evm.codes/#15"
            };

        case "AND":
            return {
                "html": "Bitwise AND operation\nInput: <code>a | b</code>\nOutput: <code>a & b</code>",
                "tooltip": "Bitwise AND operation",
                "url": "https://www.evm.codes/#16"
            };

        case "OR":
            return {
                "html": "Bitwise OR operation\nInput: <code>a | b</code>\nOutput: <code>a \\| b</code>",
                "tooltip": "Bitwise OR operation",
                "url": "https://www.evm.codes/#17"
            };

        case "XOR":
            return {
                "html": "Bitwise XOR operation\nInput: <code>a | b</code>\nOutput: <code>a ^ b</code>",
                "tooltip": "Bitwise XOR operation",
                "url": "https://www.evm.codes/#18"
            };

        case "NOT":
            return {
                "html": "Bitwise NOT operation\nInput: <code>a</code>\nOutput: <code>~a</code>",
                "tooltip": "Bitwise NOT operation",
                "url": "https://www.evm.codes/#19"
            };

        case "BYTE":
            return {
                "html": "Retrieve single byte from word\nInput: <code>i | x</code>\nOutput: <code>y</code>",
                "tooltip": "Retrieve single byte from word",
                "url": "https://www.evm.codes/#1a"
            };

        case "SHL":
            return {
                "html": "Left shift operation\nInput: <code>shift | value</code>\nOutput: <code>value << shift</code>",
                "tooltip": "Left shift operation",
                "url": "https://www.evm.codes/#1b"
            };

        case "SHR":
            return {
                "html": "Logical right shift operation\nInput: <code>shift | value</code>\nOutput: <code>value >> shift</code>",
                "tooltip": "Logical right shift operation",
                "url": "https://www.evm.codes/#1c"
            };

        case "SAR":
            return {
                "html": "Arithmetic (signed) right shift operation\nInput: <code>shift | value</code>\nOutput: <code>value >> shift</code>",
                "tooltip": "Arithmetic (signed) right shift operation",
                "url": "https://www.evm.codes/#1d"
            };

        case "KECCAK256":
            return {
                "html": "Compute Keccak-256 hash\nInput: <code>offset | size</code>\nOutput: <code>hash</code>",
                "tooltip": "Compute Keccak-256 hash",
                "url": "https://www.evm.codes/#20"
            };

        case "ADDRESS":
            return {
                "html": "Get address of currently executing account\nInput: -\nOutput: <code>address</code>",
                "tooltip": "Get address of currently executing account",
                "url": "https://www.evm.codes/#30"
            };

        case "BALANCE":
            return {
                "html": "Get balance of the given account\nInput: <code>address</code>\nOutput: <code>balance</code>",
                "tooltip": "Get balance of the given account",
                "url": "https://www.evm.codes/#31"
            };

        case "ORIGIN":
            return {
                "html": "Get execution origination address\nInput: -\nOutput: <code>address</code>",
                "tooltip": "Get execution origination address",
                "url": "https://www.evm.codes/#32"
            };

        case "CALLER":
            return {
                "html": "Get caller address\nInput: -\nOutput: <code>address</code>",
                "tooltip": "Get caller address",
                "url": "https://www.evm.codes/#33"
            };

        case "CALLVALUE":
            return {
                "html": "Get deposited value by the instruction/transaction responsible for this execution\nInput: -\nOutput: <code>value</code>",
                "tooltip": "Get deposited value by the instruction/transaction responsible for this execution",
                "url": "https://www.evm.codes/#34"
            };

        case "CALLDATALOAD":
            return {
                "html": "Get input data of current environment\nInput: <code>i</code>\nOutput: <code>data[i]</code>",
                "tooltip": "Get input data of current environment",
                "url": "https://www.evm.codes/#35"
            };

        case "CALLDATASIZE":
            return {
                "html": "Get size of input data in current environment\nInput: -\nOutput: <code>size</code>",
                "tooltip": "Get size of input data in current environment",
                "url": "https://www.evm.codes/#36"
            };

        case "CALLDATACOPY":
            return {
                "html": "Copy input data in current environment to memory\nInput: <code>destOffset | offset | size</code>\nOutput: -",
                "tooltip": "Copy input data in current environment to memory",
                "url": "https://www.evm.codes/#37"
            };

        case "CODESIZE":
            return {
                "html": "Get size of code running in current environment\nInput: -\nOutput: <code>size</code>",
                "tooltip": "Get size of code running in current environment",
                "url": "https://www.evm.codes/#38"
            };

        case "CODECOPY":
            return {
                "html": "Copy code running in current environment to memory\nInput: <code>destOffset | offset | size</code>\nOutput: -",
                "tooltip": "Copy code running in current environment to memory",
                "url": "https://www.evm.codes/#39"
            };

        case "GASPRICE":
            return {
                "html": "Get price of gas in current environment\nInput: -\nOutput: <code>price</code>",
                "tooltip": "Get price of gas in current environment",
                "url": "https://www.evm.codes/#3a"
            };

        case "EXTCODESIZE":
            return {
                "html": "Get size of an account\u2019s code\nInput: <code>address</code>\nOutput: <code>size</code>",
                "tooltip": "Get size of an account\u2019s code",
                "url": "https://www.evm.codes/#3b"
            };

        case "EXTCODECOPY":
            return {
                "html": "Copy an account\u2019s code to memory\nInput: <code>address | destOffset | offset | size</code>\nOutput: -",
                "tooltip": "Copy an account\u2019s code to memory",
                "url": "https://www.evm.codes/#3c"
            };

        case "RETURNDATASIZE":
            return {
                "html": "Get size of output data from the previous call from the current environment\nInput: -\nOutput: <code>size</code>",
                "tooltip": "Get size of output data from the previous call from the current environment",
                "url": "https://www.evm.codes/#3d"
            };

        case "RETURNDATACOPY":
            return {
                "html": "Copy output data from the previous call to memory\nInput: <code>destOffset | offset | size</code>\nOutput: -",
                "tooltip": "Copy output data from the previous call to memory",
                "url": "https://www.evm.codes/#3e"
            };

        case "EXTCODEHASH":
            return {
                "html": "Get hash of an account\u2019s code\nInput: <code>address</code>\nOutput: <code>hash</code>",
                "tooltip": "Get hash of an account\u2019s code",
                "url": "https://www.evm.codes/#3f"
            };

        case "BLOCKHASH":
            return {
                "html": "Get the hash of one of the 256 most recent complete blocks\nInput: <code>blockNumber</code>\nOutput: <code>hash</code>",
                "tooltip": "Get the hash of one of the 256 most recent complete blocks",
                "url": "https://www.evm.codes/#40"
            };

        case "COINBASE":
            return {
                "html": "Get the block\u2019s beneficiary address\nInput: -\nOutput: <code>address</code>",
                "tooltip": "Get the block\u2019s beneficiary address",
                "url": "https://www.evm.codes/#41"
            };

        case "TIMESTAMP":
            return {
                "html": "Get the block\u2019s timestamp\nInput: -\nOutput: <code>timestamp</code>",
                "tooltip": "Get the block\u2019s timestamp",
                "url": "https://www.evm.codes/#42"
            };

        case "NUMBER":
            return {
                "html": "Get the block\u2019s number\nInput: -\nOutput: <code>blockNumber</code>",
                "tooltip": "Get the block\u2019s number",
                "url": "https://www.evm.codes/#43"
            };

        case "DIFFICULTY":
            return {
                "html": "Get the block\u2019s difficulty\nInput: -\nOutput: <code>difficulty</code>",
                "tooltip": "Get the block\u2019s difficulty",
                "url": "https://www.evm.codes/#44"
            };

        case "GASLIMIT":
            return {
                "html": "Get the block\u2019s gas limit\nInput: -\nOutput: <code>gasLimit</code>",
                "tooltip": "Get the block\u2019s gas limit",
                "url": "https://www.evm.codes/#45"
            };

        case "CHAINID":
            return {
                "html": "Get the chain ID\nInput: -\nOutput: <code>chainId</code>",
                "tooltip": "Get the chain ID",
                "url": "https://www.evm.codes/#46"
            };

        case "BASEFEE":
            return {
                "html": "Get the base fee\nInput: -\nOutput: <code>baseFee</code>",
                "tooltip": "Get the base fee",
                "url": "https://www.evm.codes/#48"
            };

        case "POP":
            return {
                "html": "Remove item from stack\nInput: <code>y</code>\nOutput: -",
                "tooltip": "Remove item from stack",
                "url": "https://www.evm.codes/#50"
            };

        case "MLOAD":
            return {
                "html": "Load word from memory\nInput: <code>offset</code>\nOutput: <code>value</code>",
                "tooltip": "Load word from memory",
                "url": "https://www.evm.codes/#51"
            };

        case "MSTORE":
            return {
                "html": "Save word to memory\nInput: <code>offset | value</code>\nOutput: -",
                "tooltip": "Save word to memory",
                "url": "https://www.evm.codes/#52"
            };

        case "MSTORE8":
            return {
                "html": "Save byte to memory\nInput: <code>offset | value</code>\nOutput: -",
                "tooltip": "Save byte to memory",
                "url": "https://www.evm.codes/#53"
            };

        case "SLOAD":
            return {
                "html": "Load word from storage\nInput: <code>key</code>\nOutput: <code>value</code>",
                "tooltip": "Load word from storage",
                "url": "https://www.evm.codes/#54"
            };

        case "SSTORE":
            return {
                "html": "Save word to storage\nInput: <code>key | value</code>\nOutput: -",
                "tooltip": "Save word to storage",
                "url": "https://www.evm.codes/#55"
            };

        case "JUMP":
            return {
                "html": "Alter the program counter\nInput: <code>counter</code>\nOutput: -",
                "tooltip": "Alter the program counter",
                "url": "https://www.evm.codes/#56"
            };

        case "JUMPI":
            return {
                "html": "Conditionally alter the program counter\nInput: <code>counter | b</code>\nOutput: -",
                "tooltip": "Conditionally alter the program counter",
                "url": "https://www.evm.codes/#57"
            };

        case "GETPC":
            return {
                "html": "Get the value of the program counter prior to the increment corresponding to this instruction\nInput: -\nOutput: <code>counter</code>",
                "tooltip": "Get the value of the program counter prior to the increment corresponding to this instruction",
                "url": "https://www.evm.codes/#58"
            };

        case "MSIZE":
            return {
                "html": "Get the size of active memory in bytes\nInput: -\nOutput: <code>size</code>",
                "tooltip": "Get the size of active memory in bytes",
                "url": "https://www.evm.codes/#59"
            };

        case "GAS":
            return {
                "html": "Get the amount of available gas, including the corresponding reduction for the cost of this instruction\nInput: -\nOutput: <code>gas</code>",
                "tooltip": "Get the amount of available gas, including the corresponding reduction for the cost of this instruction",
                "url": "https://www.evm.codes/#5a"
            };

        case "JUMPDEST":
            return {
                "html": "Mark a valid destination for jumps\nInput: -\nOutput: -",
                "tooltip": "Mark a valid destination for jumps",
                "url": "https://www.evm.codes/#5b"
            };

        case "PUSH1":
            return {
                "html": "Place 1 byte item on stack\nInput: -\nOutput: <code>value</code>",
                "tooltip": "Place 1 byte item on stack",
                "url": "https://www.evm.codes/#60"
            };

        case "PUSH2":
            return {
                "html": "Place 2 byte item on stack\nInput: -\nOutput: <code>value</code>",
                "tooltip": "Place 2 byte item on stack",
                "url": "https://www.evm.codes/#61"
            };

        case "PUSH3":
            return {
                "html": "Place 3 byte item on stack\nInput: -\nOutput: <code>value</code>",
                "tooltip": "Place 3 byte item on stack",
                "url": "https://www.evm.codes/#62"
            };

        case "PUSH4":
            return {
                "html": "Place 4 byte item on stack\nInput: -\nOutput: <code>value</code>",
                "tooltip": "Place 4 byte item on stack",
                "url": "https://www.evm.codes/#63"
            };

        case "PUSH5":
            return {
                "html": "Place 5 byte item on stack\nInput: -\nOutput: <code>value</code>",
                "tooltip": "Place 5 byte item on stack",
                "url": "https://www.evm.codes/#64"
            };

        case "PUSH6":
            return {
                "html": "Place 6 byte item on stack\nInput: -\nOutput: <code>value</code>",
                "tooltip": "Place 6 byte item on stack",
                "url": "https://www.evm.codes/#65"
            };

        case "PUSH7":
            return {
                "html": "Place 7 byte item on stack\nInput: -\nOutput: <code>value</code>",
                "tooltip": "Place 7 byte item on stack",
                "url": "https://www.evm.codes/#66"
            };

        case "PUSH8":
            return {
                "html": "Place 8 byte item on stack\nInput: -\nOutput: <code>value</code>",
                "tooltip": "Place 8 byte item on stack",
                "url": "https://www.evm.codes/#67"
            };

        case "PUSH9":
            return {
                "html": "Place 9 byte item on stack\nInput: -\nOutput: <code>value</code>",
                "tooltip": "Place 9 byte item on stack",
                "url": "https://www.evm.codes/#68"
            };

        case "PUSH10":
            return {
                "html": "Place 10 byte item on stack\nInput: -\nOutput: <code>value</code>",
                "tooltip": "Place 10 byte item on stack",
                "url": "https://www.evm.codes/#69"
            };

        case "PUSH11":
            return {
                "html": "Place 11 byte item on stack\nInput: -\nOutput: <code>value</code>",
                "tooltip": "Place 11 byte item on stack",
                "url": "https://www.evm.codes/#6a"
            };

        case "PUSH12":
            return {
                "html": "Place 12 byte item on stack\nInput: -\nOutput: <code>value</code>",
                "tooltip": "Place 12 byte item on stack",
                "url": "https://www.evm.codes/#6b"
            };

        case "PUSH13":
            return {
                "html": "Place 13 byte item on stack\nInput: -\nOutput: <code>value</code>",
                "tooltip": "Place 13 byte item on stack",
                "url": "https://www.evm.codes/#6c"
            };

        case "PUSH14":
            return {
                "html": "Place 14 byte item on stack\nInput: -\nOutput: <code>value</code>",
                "tooltip": "Place 14 byte item on stack",
                "url": "https://www.evm.codes/#6d"
            };

        case "PUSH15":
            return {
                "html": "Place 15 byte item on stack\nInput: -\nOutput: <code>value</code>",
                "tooltip": "Place 15 byte item on stack",
                "url": "https://www.evm.codes/#6e"
            };

        case "PUSH16":
            return {
                "html": "Place 16 byte item on stack\nInput: -\nOutput: <code>value</code>",
                "tooltip": "Place 16 byte item on stack",
                "url": "https://www.evm.codes/#6f"
            };

        case "PUSH17":
            return {
                "html": "Place 17 byte item on stack\nInput: -\nOutput: <code>value</code>",
                "tooltip": "Place 17 byte item on stack",
                "url": "https://www.evm.codes/#70"
            };

        case "PUSH18":
            return {
                "html": "Place 18 byte item on stack\nInput: -\nOutput: <code>value</code>",
                "tooltip": "Place 18 byte item on stack",
                "url": "https://www.evm.codes/#71"
            };

        case "PUSH19":
            return {
                "html": "Place 19 byte item on stack\nInput: -\nOutput: <code>value</code>",
                "tooltip": "Place 19 byte item on stack",
                "url": "https://www.evm.codes/#72"
            };

        case "PUSH20":
            return {
                "html": "Place 20 byte item on stack\nInput: -\nOutput: <code>value</code>",
                "tooltip": "Place 20 byte item on stack",
                "url": "https://www.evm.codes/#73"
            };

        case "PUSH21":
            return {
                "html": "Place 21 byte item on stack\nInput: -\nOutput: <code>value</code>",
                "tooltip": "Place 21 byte item on stack",
                "url": "https://www.evm.codes/#74"
            };

        case "PUSH22":
            return {
                "html": "Place 22 byte item on stack\nInput: -\nOutput: <code>value</code>",
                "tooltip": "Place 22 byte item on stack",
                "url": "https://www.evm.codes/#75"
            };

        case "PUSH23":
            return {
                "html": "Place 23 byte item on stack\nInput: -\nOutput: <code>value</code>",
                "tooltip": "Place 23 byte item on stack",
                "url": "https://www.evm.codes/#76"
            };

        case "PUSH24":
            return {
                "html": "Place 24 byte item on stack\nInput: -\nOutput: <code>value</code>",
                "tooltip": "Place 24 byte item on stack",
                "url": "https://www.evm.codes/#77"
            };

        case "PUSH25":
            return {
                "html": "Place 25 byte item on stack\nInput: -\nOutput: <code>value</code>",
                "tooltip": "Place 25 byte item on stack",
                "url": "https://www.evm.codes/#78"
            };

        case "PUSH26":
            return {
                "html": "Place 26 byte item on stack\nInput: -\nOutput: <code>value</code>",
                "tooltip": "Place 26 byte item on stack",
                "url": "https://www.evm.codes/#79"
            };

        case "PUSH27":
            return {
                "html": "Place 27 byte item on stack\nInput: -\nOutput: <code>value</code>",
                "tooltip": "Place 27 byte item on stack",
                "url": "https://www.evm.codes/#7a"
            };

        case "PUSH28":
            return {
                "html": "Place 28 byte item on stack\nInput: -\nOutput: <code>value</code>",
                "tooltip": "Place 28 byte item on stack",
                "url": "https://www.evm.codes/#7b"
            };

        case "PUSH29":
            return {
                "html": "Place 29 byte item on stack\nInput: -\nOutput: <code>value</code>",
                "tooltip": "Place 29 byte item on stack",
                "url": "https://www.evm.codes/#7c"
            };

        case "PUSH30":
            return {
                "html": "Place 30 byte item on stack\nInput: -\nOutput: <code>value</code>",
                "tooltip": "Place 30 byte item on stack",
                "url": "https://www.evm.codes/#7d"
            };

        case "PUSH31":
            return {
                "html": "Place 31 byte item on stack\nInput: -\nOutput: <code>value</code>",
                "tooltip": "Place 31 byte item on stack",
                "url": "https://www.evm.codes/#7e"
            };

        case "PUSH32":
            return {
                "html": "Place 32 byte (full word) item on stack\nInput: -\nOutput: <code>value</code>",
                "tooltip": "Place 32 byte (full word) item on stack",
                "url": "https://www.evm.codes/#7f"
            };

        case "DUP1":
            return {
                "html": "Duplicate 1st stack item\nInput: <code>value</code>\nOutput: <code>value | value</code>",
                "tooltip": "Duplicate 1st stack item",
                "url": "https://www.evm.codes/#80"
            };

        case "DUP2":
            return {
                "html": "Duplicate 2nd stack item\nInput: <code>a | b</code>\nOutput: <code>b | a | b</code>",
                "tooltip": "Duplicate 2nd stack item",
                "url": "https://www.evm.codes/#81"
            };

        case "DUP3":
            return {
                "html": "Duplicate 3rd stack item\nInput: <code>a | b | c</code>\nOutput: <code>c | a | b | c</code>",
                "tooltip": "Duplicate 3rd stack item",
                "url": "https://www.evm.codes/#82"
            };

        case "DUP4":
            return {
                "html": "Duplicate 4th stack item\nInput: <code>... | value</code>\nOutput: <code>value | ... | value</code>",
                "tooltip": "Duplicate 4th stack item",
                "url": "https://www.evm.codes/#83"
            };

        case "DUP5":
            return {
                "html": "Duplicate 5th stack item\nInput: <code>... | value</code>\nOutput: <code>value | ... | value</code>",
                "tooltip": "Duplicate 5th stack item",
                "url": "https://www.evm.codes/#84"
            };

        case "DUP6":
            return {
                "html": "Duplicate 6th stack item\nInput: <code>... | value</code>\nOutput: <code>value | ... | value</code>",
                "tooltip": "Duplicate 6th stack item",
                "url": "https://www.evm.codes/#85"
            };

        case "DUP7":
            return {
                "html": "Duplicate 7th stack item\nInput: <code>... | value</code>\nOutput: <code>value | ... | value</code>",
                "tooltip": "Duplicate 7th stack item",
                "url": "https://www.evm.codes/#86"
            };

        case "DUP8":
            return {
                "html": "Duplicate 8th stack item\nInput: <code>... | value</code>\nOutput: <code>value | ... | value</code>",
                "tooltip": "Duplicate 8th stack item",
                "url": "https://www.evm.codes/#87"
            };

        case "DUP9":
            return {
                "html": "Duplicate 9th stack item\nInput: <code>... | value</code>\nOutput: <code>value | ... | value</code>",
                "tooltip": "Duplicate 9th stack item",
                "url": "https://www.evm.codes/#88"
            };

        case "DUP10":
            return {
                "html": "Duplicate 10th stack item\nInput: <code>... | value</code>\nOutput: <code>value | ... | value</code>",
                "tooltip": "Duplicate 10th stack item",
                "url": "https://www.evm.codes/#89"
            };

        case "DUP11":
            return {
                "html": "Duplicate 11th stack item\nInput: <code>... | value</code>\nOutput: <code>value | ... | value</code>",
                "tooltip": "Duplicate 11th stack item",
                "url": "https://www.evm.codes/#8a"
            };

        case "DUP12":
            return {
                "html": "Duplicate 12th stack item\nInput: <code>... | value</code>\nOutput: <code>value | ... | value</code>",
                "tooltip": "Duplicate 12th stack item",
                "url": "https://www.evm.codes/#8b"
            };

        case "DUP13":
            return {
                "html": "Duplicate 13th stack item\nInput: <code>... | value</code>\nOutput: <code>value | ... | value</code>",
                "tooltip": "Duplicate 13th stack item",
                "url": "https://www.evm.codes/#8c"
            };

        case "DUP14":
            return {
                "html": "Duplicate 14th stack item\nInput: <code>... | value</code>\nOutput: <code>value | ... | value</code>",
                "tooltip": "Duplicate 14th stack item",
                "url": "https://www.evm.codes/#8d"
            };

        case "DUP15":
            return {
                "html": "Duplicate 15th stack item\nInput: <code>... | value</code>\nOutput: <code>value | ... | value</code>",
                "tooltip": "Duplicate 15th stack item",
                "url": "https://www.evm.codes/#8e"
            };

        case "DUP16":
            return {
                "html": "Duplicate 16th stack item\nInput: <code>... | value</code>\nOutput: <code>value | ... | value</code>",
                "tooltip": "Duplicate 16th stack item",
                "url": "https://www.evm.codes/#8f"
            };

        case "SWAP1":
            return {
                "html": "Exchange 1st and 2nd stack items\nInput: <code>a | b</code>\nOutput: <code>b | a</code>",
                "tooltip": "Exchange 1st and 2nd stack items",
                "url": "https://www.evm.codes/#90"
            };

        case "SWAP2":
            return {
                "html": "Exchange 1st and 3rd stack items\nInput: <code>a | b | c</code>\nOutput: <code>c | b | a</code>",
                "tooltip": "Exchange 1st and 3rd stack items",
                "url": "https://www.evm.codes/#91"
            };

        case "SWAP3":
            return {
                "html": "Exchange 1st and 4th stack items\nInput: <code>a | ... | b</code>\nOutput: <code>b | ... | a</code>",
                "tooltip": "Exchange 1st and 4th stack items",
                "url": "https://www.evm.codes/#92"
            };

        case "SWAP4":
            return {
                "html": "Exchange 1st and 5th stack items\nInput: <code>a | ... | b</code>\nOutput: <code>b | ... | a</code>",
                "tooltip": "Exchange 1st and 5th stack items",
                "url": "https://www.evm.codes/#93"
            };

        case "SWAP5":
            return {
                "html": "Exchange 1st and 6th stack items\nInput: <code>a | ... | b</code>\nOutput: <code>b | ... | a</code>",
                "tooltip": "Exchange 1st and 6th stack items",
                "url": "https://www.evm.codes/#94"
            };

        case "SWAP6":
            return {
                "html": "Exchange 1st and 7th stack items\nInput: <code>a | ... | b</code>\nOutput: <code>b | ... | a</code>",
                "tooltip": "Exchange 1st and 7th stack items",
                "url": "https://www.evm.codes/#95"
            };

        case "SWAP7":
            return {
                "html": "Exchange 1st and 8th stack items\nInput: <code>a | ... | b</code>\nOutput: <code>b | ... | a</code>",
                "tooltip": "Exchange 1st and 8th stack items",
                "url": "https://www.evm.codes/#96"
            };

        case "SWAP8":
            return {
                "html": "Exchange 1st and 9th stack items\nInput: <code>a | ... | b</code>\nOutput: <code>b | ... | a</code>",
                "tooltip": "Exchange 1st and 9th stack items",
                "url": "https://www.evm.codes/#97"
            };

        case "SWAP9":
            return {
                "html": "Exchange 1st and 10th stack items\nInput: <code>a | ... | b</code>\nOutput: <code>b | ... | a</code>",
                "tooltip": "Exchange 1st and 10th stack items",
                "url": "https://www.evm.codes/#98"
            };

        case "SWAP10":
            return {
                "html": "Exchange 1st and 11th stack items\nInput: <code>a | ... | b</code>\nOutput: <code>b | ... | a</code>",
                "tooltip": "Exchange 1st and 11th stack items",
                "url": "https://www.evm.codes/#99"
            };

        case "SWAP11":
            return {
                "html": "Exchange 1st and 12th stack items\nInput: <code>a | ... | b</code>\nOutput: <code>b | ... | a</code>",
                "tooltip": "Exchange 1st and 12th stack items",
                "url": "https://www.evm.codes/#9a"
            };

        case "SWAP12":
            return {
                "html": "Exchange 1st and 13th stack items\nInput: <code>a | ... | b</code>\nOutput: <code>b | ... | a</code>",
                "tooltip": "Exchange 1st and 13th stack items",
                "url": "https://www.evm.codes/#9b"
            };

        case "SWAP13":
            return {
                "html": "Exchange 1st and 14th stack items\nInput: <code>a | ... | b</code>\nOutput: <code>b | ... | a</code>",
                "tooltip": "Exchange 1st and 14th stack items",
                "url": "https://www.evm.codes/#9c"
            };

        case "SWAP14":
            return {
                "html": "Exchange 1st and 15th stack items\nInput: <code>a | ... | b</code>\nOutput: <code>b | ... | a</code>",
                "tooltip": "Exchange 1st and 15th stack items",
                "url": "https://www.evm.codes/#9d"
            };

        case "SWAP15":
            return {
                "html": "Exchange 1st and 16th stack items\nInput: <code>a | ... | b</code>\nOutput: <code>b | ... | a</code>",
                "tooltip": "Exchange 1st and 16th stack items",
                "url": "https://www.evm.codes/#9e"
            };

        case "SWAP16":
            return {
                "html": "Exchange 1st and 17th stack items\nInput: <code>a | ... | b</code>\nOutput: <code>b | ... | a</code>",
                "tooltip": "Exchange 1st and 17th stack items",
                "url": "https://www.evm.codes/#9f"
            };

        case "LOG0":
            return {
                "html": "Append log record with no topics\nInput: <code>offset | size</code>\nOutput: -",
                "tooltip": "Append log record with no topics",
                "url": "https://www.evm.codes/#a0"
            };

        case "LOG1":
            return {
                "html": "Append log record with one topic\nInput: <code>offset | size | topic</code>\nOutput: -",
                "tooltip": "Append log record with one topic",
                "url": "https://www.evm.codes/#a1"
            };

        case "LOG2":
            return {
                "html": "Append log record with two topics\nInput: <code>offset | size | topic1 | topic2</code>\nOutput: -",
                "tooltip": "Append log record with two topics",
                "url": "https://www.evm.codes/#a2"
            };

        case "LOG3":
            return {
                "html": "Append log record with three topics\nInput: <code>offset | size | topic1 | topic2 | topic3</code>\nOutput: -",
                "tooltip": "Append log record with three topics",
                "url": "https://www.evm.codes/#a3"
            };

        case "LOG4":
            return {
                "html": "Append log record with four topics\nInput: <code>offset | size | topic1 | topic2 | topic3 | topic4</code>\nOutput: -",
                "tooltip": "Append log record with four topics",
                "url": "https://www.evm.codes/#a4"
            };

        case "SLOADBYTES":
            return {
                "html": "\nInput: -\nOutput: -",
                "tooltip": "",
                "url": "https://www.evm.codes/#e1"
            };

        case "SSTOREBYTES":
            return {
                "html": "\nInput: -\nOutput: -",
                "tooltip": "",
                "url": "https://www.evm.codes/#e2"
            };

        case "SSIZE":
            return {
                "html": "\nInput: -\nOutput: -",
                "tooltip": "",
                "url": "https://www.evm.codes/#e3"
            };

        case "CREATE":
            return {
                "html": "Create a new account with associated code\nInput: <code>value | offset | size</code>\nOutput: <code>address</code>",
                "tooltip": "Create a new account with associated code",
                "url": "https://www.evm.codes/#f0"
            };

        case "CALL":
            return {
                "html": "Message-call into an account\nInput: <code>gas | address | value | argsOffset | argsSize | retOffset | retSize</code>\nOutput: <code>success</code>",
                "tooltip": "Message-call into an account",
                "url": "https://www.evm.codes/#f1"
            };

        case "CALLCODE":
            return {
                "html": "Message-call into this account with alternative account\u2019s code\nInput: <code>gas | address | value | argsOffset | argsSize | retOffset | retSize</code>\nOutput: <code>success</code>",
                "tooltip": "Message-call into this account with alternative account\u2019s code",
                "url": "https://www.evm.codes/#f2"
            };

        case "RETURN":
            return {
                "html": "Halt execution returning output data\nInput: <code>offset | size</code>\nOutput: -",
                "tooltip": "Halt execution returning output data",
                "url": "https://www.evm.codes/#f3"
            };

        case "DELEGATECALL":
            return {
                "html": "Message-call into this account with an alternative account\u2019s code, but persisting the current values for sender and value\nInput: <code>gas | address | argsOffset | argsSize | retOffset | retSize</code>\nOutput: <code>success</code>",
                "tooltip": "Message-call into this account with an alternative account\u2019s code, but persisting the current values for sender and value",
                "url": "https://www.evm.codes/#f4"
            };

        case "STATICCALL":
            return {
                "html": "Static message-call into an account\nInput: <code>gas | address | argsOffset | argsSize | retOffset | retSize</code>\nOutput: <code>success</code>",
                "tooltip": "Static message-call into an account",
                "url": "https://www.evm.codes/#fa"
            };

        case "TXEXECGAS":
            return {
                "html": "\nInput: -\nOutput: -",
                "tooltip": "",
                "url": "https://www.evm.codes/#fc"
            };

        case "REVERT":
            return {
                "html": "Halt execution reverting state changes but returning data and remaining gas\nInput: <code>offset | size</code>\nOutput: -",
                "tooltip": "Halt execution reverting state changes but returning data and remaining gas",
                "url": "https://www.evm.codes/#fd"
            };

        case "INVALID":
            return {
                "html": "Designated invalid instruction\nInput: -\nOutput: -",
                "tooltip": "Designated invalid instruction",
                "url": "https://www.evm.codes/#fe"
            };


    }
}
