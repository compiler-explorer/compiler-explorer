import type {AssemblyInstructionInfo} from '../../../types/assembly-docs.interfaces.js';

export function getAsmOpcode(opcode: string | undefined): AssemblyInstructionInfo | undefined {
    if (!opcode) return;
    switch (opcode.toLowerCase()) {
        case "aassign":
            return {
                "html": "list assignment",
                "tooltip": "list assignment",
                "url": ""
            };

        case "abs":
            return {
                "html": "abs",
                "tooltip": "abs",
                "url": ""
            };

        case "accept":
            return {
                "html": "accept",
                "tooltip": "accept",
                "url": ""
            };

        case "add":
            return {
                "html": "addition (+)",
                "tooltip": "addition (+)",
                "url": ""
            };

        case "aeach":
            return {
                "html": "each on array",
                "tooltip": "each on array",
                "url": ""
            };

        case "aelem":
            return {
                "html": "array element",
                "tooltip": "array element",
                "url": ""
            };

        case "aelemfast":
            return {
                "html": "constant array element",
                "tooltip": "constant array element",
                "url": ""
            };

        case "aelemfast_lex":
            return {
                "html": "constant lexical array element",
                "tooltip": "constant lexical array element",
                "url": ""
            };

        case "aelemfastlex_store":
            return {
                "html": "const lexical array element store",
                "tooltip": "const lexical array element store",
                "url": ""
            };

        case "akeys":
            return {
                "html": "keys on array",
                "tooltip": "keys on array",
                "url": ""
            };

        case "alarm":
            return {
                "html": "alarm",
                "tooltip": "alarm",
                "url": ""
            };

        case "allstart":
            return {
                "html": "all",
                "tooltip": "all",
                "url": ""
            };

        case "and":
            return {
                "html": "logical and (&&)",
                "tooltip": "logical and (&&)",
                "url": ""
            };

        case "andassign":
            return {
                "html": "logical and assignment (&&=)",
                "tooltip": "logical and assignment (&&=)",
                "url": ""
            };

        case "anoncode":
            return {
                "html": "anonymous subroutine",
                "tooltip": "anonymous subroutine",
                "url": ""
            };

        case "anonconst":
            return {
                "html": "anonymous constant",
                "tooltip": "anonymous constant",
                "url": ""
            };

        case "anonhash":
            return {
                "html": "anonymous hash ({})",
                "tooltip": "anonymous hash ({})",
                "url": ""
            };

        case "anonlist":
            return {
                "html": "anonymous array ([])",
                "tooltip": "anonymous array ([])",
                "url": ""
            };

        case "anystart":
            return {
                "html": "any",
                "tooltip": "any",
                "url": ""
            };

        case "anywhile":
            return {
                "html": "any/all iterator",
                "tooltip": "any/all iterator",
                "url": ""
            };

        case "argcheck":
            return {
                "html": "check subroutine arguments",
                "tooltip": "check subroutine arguments",
                "url": ""
            };

        case "argdefelem":
            return {
                "html": "subroutine argument default value",
                "tooltip": "subroutine argument default value",
                "url": ""
            };

        case "argelem":
            return {
                "html": "subroutine argument",
                "tooltip": "subroutine argument",
                "url": ""
            };

        case "aslice":
            return {
                "html": "array slice",
                "tooltip": "array slice",
                "url": ""
            };

        case "atan2":
            return {
                "html": "atan2",
                "tooltip": "atan2",
                "url": ""
            };

        case "av2arylen":
            return {
                "html": "array length",
                "tooltip": "array length",
                "url": ""
            };

        case "avalues":
            return {
                "html": "values on array",
                "tooltip": "values on array",
                "url": ""
            };

        case "avhvswitch":
            return {
                "html": "Array/hash switch",
                "tooltip": "Array/hash switch",
                "url": ""
            };

        case "backtick":
            return {
                "html": "quoted execution (``, qx)",
                "tooltip": "quoted execution (``, qx)",
                "url": ""
            };

        case "bind":
            return {
                "html": "bind",
                "tooltip": "bind",
                "url": ""
            };

        case "binmode":
            return {
                "html": "binmode",
                "tooltip": "binmode",
                "url": ""
            };

        case "bit_and":
            return {
                "html": "bitwise and (&)",
                "tooltip": "bitwise and (&)",
                "url": ""
            };

        case "bit_or":
            return {
                "html": "bitwise or (|)",
                "tooltip": "bitwise or (|)",
                "url": ""
            };

        case "bit_xor":
            return {
                "html": "bitwise xor (^)",
                "tooltip": "bitwise xor (^)",
                "url": ""
            };

        case "bless":
            return {
                "html": "bless",
                "tooltip": "bless",
                "url": ""
            };

        case "blessed":
            return {
                "html": "blessed",
                "tooltip": "blessed",
                "url": ""
            };

        case "break":
            return {
                "html": "break",
                "tooltip": "break",
                "url": ""
            };

        case "caller":
            return {
                "html": "caller",
                "tooltip": "caller",
                "url": ""
            };

        case "catch":
            return {
                "html": "catch {} block",
                "tooltip": "catch {} block",
                "url": ""
            };

        case "ceil":
            return {
                "html": "ceil",
                "tooltip": "ceil",
                "url": ""
            };

        case "chdir":
            return {
                "html": "chdir",
                "tooltip": "chdir",
                "url": ""
            };

        case "chmod":
            return {
                "html": "chmod",
                "tooltip": "chmod",
                "url": ""
            };

        case "chomp":
            return {
                "html": "chomp",
                "tooltip": "chomp",
                "url": ""
            };

        case "chop":
            return {
                "html": "chop",
                "tooltip": "chop",
                "url": ""
            };

        case "chown":
            return {
                "html": "chown",
                "tooltip": "chown",
                "url": ""
            };

        case "chr":
            return {
                "html": "chr",
                "tooltip": "chr",
                "url": ""
            };

        case "chroot":
            return {
                "html": "chroot",
                "tooltip": "chroot",
                "url": ""
            };

        case "classname":
            return {
                "html": "class name",
                "tooltip": "class name",
                "url": ""
            };

        case "clonecv":
            return {
                "html": "private subroutine",
                "tooltip": "private subroutine",
                "url": ""
            };

        case "close":
            return {
                "html": "close",
                "tooltip": "close",
                "url": ""
            };

        case "closedir":
            return {
                "html": "closedir",
                "tooltip": "closedir",
                "url": ""
            };

        case "cmpchain_and":
            return {
                "html": "comparison chaining",
                "tooltip": "comparison chaining",
                "url": ""
            };

        case "cmpchain_dup":
            return {
                "html": "comparand shuffling",
                "tooltip": "comparand shuffling",
                "url": ""
            };

        case "complement":
            return {
                "html": "1's complement (~)",
                "tooltip": "1's complement (~)",
                "url": ""
            };

        case "concat":
            return {
                "html": "concatenation (.) or string",
                "tooltip": "concatenation (.) or string",
                "url": ""
            };

        case "cond_expr":
            return {
                "html": "conditional expression",
                "tooltip": "conditional expression",
                "url": ""
            };

        case "connect":
            return {
                "html": "connect",
                "tooltip": "connect",
                "url": ""
            };

        case "const":
            return {
                "html": "constant item",
                "tooltip": "constant item",
                "url": ""
            };

        case "continue":
            return {
                "html": "continue",
                "tooltip": "continue",
                "url": ""
            };

        case "coreargs":
            return {
                "html": "CORE:: subroutine",
                "tooltip": "CORE:: subroutine",
                "url": ""
            };

        case "cos":
            return {
                "html": "cos",
                "tooltip": "cos",
                "url": ""
            };

        case "crypt":
            return {
                "html": "crypt",
                "tooltip": "crypt",
                "url": ""
            };

        case "custom":
            return {
                "html": "unknown custom operator",
                "tooltip": "unknown custom operator",
                "url": ""
            };

        case "dbmclose":
            return {
                "html": "dbmclose",
                "tooltip": "dbmclose",
                "url": ""
            };

        case "dbmopen":
            return {
                "html": "dbmopen",
                "tooltip": "dbmopen",
                "url": ""
            };

        case "dbstate":
            return {
                "html": "debug next statement",
                "tooltip": "debug next statement",
                "url": ""
            };

        case "defined":
            return {
                "html": "defined operator",
                "tooltip": "defined operator",
                "url": ""
            };

        case "delete":
            return {
                "html": "delete",
                "tooltip": "delete",
                "url": ""
            };

        case "die":
            return {
                "html": "die",
                "tooltip": "die",
                "url": ""
            };

        case "divide":
            return {
                "html": "division (/)",
                "tooltip": "division (/)",
                "url": ""
            };

        case "dofile":
            return {
                "html": "do \"file\"",
                "tooltip": "do \"file\"",
                "url": ""
            };

        case "dor":
            return {
                "html": "defined or (//)",
                "tooltip": "defined or (//)",
                "url": ""
            };

        case "dorassign":
            return {
                "html": "defined or assignment (//=)",
                "tooltip": "defined or assignment (//=)",
                "url": ""
            };

        case "dump":
            return {
                "html": "dump",
                "tooltip": "dump",
                "url": ""
            };

        case "each":
            return {
                "html": "each",
                "tooltip": "each",
                "url": ""
            };

        case "egrent":
            return {
                "html": "endgrent",
                "tooltip": "endgrent",
                "url": ""
            };

        case "ehostent":
            return {
                "html": "endhostent",
                "tooltip": "endhostent",
                "url": ""
            };

        case "emptyavhv":
            return {
                "html": "empty anon hash/array",
                "tooltip": "empty anon hash/array",
                "url": ""
            };

        case "enetent":
            return {
                "html": "endnetent",
                "tooltip": "endnetent",
                "url": ""
            };

        case "enter":
            return {
                "html": "block entry",
                "tooltip": "block entry",
                "url": ""
            };

        case "entereval":
            return {
                "html": "eval \"string\"",
                "tooltip": "eval \"string\"",
                "url": ""
            };

        case "entergiven":
            return {
                "html": "given()",
                "tooltip": "given()",
                "url": ""
            };

        case "enteriter":
            return {
                "html": "foreach loop entry",
                "tooltip": "foreach loop entry",
                "url": ""
            };

        case "enterloop":
            return {
                "html": "loop entry",
                "tooltip": "loop entry",
                "url": ""
            };

        case "entersub":
            return {
                "html": "subroutine entry",
                "tooltip": "subroutine entry",
                "url": ""
            };

        case "entertry":
            return {
                "html": "eval {block}",
                "tooltip": "eval {block}",
                "url": ""
            };

        case "entertrycatch":
            return {
                "html": "try {block}",
                "tooltip": "try {block}",
                "url": ""
            };

        case "enterwhen":
            return {
                "html": "when()",
                "tooltip": "when()",
                "url": ""
            };

        case "enterwrite":
            return {
                "html": "write",
                "tooltip": "write",
                "url": ""
            };

        case "eof":
            return {
                "html": "eof",
                "tooltip": "eof",
                "url": ""
            };

        case "eprotoent":
            return {
                "html": "endprotoent",
                "tooltip": "endprotoent",
                "url": ""
            };

        case "epwent":
            return {
                "html": "endpwent",
                "tooltip": "endpwent",
                "url": ""
            };

        case "eq":
            return {
                "html": "numeric eq (==)",
                "tooltip": "numeric eq (==)",
                "url": ""
            };

        case "eservent":
            return {
                "html": "endservent",
                "tooltip": "endservent",
                "url": ""
            };

        case "exec":
            return {
                "html": "exec",
                "tooltip": "exec",
                "url": ""
            };

        case "exists":
            return {
                "html": "exists",
                "tooltip": "exists",
                "url": ""
            };

        case "exit":
            return {
                "html": "exit",
                "tooltip": "exit",
                "url": ""
            };

        case "exp":
            return {
                "html": "exp",
                "tooltip": "exp",
                "url": ""
            };

        case "fc":
            return {
                "html": "fc",
                "tooltip": "fc",
                "url": ""
            };

        case "fcntl":
            return {
                "html": "fcntl",
                "tooltip": "fcntl",
                "url": ""
            };

        case "fileno":
            return {
                "html": "fileno",
                "tooltip": "fileno",
                "url": ""
            };

        case "flip":
            return {
                "html": "range (or flip)",
                "tooltip": "range (or flip)",
                "url": ""
            };

        case "flock":
            return {
                "html": "flock",
                "tooltip": "flock",
                "url": ""
            };

        case "floor":
            return {
                "html": "floor",
                "tooltip": "floor",
                "url": ""
            };

        case "flop":
            return {
                "html": "range (or flop)",
                "tooltip": "range (or flop)",
                "url": ""
            };

        case "fork":
            return {
                "html": "fork",
                "tooltip": "fork",
                "url": ""
            };

        case "formline":
            return {
                "html": "formline",
                "tooltip": "formline",
                "url": ""
            };

        case "ftatime":
            return {
                "html": "-A",
                "tooltip": "-A",
                "url": ""
            };

        case "ftbinary":
            return {
                "html": "-B",
                "tooltip": "-B",
                "url": ""
            };

        case "ftblk":
            return {
                "html": "-b",
                "tooltip": "-b",
                "url": ""
            };

        case "ftchr":
            return {
                "html": "-c",
                "tooltip": "-c",
                "url": ""
            };

        case "ftctime":
            return {
                "html": "-C",
                "tooltip": "-C",
                "url": ""
            };

        case "ftdir":
            return {
                "html": "-d",
                "tooltip": "-d",
                "url": ""
            };

        case "fteexec":
            return {
                "html": "-x",
                "tooltip": "-x",
                "url": ""
            };

        case "fteowned":
            return {
                "html": "-o",
                "tooltip": "-o",
                "url": ""
            };

        case "fteread":
            return {
                "html": "-r",
                "tooltip": "-r",
                "url": ""
            };

        case "ftewrite":
            return {
                "html": "-w",
                "tooltip": "-w",
                "url": ""
            };

        case "ftfile":
            return {
                "html": "-f",
                "tooltip": "-f",
                "url": ""
            };

        case "ftis":
            return {
                "html": "-e",
                "tooltip": "-e",
                "url": ""
            };

        case "ftlink":
            return {
                "html": "-l",
                "tooltip": "-l",
                "url": ""
            };

        case "ftmtime":
            return {
                "html": "-M",
                "tooltip": "-M",
                "url": ""
            };

        case "ftpipe":
            return {
                "html": "-p",
                "tooltip": "-p",
                "url": ""
            };

        case "ftrexec":
            return {
                "html": "-X",
                "tooltip": "-X",
                "url": ""
            };

        case "ftrowned":
            return {
                "html": "-O",
                "tooltip": "-O",
                "url": ""
            };

        case "ftrread":
            return {
                "html": "-R",
                "tooltip": "-R",
                "url": ""
            };

        case "ftrwrite":
            return {
                "html": "-W",
                "tooltip": "-W",
                "url": ""
            };

        case "ftsgid":
            return {
                "html": "-g",
                "tooltip": "-g",
                "url": ""
            };

        case "ftsize":
            return {
                "html": "-s",
                "tooltip": "-s",
                "url": ""
            };

        case "ftsock":
            return {
                "html": "-S",
                "tooltip": "-S",
                "url": ""
            };

        case "ftsuid":
            return {
                "html": "-u",
                "tooltip": "-u",
                "url": ""
            };

        case "ftsvtx":
            return {
                "html": "-k",
                "tooltip": "-k",
                "url": ""
            };

        case "fttext":
            return {
                "html": "-T",
                "tooltip": "-T",
                "url": ""
            };

        case "fttty":
            return {
                "html": "-t",
                "tooltip": "-t",
                "url": ""
            };

        case "ftzero":
            return {
                "html": "-z",
                "tooltip": "-z",
                "url": ""
            };

        case "ge":
            return {
                "html": "numeric ge (>=)",
                "tooltip": "numeric ge (>=)",
                "url": ""
            };

        case "gelem":
            return {
                "html": "glob elem",
                "tooltip": "glob elem",
                "url": ""
            };

        case "getc":
            return {
                "html": "getc",
                "tooltip": "getc",
                "url": ""
            };

        case "getlogin":
            return {
                "html": "getlogin",
                "tooltip": "getlogin",
                "url": ""
            };

        case "getpeername":
            return {
                "html": "getpeername",
                "tooltip": "getpeername",
                "url": ""
            };

        case "getpgrp":
            return {
                "html": "getpgrp",
                "tooltip": "getpgrp",
                "url": ""
            };

        case "getppid":
            return {
                "html": "getppid",
                "tooltip": "getppid",
                "url": ""
            };

        case "getpriority":
            return {
                "html": "getpriority",
                "tooltip": "getpriority",
                "url": ""
            };

        case "getsockname":
            return {
                "html": "getsockname",
                "tooltip": "getsockname",
                "url": ""
            };

        case "ggrent":
            return {
                "html": "getgrent",
                "tooltip": "getgrent",
                "url": ""
            };

        case "ggrgid":
            return {
                "html": "getgrgid",
                "tooltip": "getgrgid",
                "url": ""
            };

        case "ggrnam":
            return {
                "html": "getgrnam",
                "tooltip": "getgrnam",
                "url": ""
            };

        case "ghbyaddr":
            return {
                "html": "gethostbyaddr",
                "tooltip": "gethostbyaddr",
                "url": ""
            };

        case "ghbyname":
            return {
                "html": "gethostbyname",
                "tooltip": "gethostbyname",
                "url": ""
            };

        case "ghostent":
            return {
                "html": "gethostent",
                "tooltip": "gethostent",
                "url": ""
            };

        case "glob":
            return {
                "html": "glob",
                "tooltip": "glob",
                "url": ""
            };

        case "gmtime":
            return {
                "html": "gmtime",
                "tooltip": "gmtime",
                "url": ""
            };

        case "gnbyaddr":
            return {
                "html": "getnetbyaddr",
                "tooltip": "getnetbyaddr",
                "url": ""
            };

        case "gnbyname":
            return {
                "html": "getnetbyname",
                "tooltip": "getnetbyname",
                "url": ""
            };

        case "gnetent":
            return {
                "html": "getnetent",
                "tooltip": "getnetent",
                "url": ""
            };

        case "goto":
            return {
                "html": "goto",
                "tooltip": "goto",
                "url": ""
            };

        case "gpbyname":
            return {
                "html": "getprotobyname",
                "tooltip": "getprotobyname",
                "url": ""
            };

        case "gpbynumber":
            return {
                "html": "getprotobynumber",
                "tooltip": "getprotobynumber",
                "url": ""
            };

        case "gprotoent":
            return {
                "html": "getprotoent",
                "tooltip": "getprotoent",
                "url": ""
            };

        case "gpwent":
            return {
                "html": "getpwent",
                "tooltip": "getpwent",
                "url": ""
            };

        case "gpwnam":
            return {
                "html": "getpwnam",
                "tooltip": "getpwnam",
                "url": ""
            };

        case "gpwuid":
            return {
                "html": "getpwuid",
                "tooltip": "getpwuid",
                "url": ""
            };

        case "grepstart":
            return {
                "html": "grep",
                "tooltip": "grep",
                "url": ""
            };

        case "grepwhile":
            return {
                "html": "grep iterator",
                "tooltip": "grep iterator",
                "url": ""
            };

        case "gsbyname":
            return {
                "html": "getservbyname",
                "tooltip": "getservbyname",
                "url": ""
            };

        case "gsbyport":
            return {
                "html": "getservbyport",
                "tooltip": "getservbyport",
                "url": ""
            };

        case "gservent":
            return {
                "html": "getservent",
                "tooltip": "getservent",
                "url": ""
            };

        case "gsockopt":
            return {
                "html": "getsockopt",
                "tooltip": "getsockopt",
                "url": ""
            };

        case "gt":
            return {
                "html": "numeric gt (>)",
                "tooltip": "numeric gt (>)",
                "url": ""
            };

        case "gv":
            return {
                "html": "glob value",
                "tooltip": "glob value",
                "url": ""
            };

        case "gvsv":
            return {
                "html": "scalar variable",
                "tooltip": "scalar variable",
                "url": ""
            };

        case "helem":
            return {
                "html": "hash element",
                "tooltip": "hash element",
                "url": ""
            };

        case "helemexistsor":
            return {
                "html": "hash element exists or",
                "tooltip": "hash element exists or",
                "url": ""
            };

        case "hex":
            return {
                "html": "hex",
                "tooltip": "hex",
                "url": ""
            };

        case "hintseval":
            return {
                "html": "eval hints",
                "tooltip": "eval hints",
                "url": ""
            };

        case "hslice":
            return {
                "html": "hash slice",
                "tooltip": "hash slice",
                "url": ""
            };

        case "i_add":
            return {
                "html": "integer addition (+)",
                "tooltip": "integer addition (+)",
                "url": ""
            };

        case "i_divide":
            return {
                "html": "integer division (/)",
                "tooltip": "integer division (/)",
                "url": ""
            };

        case "i_eq":
            return {
                "html": "integer eq (==)",
                "tooltip": "integer eq (==)",
                "url": ""
            };

        case "i_ge":
            return {
                "html": "integer ge (>=)",
                "tooltip": "integer ge (>=)",
                "url": ""
            };

        case "i_gt":
            return {
                "html": "integer gt (>)",
                "tooltip": "integer gt (>)",
                "url": ""
            };

        case "i_le":
            return {
                "html": "integer le (<=)",
                "tooltip": "integer le (<=)",
                "url": ""
            };

        case "i_lt":
            return {
                "html": "integer lt (<)",
                "tooltip": "integer lt (<)",
                "url": ""
            };

        case "i_modulo":
            return {
                "html": "integer modulus (%)",
                "tooltip": "integer modulus (%)",
                "url": ""
            };

        case "i_multiply":
            return {
                "html": "integer multiplication (*)",
                "tooltip": "integer multiplication (*)",
                "url": ""
            };

        case "i_ncmp":
            return {
                "html": "integer comparison (<=>)",
                "tooltip": "integer comparison (<=>)",
                "url": ""
            };

        case "i_ne":
            return {
                "html": "integer ne (!=)",
                "tooltip": "integer ne (!=)",
                "url": ""
            };

        case "i_negate":
            return {
                "html": "integer negation (-)",
                "tooltip": "integer negation (-)",
                "url": ""
            };

        case "i_postdec":
            return {
                "html": "integer postdecrement (--)",
                "tooltip": "integer postdecrement (--)",
                "url": ""
            };

        case "i_postinc":
            return {
                "html": "integer postincrement (++)",
                "tooltip": "integer postincrement (++)",
                "url": ""
            };

        case "i_predec":
            return {
                "html": "integer predecrement (--)",
                "tooltip": "integer predecrement (--)",
                "url": ""
            };

        case "i_preinc":
            return {
                "html": "integer preincrement (++)",
                "tooltip": "integer preincrement (++)",
                "url": ""
            };

        case "i_subtract":
            return {
                "html": "integer subtraction (-)",
                "tooltip": "integer subtraction (-)",
                "url": ""
            };

        case "index":
            return {
                "html": "index",
                "tooltip": "index",
                "url": ""
            };

        case "initfield":
            return {
                "html": "initialise field",
                "tooltip": "initialise field",
                "url": ""
            };

        case "int":
            return {
                "html": "int",
                "tooltip": "int",
                "url": ""
            };

        case "introcv":
            return {
                "html": "private subroutine",
                "tooltip": "private subroutine",
                "url": ""
            };

        case "ioctl":
            return {
                "html": "ioctl",
                "tooltip": "ioctl",
                "url": ""
            };

        case "is_bool":
            return {
                "html": "boolean type test",
                "tooltip": "boolean type test",
                "url": ""
            };

        case "is_tainted":
            return {
                "html": "is_tainted",
                "tooltip": "is_tainted",
                "url": ""
            };

        case "is_weak":
            return {
                "html": "weakref type test",
                "tooltip": "weakref type test",
                "url": ""
            };

        case "isa":
            return {
                "html": "derived class test",
                "tooltip": "derived class test",
                "url": ""
            };

        case "iter":
            return {
                "html": "foreach loop iterator",
                "tooltip": "foreach loop iterator",
                "url": ""
            };

        case "join":
            return {
                "html": "join or string",
                "tooltip": "join or string",
                "url": ""
            };

        case "keys":
            return {
                "html": "keys",
                "tooltip": "keys",
                "url": ""
            };

        case "kill":
            return {
                "html": "kill",
                "tooltip": "kill",
                "url": ""
            };

        case "kvaslice":
            return {
                "html": "index/value array slice",
                "tooltip": "index/value array slice",
                "url": ""
            };

        case "kvhslice":
            return {
                "html": "key/value hash slice",
                "tooltip": "key/value hash slice",
                "url": ""
            };

        case "last":
            return {
                "html": "last",
                "tooltip": "last",
                "url": ""
            };

        case "lc":
            return {
                "html": "lc",
                "tooltip": "lc",
                "url": ""
            };

        case "lcfirst":
            return {
                "html": "lcfirst",
                "tooltip": "lcfirst",
                "url": ""
            };

        case "le":
            return {
                "html": "numeric le (<=)",
                "tooltip": "numeric le (<=)",
                "url": ""
            };

        case "leave":
            return {
                "html": "block exit",
                "tooltip": "block exit",
                "url": ""
            };

        case "leaveeval":
            return {
                "html": "eval \"string\" exit",
                "tooltip": "eval \"string\" exit",
                "url": ""
            };

        case "leavegiven":
            return {
                "html": "leave given block",
                "tooltip": "leave given block",
                "url": ""
            };

        case "leaveloop":
            return {
                "html": "loop exit",
                "tooltip": "loop exit",
                "url": ""
            };

        case "leavesub":
            return {
                "html": "subroutine exit",
                "tooltip": "subroutine exit",
                "url": ""
            };

        case "leavesublv":
            return {
                "html": "lvalue subroutine return",
                "tooltip": "lvalue subroutine return",
                "url": ""
            };

        case "leavetry":
            return {
                "html": "eval {block} exit",
                "tooltip": "eval {block} exit",
                "url": ""
            };

        case "leavetrycatch":
            return {
                "html": "try {block} exit",
                "tooltip": "try {block} exit",
                "url": ""
            };

        case "leavewhen":
            return {
                "html": "leave when block",
                "tooltip": "leave when block",
                "url": ""
            };

        case "leavewrite":
            return {
                "html": "write exit",
                "tooltip": "write exit",
                "url": ""
            };

        case "left_shift":
            return {
                "html": "left bitshift (<<)",
                "tooltip": "left bitshift (<<)",
                "url": ""
            };

        case "length":
            return {
                "html": "length",
                "tooltip": "length",
                "url": ""
            };

        case "lineseq":
            return {
                "html": "line sequence",
                "tooltip": "line sequence",
                "url": ""
            };

        case "link":
            return {
                "html": "link",
                "tooltip": "link",
                "url": ""
            };

        case "list":
            return {
                "html": "list",
                "tooltip": "list",
                "url": ""
            };

        case "listen":
            return {
                "html": "listen",
                "tooltip": "listen",
                "url": ""
            };

        case "localtime":
            return {
                "html": "localtime",
                "tooltip": "localtime",
                "url": ""
            };

        case "lock":
            return {
                "html": "lock",
                "tooltip": "lock",
                "url": ""
            };

        case "log":
            return {
                "html": "log",
                "tooltip": "log",
                "url": ""
            };

        case "lslice":
            return {
                "html": "list slice",
                "tooltip": "list slice",
                "url": ""
            };

        case "lstat":
            return {
                "html": "lstat",
                "tooltip": "lstat",
                "url": ""
            };

        case "lt":
            return {
                "html": "numeric lt (<)",
                "tooltip": "numeric lt (<)",
                "url": ""
            };

        case "lvavref":
            return {
                "html": "lvalue array reference",
                "tooltip": "lvalue array reference",
                "url": ""
            };

        case "lvref":
            return {
                "html": "lvalue ref assignment",
                "tooltip": "lvalue ref assignment",
                "url": ""
            };

        case "lvrefslice":
            return {
                "html": "lvalue ref assignment",
                "tooltip": "lvalue ref assignment",
                "url": ""
            };

        case "mapstart":
            return {
                "html": "map",
                "tooltip": "map",
                "url": ""
            };

        case "mapwhile":
            return {
                "html": "map iterator",
                "tooltip": "map iterator",
                "url": ""
            };

        case "match":
            return {
                "html": "pattern match (m//)",
                "tooltip": "pattern match (m//)",
                "url": ""
            };

        case "method":
            return {
                "html": "method lookup",
                "tooltip": "method lookup",
                "url": ""
            };

        case "method_named":
            return {
                "html": "method with known name",
                "tooltip": "method with known name",
                "url": ""
            };

        case "method_redir":
            return {
                "html": "redirect method with known name",
                "tooltip": "redirect method with known name",
                "url": ""
            };

        case "method_redir_super":
            return {
                "html": "redirect super method with known name",
                "tooltip": "redirect super method with known name",
                "url": ""
            };

        case "method_super":
            return {
                "html": "super with known name",
                "tooltip": "super with known name",
                "url": ""
            };

        case "methstart":
            return {
                "html": "method start",
                "tooltip": "method start",
                "url": ""
            };

        case "mkdir":
            return {
                "html": "mkdir",
                "tooltip": "mkdir",
                "url": ""
            };

        case "modulo":
            return {
                "html": "modulus (%)",
                "tooltip": "modulus (%)",
                "url": ""
            };

        case "msgctl":
            return {
                "html": "msgctl",
                "tooltip": "msgctl",
                "url": ""
            };

        case "msgget":
            return {
                "html": "msgget",
                "tooltip": "msgget",
                "url": ""
            };

        case "msgrcv":
            return {
                "html": "msgrcv",
                "tooltip": "msgrcv",
                "url": ""
            };

        case "msgsnd":
            return {
                "html": "msgsnd",
                "tooltip": "msgsnd",
                "url": ""
            };

        case "multiconcat":
            return {
                "html": "concatenation (.) or string",
                "tooltip": "concatenation (.) or string",
                "url": ""
            };

        case "multideref":
            return {
                "html": "array or hash lookup",
                "tooltip": "array or hash lookup",
                "url": ""
            };

        case "multiparam":
            return {
                "html": "signature processing",
                "tooltip": "signature processing",
                "url": ""
            };

        case "multiply":
            return {
                "html": "multiplication (*)",
                "tooltip": "multiplication (*)",
                "url": ""
            };

        case "nbit_and":
            return {
                "html": "numeric bitwise and (&)",
                "tooltip": "numeric bitwise and (&)",
                "url": ""
            };

        case "nbit_or":
            return {
                "html": "numeric bitwise or (|)",
                "tooltip": "numeric bitwise or (|)",
                "url": ""
            };

        case "nbit_xor":
            return {
                "html": "numeric bitwise xor (^)",
                "tooltip": "numeric bitwise xor (^)",
                "url": ""
            };

        case "ncmp":
            return {
                "html": "numeric comparison (<=>)",
                "tooltip": "numeric comparison (<=>)",
                "url": ""
            };

        case "ncomplement":
            return {
                "html": "numeric 1's complement (~)",
                "tooltip": "numeric 1's complement (~)",
                "url": ""
            };

        case "ne":
            return {
                "html": "numeric ne (!=)",
                "tooltip": "numeric ne (!=)",
                "url": ""
            };

        case "negate":
            return {
                "html": "negation (-)",
                "tooltip": "negation (-)",
                "url": ""
            };

        case "next":
            return {
                "html": "next",
                "tooltip": "next",
                "url": ""
            };

        case "nextstate":
            return {
                "html": "next statement",
                "tooltip": "next statement",
                "url": ""
            };

        case "not":
            return {
                "html": "not",
                "tooltip": "not",
                "url": ""
            };

        case "null":
            return {
                "html": "null operation",
                "tooltip": "null operation",
                "url": ""
            };

        case "oct":
            return {
                "html": "oct",
                "tooltip": "oct",
                "url": ""
            };

        case "once":
            return {
                "html": "once",
                "tooltip": "once",
                "url": ""
            };

        case "open":
            return {
                "html": "open",
                "tooltip": "open",
                "url": ""
            };

        case "open_dir":
            return {
                "html": "opendir",
                "tooltip": "opendir",
                "url": ""
            };

        case "or":
            return {
                "html": "logical or (||)",
                "tooltip": "logical or (||)",
                "url": ""
            };

        case "orassign":
            return {
                "html": "logical or assignment (||=)",
                "tooltip": "logical or assignment (||=)",
                "url": ""
            };

        case "ord":
            return {
                "html": "ord",
                "tooltip": "ord",
                "url": ""
            };

        case "pack":
            return {
                "html": "pack",
                "tooltip": "pack",
                "url": ""
            };

        case "padany":
            return {
                "html": "private value",
                "tooltip": "private value",
                "url": ""
            };

        case "padav":
            return {
                "html": "private array",
                "tooltip": "private array",
                "url": ""
            };

        case "padcv":
            return {
                "html": "private subroutine",
                "tooltip": "private subroutine",
                "url": ""
            };

        case "padhv":
            return {
                "html": "private hash",
                "tooltip": "private hash",
                "url": ""
            };

        case "padrange":
            return {
                "html": "list of private variables",
                "tooltip": "list of private variables",
                "url": ""
            };

        case "padsv":
            return {
                "html": "private variable",
                "tooltip": "private variable",
                "url": ""
            };

        case "padsv_store":
            return {
                "html": "padsv scalar assignment",
                "tooltip": "padsv scalar assignment",
                "url": ""
            };

        case "paramstore":
            return {
                "html": "signature parameter default expression",
                "tooltip": "signature parameter default expression",
                "url": ""
            };

        case "paramtest":
            return {
                "html": "signature argument value test",
                "tooltip": "signature argument value test",
                "url": ""
            };

        case "pipe_op":
            return {
                "html": "pipe",
                "tooltip": "pipe",
                "url": ""
            };

        case "pop":
            return {
                "html": "pop",
                "tooltip": "pop",
                "url": ""
            };

        case "poptry":
            return {
                "html": "pop try",
                "tooltip": "pop try",
                "url": ""
            };

        case "pos":
            return {
                "html": "match position",
                "tooltip": "match position",
                "url": ""
            };

        case "postdec":
            return {
                "html": "postdecrement (--)",
                "tooltip": "postdecrement (--)",
                "url": ""
            };

        case "postinc":
            return {
                "html": "postincrement (++)",
                "tooltip": "postincrement (++)",
                "url": ""
            };

        case "pow":
            return {
                "html": "exponentiation (**)",
                "tooltip": "exponentiation (**)",
                "url": ""
            };

        case "predec":
            return {
                "html": "predecrement (--)",
                "tooltip": "predecrement (--)",
                "url": ""
            };

        case "preinc":
            return {
                "html": "preincrement (++)",
                "tooltip": "preincrement (++)",
                "url": ""
            };

        case "print":
            return {
                "html": "print",
                "tooltip": "print",
                "url": ""
            };

        case "prototype":
            return {
                "html": "subroutine prototype",
                "tooltip": "subroutine prototype",
                "url": ""
            };

        case "prtf":
            return {
                "html": "printf",
                "tooltip": "printf",
                "url": ""
            };

        case "push":
            return {
                "html": "push",
                "tooltip": "push",
                "url": ""
            };

        case "pushdefer":
            return {
                "html": "push defer {} block",
                "tooltip": "push defer {} block",
                "url": ""
            };

        case "pushmark":
            return {
                "html": "pushmark",
                "tooltip": "pushmark",
                "url": ""
            };

        case "qr":
            return {
                "html": "pattern quote (qr//)",
                "tooltip": "pattern quote (qr//)",
                "url": ""
            };

        case "quotemeta":
            return {
                "html": "quotemeta",
                "tooltip": "quotemeta",
                "url": ""
            };

        case "rand":
            return {
                "html": "rand",
                "tooltip": "rand",
                "url": ""
            };

        case "range":
            return {
                "html": "flipflop",
                "tooltip": "flipflop",
                "url": ""
            };

        case "rcatline":
            return {
                "html": "append I/O operator",
                "tooltip": "append I/O operator",
                "url": ""
            };

        case "read":
            return {
                "html": "read",
                "tooltip": "read",
                "url": ""
            };

        case "readdir":
            return {
                "html": "readdir",
                "tooltip": "readdir",
                "url": ""
            };

        case "readline":
            return {
                "html": "<HANDLE>",
                "tooltip": "<HANDLE>",
                "url": ""
            };

        case "readlink":
            return {
                "html": "readlink",
                "tooltip": "readlink",
                "url": ""
            };

        case "recv":
            return {
                "html": "recv",
                "tooltip": "recv",
                "url": ""
            };

        case "redo":
            return {
                "html": "redo",
                "tooltip": "redo",
                "url": ""
            };

        case "ref":
            return {
                "html": "reference-type operator",
                "tooltip": "reference-type operator",
                "url": ""
            };

        case "refaddr":
            return {
                "html": "refaddr",
                "tooltip": "refaddr",
                "url": ""
            };

        case "refassign":
            return {
                "html": "lvalue ref assignment",
                "tooltip": "lvalue ref assignment",
                "url": ""
            };

        case "refgen":
            return {
                "html": "reference constructor",
                "tooltip": "reference constructor",
                "url": ""
            };

        case "reftype":
            return {
                "html": "reftype",
                "tooltip": "reftype",
                "url": ""
            };

        case "regcmaybe":
            return {
                "html": "regexp internal guard",
                "tooltip": "regexp internal guard",
                "url": ""
            };

        case "regcomp":
            return {
                "html": "regexp compilation",
                "tooltip": "regexp compilation",
                "url": ""
            };

        case "regcreset":
            return {
                "html": "regexp internal reset",
                "tooltip": "regexp internal reset",
                "url": ""
            };

        case "rename":
            return {
                "html": "rename",
                "tooltip": "rename",
                "url": ""
            };

        case "repeat":
            return {
                "html": "repeat (x)",
                "tooltip": "repeat (x)",
                "url": ""
            };

        case "require":
            return {
                "html": "require",
                "tooltip": "require",
                "url": ""
            };

        case "reset":
            return {
                "html": "symbol reset",
                "tooltip": "symbol reset",
                "url": ""
            };

        case "return":
            return {
                "html": "return",
                "tooltip": "return",
                "url": ""
            };

        case "reverse":
            return {
                "html": "reverse",
                "tooltip": "reverse",
                "url": ""
            };

        case "rewinddir":
            return {
                "html": "rewinddir",
                "tooltip": "rewinddir",
                "url": ""
            };

        case "right_shift":
            return {
                "html": "right bitshift (>>)",
                "tooltip": "right bitshift (>>)",
                "url": ""
            };

        case "rindex":
            return {
                "html": "rindex",
                "tooltip": "rindex",
                "url": ""
            };

        case "rmdir":
            return {
                "html": "rmdir",
                "tooltip": "rmdir",
                "url": ""
            };

        case "runcv":
            return {
                "html": "__SUB__",
                "tooltip": "__SUB__",
                "url": ""
            };

        case "rv2av":
            return {
                "html": "array dereference",
                "tooltip": "array dereference",
                "url": ""
            };

        case "rv2cv":
            return {
                "html": "subroutine dereference",
                "tooltip": "subroutine dereference",
                "url": ""
            };

        case "rv2gv":
            return {
                "html": "ref-to-glob cast",
                "tooltip": "ref-to-glob cast",
                "url": ""
            };

        case "rv2hv":
            return {
                "html": "hash dereference",
                "tooltip": "hash dereference",
                "url": ""
            };

        case "rv2sv":
            return {
                "html": "scalar dereference",
                "tooltip": "scalar dereference",
                "url": ""
            };

        case "sassign":
            return {
                "html": "scalar assignment",
                "tooltip": "scalar assignment",
                "url": ""
            };

        case "say":
            return {
                "html": "say",
                "tooltip": "say",
                "url": ""
            };

        case "sbit_and":
            return {
                "html": "string bitwise and (&.)",
                "tooltip": "string bitwise and (&.)",
                "url": ""
            };

        case "sbit_or":
            return {
                "html": "string bitwise or (|.)",
                "tooltip": "string bitwise or (|.)",
                "url": ""
            };

        case "sbit_xor":
            return {
                "html": "string bitwise xor (^.)",
                "tooltip": "string bitwise xor (^.)",
                "url": ""
            };

        case "scalar":
            return {
                "html": "scalar",
                "tooltip": "scalar",
                "url": ""
            };

        case "schomp":
            return {
                "html": "scalar chomp",
                "tooltip": "scalar chomp",
                "url": ""
            };

        case "schop":
            return {
                "html": "scalar chop",
                "tooltip": "scalar chop",
                "url": ""
            };

        case "scmp":
            return {
                "html": "string comparison (cmp)",
                "tooltip": "string comparison (cmp)",
                "url": ""
            };

        case "scomplement":
            return {
                "html": "string 1's complement (~)",
                "tooltip": "string 1's complement (~)",
                "url": ""
            };

        case "scope":
            return {
                "html": "block",
                "tooltip": "block",
                "url": ""
            };

        case "seek":
            return {
                "html": "seek",
                "tooltip": "seek",
                "url": ""
            };

        case "seekdir":
            return {
                "html": "seekdir",
                "tooltip": "seekdir",
                "url": ""
            };

        case "select":
            return {
                "html": "select",
                "tooltip": "select",
                "url": ""
            };

        case "semctl":
            return {
                "html": "semctl",
                "tooltip": "semctl",
                "url": ""
            };

        case "semget":
            return {
                "html": "semget",
                "tooltip": "semget",
                "url": ""
            };

        case "semop":
            return {
                "html": "semop",
                "tooltip": "semop",
                "url": ""
            };

        case "send":
            return {
                "html": "send",
                "tooltip": "send",
                "url": ""
            };

        case "seq":
            return {
                "html": "string eq",
                "tooltip": "string eq",
                "url": ""
            };

        case "setpgrp":
            return {
                "html": "setpgrp",
                "tooltip": "setpgrp",
                "url": ""
            };

        case "setpriority":
            return {
                "html": "setpriority",
                "tooltip": "setpriority",
                "url": ""
            };

        case "sge":
            return {
                "html": "string ge",
                "tooltip": "string ge",
                "url": ""
            };

        case "sgrent":
            return {
                "html": "setgrent",
                "tooltip": "setgrent",
                "url": ""
            };

        case "sgt":
            return {
                "html": "string gt",
                "tooltip": "string gt",
                "url": ""
            };

        case "shift":
            return {
                "html": "shift",
                "tooltip": "shift",
                "url": ""
            };

        case "shmctl":
            return {
                "html": "shmctl",
                "tooltip": "shmctl",
                "url": ""
            };

        case "shmget":
            return {
                "html": "shmget",
                "tooltip": "shmget",
                "url": ""
            };

        case "shmread":
            return {
                "html": "shmread",
                "tooltip": "shmread",
                "url": ""
            };

        case "shmwrite":
            return {
                "html": "shmwrite",
                "tooltip": "shmwrite",
                "url": ""
            };

        case "shostent":
            return {
                "html": "sethostent",
                "tooltip": "sethostent",
                "url": ""
            };

        case "shutdown":
            return {
                "html": "shutdown",
                "tooltip": "shutdown",
                "url": ""
            };

        case "sin":
            return {
                "html": "sin",
                "tooltip": "sin",
                "url": ""
            };

        case "sle":
            return {
                "html": "string le",
                "tooltip": "string le",
                "url": ""
            };

        case "sleep":
            return {
                "html": "sleep",
                "tooltip": "sleep",
                "url": ""
            };

        case "slt":
            return {
                "html": "string lt",
                "tooltip": "string lt",
                "url": ""
            };

        case "smartmatch":
            return {
                "html": "smart match",
                "tooltip": "smart match",
                "url": ""
            };

        case "sne":
            return {
                "html": "string ne",
                "tooltip": "string ne",
                "url": ""
            };

        case "snetent":
            return {
                "html": "setnetent",
                "tooltip": "setnetent",
                "url": ""
            };

        case "socket":
            return {
                "html": "socket",
                "tooltip": "socket",
                "url": ""
            };

        case "sockpair":
            return {
                "html": "socketpair",
                "tooltip": "socketpair",
                "url": ""
            };

        case "sort":
            return {
                "html": "sort",
                "tooltip": "sort",
                "url": ""
            };

        case "splice":
            return {
                "html": "splice",
                "tooltip": "splice",
                "url": ""
            };

        case "split":
            return {
                "html": "split",
                "tooltip": "split",
                "url": ""
            };

        case "sprintf":
            return {
                "html": "sprintf",
                "tooltip": "sprintf",
                "url": ""
            };

        case "sprotoent":
            return {
                "html": "setprotoent",
                "tooltip": "setprotoent",
                "url": ""
            };

        case "spwent":
            return {
                "html": "setpwent",
                "tooltip": "setpwent",
                "url": ""
            };

        case "sqrt":
            return {
                "html": "sqrt",
                "tooltip": "sqrt",
                "url": ""
            };

        case "srand":
            return {
                "html": "srand",
                "tooltip": "srand",
                "url": ""
            };

        case "srefgen":
            return {
                "html": "single ref constructor",
                "tooltip": "single ref constructor",
                "url": ""
            };

        case "sselect":
            return {
                "html": "select system call",
                "tooltip": "select system call",
                "url": ""
            };

        case "sservent":
            return {
                "html": "setservent",
                "tooltip": "setservent",
                "url": ""
            };

        case "ssockopt":
            return {
                "html": "setsockopt",
                "tooltip": "setsockopt",
                "url": ""
            };

        case "stat":
            return {
                "html": "stat",
                "tooltip": "stat",
                "url": ""
            };

        case "stringify":
            return {
                "html": "string",
                "tooltip": "string",
                "url": ""
            };

        case "stub":
            return {
                "html": "stub",
                "tooltip": "stub",
                "url": ""
            };

        case "study":
            return {
                "html": "study",
                "tooltip": "study",
                "url": ""
            };

        case "subst":
            return {
                "html": "substitution (s///)",
                "tooltip": "substitution (s///)",
                "url": ""
            };

        case "substcont":
            return {
                "html": "substitution iterator",
                "tooltip": "substitution iterator",
                "url": ""
            };

        case "substr":
            return {
                "html": "substr",
                "tooltip": "substr",
                "url": ""
            };

        case "substr_left":
            return {
                "html": "substr left",
                "tooltip": "substr left",
                "url": ""
            };

        case "subtract":
            return {
                "html": "subtraction (-)",
                "tooltip": "subtraction (-)",
                "url": ""
            };

        case "symlink":
            return {
                "html": "symlink",
                "tooltip": "symlink",
                "url": ""
            };

        case "syscall":
            return {
                "html": "syscall",
                "tooltip": "syscall",
                "url": ""
            };

        case "sysopen":
            return {
                "html": "sysopen",
                "tooltip": "sysopen",
                "url": ""
            };

        case "sysread":
            return {
                "html": "sysread",
                "tooltip": "sysread",
                "url": ""
            };

        case "sysseek":
            return {
                "html": "sysseek",
                "tooltip": "sysseek",
                "url": ""
            };

        case "system":
            return {
                "html": "system",
                "tooltip": "system",
                "url": ""
            };

        case "syswrite":
            return {
                "html": "syswrite",
                "tooltip": "syswrite",
                "url": ""
            };

        case "tell":
            return {
                "html": "tell",
                "tooltip": "tell",
                "url": ""
            };

        case "telldir":
            return {
                "html": "telldir",
                "tooltip": "telldir",
                "url": ""
            };

        case "tie":
            return {
                "html": "tie",
                "tooltip": "tie",
                "url": ""
            };

        case "tied":
            return {
                "html": "tied",
                "tooltip": "tied",
                "url": ""
            };

        case "time":
            return {
                "html": "time",
                "tooltip": "time",
                "url": ""
            };

        case "tms":
            return {
                "html": "times",
                "tooltip": "times",
                "url": ""
            };

        case "trans":
            return {
                "html": "transliteration (tr///)",
                "tooltip": "transliteration (tr///)",
                "url": ""
            };

        case "transr":
            return {
                "html": "transliteration (tr///)",
                "tooltip": "transliteration (tr///)",
                "url": ""
            };

        case "truncate":
            return {
                "html": "truncate",
                "tooltip": "truncate",
                "url": ""
            };

        case "uc":
            return {
                "html": "uc",
                "tooltip": "uc",
                "url": ""
            };

        case "ucfirst":
            return {
                "html": "ucfirst",
                "tooltip": "ucfirst",
                "url": ""
            };

        case "umask":
            return {
                "html": "umask",
                "tooltip": "umask",
                "url": ""
            };

        case "undef":
            return {
                "html": "undef operator",
                "tooltip": "undef operator",
                "url": ""
            };

        case "unlink":
            return {
                "html": "unlink",
                "tooltip": "unlink",
                "url": ""
            };

        case "unpack":
            return {
                "html": "unpack",
                "tooltip": "unpack",
                "url": ""
            };

        case "unshift":
            return {
                "html": "unshift",
                "tooltip": "unshift",
                "url": ""
            };

        case "unstack":
            return {
                "html": "iteration finalizer",
                "tooltip": "iteration finalizer",
                "url": ""
            };

        case "untie":
            return {
                "html": "untie",
                "tooltip": "untie",
                "url": ""
            };

        case "unweaken":
            return {
                "html": "reference unweaken",
                "tooltip": "reference unweaken",
                "url": ""
            };

        case "utime":
            return {
                "html": "utime",
                "tooltip": "utime",
                "url": ""
            };

        case "values":
            return {
                "html": "values",
                "tooltip": "values",
                "url": ""
            };

        case "vec":
            return {
                "html": "vec",
                "tooltip": "vec",
                "url": ""
            };

        case "wait":
            return {
                "html": "wait",
                "tooltip": "wait",
                "url": ""
            };

        case "waitpid":
            return {
                "html": "waitpid",
                "tooltip": "waitpid",
                "url": ""
            };

        case "wantarray":
            return {
                "html": "wantarray",
                "tooltip": "wantarray",
                "url": ""
            };

        case "warn":
            return {
                "html": "warn",
                "tooltip": "warn",
                "url": ""
            };

        case "weaken":
            return {
                "html": "reference weaken",
                "tooltip": "reference weaken",
                "url": ""
            };

        case "xor":
            return {
                "html": "logical xor",
                "tooltip": "logical xor",
                "url": ""
            };


    }
}
