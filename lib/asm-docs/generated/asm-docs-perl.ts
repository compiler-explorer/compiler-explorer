import type {AssemblyInstructionInfo} from '../../../types/assembly-docs.interfaces.js';

export function getAsmOpcode(opcode: string | undefined): AssemblyInstructionInfo | undefined {
    if (!opcode) return;
    switch (opcode.toLowerCase()) {
        case "aassign":
            return {
                "html": "<p>list assignment</p>",
                "tooltip": "list assignment",
                "url": ""
            };

        case "abs":
            return {
                "html": "<p>abs</p>",
                "tooltip": "abs",
                "url": ""
            };

        case "accept":
            return {
                "html": "<p>accept</p>",
                "tooltip": "accept",
                "url": ""
            };

        case "add":
            return {
                "html": "<p>addition (+)</p>",
                "tooltip": "addition (+)",
                "url": ""
            };

        case "aeach":
            return {
                "html": "<p>each on array</p>",
                "tooltip": "each on array",
                "url": ""
            };

        case "aelem":
            return {
                "html": "<p>array element</p>",
                "tooltip": "array element",
                "url": ""
            };

        case "aelemfast":
            return {
                "html": "<p>constant array element</p>",
                "tooltip": "constant array element",
                "url": ""
            };

        case "aelemfast_lex":
            return {
                "html": "<p>constant lexical array element</p>",
                "tooltip": "constant lexical array element",
                "url": ""
            };

        case "aelemfastlex_store":
            return {
                "html": "<p>const lexical array element store</p>",
                "tooltip": "const lexical array element store",
                "url": ""
            };

        case "akeys":
            return {
                "html": "<p>keys on array</p>",
                "tooltip": "keys on array",
                "url": ""
            };

        case "alarm":
            return {
                "html": "<p>alarm</p>",
                "tooltip": "alarm",
                "url": ""
            };

        case "allstart":
            return {
                "html": "<p>all</p>",
                "tooltip": "all",
                "url": ""
            };

        case "and":
            return {
                "html": "<p>logical and (&amp;&amp;)</p>",
                "tooltip": "logical and (&amp;&amp;)",
                "url": ""
            };

        case "andassign":
            return {
                "html": "<p>logical and assignment (&amp;&amp;=)</p>",
                "tooltip": "logical and assignment (&amp;&amp;=)",
                "url": ""
            };

        case "anoncode":
            return {
                "html": "<p>anonymous subroutine</p>",
                "tooltip": "anonymous subroutine",
                "url": ""
            };

        case "anonconst":
            return {
                "html": "<p>anonymous constant</p>",
                "tooltip": "anonymous constant",
                "url": ""
            };

        case "anonhash":
            return {
                "html": "<p>anonymous hash ({})</p>",
                "tooltip": "anonymous hash ({})",
                "url": ""
            };

        case "anonlist":
            return {
                "html": "<p>anonymous array ([])</p>",
                "tooltip": "anonymous array ([])",
                "url": ""
            };

        case "anystart":
            return {
                "html": "<p>any</p>",
                "tooltip": "any",
                "url": ""
            };

        case "anywhile":
            return {
                "html": "<p>any/all iterator</p>",
                "tooltip": "any/all iterator",
                "url": ""
            };

        case "argcheck":
            return {
                "html": "<p>check subroutine arguments</p>",
                "tooltip": "check subroutine arguments",
                "url": ""
            };

        case "argdefelem":
            return {
                "html": "<p>subroutine argument default value</p>",
                "tooltip": "subroutine argument default value",
                "url": ""
            };

        case "argelem":
            return {
                "html": "<p>subroutine argument</p>",
                "tooltip": "subroutine argument",
                "url": ""
            };

        case "aslice":
            return {
                "html": "<p>array slice</p>",
                "tooltip": "array slice",
                "url": ""
            };

        case "atan2":
            return {
                "html": "<p>atan2</p>",
                "tooltip": "atan2",
                "url": ""
            };

        case "av2arylen":
            return {
                "html": "<p>array length</p>",
                "tooltip": "array length",
                "url": ""
            };

        case "avalues":
            return {
                "html": "<p>values on array</p>",
                "tooltip": "values on array",
                "url": ""
            };

        case "avhvswitch":
            return {
                "html": "<p>Array/hash switch</p>",
                "tooltip": "Array/hash switch",
                "url": ""
            };

        case "backtick":
            return {
                "html": "<p>quoted execution (``, qx)</p>",
                "tooltip": "quoted execution (``, qx)",
                "url": ""
            };

        case "bind":
            return {
                "html": "<p>bind</p>",
                "tooltip": "bind",
                "url": ""
            };

        case "binmode":
            return {
                "html": "<p>binmode</p>",
                "tooltip": "binmode",
                "url": ""
            };

        case "bit_and":
            return {
                "html": "<p>bitwise and (&amp;)</p>",
                "tooltip": "bitwise and (&amp;)",
                "url": ""
            };

        case "bit_or":
            return {
                "html": "<p>bitwise or (|)</p>",
                "tooltip": "bitwise or (|)",
                "url": ""
            };

        case "bit_xor":
            return {
                "html": "<p>bitwise xor (^)</p>",
                "tooltip": "bitwise xor (^)",
                "url": ""
            };

        case "bless":
            return {
                "html": "<p>bless</p>",
                "tooltip": "bless",
                "url": ""
            };

        case "blessed":
            return {
                "html": "<p>blessed</p>",
                "tooltip": "blessed",
                "url": ""
            };

        case "break":
            return {
                "html": "<p>break</p>",
                "tooltip": "break",
                "url": ""
            };

        case "caller":
            return {
                "html": "<p>caller</p>",
                "tooltip": "caller",
                "url": ""
            };

        case "catch":
            return {
                "html": "<p>catch {} block</p>",
                "tooltip": "catch {} block",
                "url": ""
            };

        case "ceil":
            return {
                "html": "<p>ceil</p>",
                "tooltip": "ceil",
                "url": ""
            };

        case "chdir":
            return {
                "html": "<p>chdir</p>",
                "tooltip": "chdir",
                "url": ""
            };

        case "chmod":
            return {
                "html": "<p>chmod</p>",
                "tooltip": "chmod",
                "url": ""
            };

        case "chomp":
            return {
                "html": "<p>chomp</p>",
                "tooltip": "chomp",
                "url": ""
            };

        case "chop":
            return {
                "html": "<p>chop</p>",
                "tooltip": "chop",
                "url": ""
            };

        case "chown":
            return {
                "html": "<p>chown</p>",
                "tooltip": "chown",
                "url": ""
            };

        case "chr":
            return {
                "html": "<p>chr</p>",
                "tooltip": "chr",
                "url": ""
            };

        case "chroot":
            return {
                "html": "<p>chroot</p>",
                "tooltip": "chroot",
                "url": ""
            };

        case "classname":
            return {
                "html": "<p>class name</p>",
                "tooltip": "class name",
                "url": ""
            };

        case "clonecv":
            return {
                "html": "<p>private subroutine</p>",
                "tooltip": "private subroutine",
                "url": ""
            };

        case "close":
            return {
                "html": "<p>close</p>",
                "tooltip": "close",
                "url": ""
            };

        case "closedir":
            return {
                "html": "<p>closedir</p>",
                "tooltip": "closedir",
                "url": ""
            };

        case "cmpchain_and":
            return {
                "html": "<p>comparison chaining</p>",
                "tooltip": "comparison chaining",
                "url": ""
            };

        case "cmpchain_dup":
            return {
                "html": "<p>comparand shuffling</p>",
                "tooltip": "comparand shuffling",
                "url": ""
            };

        case "complement":
            return {
                "html": "<p>1&#x27;s complement (~)</p>",
                "tooltip": "1&#x27;s complement (~)",
                "url": ""
            };

        case "concat":
            return {
                "html": "<p>concatenation (.) or string</p>",
                "tooltip": "concatenation (.) or string",
                "url": ""
            };

        case "cond_expr":
            return {
                "html": "<p>conditional expression</p>",
                "tooltip": "conditional expression",
                "url": ""
            };

        case "connect":
            return {
                "html": "<p>connect</p>",
                "tooltip": "connect",
                "url": ""
            };

        case "const":
            return {
                "html": "<p>constant item</p>",
                "tooltip": "constant item",
                "url": ""
            };

        case "continue":
            return {
                "html": "<p>continue</p>",
                "tooltip": "continue",
                "url": ""
            };

        case "coreargs":
            return {
                "html": "<p>CORE:: subroutine</p>",
                "tooltip": "CORE:: subroutine",
                "url": ""
            };

        case "cos":
            return {
                "html": "<p>cos</p>",
                "tooltip": "cos",
                "url": ""
            };

        case "crypt":
            return {
                "html": "<p>crypt</p>",
                "tooltip": "crypt",
                "url": ""
            };

        case "custom":
            return {
                "html": "<p>unknown custom operator</p>",
                "tooltip": "unknown custom operator",
                "url": ""
            };

        case "dbmclose":
            return {
                "html": "<p>dbmclose</p>",
                "tooltip": "dbmclose",
                "url": ""
            };

        case "dbmopen":
            return {
                "html": "<p>dbmopen</p>",
                "tooltip": "dbmopen",
                "url": ""
            };

        case "dbstate":
            return {
                "html": "<p>debug next statement</p>",
                "tooltip": "debug next statement",
                "url": ""
            };

        case "defined":
            return {
                "html": "<p>defined operator</p>",
                "tooltip": "defined operator",
                "url": ""
            };

        case "delete":
            return {
                "html": "<p>delete</p>",
                "tooltip": "delete",
                "url": ""
            };

        case "die":
            return {
                "html": "<p>die</p>",
                "tooltip": "die",
                "url": ""
            };

        case "divide":
            return {
                "html": "<p>division (/)</p>",
                "tooltip": "division (/)",
                "url": ""
            };

        case "dofile":
            return {
                "html": "<p>do &quot;file&quot;</p>",
                "tooltip": "do &quot;file&quot;",
                "url": ""
            };

        case "dor":
            return {
                "html": "<p>defined or (//)</p>",
                "tooltip": "defined or (//)",
                "url": ""
            };

        case "dorassign":
            return {
                "html": "<p>defined or assignment (//=)</p>",
                "tooltip": "defined or assignment (//=)",
                "url": ""
            };

        case "dump":
            return {
                "html": "<p>dump</p>",
                "tooltip": "dump",
                "url": ""
            };

        case "each":
            return {
                "html": "<p>each</p>",
                "tooltip": "each",
                "url": ""
            };

        case "egrent":
            return {
                "html": "<p>endgrent</p>",
                "tooltip": "endgrent",
                "url": ""
            };

        case "ehostent":
            return {
                "html": "<p>endhostent</p>",
                "tooltip": "endhostent",
                "url": ""
            };

        case "emptyavhv":
            return {
                "html": "<p>empty anon hash/array</p>",
                "tooltip": "empty anon hash/array",
                "url": ""
            };

        case "enetent":
            return {
                "html": "<p>endnetent</p>",
                "tooltip": "endnetent",
                "url": ""
            };

        case "enter":
            return {
                "html": "<p>block entry</p>",
                "tooltip": "block entry",
                "url": ""
            };

        case "entereval":
            return {
                "html": "<p>eval &quot;string&quot;</p>",
                "tooltip": "eval &quot;string&quot;",
                "url": ""
            };

        case "entergiven":
            return {
                "html": "<p>given()</p>",
                "tooltip": "given()",
                "url": ""
            };

        case "enteriter":
            return {
                "html": "<p>foreach loop entry</p>",
                "tooltip": "foreach loop entry",
                "url": ""
            };

        case "enterloop":
            return {
                "html": "<p>loop entry</p>",
                "tooltip": "loop entry",
                "url": ""
            };

        case "entersub":
            return {
                "html": "<p>subroutine entry</p>",
                "tooltip": "subroutine entry",
                "url": ""
            };

        case "entertry":
            return {
                "html": "<p>eval {block}</p>",
                "tooltip": "eval {block}",
                "url": ""
            };

        case "entertrycatch":
            return {
                "html": "<p>try {block}</p>",
                "tooltip": "try {block}",
                "url": ""
            };

        case "enterwhen":
            return {
                "html": "<p>when()</p>",
                "tooltip": "when()",
                "url": ""
            };

        case "enterwrite":
            return {
                "html": "<p>write</p>",
                "tooltip": "write",
                "url": ""
            };

        case "eof":
            return {
                "html": "<p>eof</p>",
                "tooltip": "eof",
                "url": ""
            };

        case "eprotoent":
            return {
                "html": "<p>endprotoent</p>",
                "tooltip": "endprotoent",
                "url": ""
            };

        case "epwent":
            return {
                "html": "<p>endpwent</p>",
                "tooltip": "endpwent",
                "url": ""
            };

        case "eq":
            return {
                "html": "<p>numeric eq (==)</p>",
                "tooltip": "numeric eq (==)",
                "url": ""
            };

        case "eservent":
            return {
                "html": "<p>endservent</p>",
                "tooltip": "endservent",
                "url": ""
            };

        case "exec":
            return {
                "html": "<p>exec</p>",
                "tooltip": "exec",
                "url": ""
            };

        case "exists":
            return {
                "html": "<p>exists</p>",
                "tooltip": "exists",
                "url": ""
            };

        case "exit":
            return {
                "html": "<p>exit</p>",
                "tooltip": "exit",
                "url": ""
            };

        case "exp":
            return {
                "html": "<p>exp</p>",
                "tooltip": "exp",
                "url": ""
            };

        case "fc":
            return {
                "html": "<p>fc</p>",
                "tooltip": "fc",
                "url": ""
            };

        case "fcntl":
            return {
                "html": "<p>fcntl</p>",
                "tooltip": "fcntl",
                "url": ""
            };

        case "fileno":
            return {
                "html": "<p>fileno</p>",
                "tooltip": "fileno",
                "url": ""
            };

        case "flip":
            return {
                "html": "<p>range (or flip)</p>",
                "tooltip": "range (or flip)",
                "url": ""
            };

        case "flock":
            return {
                "html": "<p>flock</p>",
                "tooltip": "flock",
                "url": ""
            };

        case "floor":
            return {
                "html": "<p>floor</p>",
                "tooltip": "floor",
                "url": ""
            };

        case "flop":
            return {
                "html": "<p>range (or flop)</p>",
                "tooltip": "range (or flop)",
                "url": ""
            };

        case "fork":
            return {
                "html": "<p>fork</p>",
                "tooltip": "fork",
                "url": ""
            };

        case "formline":
            return {
                "html": "<p>formline</p>",
                "tooltip": "formline",
                "url": ""
            };

        case "ftatime":
            return {
                "html": "<p>-A</p>",
                "tooltip": "-A",
                "url": ""
            };

        case "ftbinary":
            return {
                "html": "<p>-B</p>",
                "tooltip": "-B",
                "url": ""
            };

        case "ftblk":
            return {
                "html": "<p>-b</p>",
                "tooltip": "-b",
                "url": ""
            };

        case "ftchr":
            return {
                "html": "<p>-c</p>",
                "tooltip": "-c",
                "url": ""
            };

        case "ftctime":
            return {
                "html": "<p>-C</p>",
                "tooltip": "-C",
                "url": ""
            };

        case "ftdir":
            return {
                "html": "<p>-d</p>",
                "tooltip": "-d",
                "url": ""
            };

        case "fteexec":
            return {
                "html": "<p>-x</p>",
                "tooltip": "-x",
                "url": ""
            };

        case "fteowned":
            return {
                "html": "<p>-o</p>",
                "tooltip": "-o",
                "url": ""
            };

        case "fteread":
            return {
                "html": "<p>-r</p>",
                "tooltip": "-r",
                "url": ""
            };

        case "ftewrite":
            return {
                "html": "<p>-w</p>",
                "tooltip": "-w",
                "url": ""
            };

        case "ftfile":
            return {
                "html": "<p>-f</p>",
                "tooltip": "-f",
                "url": ""
            };

        case "ftis":
            return {
                "html": "<p>-e</p>",
                "tooltip": "-e",
                "url": ""
            };

        case "ftlink":
            return {
                "html": "<p>-l</p>",
                "tooltip": "-l",
                "url": ""
            };

        case "ftmtime":
            return {
                "html": "<p>-M</p>",
                "tooltip": "-M",
                "url": ""
            };

        case "ftpipe":
            return {
                "html": "<p>-p</p>",
                "tooltip": "-p",
                "url": ""
            };

        case "ftrexec":
            return {
                "html": "<p>-X</p>",
                "tooltip": "-X",
                "url": ""
            };

        case "ftrowned":
            return {
                "html": "<p>-O</p>",
                "tooltip": "-O",
                "url": ""
            };

        case "ftrread":
            return {
                "html": "<p>-R</p>",
                "tooltip": "-R",
                "url": ""
            };

        case "ftrwrite":
            return {
                "html": "<p>-W</p>",
                "tooltip": "-W",
                "url": ""
            };

        case "ftsgid":
            return {
                "html": "<p>-g</p>",
                "tooltip": "-g",
                "url": ""
            };

        case "ftsize":
            return {
                "html": "<p>-s</p>",
                "tooltip": "-s",
                "url": ""
            };

        case "ftsock":
            return {
                "html": "<p>-S</p>",
                "tooltip": "-S",
                "url": ""
            };

        case "ftsuid":
            return {
                "html": "<p>-u</p>",
                "tooltip": "-u",
                "url": ""
            };

        case "ftsvtx":
            return {
                "html": "<p>-k</p>",
                "tooltip": "-k",
                "url": ""
            };

        case "fttext":
            return {
                "html": "<p>-T</p>",
                "tooltip": "-T",
                "url": ""
            };

        case "fttty":
            return {
                "html": "<p>-t</p>",
                "tooltip": "-t",
                "url": ""
            };

        case "ftzero":
            return {
                "html": "<p>-z</p>",
                "tooltip": "-z",
                "url": ""
            };

        case "ge":
            return {
                "html": "<p>numeric ge (&gt;=)</p>",
                "tooltip": "numeric ge (&gt;=)",
                "url": ""
            };

        case "gelem":
            return {
                "html": "<p>glob elem</p>",
                "tooltip": "glob elem",
                "url": ""
            };

        case "getc":
            return {
                "html": "<p>getc</p>",
                "tooltip": "getc",
                "url": ""
            };

        case "getlogin":
            return {
                "html": "<p>getlogin</p>",
                "tooltip": "getlogin",
                "url": ""
            };

        case "getpeername":
            return {
                "html": "<p>getpeername</p>",
                "tooltip": "getpeername",
                "url": ""
            };

        case "getpgrp":
            return {
                "html": "<p>getpgrp</p>",
                "tooltip": "getpgrp",
                "url": ""
            };

        case "getppid":
            return {
                "html": "<p>getppid</p>",
                "tooltip": "getppid",
                "url": ""
            };

        case "getpriority":
            return {
                "html": "<p>getpriority</p>",
                "tooltip": "getpriority",
                "url": ""
            };

        case "getsockname":
            return {
                "html": "<p>getsockname</p>",
                "tooltip": "getsockname",
                "url": ""
            };

        case "ggrent":
            return {
                "html": "<p>getgrent</p>",
                "tooltip": "getgrent",
                "url": ""
            };

        case "ggrgid":
            return {
                "html": "<p>getgrgid</p>",
                "tooltip": "getgrgid",
                "url": ""
            };

        case "ggrnam":
            return {
                "html": "<p>getgrnam</p>",
                "tooltip": "getgrnam",
                "url": ""
            };

        case "ghbyaddr":
            return {
                "html": "<p>gethostbyaddr</p>",
                "tooltip": "gethostbyaddr",
                "url": ""
            };

        case "ghbyname":
            return {
                "html": "<p>gethostbyname</p>",
                "tooltip": "gethostbyname",
                "url": ""
            };

        case "ghostent":
            return {
                "html": "<p>gethostent</p>",
                "tooltip": "gethostent",
                "url": ""
            };

        case "glob":
            return {
                "html": "<p>glob</p>",
                "tooltip": "glob",
                "url": ""
            };

        case "gmtime":
            return {
                "html": "<p>gmtime</p>",
                "tooltip": "gmtime",
                "url": ""
            };

        case "gnbyaddr":
            return {
                "html": "<p>getnetbyaddr</p>",
                "tooltip": "getnetbyaddr",
                "url": ""
            };

        case "gnbyname":
            return {
                "html": "<p>getnetbyname</p>",
                "tooltip": "getnetbyname",
                "url": ""
            };

        case "gnetent":
            return {
                "html": "<p>getnetent</p>",
                "tooltip": "getnetent",
                "url": ""
            };

        case "goto":
            return {
                "html": "<p>goto</p>",
                "tooltip": "goto",
                "url": ""
            };

        case "gpbyname":
            return {
                "html": "<p>getprotobyname</p>",
                "tooltip": "getprotobyname",
                "url": ""
            };

        case "gpbynumber":
            return {
                "html": "<p>getprotobynumber</p>",
                "tooltip": "getprotobynumber",
                "url": ""
            };

        case "gprotoent":
            return {
                "html": "<p>getprotoent</p>",
                "tooltip": "getprotoent",
                "url": ""
            };

        case "gpwent":
            return {
                "html": "<p>getpwent</p>",
                "tooltip": "getpwent",
                "url": ""
            };

        case "gpwnam":
            return {
                "html": "<p>getpwnam</p>",
                "tooltip": "getpwnam",
                "url": ""
            };

        case "gpwuid":
            return {
                "html": "<p>getpwuid</p>",
                "tooltip": "getpwuid",
                "url": ""
            };

        case "grepstart":
            return {
                "html": "<p>grep</p>",
                "tooltip": "grep",
                "url": ""
            };

        case "grepwhile":
            return {
                "html": "<p>grep iterator</p>",
                "tooltip": "grep iterator",
                "url": ""
            };

        case "gsbyname":
            return {
                "html": "<p>getservbyname</p>",
                "tooltip": "getservbyname",
                "url": ""
            };

        case "gsbyport":
            return {
                "html": "<p>getservbyport</p>",
                "tooltip": "getservbyport",
                "url": ""
            };

        case "gservent":
            return {
                "html": "<p>getservent</p>",
                "tooltip": "getservent",
                "url": ""
            };

        case "gsockopt":
            return {
                "html": "<p>getsockopt</p>",
                "tooltip": "getsockopt",
                "url": ""
            };

        case "gt":
            return {
                "html": "<p>numeric gt (&gt;)</p>",
                "tooltip": "numeric gt (&gt;)",
                "url": ""
            };

        case "gv":
            return {
                "html": "<p>glob value</p>",
                "tooltip": "glob value",
                "url": ""
            };

        case "gvsv":
            return {
                "html": "<p>scalar variable</p>",
                "tooltip": "scalar variable",
                "url": ""
            };

        case "helem":
            return {
                "html": "<p>hash element</p>",
                "tooltip": "hash element",
                "url": ""
            };

        case "helemexistsor":
            return {
                "html": "<p>hash element exists or</p>",
                "tooltip": "hash element exists or",
                "url": ""
            };

        case "hex":
            return {
                "html": "<p>hex</p>",
                "tooltip": "hex",
                "url": ""
            };

        case "hintseval":
            return {
                "html": "<p>eval hints</p>",
                "tooltip": "eval hints",
                "url": ""
            };

        case "hslice":
            return {
                "html": "<p>hash slice</p>",
                "tooltip": "hash slice",
                "url": ""
            };

        case "i_add":
            return {
                "html": "<p>integer addition (+)</p>",
                "tooltip": "integer addition (+)",
                "url": ""
            };

        case "i_divide":
            return {
                "html": "<p>integer division (/)</p>",
                "tooltip": "integer division (/)",
                "url": ""
            };

        case "i_eq":
            return {
                "html": "<p>integer eq (==)</p>",
                "tooltip": "integer eq (==)",
                "url": ""
            };

        case "i_ge":
            return {
                "html": "<p>integer ge (&gt;=)</p>",
                "tooltip": "integer ge (&gt;=)",
                "url": ""
            };

        case "i_gt":
            return {
                "html": "<p>integer gt (&gt;)</p>",
                "tooltip": "integer gt (&gt;)",
                "url": ""
            };

        case "i_le":
            return {
                "html": "<p>integer le (&lt;=)</p>",
                "tooltip": "integer le (&lt;=)",
                "url": ""
            };

        case "i_lt":
            return {
                "html": "<p>integer lt (&lt;)</p>",
                "tooltip": "integer lt (&lt;)",
                "url": ""
            };

        case "i_modulo":
            return {
                "html": "<p>integer modulus (%)</p>",
                "tooltip": "integer modulus (%)",
                "url": ""
            };

        case "i_multiply":
            return {
                "html": "<p>integer multiplication (*)</p>",
                "tooltip": "integer multiplication (*)",
                "url": ""
            };

        case "i_ncmp":
            return {
                "html": "<p>integer comparison (&lt;=&gt;)</p>",
                "tooltip": "integer comparison (&lt;=&gt;)",
                "url": ""
            };

        case "i_ne":
            return {
                "html": "<p>integer ne (!=)</p>",
                "tooltip": "integer ne (!=)",
                "url": ""
            };

        case "i_negate":
            return {
                "html": "<p>integer negation (-)</p>",
                "tooltip": "integer negation (-)",
                "url": ""
            };

        case "i_postdec":
            return {
                "html": "<p>integer postdecrement (--)</p>",
                "tooltip": "integer postdecrement (--)",
                "url": ""
            };

        case "i_postinc":
            return {
                "html": "<p>integer postincrement (++)</p>",
                "tooltip": "integer postincrement (++)",
                "url": ""
            };

        case "i_predec":
            return {
                "html": "<p>integer predecrement (--)</p>",
                "tooltip": "integer predecrement (--)",
                "url": ""
            };

        case "i_preinc":
            return {
                "html": "<p>integer preincrement (++)</p>",
                "tooltip": "integer preincrement (++)",
                "url": ""
            };

        case "i_subtract":
            return {
                "html": "<p>integer subtraction (-)</p>",
                "tooltip": "integer subtraction (-)",
                "url": ""
            };

        case "index":
            return {
                "html": "<p>index</p>",
                "tooltip": "index",
                "url": ""
            };

        case "initfield":
            return {
                "html": "<p>initialise field</p>",
                "tooltip": "initialise field",
                "url": ""
            };

        case "int":
            return {
                "html": "<p>int</p>",
                "tooltip": "int",
                "url": ""
            };

        case "introcv":
            return {
                "html": "<p>private subroutine</p>",
                "tooltip": "private subroutine",
                "url": ""
            };

        case "ioctl":
            return {
                "html": "<p>ioctl</p>",
                "tooltip": "ioctl",
                "url": ""
            };

        case "is_bool":
            return {
                "html": "<p>boolean type test</p>",
                "tooltip": "boolean type test",
                "url": ""
            };

        case "is_tainted":
            return {
                "html": "<p>is_tainted</p>",
                "tooltip": "is_tainted",
                "url": ""
            };

        case "is_weak":
            return {
                "html": "<p>weakref type test</p>",
                "tooltip": "weakref type test",
                "url": ""
            };

        case "isa":
            return {
                "html": "<p>derived class test</p>",
                "tooltip": "derived class test",
                "url": ""
            };

        case "iter":
            return {
                "html": "<p>foreach loop iterator</p>",
                "tooltip": "foreach loop iterator",
                "url": ""
            };

        case "join":
            return {
                "html": "<p>join or string</p>",
                "tooltip": "join or string",
                "url": ""
            };

        case "keys":
            return {
                "html": "<p>keys</p>",
                "tooltip": "keys",
                "url": ""
            };

        case "kill":
            return {
                "html": "<p>kill</p>",
                "tooltip": "kill",
                "url": ""
            };

        case "kvaslice":
            return {
                "html": "<p>index/value array slice</p>",
                "tooltip": "index/value array slice",
                "url": ""
            };

        case "kvhslice":
            return {
                "html": "<p>key/value hash slice</p>",
                "tooltip": "key/value hash slice",
                "url": ""
            };

        case "last":
            return {
                "html": "<p>last</p>",
                "tooltip": "last",
                "url": ""
            };

        case "lc":
            return {
                "html": "<p>lc</p>",
                "tooltip": "lc",
                "url": ""
            };

        case "lcfirst":
            return {
                "html": "<p>lcfirst</p>",
                "tooltip": "lcfirst",
                "url": ""
            };

        case "le":
            return {
                "html": "<p>numeric le (&lt;=)</p>",
                "tooltip": "numeric le (&lt;=)",
                "url": ""
            };

        case "leave":
            return {
                "html": "<p>block exit</p>",
                "tooltip": "block exit",
                "url": ""
            };

        case "leaveeval":
            return {
                "html": "<p>eval &quot;string&quot; exit</p>",
                "tooltip": "eval &quot;string&quot; exit",
                "url": ""
            };

        case "leavegiven":
            return {
                "html": "<p>leave given block</p>",
                "tooltip": "leave given block",
                "url": ""
            };

        case "leaveloop":
            return {
                "html": "<p>loop exit</p>",
                "tooltip": "loop exit",
                "url": ""
            };

        case "leavesub":
            return {
                "html": "<p>subroutine exit</p>",
                "tooltip": "subroutine exit",
                "url": ""
            };

        case "leavesublv":
            return {
                "html": "<p>lvalue subroutine return</p>",
                "tooltip": "lvalue subroutine return",
                "url": ""
            };

        case "leavetry":
            return {
                "html": "<p>eval {block} exit</p>",
                "tooltip": "eval {block} exit",
                "url": ""
            };

        case "leavetrycatch":
            return {
                "html": "<p>try {block} exit</p>",
                "tooltip": "try {block} exit",
                "url": ""
            };

        case "leavewhen":
            return {
                "html": "<p>leave when block</p>",
                "tooltip": "leave when block",
                "url": ""
            };

        case "leavewrite":
            return {
                "html": "<p>write exit</p>",
                "tooltip": "write exit",
                "url": ""
            };

        case "left_shift":
            return {
                "html": "<p>left bitshift (&lt;&lt;)</p>",
                "tooltip": "left bitshift (&lt;&lt;)",
                "url": ""
            };

        case "length":
            return {
                "html": "<p>length</p>",
                "tooltip": "length",
                "url": ""
            };

        case "lineseq":
            return {
                "html": "<p>line sequence</p>",
                "tooltip": "line sequence",
                "url": ""
            };

        case "link":
            return {
                "html": "<p>link</p>",
                "tooltip": "link",
                "url": ""
            };

        case "list":
            return {
                "html": "<p>list</p>",
                "tooltip": "list",
                "url": ""
            };

        case "listen":
            return {
                "html": "<p>listen</p>",
                "tooltip": "listen",
                "url": ""
            };

        case "localtime":
            return {
                "html": "<p>localtime</p>",
                "tooltip": "localtime",
                "url": ""
            };

        case "lock":
            return {
                "html": "<p>lock</p>",
                "tooltip": "lock",
                "url": ""
            };

        case "log":
            return {
                "html": "<p>log</p>",
                "tooltip": "log",
                "url": ""
            };

        case "lslice":
            return {
                "html": "<p>list slice</p>",
                "tooltip": "list slice",
                "url": ""
            };

        case "lstat":
            return {
                "html": "<p>lstat</p>",
                "tooltip": "lstat",
                "url": ""
            };

        case "lt":
            return {
                "html": "<p>numeric lt (&lt;)</p>",
                "tooltip": "numeric lt (&lt;)",
                "url": ""
            };

        case "lvavref":
            return {
                "html": "<p>lvalue array reference</p>",
                "tooltip": "lvalue array reference",
                "url": ""
            };

        case "lvref":
            return {
                "html": "<p>lvalue ref assignment</p>",
                "tooltip": "lvalue ref assignment",
                "url": ""
            };

        case "lvrefslice":
            return {
                "html": "<p>lvalue ref assignment</p>",
                "tooltip": "lvalue ref assignment",
                "url": ""
            };

        case "mapstart":
            return {
                "html": "<p>map</p>",
                "tooltip": "map",
                "url": ""
            };

        case "mapwhile":
            return {
                "html": "<p>map iterator</p>",
                "tooltip": "map iterator",
                "url": ""
            };

        case "match":
            return {
                "html": "<p>pattern match (m//)</p>",
                "tooltip": "pattern match (m//)",
                "url": ""
            };

        case "method":
            return {
                "html": "<p>method lookup</p>",
                "tooltip": "method lookup",
                "url": ""
            };

        case "method_named":
            return {
                "html": "<p>method with known name</p>",
                "tooltip": "method with known name",
                "url": ""
            };

        case "method_redir":
            return {
                "html": "<p>redirect method with known name</p>",
                "tooltip": "redirect method with known name",
                "url": ""
            };

        case "method_redir_super":
            return {
                "html": "<p>redirect super method with known name</p>",
                "tooltip": "redirect super method with known name",
                "url": ""
            };

        case "method_super":
            return {
                "html": "<p>super with known name</p>",
                "tooltip": "super with known name",
                "url": ""
            };

        case "methstart":
            return {
                "html": "<p>method start</p>",
                "tooltip": "method start",
                "url": ""
            };

        case "mkdir":
            return {
                "html": "<p>mkdir</p>",
                "tooltip": "mkdir",
                "url": ""
            };

        case "modulo":
            return {
                "html": "<p>modulus (%)</p>",
                "tooltip": "modulus (%)",
                "url": ""
            };

        case "msgctl":
            return {
                "html": "<p>msgctl</p>",
                "tooltip": "msgctl",
                "url": ""
            };

        case "msgget":
            return {
                "html": "<p>msgget</p>",
                "tooltip": "msgget",
                "url": ""
            };

        case "msgrcv":
            return {
                "html": "<p>msgrcv</p>",
                "tooltip": "msgrcv",
                "url": ""
            };

        case "msgsnd":
            return {
                "html": "<p>msgsnd</p>",
                "tooltip": "msgsnd",
                "url": ""
            };

        case "multiconcat":
            return {
                "html": "<p>concatenation (.) or string</p>",
                "tooltip": "concatenation (.) or string",
                "url": ""
            };

        case "multideref":
            return {
                "html": "<p>array or hash lookup</p>",
                "tooltip": "array or hash lookup",
                "url": ""
            };

        case "multiparam":
            return {
                "html": "<p>signature processing</p>",
                "tooltip": "signature processing",
                "url": ""
            };

        case "multiply":
            return {
                "html": "<p>multiplication (*)</p>",
                "tooltip": "multiplication (*)",
                "url": ""
            };

        case "nbit_and":
            return {
                "html": "<p>numeric bitwise and (&amp;)</p>",
                "tooltip": "numeric bitwise and (&amp;)",
                "url": ""
            };

        case "nbit_or":
            return {
                "html": "<p>numeric bitwise or (|)</p>",
                "tooltip": "numeric bitwise or (|)",
                "url": ""
            };

        case "nbit_xor":
            return {
                "html": "<p>numeric bitwise xor (^)</p>",
                "tooltip": "numeric bitwise xor (^)",
                "url": ""
            };

        case "ncmp":
            return {
                "html": "<p>numeric comparison (&lt;=&gt;)</p>",
                "tooltip": "numeric comparison (&lt;=&gt;)",
                "url": ""
            };

        case "ncomplement":
            return {
                "html": "<p>numeric 1&#x27;s complement (~)</p>",
                "tooltip": "numeric 1&#x27;s complement (~)",
                "url": ""
            };

        case "ne":
            return {
                "html": "<p>numeric ne (!=)</p>",
                "tooltip": "numeric ne (!=)",
                "url": ""
            };

        case "negate":
            return {
                "html": "<p>negation (-)</p>",
                "tooltip": "negation (-)",
                "url": ""
            };

        case "next":
            return {
                "html": "<p>next</p>",
                "tooltip": "next",
                "url": ""
            };

        case "nextstate":
            return {
                "html": "<p>next statement</p>",
                "tooltip": "next statement",
                "url": ""
            };

        case "not":
            return {
                "html": "<p>not</p>",
                "tooltip": "not",
                "url": ""
            };

        case "null":
            return {
                "html": "<p>null operation</p>",
                "tooltip": "null operation",
                "url": ""
            };

        case "oct":
            return {
                "html": "<p>oct</p>",
                "tooltip": "oct",
                "url": ""
            };

        case "once":
            return {
                "html": "<p>once</p>",
                "tooltip": "once",
                "url": ""
            };

        case "open":
            return {
                "html": "<p>open</p>",
                "tooltip": "open",
                "url": ""
            };

        case "open_dir":
            return {
                "html": "<p>opendir</p>",
                "tooltip": "opendir",
                "url": ""
            };

        case "or":
            return {
                "html": "<p>logical or (||)</p>",
                "tooltip": "logical or (||)",
                "url": ""
            };

        case "orassign":
            return {
                "html": "<p>logical or assignment (||=)</p>",
                "tooltip": "logical or assignment (||=)",
                "url": ""
            };

        case "ord":
            return {
                "html": "<p>ord</p>",
                "tooltip": "ord",
                "url": ""
            };

        case "pack":
            return {
                "html": "<p>pack</p>",
                "tooltip": "pack",
                "url": ""
            };

        case "padany":
            return {
                "html": "<p>private value</p>",
                "tooltip": "private value",
                "url": ""
            };

        case "padav":
            return {
                "html": "<p>private array</p>",
                "tooltip": "private array",
                "url": ""
            };

        case "padcv":
            return {
                "html": "<p>private subroutine</p>",
                "tooltip": "private subroutine",
                "url": ""
            };

        case "padhv":
            return {
                "html": "<p>private hash</p>",
                "tooltip": "private hash",
                "url": ""
            };

        case "padrange":
            return {
                "html": "<p>list of private variables</p>",
                "tooltip": "list of private variables",
                "url": ""
            };

        case "padsv":
            return {
                "html": "<p>private variable</p>",
                "tooltip": "private variable",
                "url": ""
            };

        case "padsv_store":
            return {
                "html": "<p>padsv scalar assignment</p>",
                "tooltip": "padsv scalar assignment",
                "url": ""
            };

        case "paramstore":
            return {
                "html": "<p>signature parameter default expression</p>",
                "tooltip": "signature parameter default expression",
                "url": ""
            };

        case "paramtest":
            return {
                "html": "<p>signature argument value test</p>",
                "tooltip": "signature argument value test",
                "url": ""
            };

        case "pipe_op":
            return {
                "html": "<p>pipe</p>",
                "tooltip": "pipe",
                "url": ""
            };

        case "pop":
            return {
                "html": "<p>pop</p>",
                "tooltip": "pop",
                "url": ""
            };

        case "poptry":
            return {
                "html": "<p>pop try</p>",
                "tooltip": "pop try",
                "url": ""
            };

        case "pos":
            return {
                "html": "<p>match position</p>",
                "tooltip": "match position",
                "url": ""
            };

        case "postdec":
            return {
                "html": "<p>postdecrement (--)</p>",
                "tooltip": "postdecrement (--)",
                "url": ""
            };

        case "postinc":
            return {
                "html": "<p>postincrement (++)</p>",
                "tooltip": "postincrement (++)",
                "url": ""
            };

        case "pow":
            return {
                "html": "<p>exponentiation (**)</p>",
                "tooltip": "exponentiation (**)",
                "url": ""
            };

        case "predec":
            return {
                "html": "<p>predecrement (--)</p>",
                "tooltip": "predecrement (--)",
                "url": ""
            };

        case "preinc":
            return {
                "html": "<p>preincrement (++)</p>",
                "tooltip": "preincrement (++)",
                "url": ""
            };

        case "print":
            return {
                "html": "<p>print</p>",
                "tooltip": "print",
                "url": ""
            };

        case "prototype":
            return {
                "html": "<p>subroutine prototype</p>",
                "tooltip": "subroutine prototype",
                "url": ""
            };

        case "prtf":
            return {
                "html": "<p>printf</p>",
                "tooltip": "printf",
                "url": ""
            };

        case "push":
            return {
                "html": "<p>push</p>",
                "tooltip": "push",
                "url": ""
            };

        case "pushdefer":
            return {
                "html": "<p>push defer {} block</p>",
                "tooltip": "push defer {} block",
                "url": ""
            };

        case "pushmark":
            return {
                "html": "<p>pushmark</p>",
                "tooltip": "pushmark",
                "url": ""
            };

        case "qr":
            return {
                "html": "<p>pattern quote (qr//)</p>",
                "tooltip": "pattern quote (qr//)",
                "url": ""
            };

        case "quotemeta":
            return {
                "html": "<p>quotemeta</p>",
                "tooltip": "quotemeta",
                "url": ""
            };

        case "rand":
            return {
                "html": "<p>rand</p>",
                "tooltip": "rand",
                "url": ""
            };

        case "range":
            return {
                "html": "<p>flipflop</p>",
                "tooltip": "flipflop",
                "url": ""
            };

        case "rcatline":
            return {
                "html": "<p>append I/O operator</p>",
                "tooltip": "append I/O operator",
                "url": ""
            };

        case "read":
            return {
                "html": "<p>read</p>",
                "tooltip": "read",
                "url": ""
            };

        case "readdir":
            return {
                "html": "<p>readdir</p>",
                "tooltip": "readdir",
                "url": ""
            };

        case "readline":
            return {
                "html": "<p>&lt;HANDLE&gt;</p>",
                "tooltip": "&lt;HANDLE&gt;",
                "url": ""
            };

        case "readlink":
            return {
                "html": "<p>readlink</p>",
                "tooltip": "readlink",
                "url": ""
            };

        case "recv":
            return {
                "html": "<p>recv</p>",
                "tooltip": "recv",
                "url": ""
            };

        case "redo":
            return {
                "html": "<p>redo</p>",
                "tooltip": "redo",
                "url": ""
            };

        case "ref":
            return {
                "html": "<p>reference-type operator</p>",
                "tooltip": "reference-type operator",
                "url": ""
            };

        case "refaddr":
            return {
                "html": "<p>refaddr</p>",
                "tooltip": "refaddr",
                "url": ""
            };

        case "refassign":
            return {
                "html": "<p>lvalue ref assignment</p>",
                "tooltip": "lvalue ref assignment",
                "url": ""
            };

        case "refgen":
            return {
                "html": "<p>reference constructor</p>",
                "tooltip": "reference constructor",
                "url": ""
            };

        case "reftype":
            return {
                "html": "<p>reftype</p>",
                "tooltip": "reftype",
                "url": ""
            };

        case "regcmaybe":
            return {
                "html": "<p>regexp internal guard</p>",
                "tooltip": "regexp internal guard",
                "url": ""
            };

        case "regcomp":
            return {
                "html": "<p>regexp compilation</p>",
                "tooltip": "regexp compilation",
                "url": ""
            };

        case "regcreset":
            return {
                "html": "<p>regexp internal reset</p>",
                "tooltip": "regexp internal reset",
                "url": ""
            };

        case "rename":
            return {
                "html": "<p>rename</p>",
                "tooltip": "rename",
                "url": ""
            };

        case "repeat":
            return {
                "html": "<p>repeat (x)</p>",
                "tooltip": "repeat (x)",
                "url": ""
            };

        case "require":
            return {
                "html": "<p>require</p>",
                "tooltip": "require",
                "url": ""
            };

        case "reset":
            return {
                "html": "<p>symbol reset</p>",
                "tooltip": "symbol reset",
                "url": ""
            };

        case "return":
            return {
                "html": "<p>return</p>",
                "tooltip": "return",
                "url": ""
            };

        case "reverse":
            return {
                "html": "<p>reverse</p>",
                "tooltip": "reverse",
                "url": ""
            };

        case "rewinddir":
            return {
                "html": "<p>rewinddir</p>",
                "tooltip": "rewinddir",
                "url": ""
            };

        case "right_shift":
            return {
                "html": "<p>right bitshift (&gt;&gt;)</p>",
                "tooltip": "right bitshift (&gt;&gt;)",
                "url": ""
            };

        case "rindex":
            return {
                "html": "<p>rindex</p>",
                "tooltip": "rindex",
                "url": ""
            };

        case "rmdir":
            return {
                "html": "<p>rmdir</p>",
                "tooltip": "rmdir",
                "url": ""
            };

        case "runcv":
            return {
                "html": "<p>__SUB__</p>",
                "tooltip": "__SUB__",
                "url": ""
            };

        case "rv2av":
            return {
                "html": "<p>array dereference</p>",
                "tooltip": "array dereference",
                "url": ""
            };

        case "rv2cv":
            return {
                "html": "<p>subroutine dereference</p>",
                "tooltip": "subroutine dereference",
                "url": ""
            };

        case "rv2gv":
            return {
                "html": "<p>ref-to-glob cast</p>",
                "tooltip": "ref-to-glob cast",
                "url": ""
            };

        case "rv2hv":
            return {
                "html": "<p>hash dereference</p>",
                "tooltip": "hash dereference",
                "url": ""
            };

        case "rv2sv":
            return {
                "html": "<p>scalar dereference</p>",
                "tooltip": "scalar dereference",
                "url": ""
            };

        case "sassign":
            return {
                "html": "<p>scalar assignment</p>",
                "tooltip": "scalar assignment",
                "url": ""
            };

        case "say":
            return {
                "html": "<p>say</p>",
                "tooltip": "say",
                "url": ""
            };

        case "sbit_and":
            return {
                "html": "<p>string bitwise and (&amp;.)</p>",
                "tooltip": "string bitwise and (&amp;.)",
                "url": ""
            };

        case "sbit_or":
            return {
                "html": "<p>string bitwise or (|.)</p>",
                "tooltip": "string bitwise or (|.)",
                "url": ""
            };

        case "sbit_xor":
            return {
                "html": "<p>string bitwise xor (^.)</p>",
                "tooltip": "string bitwise xor (^.)",
                "url": ""
            };

        case "scalar":
            return {
                "html": "<p>scalar</p>",
                "tooltip": "scalar",
                "url": ""
            };

        case "schomp":
            return {
                "html": "<p>scalar chomp</p>",
                "tooltip": "scalar chomp",
                "url": ""
            };

        case "schop":
            return {
                "html": "<p>scalar chop</p>",
                "tooltip": "scalar chop",
                "url": ""
            };

        case "scmp":
            return {
                "html": "<p>string comparison (cmp)</p>",
                "tooltip": "string comparison (cmp)",
                "url": ""
            };

        case "scomplement":
            return {
                "html": "<p>string 1&#x27;s complement (~)</p>",
                "tooltip": "string 1&#x27;s complement (~)",
                "url": ""
            };

        case "scope":
            return {
                "html": "<p>block</p>",
                "tooltip": "block",
                "url": ""
            };

        case "seek":
            return {
                "html": "<p>seek</p>",
                "tooltip": "seek",
                "url": ""
            };

        case "seekdir":
            return {
                "html": "<p>seekdir</p>",
                "tooltip": "seekdir",
                "url": ""
            };

        case "select":
            return {
                "html": "<p>select</p>",
                "tooltip": "select",
                "url": ""
            };

        case "semctl":
            return {
                "html": "<p>semctl</p>",
                "tooltip": "semctl",
                "url": ""
            };

        case "semget":
            return {
                "html": "<p>semget</p>",
                "tooltip": "semget",
                "url": ""
            };

        case "semop":
            return {
                "html": "<p>semop</p>",
                "tooltip": "semop",
                "url": ""
            };

        case "send":
            return {
                "html": "<p>send</p>",
                "tooltip": "send",
                "url": ""
            };

        case "seq":
            return {
                "html": "<p>string eq</p>",
                "tooltip": "string eq",
                "url": ""
            };

        case "setpgrp":
            return {
                "html": "<p>setpgrp</p>",
                "tooltip": "setpgrp",
                "url": ""
            };

        case "setpriority":
            return {
                "html": "<p>setpriority</p>",
                "tooltip": "setpriority",
                "url": ""
            };

        case "sge":
            return {
                "html": "<p>string ge</p>",
                "tooltip": "string ge",
                "url": ""
            };

        case "sgrent":
            return {
                "html": "<p>setgrent</p>",
                "tooltip": "setgrent",
                "url": ""
            };

        case "sgt":
            return {
                "html": "<p>string gt</p>",
                "tooltip": "string gt",
                "url": ""
            };

        case "shift":
            return {
                "html": "<p>shift</p>",
                "tooltip": "shift",
                "url": ""
            };

        case "shmctl":
            return {
                "html": "<p>shmctl</p>",
                "tooltip": "shmctl",
                "url": ""
            };

        case "shmget":
            return {
                "html": "<p>shmget</p>",
                "tooltip": "shmget",
                "url": ""
            };

        case "shmread":
            return {
                "html": "<p>shmread</p>",
                "tooltip": "shmread",
                "url": ""
            };

        case "shmwrite":
            return {
                "html": "<p>shmwrite</p>",
                "tooltip": "shmwrite",
                "url": ""
            };

        case "shostent":
            return {
                "html": "<p>sethostent</p>",
                "tooltip": "sethostent",
                "url": ""
            };

        case "shutdown":
            return {
                "html": "<p>shutdown</p>",
                "tooltip": "shutdown",
                "url": ""
            };

        case "sin":
            return {
                "html": "<p>sin</p>",
                "tooltip": "sin",
                "url": ""
            };

        case "sle":
            return {
                "html": "<p>string le</p>",
                "tooltip": "string le",
                "url": ""
            };

        case "sleep":
            return {
                "html": "<p>sleep</p>",
                "tooltip": "sleep",
                "url": ""
            };

        case "slt":
            return {
                "html": "<p>string lt</p>",
                "tooltip": "string lt",
                "url": ""
            };

        case "smartmatch":
            return {
                "html": "<p>smart match</p>",
                "tooltip": "smart match",
                "url": ""
            };

        case "sne":
            return {
                "html": "<p>string ne</p>",
                "tooltip": "string ne",
                "url": ""
            };

        case "snetent":
            return {
                "html": "<p>setnetent</p>",
                "tooltip": "setnetent",
                "url": ""
            };

        case "socket":
            return {
                "html": "<p>socket</p>",
                "tooltip": "socket",
                "url": ""
            };

        case "sockpair":
            return {
                "html": "<p>socketpair</p>",
                "tooltip": "socketpair",
                "url": ""
            };

        case "sort":
            return {
                "html": "<p>sort</p>",
                "tooltip": "sort",
                "url": ""
            };

        case "splice":
            return {
                "html": "<p>splice</p>",
                "tooltip": "splice",
                "url": ""
            };

        case "split":
            return {
                "html": "<p>split</p>",
                "tooltip": "split",
                "url": ""
            };

        case "sprintf":
            return {
                "html": "<p>sprintf</p>",
                "tooltip": "sprintf",
                "url": ""
            };

        case "sprotoent":
            return {
                "html": "<p>setprotoent</p>",
                "tooltip": "setprotoent",
                "url": ""
            };

        case "spwent":
            return {
                "html": "<p>setpwent</p>",
                "tooltip": "setpwent",
                "url": ""
            };

        case "sqrt":
            return {
                "html": "<p>sqrt</p>",
                "tooltip": "sqrt",
                "url": ""
            };

        case "srand":
            return {
                "html": "<p>srand</p>",
                "tooltip": "srand",
                "url": ""
            };

        case "srefgen":
            return {
                "html": "<p>single ref constructor</p>",
                "tooltip": "single ref constructor",
                "url": ""
            };

        case "sselect":
            return {
                "html": "<p>select system call</p>",
                "tooltip": "select system call",
                "url": ""
            };

        case "sservent":
            return {
                "html": "<p>setservent</p>",
                "tooltip": "setservent",
                "url": ""
            };

        case "ssockopt":
            return {
                "html": "<p>setsockopt</p>",
                "tooltip": "setsockopt",
                "url": ""
            };

        case "stat":
            return {
                "html": "<p>stat</p>",
                "tooltip": "stat",
                "url": ""
            };

        case "stringify":
            return {
                "html": "<p>string</p>",
                "tooltip": "string",
                "url": ""
            };

        case "stub":
            return {
                "html": "<p>stub</p>",
                "tooltip": "stub",
                "url": ""
            };

        case "study":
            return {
                "html": "<p>study</p>",
                "tooltip": "study",
                "url": ""
            };

        case "subst":
            return {
                "html": "<p>substitution (s///)</p>",
                "tooltip": "substitution (s///)",
                "url": ""
            };

        case "substcont":
            return {
                "html": "<p>substitution iterator</p>",
                "tooltip": "substitution iterator",
                "url": ""
            };

        case "substr":
            return {
                "html": "<p>substr</p>",
                "tooltip": "substr",
                "url": ""
            };

        case "substr_left":
            return {
                "html": "<p>substr left</p>",
                "tooltip": "substr left",
                "url": ""
            };

        case "subtract":
            return {
                "html": "<p>subtraction (-)</p>",
                "tooltip": "subtraction (-)",
                "url": ""
            };

        case "symlink":
            return {
                "html": "<p>symlink</p>",
                "tooltip": "symlink",
                "url": ""
            };

        case "syscall":
            return {
                "html": "<p>syscall</p>",
                "tooltip": "syscall",
                "url": ""
            };

        case "sysopen":
            return {
                "html": "<p>sysopen</p>",
                "tooltip": "sysopen",
                "url": ""
            };

        case "sysread":
            return {
                "html": "<p>sysread</p>",
                "tooltip": "sysread",
                "url": ""
            };

        case "sysseek":
            return {
                "html": "<p>sysseek</p>",
                "tooltip": "sysseek",
                "url": ""
            };

        case "system":
            return {
                "html": "<p>system</p>",
                "tooltip": "system",
                "url": ""
            };

        case "syswrite":
            return {
                "html": "<p>syswrite</p>",
                "tooltip": "syswrite",
                "url": ""
            };

        case "tell":
            return {
                "html": "<p>tell</p>",
                "tooltip": "tell",
                "url": ""
            };

        case "telldir":
            return {
                "html": "<p>telldir</p>",
                "tooltip": "telldir",
                "url": ""
            };

        case "tie":
            return {
                "html": "<p>tie</p>",
                "tooltip": "tie",
                "url": ""
            };

        case "tied":
            return {
                "html": "<p>tied</p>",
                "tooltip": "tied",
                "url": ""
            };

        case "time":
            return {
                "html": "<p>time</p>",
                "tooltip": "time",
                "url": ""
            };

        case "tms":
            return {
                "html": "<p>times</p>",
                "tooltip": "times",
                "url": ""
            };

        case "trans":
            return {
                "html": "<p>transliteration (tr///)</p>",
                "tooltip": "transliteration (tr///)",
                "url": ""
            };

        case "transr":
            return {
                "html": "<p>transliteration (tr///)</p>",
                "tooltip": "transliteration (tr///)",
                "url": ""
            };

        case "truncate":
            return {
                "html": "<p>truncate</p>",
                "tooltip": "truncate",
                "url": ""
            };

        case "uc":
            return {
                "html": "<p>uc</p>",
                "tooltip": "uc",
                "url": ""
            };

        case "ucfirst":
            return {
                "html": "<p>ucfirst</p>",
                "tooltip": "ucfirst",
                "url": ""
            };

        case "umask":
            return {
                "html": "<p>umask</p>",
                "tooltip": "umask",
                "url": ""
            };

        case "undef":
            return {
                "html": "<p>undef operator</p>",
                "tooltip": "undef operator",
                "url": ""
            };

        case "unlink":
            return {
                "html": "<p>unlink</p>",
                "tooltip": "unlink",
                "url": ""
            };

        case "unpack":
            return {
                "html": "<p>unpack</p>",
                "tooltip": "unpack",
                "url": ""
            };

        case "unshift":
            return {
                "html": "<p>unshift</p>",
                "tooltip": "unshift",
                "url": ""
            };

        case "unstack":
            return {
                "html": "<p>iteration finalizer</p>",
                "tooltip": "iteration finalizer",
                "url": ""
            };

        case "untie":
            return {
                "html": "<p>untie</p>",
                "tooltip": "untie",
                "url": ""
            };

        case "unweaken":
            return {
                "html": "<p>reference unweaken</p>",
                "tooltip": "reference unweaken",
                "url": ""
            };

        case "utime":
            return {
                "html": "<p>utime</p>",
                "tooltip": "utime",
                "url": ""
            };

        case "values":
            return {
                "html": "<p>values</p>",
                "tooltip": "values",
                "url": ""
            };

        case "vec":
            return {
                "html": "<p>vec</p>",
                "tooltip": "vec",
                "url": ""
            };

        case "wait":
            return {
                "html": "<p>wait</p>",
                "tooltip": "wait",
                "url": ""
            };

        case "waitpid":
            return {
                "html": "<p>waitpid</p>",
                "tooltip": "waitpid",
                "url": ""
            };

        case "wantarray":
            return {
                "html": "<p>wantarray</p>",
                "tooltip": "wantarray",
                "url": ""
            };

        case "warn":
            return {
                "html": "<p>warn</p>",
                "tooltip": "warn",
                "url": ""
            };

        case "weaken":
            return {
                "html": "<p>reference weaken</p>",
                "tooltip": "reference weaken",
                "url": ""
            };

        case "xor":
            return {
                "html": "<p>logical xor</p>",
                "tooltip": "logical xor",
                "url": ""
            };


    }
}
