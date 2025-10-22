import type {AssemblyInstructionInfo} from '../../../types/assembly-docs.interfaces.js';

    export function getAsmOpcode(opcode: string | undefined): AssemblyInstructionInfo | undefined {
        if (!opcode) return;
        switch (opcode) {
            case "@":
            return {
                "html": "<p>Execute an instruction or instruction block for threads that have the guard predicate\n<code class=\"docutils literal notranslate\"><span class=\"pre\">True</span></code>. Threads with a <code class=\"docutils literal notranslate\"><span class=\"pre\">False</span></code> guard predicate do nothing.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#control-flow-instructions-at\" target=\"_blank\" rel=\"noopener noreferrer\">@ <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Predicated execution.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#control-flow-instructions-at"
            };

        case "abs":
            return {
                "html": "<p>Take the absolute value of <code class=\"docutils literal notranslate\"><span class=\"pre\">a</span></code> and store it in <code class=\"docutils literal notranslate\"><span class=\"pre\">d</span></code>.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-abs\" target=\"_blank\" rel=\"noopener noreferrer\">abs(int) <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Absolute value.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-abs"
            };

        case "activemask":
            return {
                "html": "<p><code class=\"docutils literal notranslate\"><span class=\"pre\">activemask</span></code> queries predicated-on active threads from the executing warp and sets the destination\n<code class=\"docutils literal notranslate\"><span class=\"pre\">d</span></code> with 32-bit integer mask where bit position in the mask corresponds to the thread\u2019s\n<code class=\"docutils literal notranslate\"><span class=\"pre\">laneid</span></code>.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-activemask\" target=\"_blank\" rel=\"noopener noreferrer\">activemask <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Queries the active threads within a warp.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-activemask"
            };

        case "add":
            return {
                "html": "<p>Performs addition and writes the resulting value into a destination register.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-add\" target=\"_blank\" rel=\"noopener noreferrer\">add(int) <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Add two values.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-add"
            };

        case "addc":
            return {
                "html": "<p>Performs integer addition with carry-in and optionally writes the carry-out value into the condition\ncode register.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#extended-precision-arithmetic-instructions-addc\" target=\"_blank\" rel=\"noopener noreferrer\">addc <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Add two values with carry-in and optional carry-out.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#extended-precision-arithmetic-instructions-addc"
            };

        case "alloca":
            return {
                "html": "<p>The <code class=\"docutils literal notranslate\"><span class=\"pre\">alloca</span></code> instruction dynamically allocates memory on the stack frame of the current function\nand updates the stack pointer accordingly. The returned pointer <code class=\"docutils literal notranslate\"><span class=\"pre\">ptr</span></code> points to local memory and\ncan be used in the address operand of <code class=\"docutils literal notranslate\"><span class=\"pre\">ld.local</span></code> and <code class=\"docutils literal notranslate\"><span class=\"pre\">st.local</span></code> instructions.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#stack-manipulation-instructions-alloca\" target=\"_blank\" rel=\"noopener noreferrer\">alloca <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Dynamically allocate memory on stack.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#stack-manipulation-instructions-alloca"
            };

        case "and":
            return {
                "html": "<p>Compute the bit-wise and operation for the bits in <code class=\"docutils literal notranslate\"><span class=\"pre\">a</span></code> and <code class=\"docutils literal notranslate\"><span class=\"pre\">b</span></code>.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-and\" target=\"_blank\" rel=\"noopener noreferrer\">and <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Bitwise AND.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-and"
            };

        case "applypriority":
            return {
                "html": "<p>The <code class=\"docutils literal notranslate\"><span class=\"pre\">applypriority</span></code> instruction applies the cache eviction priority specified by the\n<code class=\"docutils literal notranslate\"><span class=\"pre\">.level::eviction_priority</span></code> qualifier to the address range <code class=\"docutils literal notranslate\"><span class=\"pre\">[a..a+size)</span></code> in the specified cache\nlevel.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-applypriority\" target=\"_blank\" rel=\"noopener noreferrer\">applypriority <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Apply the cache eviction priority to the specified address in the specified cache level.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-applypriority"
            };

        case "atom":
            return {
                "html": "<p>Atomically loads the original value at location <code class=\"docutils literal notranslate\"><span class=\"pre\">a</span></code> into destination register <code class=\"docutils literal notranslate\"><span class=\"pre\">d</span></code>, performs a\nreduction operation with operand <code class=\"docutils literal notranslate\"><span class=\"pre\">b</span></code> and the value in location <code class=\"docutils literal notranslate\"><span class=\"pre\">a</span></code>, and stores the result of the\nspecified operation at location <code class=\"docutils literal notranslate\"><span class=\"pre\">a</span></code>, overwriting the original value. Operand <code class=\"docutils literal notranslate\"><span class=\"pre\">a</span></code> specifies a\nlocation in the specified state space. If no state space is given, perform the memory accesses using\n<a class=\"reference internal\" href=\"#generic-addressing\"><span class=\"std std-ref\">Generic Addressing</span></a>. <code class=\"docutils literal notranslate\"><span class=\"pre\">atom</span></code> with scalar type may be used only\nwith <code class=\"docutils literal notranslate\"><span class=\"pre\">.global</span></code> and <code class=\"docutils literal notranslate\"><span class=\"pre\">.shared</span></code> spaces and with generic addressing, where the address points to\n<code class=\"docutils literal notranslate\"><span class=\"pre\">.global</span></code> or <code class=\"docutils literal notranslate\"><span class=\"pre\">.shared</span></code> space. <code class=\"docutils literal notranslate\"><span class=\"pre\">atom</span></code> with vector type may be used only with <code class=\"docutils literal notranslate\"><span class=\"pre\">.global</span></code> space\nand with generic addressing where the address points to <code class=\"docutils literal notranslate\"><span class=\"pre\">.global</span></code> space.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-atom\" target=\"_blank\" rel=\"noopener noreferrer\">atom <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Atomic reduction operations for thread-to-thread communication.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-atom"
            };

        case "b4x16":
            return {
                "html": "<p><code class=\"docutils literal notranslate\"><span class=\"pre\">cp.async.bulk.tensor</span></code> is a non-blocking instruction which initiates an asynchronous copy\noperation of tensor data from the location in <code class=\"docutils literal notranslate\"><span class=\"pre\">.src</span></code> state space to the location in the <code class=\"docutils literal notranslate\"><span class=\"pre\">.dst</span></code>\nstate space.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-tensor-copy\" target=\"_blank\" rel=\"noopener noreferrer\">b4x16 <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Initiates an asynchronous copy operation on the tensor data from one state space to another.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-tensor-copy"
            };

        case "bar":
            return {
                "html": "<p>Performs barrier synchronization and communication within a CTA. Each CTA instance has sixteen\nbarriers numbered <code class=\"docutils literal notranslate\"><span class=\"pre\">0..15</span></code>.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar\" target=\"_blank\" rel=\"noopener noreferrer\">bar <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Barrier synchronization.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar"
            };

        case "barrier":
            return {
                "html": "<p>Performs barrier synchronization and communication within a cluster.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-barrier-cluster\" target=\"_blank\" rel=\"noopener noreferrer\">barrier.cluster <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Barrier synchronization within a cluster.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-barrier-cluster"
            };

        case "bfe":
            return {
                "html": "<p>Extract bit field from <code class=\"docutils literal notranslate\"><span class=\"pre\">a</span></code> and place the zero or sign-extended result in <code class=\"docutils literal notranslate\"><span class=\"pre\">d</span></code>. Source <code class=\"docutils literal notranslate\"><span class=\"pre\">b</span></code> gives\nthe bit field starting bit position, and source <code class=\"docutils literal notranslate\"><span class=\"pre\">c</span></code> gives the bit field length in bits.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-bfe\" target=\"_blank\" rel=\"noopener noreferrer\">bfe(int) <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Bit Field Extract.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-bfe"
            };

        case "bfi":
            return {
                "html": "<p>Align and insert a bit field from <code class=\"docutils literal notranslate\"><span class=\"pre\">a</span></code> into <code class=\"docutils literal notranslate\"><span class=\"pre\">b</span></code>, and place the result in <code class=\"docutils literal notranslate\"><span class=\"pre\">f</span></code>. Source <code class=\"docutils literal notranslate\"><span class=\"pre\">c</span></code>\ngives the starting bit position for the insertion, and source <code class=\"docutils literal notranslate\"><span class=\"pre\">d</span></code> gives the bit field length in\nbits.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-bfi\" target=\"_blank\" rel=\"noopener noreferrer\">bfi(int) <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Bit Field Insert.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-bfi"
            };

        case "bfind":
            return {
                "html": "<p>Find the bit position of the most significant non-sign bit in <code class=\"docutils literal notranslate\"><span class=\"pre\">a</span></code> and place the result in\n<code class=\"docutils literal notranslate\"><span class=\"pre\">d</span></code>. Operand <code class=\"docutils literal notranslate\"><span class=\"pre\">a</span></code> has the instruction type, and destination <code class=\"docutils literal notranslate\"><span class=\"pre\">d</span></code> has type <code class=\"docutils literal notranslate\"><span class=\"pre\">.u32</span></code>. For unsigned\nintegers, <code class=\"docutils literal notranslate\"><span class=\"pre\">bfind</span></code> returns the bit position of the most significant <code class=\"docutils literal notranslate\"><span class=\"pre\">1</span></code>. For signed integers,\n<code class=\"docutils literal notranslate\"><span class=\"pre\">bfind</span></code> returns the bit position of the most significant <code class=\"docutils literal notranslate\"><span class=\"pre\">0</span></code> for negative inputs and the most\nsignificant <code class=\"docutils literal notranslate\"><span class=\"pre\">1</span></code> for non-negative inputs.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-bfind\" target=\"_blank\" rel=\"noopener noreferrer\">bfind(int) <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Find most significant non-sign bit.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-bfind"
            };

        case "bmsk":
            return {
                "html": "<p>Generates a 32-bit mask starting from the bit position specified in operand <code class=\"docutils literal notranslate\"><span class=\"pre\">a</span></code>, and of the width\nspecified in operand <code class=\"docutils literal notranslate\"><span class=\"pre\">b</span></code>. The generated bitmask is stored in the destination operand <code class=\"docutils literal notranslate\"><span class=\"pre\">d</span></code>.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-bmsk\" target=\"_blank\" rel=\"noopener noreferrer\">bmsk(int) <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Bit Field Mask.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-bmsk"
            };

        case "bra":
            return {
                "html": "<p>Continue execution at the target. Conditional branches are specified by using a guard predicate. The\nbranch target must be a label.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#control-flow-instructions-bra\" target=\"_blank\" rel=\"noopener noreferrer\">bra <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Branch to a target and continue execution there.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#control-flow-instructions-bra"
            };

        case "brev":
            return {
                "html": "<p>Perform bitwise reversal of input.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-brev\" target=\"_blank\" rel=\"noopener noreferrer\">brev(int) <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Bit reverse.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-brev"
            };

        case "brkpt":
            return {
                "html": "<p>Suspends execution.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-brkpt\" target=\"_blank\" rel=\"noopener noreferrer\">brkpt <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Breakpoint.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-brkpt"
            };

        case "brx":
            return {
                "html": "<p>Index into a list of possible destination labels, and continue execution from the chosen\nlabel. Conditional branches are specified by using a guard predicate.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#control-flow-instructions-brx-idx\" target=\"_blank\" rel=\"noopener noreferrer\">brx.idx <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Branch to a label indexed from a list of potential branch targets.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#control-flow-instructions-brx-idx"
            };

        case "call":
            return {
                "html": "<p>The <code class=\"docutils literal notranslate\"><span class=\"pre\">call</span></code> instruction stores the address of the next instruction, so execution can resume at that\npoint after executing a <code class=\"docutils literal notranslate\"><span class=\"pre\">ret</span></code> instruction. A <code class=\"docutils literal notranslate\"><span class=\"pre\">call</span></code> is assumed to be divergent unless the\n<code class=\"docutils literal notranslate\"><span class=\"pre\">.uni</span></code> suffix is present. The <code class=\"docutils literal notranslate\"><span class=\"pre\">.uni</span></code> suffix indicates that the <code class=\"docutils literal notranslate\"><span class=\"pre\">call</span></code> is guaranteed to be\nnon-divergent, i.e. all active threads in a warp that are currently executing this instruction have\nidentical values for the guard predicate and <code class=\"docutils literal notranslate\"><span class=\"pre\">call</span></code> target.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#control-flow-instructions-call\" target=\"_blank\" rel=\"noopener noreferrer\">call <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Call a function, recording the return location.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#control-flow-instructions-call"
            };

        case "clusterlaunchcontrol":
            return {
                "html": "<p>The <code class=\"docutils literal notranslate\"><span class=\"pre\">clusterlaunchcontrol.try_cancel</span></code> instruction requests atomically cancelling the launch of\na cluster that has not started running yet. It asynchronously writes an opaque response to shared\nmemory indicating whether the operation succeeded or failed. The completion of the asynchronous\noperation is tracked using the mbarrier completion mechanism at <code class=\"docutils literal notranslate\"><span class=\"pre\">.cluster</span></code> scope.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-clusterlaunchcontrol-try-cancel\" target=\"_blank\" rel=\"noopener noreferrer\">clusterlaunchcontrol.try_cancel <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Requests cancellation of cluster which is not launched yet.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-clusterlaunchcontrol-try-cancel"
            };

        case "clz":
            return {
                "html": "<p>Count the number of leading zeros in <code class=\"docutils literal notranslate\"><span class=\"pre\">a</span></code> starting with the most-significant bit and place the\nresult in 32-bit destination register <code class=\"docutils literal notranslate\"><span class=\"pre\">d</span></code>. Operand <code class=\"docutils literal notranslate\"><span class=\"pre\">a</span></code> has the instruction type, and destination\n<code class=\"docutils literal notranslate\"><span class=\"pre\">d</span></code> has type <code class=\"docutils literal notranslate\"><span class=\"pre\">.u32</span></code>. For <code class=\"docutils literal notranslate\"><span class=\"pre\">.b32</span></code> type, the number of leading zeros is between 0 and 32,\ninclusively. For <code class=\"docutils literal notranslate\"><span class=\"pre\">.b64</span></code> type, the number of leading zeros is between 0 and 64, inclusively.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-clz\" target=\"_blank\" rel=\"noopener noreferrer\">clz(int) <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Count leading zeros.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-clz"
            };

        case "cnot":
            return {
                "html": "<p>Compute the logical negation using C/C++ semantics.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-cnot\" target=\"_blank\" rel=\"noopener noreferrer\">cnot <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "C/C++ style logical negation.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-cnot"
            };

        case "copysign":
            return {
                "html": "<p>Copy sign bit of <code class=\"docutils literal notranslate\"><span class=\"pre\">a</span></code> into value of <code class=\"docutils literal notranslate\"><span class=\"pre\">b</span></code>, and return the result as <code class=\"docutils literal notranslate\"><span class=\"pre\">d</span></code>.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-copysign\" target=\"_blank\" rel=\"noopener noreferrer\">copysign(fp) <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Copy sign of one input to another.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-copysign"
            };

        case "cos":
            return {
                "html": "<p>Find the cosine of the angle <code class=\"docutils literal notranslate\"><span class=\"pre\">a</span></code> (in radians).</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-cos\" target=\"_blank\" rel=\"noopener noreferrer\">cos(fp) <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Find the cosine of a value.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-cos"
            };

        case "cp":
            return {
                "html": "<p><code class=\"docutils literal notranslate\"><span class=\"pre\">cp.async</span></code> is a non-blocking instruction which initiates an asynchronous copy operation of data\nfrom the location specified by source address operand <code class=\"docutils literal notranslate\"><span class=\"pre\">src</span></code> to the location specified by\ndestination address operand <code class=\"docutils literal notranslate\"><span class=\"pre\">dst</span></code>. Operand <code class=\"docutils literal notranslate\"><span class=\"pre\">src</span></code> specifies a location in the global state space\nand <code class=\"docutils literal notranslate\"><span class=\"pre\">dst</span></code> specifies a location in the shared state space.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-non-bulk-copy\" target=\"_blank\" rel=\"noopener noreferrer\">cp.async <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Initiates an asynchronous copy operation from one state space to another.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-non-bulk-copy"
            };

        case "createpolicy":
            return {
                "html": "<p>The <code class=\"docutils literal notranslate\"><span class=\"pre\">createpolicy</span></code> instruction creates a cache eviction policy for the specified cache level in an\nopaque 64-bit register specified by the destination operand <code class=\"docutils literal notranslate\"><span class=\"pre\">cache-policy</span></code>. The cache eviction\npolicy specifies how cache eviction priorities are applied to global memory addresses used in memory\noperations with <code class=\"docutils literal notranslate\"><span class=\"pre\">.level::cache_hint</span></code> qualifier.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-createpolicy\" target=\"_blank\" rel=\"noopener noreferrer\">createpolicy <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Create a cache eviction policy for the specified cache level.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-createpolicy"
            };

        case "cvt":
            return {
                "html": "<p>Convert between different types and sizes.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt\" target=\"_blank\" rel=\"noopener noreferrer\">cvt <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Convert a value from one type to another.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt"
            };

        case "cvta":
            return {
                "html": "<p>Convert a <code class=\"docutils literal notranslate\"><span class=\"pre\">const</span></code>, <a class=\"reference internal\" href=\"#kernel-function-parameters\"><span class=\"std std-ref\">Kernel Function Parameters</span></a>\n(<code class=\"docutils literal notranslate\"><span class=\"pre\">.param</span></code>), <code class=\"docutils literal notranslate\"><span class=\"pre\">global</span></code>, <code class=\"docutils literal notranslate\"><span class=\"pre\">local</span></code>, or <code class=\"docutils literal notranslate\"><span class=\"pre\">shared</span></code> address to a generic address, or vice-versa. The\nsource and destination addresses must be the same size. Use <code class=\"docutils literal notranslate\"><span class=\"pre\">cvt.u32.u64</span></code> or <code class=\"docutils literal notranslate\"><span class=\"pre\">cvt.u64.u32</span></code> to\ntruncate or zero-extend addresses.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvta\" target=\"_blank\" rel=\"noopener noreferrer\">cvta <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Convert aconst,Kernel Function Parameters(.param),global,local, orsharedaddress to a generic address, or vice-versa. The\n\nsource and destination addresses must be the same size. Usecvt.u32.u64orcvt.u64...",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvta"
            };

        case "discard":
            return {
                "html": "<p>Semantically, this behaves like a weak write of an <em>unstable indeterminate value</em>:\nreads of memory locations with <em>unstable indeterminate values</em> may return different\nbit patterns each time until the memory is overwritten.\nThis operation <em>hints</em> to the implementation that data in the specified cache <code class=\"docutils literal notranslate\"><span class=\"pre\">.level</span></code>\ncan be destructively discarded without writing it back to memory.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-discard\" target=\"_blank\" rel=\"noopener noreferrer\">discard <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Discard the data at the specified address range and cache level.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-discard"
            };

        case "div":
            return {
                "html": "<p>Divides <code class=\"docutils literal notranslate\"><span class=\"pre\">a</span></code> by <code class=\"docutils literal notranslate\"><span class=\"pre\">b</span></code>, stores result in <code class=\"docutils literal notranslate\"><span class=\"pre\">d</span></code>.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-div\" target=\"_blank\" rel=\"noopener noreferrer\">div(int) <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Divide one value by another.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-div"
            };

        case "dp2a":
            return {
                "html": "<p>Two-way 16-bit to 8-bit dot product which is accumulated in 32-bit result.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-dp2a\" target=\"_blank\" rel=\"noopener noreferrer\">dp2a(int) <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Two-way dot product-accumulate.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-dp2a"
            };

        case "dp4a":
            return {
                "html": "<p>Four-way byte dot product which is accumulated in 32-bit result.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-dp4a\" target=\"_blank\" rel=\"noopener noreferrer\">dp4a(int) <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Four-way byte dot product-accumulate.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-dp4a"
            };

        case "elect":
            return {
                "html": "<p><code class=\"docutils literal notranslate\"><span class=\"pre\">elect.sync</span></code> elects one predicated active leader thread from among a set of threads specified by\n<code class=\"docutils literal notranslate\"><span class=\"pre\">membermask</span></code>. <code class=\"docutils literal notranslate\"><span class=\"pre\">laneid</span></code> of the elected thread is returned in the 32-bit destination operand\n<code class=\"docutils literal notranslate\"><span class=\"pre\">d</span></code>. The sink symbol \u2018_\u2019 can be used for destination operand <code class=\"docutils literal notranslate\"><span class=\"pre\">d</span></code>. The predicate destination\n<code class=\"docutils literal notranslate\"><span class=\"pre\">p</span></code> is set to <code class=\"docutils literal notranslate\"><span class=\"pre\">True</span></code> for the leader thread, and <code class=\"docutils literal notranslate\"><span class=\"pre\">False</span></code> for all other threads.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-elect-sync\" target=\"_blank\" rel=\"noopener noreferrer\">elect.sync <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Elect a leader thread from a set of threads.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-elect-sync"
            };

        case "ex2":
            return {
                "html": "<p>Raise 2 to the power <code class=\"docutils literal notranslate\"><span class=\"pre\">a</span></code>.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-ex2\" target=\"_blank\" rel=\"noopener noreferrer\">ex2(fp) <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Find the base-2 exponential of a value.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-ex2"
            };

        case "exit":
            return {
                "html": "<p>Ends execution of a thread.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#control-flow-instructions-exit\" target=\"_blank\" rel=\"noopener noreferrer\">exit <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Terminate a thread.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#control-flow-instructions-exit"
            };

        case "fma":
            return {
                "html": "<p>Performs a fused multiply-add with no loss of precision in the intermediate product and addition.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-fma\" target=\"_blank\" rel=\"noopener noreferrer\">fma(fp) <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Fused multiply-add.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-fma"
            };

        case "fns":
            return {
                "html": "<p>Given a 32-bit value <code class=\"docutils literal notranslate\"><span class=\"pre\">mask</span></code> and an integer value <code class=\"docutils literal notranslate\"><span class=\"pre\">base</span></code> (between 0 and 31), find the n-th (given\nby offset) set bit in <code class=\"docutils literal notranslate\"><span class=\"pre\">mask</span></code> from the <code class=\"docutils literal notranslate\"><span class=\"pre\">base</span></code> bit, and store the bit position in <code class=\"docutils literal notranslate\"><span class=\"pre\">d</span></code>. If not\nfound, store 0xffffffff in <code class=\"docutils literal notranslate\"><span class=\"pre\">d</span></code>.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-fns\" target=\"_blank\" rel=\"noopener noreferrer\">fns(int) <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Find the n-th set bit",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-fns"
            };

        case "getctarank":
            return {
                "html": "<p>Write the destination register <code class=\"docutils literal notranslate\"><span class=\"pre\">d</span></code> with the rank of the CTA which contains the address specified\nin operand <code class=\"docutils literal notranslate\"><span class=\"pre\">a</span></code>.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-getctarank\" target=\"_blank\" rel=\"noopener noreferrer\">getctarank <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Generate the CTA rank of the address.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-getctarank"
            };

        case "griddepcontrol":
            return {
                "html": "<p>The <code class=\"docutils literal notranslate\"><span class=\"pre\">griddepcontrol</span></code> instruction allows the dependent grids and prerequisite grids as defined by\nthe runtime, to control execution in the following way:</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-griddepcontrol\" target=\"_blank\" rel=\"noopener noreferrer\">griddepcontrol <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Control execution of dependent grids.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-griddepcontrol"
            };

        case "isspacep":
            return {
                "html": "<p>Write predicate register <code class=\"docutils literal notranslate\"><span class=\"pre\">p</span></code> with <code class=\"docutils literal notranslate\"><span class=\"pre\">1</span></code> if generic address a falls within the specified state\nspace window and with <code class=\"docutils literal notranslate\"><span class=\"pre\">0</span></code> otherwise. Destination <code class=\"docutils literal notranslate\"><span class=\"pre\">p</span></code> has type <code class=\"docutils literal notranslate\"><span class=\"pre\">.pred</span></code>; the source address\noperand must be of type <code class=\"docutils literal notranslate\"><span class=\"pre\">.u32</span></code> or <code class=\"docutils literal notranslate\"><span class=\"pre\">.u64</span></code>.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-isspacep\" target=\"_blank\" rel=\"noopener noreferrer\">isspacep <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Query whether a generic address falls within a specified state space window.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-isspacep"
            };

        case "istypep":
            return {
                "html": "<p>Write predicate register <code class=\"docutils literal notranslate\"><span class=\"pre\">p</span></code> with 1 if register <code class=\"docutils literal notranslate\"><span class=\"pre\">a</span></code> points to an opaque variable of the\nspecified type, and with 0 otherwise. Destination <code class=\"docutils literal notranslate\"><span class=\"pre\">p</span></code> has type <code class=\"docutils literal notranslate\"><span class=\"pre\">.pred</span></code>; the source address\noperand must be of type <code class=\"docutils literal notranslate\"><span class=\"pre\">.u64</span></code>.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#texture-instructions-istypep\" target=\"_blank\" rel=\"noopener noreferrer\">istypep <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Query whether a register points to an opaque variable of a specified type.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#texture-instructions-istypep"
            };

        case "ld":
            return {
                "html": "<p>Load register variable <code class=\"docutils literal notranslate\"><span class=\"pre\">d</span></code> from the location specified by the source address operand <code class=\"docutils literal notranslate\"><span class=\"pre\">a</span></code> in\nspecified state space. If no state space is given, perform the load using <a class=\"reference internal\" href=\"#generic-addressing\"><span class=\"std std-ref\">Generic Addressing</span></a>.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld\" target=\"_blank\" rel=\"noopener noreferrer\">ld <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Load a register variable from an addressable state space variable.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld"
            };

        case "ldmatrix":
            return {
                "html": "<p>Collectively load one or more matrices across all threads in a warp from the location indicated by\nthe address operand <code class=\"docutils literal notranslate\"><span class=\"pre\">p</span></code>, from <code class=\"docutils literal notranslate\"><span class=\"pre\">.shared</span></code> state space into destination register <code class=\"docutils literal notranslate\"><span class=\"pre\">r</span></code>. If no state\nspace is provided, generic addressing is used, such that the address in <code class=\"docutils literal notranslate\"><span class=\"pre\">p</span></code> points into\n<code class=\"docutils literal notranslate\"><span class=\"pre\">.shared</span></code> space. If the generic address doesn\u2019t fall in <code class=\"docutils literal notranslate\"><span class=\"pre\">.shared</span></code> state space, then the behavior\nis undefined.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-ldmatrix\" target=\"_blank\" rel=\"noopener noreferrer\">ldmatrix <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Collectively load one or more matrices from shared memory formmainstruction",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-ldmatrix"
            };

        case "ldu":
            return {
                "html": "<p>Load <em>read-only</em> data into register variable <code class=\"docutils literal notranslate\"><span class=\"pre\">d</span></code> from the location specified by the source address\noperand <code class=\"docutils literal notranslate\"><span class=\"pre\">a</span></code> in the global state space, where the address is guaranteed to be the same across all\nthreads in the warp. If no state space is given, perform the load using <a class=\"reference internal\" href=\"#generic-addressing\"><span class=\"std std-ref\">Generic Addressing</span></a>.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ldu\" target=\"_blank\" rel=\"noopener noreferrer\">ldu <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Load read-only data from an address that is common across threads in the warp.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ldu"
            };

        case "lg2":
            return {
                "html": "<p>Determine the log<sub>2</sub> of <code class=\"docutils literal notranslate\"><span class=\"pre\">a</span></code>.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-lg2\" target=\"_blank\" rel=\"noopener noreferrer\">lg2(fp) <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Find the base-2 logarithm of a value.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-lg2"
            };

        case "lop3":
            return {
                "html": "<p>Compute bitwise logical operation on inputs <code class=\"docutils literal notranslate\"><span class=\"pre\">a</span></code>, <code class=\"docutils literal notranslate\"><span class=\"pre\">b</span></code>, <code class=\"docutils literal notranslate\"><span class=\"pre\">c</span></code> and store the result in destination\n<code class=\"docutils literal notranslate\"><span class=\"pre\">d</span></code>.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-lop3\" target=\"_blank\" rel=\"noopener noreferrer\">lop3 <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Arbitrary logical operation on 3 inputs.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-lop3"
            };

        case "mad":
            return {
                "html": "<p>Multiplies two values, optionally extracts the high or low half of the intermediate result, and adds\na third value. Writes the result into a destination register.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-mad\" target=\"_blank\" rel=\"noopener noreferrer\">mad(int) <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Multiplies two values, optionally extracts the high or low half of the intermediate result, and adds\n\na third value. Writes the result into a destination register.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-mad"
            };

        case "mad24":
            return {
                "html": "<p>Compute the product of two 24-bit integer values held in 32-bit source registers, and add a third,\n32-bit value to either the high or low 32-bits of the 48-bit result. Return either the high or low\n32-bits of the 48-bit result.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-mad24\" target=\"_blank\" rel=\"noopener noreferrer\">mad24(int) <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Multiply two 24-bit integer values and add a third value.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-mad24"
            };

        case "madc":
            return {
                "html": "<p>Multiplies two values, extracts either the high or low part of the result, and adds a third value\nalong with carry-in. Writes the result to the destination register and optionally writes the\ncarry-out from the addition into the condition code register.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#extended-precision-arithmetic-instructions-madc\" target=\"_blank\" rel=\"noopener noreferrer\">madc <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Multiplies two values, extracts either the high or low part of the result, and adds a third value\n\nalong with carry-in. Writes the result to the destination register and optionally writes the\n\ncarry-out...",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#extended-precision-arithmetic-instructions-madc"
            };

        case "mapa":
            return {
                "html": "<p>Get address in the CTA specified by operand <code class=\"docutils literal notranslate\"><span class=\"pre\">b</span></code> which corresponds to the address specified by\noperand <code class=\"docutils literal notranslate\"><span class=\"pre\">a</span></code>.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-mapa\" target=\"_blank\" rel=\"noopener noreferrer\">mapa <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Map the address of the shared variable in the target CTA.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-mapa"
            };

        case "match":
            return {
                "html": "<p><code class=\"docutils literal notranslate\"><span class=\"pre\">match.sync</span></code> will cause executing thread to wait until all non-exited threads from <code class=\"docutils literal notranslate\"><span class=\"pre\">membermask</span></code>\nhave executed <code class=\"docutils literal notranslate\"><span class=\"pre\">match.sync</span></code> with the same qualifiers and same <code class=\"docutils literal notranslate\"><span class=\"pre\">membermask</span></code> value before resuming\nexecution.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-match-sync\" target=\"_blank\" rel=\"noopener noreferrer\">match.sync <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Broadcast and compare a value across threads in warp.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-match-sync"
            };

        case "max":
            return {
                "html": "<p>Store the maximum of <code class=\"docutils literal notranslate\"><span class=\"pre\">a</span></code> and <code class=\"docutils literal notranslate\"><span class=\"pre\">b</span></code> in <code class=\"docutils literal notranslate\"><span class=\"pre\">d</span></code>.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-max\" target=\"_blank\" rel=\"noopener noreferrer\">max(int) <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Find the maximum of two values.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-max"
            };

        case "mbarrier":
            return {
                "html": "<p><code class=\"docutils literal notranslate\"><span class=\"pre\">cp.async</span></code> is a non-blocking instruction which initiates an asynchronous copy operation of data\nfrom the location specified by source address operand <code class=\"docutils literal notranslate\"><span class=\"pre\">src</span></code> to the location specified by\ndestination address operand <code class=\"docutils literal notranslate\"><span class=\"pre\">dst</span></code>. Operand <code class=\"docutils literal notranslate\"><span class=\"pre\">src</span></code> specifies a location in the global state space\nand <code class=\"docutils literal notranslate\"><span class=\"pre\">dst</span></code> specifies a location in the shared state space.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-asynchronous-copy\" target=\"_blank\" rel=\"noopener noreferrer\">mbarrier.test_wait <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Initiates an asynchronous copy operation from one state space to another.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-asynchronous-copy"
            };

        case "membar":
            return {
                "html": "<p>The <code class=\"docutils literal notranslate\"><span class=\"pre\">membar</span></code> instruction guarantees that prior memory accesses requested by this thread (<code class=\"docutils literal notranslate\"><span class=\"pre\">ld</span></code>,\n<code class=\"docutils literal notranslate\"><span class=\"pre\">st</span></code>, <code class=\"docutils literal notranslate\"><span class=\"pre\">atom</span></code> and <code class=\"docutils literal notranslate\"><span class=\"pre\">red</span></code> instructions) are performed at the specified <code class=\"docutils literal notranslate\"><span class=\"pre\">level</span></code>, before later\nmemory operations requested by this thread following the <code class=\"docutils literal notranslate\"><span class=\"pre\">membar</span></code> instruction. The <code class=\"docutils literal notranslate\"><span class=\"pre\">level</span></code>\nqualifier specifies the set of threads that may observe the ordering effect of this operation.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar\" target=\"_blank\" rel=\"noopener noreferrer\">membar <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Enforce an ordering of memory operations.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar"
            };

        case "min":
            return {
                "html": "<p>Store the minimum of <code class=\"docutils literal notranslate\"><span class=\"pre\">a</span></code> and <code class=\"docutils literal notranslate\"><span class=\"pre\">b</span></code> in <code class=\"docutils literal notranslate\"><span class=\"pre\">d</span></code>.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-min\" target=\"_blank\" rel=\"noopener noreferrer\">min(int) <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Find the minimum of two values.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-min"
            };

        case "mma":
            return {
                "html": "<p>Perform a <code class=\"docutils literal notranslate\"><span class=\"pre\">MxNxK</span></code> matrix multiply and accumulate operation, <code class=\"docutils literal notranslate\"><span class=\"pre\">D</span> <span class=\"pre\">=</span> <span class=\"pre\">A*B+C</span></code>, where the A matrix is\n<code class=\"docutils literal notranslate\"><span class=\"pre\">MxK</span></code>, the B matrix is <code class=\"docutils literal notranslate\"><span class=\"pre\">KxN</span></code>, and the C and D matrices are <code class=\"docutils literal notranslate\"><span class=\"pre\">MxN</span></code>.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-for-mma\" target=\"_blank\" rel=\"noopener noreferrer\">mma <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>, <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-for-mma\" target=\"_blank\" rel=\"noopener noreferrer\">mma.m8n8k4 <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>, <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-for-sparse-mma\" target=\"_blank\" rel=\"noopener noreferrer\">mma.sp <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>, <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-sparse-mma\" target=\"_blank\" rel=\"noopener noreferrer\">mma.sp <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>, <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-sparse-mma\" target=\"_blank\" rel=\"noopener noreferrer\">mma.sp::ordered_metadata <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>, <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-for-sparse-mma\" target=\"_blank\" rel=\"noopener noreferrer\">mma.sp{::ordered_metadata} <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Perform matrix multiply-and-accumulate operation",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-for-mma"
            };

        case "mov":
            return {
                "html": "<p>Write register <code class=\"docutils literal notranslate\"><span class=\"pre\">d</span></code> with the value of <code class=\"docutils literal notranslate\"><span class=\"pre\">a</span></code>.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-mov\" target=\"_blank\" rel=\"noopener noreferrer\">mov <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Write registerdwith the value ofa.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-mov"
            };

        case "movmatrix":
            return {
                "html": "<p>Move a row-major matrix across all threads in a warp, reading elements from source <code class=\"docutils literal notranslate\"><span class=\"pre\">a</span></code>, and\nwriting the transposed elements to destination <code class=\"docutils literal notranslate\"><span class=\"pre\">d</span></code>.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-movmatrix\" target=\"_blank\" rel=\"noopener noreferrer\">movmatrix <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Transpose a matrix in registers across the warp.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-movmatrix"
            };

        case "mul":
            return {
                "html": "<p>Compute the product of two values.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-mul\" target=\"_blank\" rel=\"noopener noreferrer\">mul(int) <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Multiply two values.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-mul"
            };

        case "mul24":
            return {
                "html": "<p>Compute the product of two 24-bit integer values held in 32-bit source registers, and return either\nthe high or low 32-bits of the 48-bit result.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-mul24\" target=\"_blank\" rel=\"noopener noreferrer\">mul24(int) <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Multiply two 24-bit integer values.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-mul24"
            };

        case "multimem":
            return {
                "html": "<p>Instruction <code class=\"docutils literal notranslate\"><span class=\"pre\">multimem.ld_reduce</span></code> performs the following operations:</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-multimem\" target=\"_blank\" rel=\"noopener noreferrer\">multimem.ld_reduce <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Perform memory operations on the multimem address.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-multimem"
            };

        case "nanosleep":
            return {
                "html": "<p>Suspends the thread for a sleep duration approximately close to the delay <code class=\"docutils literal notranslate\"><span class=\"pre\">t</span></code>, specified in\nnanoseconds. <code class=\"docutils literal notranslate\"><span class=\"pre\">t</span></code> may be a register or an immediate value.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-nanosleep\" target=\"_blank\" rel=\"noopener noreferrer\">nanosleep <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Suspend the thread for an approximate delay given in nanoseconds.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-nanosleep"
            };

        case "neg":
            return {
                "html": "<p>Negate the sign of <strong>a</strong> and store the result in <strong>d</strong>.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-neg\" target=\"_blank\" rel=\"noopener noreferrer\">neg(int) <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Arithmetic negate.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-neg"
            };

        case "not":
            return {
                "html": "<p>Invert the bits in <code class=\"docutils literal notranslate\"><span class=\"pre\">a</span></code>.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-not\" target=\"_blank\" rel=\"noopener noreferrer\">not <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Bitwise negation; one\u2019s complement.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-not"
            };

        case "or":
            return {
                "html": "<p>Compute the bit-wise or operation for the bits in <code class=\"docutils literal notranslate\"><span class=\"pre\">a</span></code> and <code class=\"docutils literal notranslate\"><span class=\"pre\">b</span></code>.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-or\" target=\"_blank\" rel=\"noopener noreferrer\">or <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Biwise OR.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-or"
            };

        case "pmevent":
            return {
                "html": "<p>Triggers one or more of a fixed number of performance monitor events, with event index or mask\nspecified by immediate operand <code class=\"docutils literal notranslate\"><span class=\"pre\">a</span></code>.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-pmevent\" target=\"_blank\" rel=\"noopener noreferrer\">pmevent <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Trigger one or more Performance Monitor events.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-pmevent"
            };

        case "popc":
            return {
                "html": "<p>Count the number of one bits in <code class=\"docutils literal notranslate\"><span class=\"pre\">a</span></code> and place the resulting <em>population count</em> in 32-bit\ndestination register <code class=\"docutils literal notranslate\"><span class=\"pre\">d</span></code>. Operand <code class=\"docutils literal notranslate\"><span class=\"pre\">a</span></code> has the instruction type and destination <code class=\"docutils literal notranslate\"><span class=\"pre\">d</span></code> has type\n<code class=\"docutils literal notranslate\"><span class=\"pre\">.u32</span></code>.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-popc\" target=\"_blank\" rel=\"noopener noreferrer\">popc(int) <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Population count.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-popc"
            };

        case "prefetch":
            return {
                "html": "<p>The <code class=\"docutils literal notranslate\"><span class=\"pre\">prefetch</span></code> instruction brings the cache line containing the specified address in global or\nlocal memory state space into the specified cache level.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-prefetch-prefetchu\" target=\"_blank\" rel=\"noopener noreferrer\">prefetch <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Theprefetchinstruction brings the cache line containing the specified address in global or\n\nlocal memory state space into the specified cache level.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-prefetch-prefetchu"
            };

        case "prmt":
            return {
                "html": "<p>Pick four arbitrary bytes from two 32-bit registers, and reassemble them into a 32-bit destination\nregister.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-prmt\" target=\"_blank\" rel=\"noopener noreferrer\">prmt <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Permute bytes from register pair.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-prmt"
            };

        case "rcp":
            return {
                "html": "<p>Compute <code class=\"docutils literal notranslate\"><span class=\"pre\">1/a</span></code>, store result in <code class=\"docutils literal notranslate\"><span class=\"pre\">d</span></code>.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-rcp\" target=\"_blank\" rel=\"noopener noreferrer\">rcp(fp) <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Take the reciprocal of a value.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-rcp"
            };

        case "red":
            return {
                "html": "<p>Performs a reduction operation with operand <code class=\"docutils literal notranslate\"><span class=\"pre\">b</span></code> and the value in location <code class=\"docutils literal notranslate\"><span class=\"pre\">a</span></code>, and stores the\nresult of the specified operation at location <code class=\"docutils literal notranslate\"><span class=\"pre\">a</span></code>, overwriting the original value. Operand <code class=\"docutils literal notranslate\"><span class=\"pre\">a</span></code>\nspecifies a location in the specified state space. If no state space is given, perform the memory\naccesses using <a class=\"reference internal\" href=\"#generic-addressing\"><span class=\"std std-ref\">Generic Addressing</span></a>. <code class=\"docutils literal notranslate\"><span class=\"pre\">red</span></code> with scalar type may\nbe used only with <code class=\"docutils literal notranslate\"><span class=\"pre\">.global</span></code> and <code class=\"docutils literal notranslate\"><span class=\"pre\">.shared</span></code> spaces and with generic addressing, where the address\npoints to <code class=\"docutils literal notranslate\"><span class=\"pre\">.global</span></code> or <code class=\"docutils literal notranslate\"><span class=\"pre\">.shared</span></code> space. <code class=\"docutils literal notranslate\"><span class=\"pre\">red</span></code> with vector type may be used only with\n<code class=\"docutils literal notranslate\"><span class=\"pre\">.global</span></code> space and with generic addressing where the address points to <code class=\"docutils literal notranslate\"><span class=\"pre\">.global</span></code> space.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-red\" target=\"_blank\" rel=\"noopener noreferrer\">red <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Reduction operations on global and shared memory.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-red"
            };

        case "redux":
            return {
                "html": "<p><code class=\"docutils literal notranslate\"><span class=\"pre\">redux.sync</span></code> will cause the executing thread to wait until all non-exited threads corresponding to\n<code class=\"docutils literal notranslate\"><span class=\"pre\">membermask</span></code> have executed <code class=\"docutils literal notranslate\"><span class=\"pre\">redux.sync</span></code> with the same qualifiers and same <code class=\"docutils literal notranslate\"><span class=\"pre\">membermask</span></code> value\nbefore resuming execution.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-redux-sync\" target=\"_blank\" rel=\"noopener noreferrer\">redux.sync <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Perform reduction operation on the data from each predicated active thread in the thread group.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-redux-sync"
            };

        case "rem":
            return {
                "html": "<p>Divides <code class=\"docutils literal notranslate\"><span class=\"pre\">a</span></code> by <code class=\"docutils literal notranslate\"><span class=\"pre\">b</span></code>, store the remainder in <code class=\"docutils literal notranslate\"><span class=\"pre\">d</span></code>.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-rem\" target=\"_blank\" rel=\"noopener noreferrer\">rem(int) <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "The remainder of integer division.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-rem"
            };

        case "ret":
            return {
                "html": "<p>Return execution to caller\u2019s environment. A divergent return suspends threads until all threads are\nready to return to the caller. This allows multiple divergent <code class=\"docutils literal notranslate\"><span class=\"pre\">ret</span></code> instructions.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#control-flow-instructions-ret\" target=\"_blank\" rel=\"noopener noreferrer\">ret <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Return from function to instruction after call.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#control-flow-instructions-ret"
            };

        case "rsqrt":
            return {
                "html": "<p>Compute <code class=\"docutils literal notranslate\"><span class=\"pre\">1/sqrt(a)</span></code> and store the result in <code class=\"docutils literal notranslate\"><span class=\"pre\">d</span></code>.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-rsqrt\" target=\"_blank\" rel=\"noopener noreferrer\">rsqrt(fp) <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Take the reciprocal of the square root of a value.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-rsqrt"
            };

        case "sad":
            return {
                "html": "<p>Adds the absolute value of <code class=\"docutils literal notranslate\"><span class=\"pre\">a-b</span></code> to <code class=\"docutils literal notranslate\"><span class=\"pre\">c</span></code> and writes the resulting value into <code class=\"docutils literal notranslate\"><span class=\"pre\">d</span></code>.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-sad\" target=\"_blank\" rel=\"noopener noreferrer\">sad(int) <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Sum of absolute differences.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-sad"
            };

        case "selp":
            return {
                "html": "<p>Conditional selection. If <code class=\"docutils literal notranslate\"><span class=\"pre\">c</span></code> is <code class=\"docutils literal notranslate\"><span class=\"pre\">True</span></code>, <code class=\"docutils literal notranslate\"><span class=\"pre\">a</span></code> is stored in <code class=\"docutils literal notranslate\"><span class=\"pre\">d</span></code>, <code class=\"docutils literal notranslate\"><span class=\"pre\">b</span></code> otherwise. Operands\n<code class=\"docutils literal notranslate\"><span class=\"pre\">d</span></code>, <code class=\"docutils literal notranslate\"><span class=\"pre\">a</span></code>, and <code class=\"docutils literal notranslate\"><span class=\"pre\">b</span></code> must be of the same type. Operand <code class=\"docutils literal notranslate\"><span class=\"pre\">c</span></code> is a predicate.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#comparison-and-selection-instructions-selp\" target=\"_blank\" rel=\"noopener noreferrer\">selp <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Select between source operands, based on the value of the predicate source operand.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#comparison-and-selection-instructions-selp"
            };

        case "set":
            return {
                "html": "<p>Compares two numeric values and optionally combines the result with another predicate value by\napplying a Boolean operator. If this result is <code class=\"docutils literal notranslate\"><span class=\"pre\">True</span></code>, <code class=\"docutils literal notranslate\"><span class=\"pre\">1.0f</span></code> is written for floating-point\ndestination types, and <code class=\"docutils literal notranslate\"><span class=\"pre\">0xffffffff</span></code> is written for integer destination types. Otherwise,\n<code class=\"docutils literal notranslate\"><span class=\"pre\">0x00000000</span></code> is written.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#comparison-and-selection-instructions-set\" target=\"_blank\" rel=\"noopener noreferrer\">set <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Compares two numeric values and optionally combines the result with another predicate value by\n\napplying a Boolean operator. If this result isTrue,1.0fis written for floating-point\n\ndestination types, a...",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#comparison-and-selection-instructions-set"
            };

        case "setmaxnreg":
            return {
                "html": "<p><code class=\"docutils literal notranslate\"><span class=\"pre\">setmaxnreg</span></code> provides a hint to the system to update the maximum number of per-thread registers\nowned by the executing warp to the value specified by the <code class=\"docutils literal notranslate\"><span class=\"pre\">imm-reg-count</span></code> operand.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-setmaxnreg\" target=\"_blank\" rel=\"noopener noreferrer\">setmaxnreg <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Hint to change the number of registers owned by the warp.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-setmaxnreg"
            };

        case "setp":
            return {
                "html": "<p>Compares two values and combines the result with another predicate value by applying a Boolean\noperator. This result is written to the first destination operand. A related value computed using\nthe complement of the compare result is written to the second destination operand.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#comparison-and-selection-instructions-setp\" target=\"_blank\" rel=\"noopener noreferrer\">setp <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Compares two values and combines the result with another predicate value by applying a Boolean\n\noperator. This result is written to the first destination operand. A related value computed using\n\nthe com...",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#comparison-and-selection-instructions-setp"
            };

        case "shf":
            return {
                "html": "<p>Shift the 64-bit value formed by concatenating operands <code class=\"docutils literal notranslate\"><span class=\"pre\">a</span></code> and <code class=\"docutils literal notranslate\"><span class=\"pre\">b</span></code> left or right by the amount\nspecified by the unsigned 32-bit value in <code class=\"docutils literal notranslate\"><span class=\"pre\">c</span></code>. Operand <code class=\"docutils literal notranslate\"><span class=\"pre\">b</span></code> holds bits <code class=\"docutils literal notranslate\"><span class=\"pre\">63:32</span></code> and operand a\nholds bits <code class=\"docutils literal notranslate\"><span class=\"pre\">31:0</span></code> of the 64-bit source value. The source is shifted left or right by the clamped\nor wrapped value in <code class=\"docutils literal notranslate\"><span class=\"pre\">c</span></code>. For <code class=\"docutils literal notranslate\"><span class=\"pre\">shf.l</span></code>, the most-significant 32-bits of the result are written\ninto <code class=\"docutils literal notranslate\"><span class=\"pre\">d</span></code>; for <code class=\"docutils literal notranslate\"><span class=\"pre\">shf.r</span></code>, the least-significant 32-bits of the result are written into <code class=\"docutils literal notranslate\"><span class=\"pre\">d</span></code>.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-shf\" target=\"_blank\" rel=\"noopener noreferrer\">shf <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Funnel shift.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-shf"
            };

        case "shfl":
            return {
                "html": "<p>Exchange register data between threads of a warp.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-shfl\" target=\"_blank\" rel=\"noopener noreferrer\">shfl <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Register data shuffle within threads of a warp.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-shfl"
            };

        case "shl":
            return {
                "html": "<p>Shift <code class=\"docutils literal notranslate\"><span class=\"pre\">a</span></code> left by the amount specified by unsigned 32-bit value in <code class=\"docutils literal notranslate\"><span class=\"pre\">b</span></code>.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-shl\" target=\"_blank\" rel=\"noopener noreferrer\">shl <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Shift bits left, zero-fill on right.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-shl"
            };

        case "shr":
            return {
                "html": "<p>Shift <code class=\"docutils literal notranslate\"><span class=\"pre\">a</span></code> right by the amount specified by unsigned 32-bit value in <code class=\"docutils literal notranslate\"><span class=\"pre\">b</span></code>. Signed shifts fill with\nthe sign bit, unsigned and untyped shifts fill with <code class=\"docutils literal notranslate\"><span class=\"pre\">0</span></code>.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-shr\" target=\"_blank\" rel=\"noopener noreferrer\">shr <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Shift bits right, sign or zero-fill on left.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-shr"
            };

        case "sin":
            return {
                "html": "<p>Find the sine of the angle <code class=\"docutils literal notranslate\"><span class=\"pre\">a</span></code> (in radians).</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-sin\" target=\"_blank\" rel=\"noopener noreferrer\">sin(fp) <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Find the sine of a value.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-sin"
            };

        case "slct":
            return {
                "html": "<p>Conditional selection. If <code class=\"docutils literal notranslate\"><span class=\"pre\">c</span></code> &gt;= 0, <code class=\"docutils literal notranslate\"><span class=\"pre\">a</span></code> is stored in <code class=\"docutils literal notranslate\"><span class=\"pre\">d</span></code>, otherwise <code class=\"docutils literal notranslate\"><span class=\"pre\">b</span></code> is stored in\n<code class=\"docutils literal notranslate\"><span class=\"pre\">d</span></code>. Operands <code class=\"docutils literal notranslate\"><span class=\"pre\">d</span></code>, <code class=\"docutils literal notranslate\"><span class=\"pre\">a</span></code>, and <code class=\"docutils literal notranslate\"><span class=\"pre\">b</span></code> are treated as a bitsize type of the same width as the first\ninstruction type; operand <code class=\"docutils literal notranslate\"><span class=\"pre\">c</span></code> must match the second instruction type (<code class=\"docutils literal notranslate\"><span class=\"pre\">.s32</span></code> or <code class=\"docutils literal notranslate\"><span class=\"pre\">.f32</span></code>). The\nselected input is copied to the output without modification.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#comparison-and-selection-instructions-slct\" target=\"_blank\" rel=\"noopener noreferrer\">slct <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Select one source operand, based on the sign of the third operand.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#comparison-and-selection-instructions-slct"
            };

        case "sqrt":
            return {
                "html": "<p>Compute sqrt(<code class=\"docutils literal notranslate\"><span class=\"pre\">a</span></code>) and store the result in <code class=\"docutils literal notranslate\"><span class=\"pre\">d</span></code>.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-sqrt\" target=\"_blank\" rel=\"noopener noreferrer\">sqrt(fp) <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Take the square root of a value.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-sqrt"
            };

        case "st":
            return {
                "html": "<p>Store the value of operand <code class=\"docutils literal notranslate\"><span class=\"pre\">b</span></code> in the location specified by the destination address\noperand <code class=\"docutils literal notranslate\"><span class=\"pre\">a</span></code> in specified state space. If no state space is given, perform the store using\n<a class=\"reference internal\" href=\"#generic-addressing\"><span class=\"std std-ref\">Generic Addressing</span></a>. Stores to const memory are illegal.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st\" target=\"_blank\" rel=\"noopener noreferrer\">st <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Store data to an addressable state space variable.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st"
            };

        case "stackrestore":
            return {
                "html": "<p>Sets the current stack pointer to source register <code class=\"docutils literal notranslate\"><span class=\"pre\">a</span></code>.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#stack-manipulation-instructions-stackrestore\" target=\"_blank\" rel=\"noopener noreferrer\">stackrestore <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Update the stack pointer with a new value.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#stack-manipulation-instructions-stackrestore"
            };

        case "stacksave":
            return {
                "html": "<p>Copies the current value of stack pointer into the destination register <code class=\"docutils literal notranslate\"><span class=\"pre\">d</span></code>. Pointer returned by\n<code class=\"docutils literal notranslate\"><span class=\"pre\">stacksave</span></code> can be used in a subsequent <code class=\"docutils literal notranslate\"><span class=\"pre\">stackrestore</span></code> instruction to restore the stack\npointer. If <code class=\"docutils literal notranslate\"><span class=\"pre\">d</span></code> is modified prior to use in <code class=\"docutils literal notranslate\"><span class=\"pre\">stackrestore</span></code> instruction, it may corrupt data in\nthe stack.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#stack-manipulation-instructions-stacksave\" target=\"_blank\" rel=\"noopener noreferrer\">stacksave <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Save the value of stack pointer into a register.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#stack-manipulation-instructions-stacksave"
            };

        case "stmatrix":
            return {
                "html": "<p>Collectively store one or more matrices across all threads in a warp to the location indicated by\nthe address operand <code class=\"docutils literal notranslate\"><span class=\"pre\">p</span></code>, in <code class=\"docutils literal notranslate\"><span class=\"pre\">.shared</span></code> state space. If no state space is provided, generic\naddressing is used, such that the address in <code class=\"docutils literal notranslate\"><span class=\"pre\">p</span></code> points into <code class=\"docutils literal notranslate\"><span class=\"pre\">.shared</span></code> space. If the generic\naddress doesn\u2019t fall in <code class=\"docutils literal notranslate\"><span class=\"pre\">.shared</span></code> state space, then the behavior is undefined.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-stmatrix\" target=\"_blank\" rel=\"noopener noreferrer\">stmatrix <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Collectively store one or more matrices to shared memory.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-stmatrix"
            };

        case "sub":
            return {
                "html": "<p>Performs subtraction and writes the resulting value into a destination register.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-sub\" target=\"_blank\" rel=\"noopener noreferrer\">sub(int) <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Subtract one value from another.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-sub"
            };

        case "subc":
            return {
                "html": "<p>Performs integer subtraction with borrow-in and optionally writes the borrow-out value into the\ncondition code register.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#extended-precision-arithmetic-instructions-subc\" target=\"_blank\" rel=\"noopener noreferrer\">subc <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Subtract one value from another, with borrow-in and optional borrow-out.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#extended-precision-arithmetic-instructions-subc"
            };

        case "suld":
            return {
                "html": "<p><code class=\"docutils literal notranslate\"><span class=\"pre\">suld.b.{1d,2d,3d}</span></code></p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#surface-instructions-suld\" target=\"_blank\" rel=\"noopener noreferrer\">suld <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Load from surface memory.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#surface-instructions-suld"
            };

        case "suq":
            return {
                "html": "<p>Query an attribute of a surface. Operand <code class=\"docutils literal notranslate\"><span class=\"pre\">a</span></code> is a <code class=\"docutils literal notranslate\"><span class=\"pre\">.surfref</span></code> variable or a <code class=\"docutils literal notranslate\"><span class=\"pre\">.u64</span></code> register.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#surface-instructions-suq\" target=\"_blank\" rel=\"noopener noreferrer\">suq <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Query a surface attribute.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#surface-instructions-suq"
            };

        case "sured":
            return {
                "html": "<p>Reduction to surface memory using a surface coordinate vector. The instruction performs a reduction\noperation with data from operand <code class=\"docutils literal notranslate\"><span class=\"pre\">c</span></code> to the surface named by operand <code class=\"docutils literal notranslate\"><span class=\"pre\">a</span></code> at coordinates given by\noperand <code class=\"docutils literal notranslate\"><span class=\"pre\">b</span></code>. Operand <code class=\"docutils literal notranslate\"><span class=\"pre\">a</span></code> is a <code class=\"docutils literal notranslate\"><span class=\"pre\">.surfref</span></code> variable or <code class=\"docutils literal notranslate\"><span class=\"pre\">.u64</span></code> register. Operand <code class=\"docutils literal notranslate\"><span class=\"pre\">b</span></code> is a\nscalar or singleton tuple for 1d surfaces; is a two-element vector for 2d surfaces; and is a\nfour-element vector for 3d surfaces, where the fourth element is ignored. Coordinate elements are of\ntype <code class=\"docutils literal notranslate\"><span class=\"pre\">.s32</span></code>.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#surface-instructions-sured\" target=\"_blank\" rel=\"noopener noreferrer\">sured <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Reduce surface memory.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#surface-instructions-sured"
            };

        case "sust":
            return {
                "html": "<p><code class=\"docutils literal notranslate\"><span class=\"pre\">sust.{1d,2d,3d}</span></code></p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#surface-instructions-sust\" target=\"_blank\" rel=\"noopener noreferrer\">sust <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Store to surface memory.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#surface-instructions-sust"
            };

        case "szext":
            return {
                "html": "<p>Sign-extends or zero-extends an N-bit value from operand <code class=\"docutils literal notranslate\"><span class=\"pre\">a</span></code> where N is specified in operand\n<code class=\"docutils literal notranslate\"><span class=\"pre\">b</span></code>. The resulting value is stored in the destination operand <code class=\"docutils literal notranslate\"><span class=\"pre\">d</span></code>.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-szext\" target=\"_blank\" rel=\"noopener noreferrer\">szext(int) <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Sign-extend or Zero-extend.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-szext"
            };

        case "tanh":
            return {
                "html": "<p>Take hyperbolic tangent value of <code class=\"docutils literal notranslate\"><span class=\"pre\">a</span></code>.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-tanh\" target=\"_blank\" rel=\"noopener noreferrer\">tanh(fp) <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Find the hyperbolic tangent of a value (in radians)",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-tanh"
            };

        case "tcgen05":
            return {
                "html": "<p><code class=\"docutils literal notranslate\"><span class=\"pre\">tcgen05.alloc</span></code> is a potentially blocking instruction which dynamically allocates\nthe specified number of columns in the <a class=\"reference internal\" href=\"#tensor-memory\"><span class=\"std std-ref\">Tensor Memory</span></a> and writes\nthe address of the allocated <a class=\"reference internal\" href=\"#tensor-memory\"><span class=\"std std-ref\">Tensor Memory</span></a> into shared memory\nat the location specified by address operand dst. The <code class=\"docutils literal notranslate\"><span class=\"pre\">tcgen05.alloc</span></code> blocks if the\nrequested amount of <a class=\"reference internal\" href=\"#tensor-memory\"><span class=\"std std-ref\">Tensor Memory</span></a> is not available and unblocks\nas soon as the requested amount of <a class=\"reference internal\" href=\"#tensor-memory\"><span class=\"std std-ref\">Tensor Memory</span></a> becomes\navailable for allocation.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-instructions-tcgen05-alloc-dealloc-relinquish-alloc-permit\" target=\"_blank\" rel=\"noopener noreferrer\">tcgen05.alloc <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "DynamicTensor Memoryallocation management instructions",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-instructions-tcgen05-alloc-dealloc-relinquish-alloc-permit"
            };

        case "tensormap":
            return {
                "html": "<p>The <code class=\"docutils literal notranslate\"><span class=\"pre\">tensormap.replace</span></code> instruction replaces the field, specified by <code class=\"docutils literal notranslate\"><span class=\"pre\">.field</span></code> qualifier,\nof the tensor-map object at the location specified by the address operand <code class=\"docutils literal notranslate\"><span class=\"pre\">addr</span></code> with a\nnew value. The new value is specified by the argument <code class=\"docutils literal notranslate\"><span class=\"pre\">new_val</span></code>.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-tensormap-replace\" target=\"_blank\" rel=\"noopener noreferrer\">tensormap.replace <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Modifies the field of a tensor-map object.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-tensormap-replace"
            };

        case "testp":
            return {
                "html": "<p><code class=\"docutils literal notranslate\"><span class=\"pre\">testp</span></code> tests common properties of floating-point numbers and returns a predicate value of <code class=\"docutils literal notranslate\"><span class=\"pre\">1</span></code>\nif <code class=\"docutils literal notranslate\"><span class=\"pre\">True</span></code> and <code class=\"docutils literal notranslate\"><span class=\"pre\">0</span></code> if <code class=\"docutils literal notranslate\"><span class=\"pre\">False</span></code>.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-testp\" target=\"_blank\" rel=\"noopener noreferrer\">testp(fp) <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Test floating-point property.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-testp"
            };

        case "tex":
            return {
                "html": "<p><code class=\"docutils literal notranslate\"><span class=\"pre\">tex.{1d,2d,3d}</span></code></p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#texture-instructions-tex\" target=\"_blank\" rel=\"noopener noreferrer\">tex <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Perform a texture memory lookup.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#texture-instructions-tex"
            };

        case "tld4":
            return {
                "html": "<p>Texture fetch of the 4-texel bilerp footprint using a texture coordinate vector. The instruction\nloads the bilerp footprint from the texture named by operand <code class=\"docutils literal notranslate\"><span class=\"pre\">a</span></code> at coordinates given by operand\n<code class=\"docutils literal notranslate\"><span class=\"pre\">c</span></code> into vector destination <code class=\"docutils literal notranslate\"><span class=\"pre\">d</span></code>. The texture component fetched for each texel sample is\nspecified by <code class=\"docutils literal notranslate\"><span class=\"pre\">.comp</span></code>. The four texel samples are placed into destination vector <code class=\"docutils literal notranslate\"><span class=\"pre\">d</span></code> in\ncounter-clockwise order starting at lower left.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#texture-instructions-tld4\" target=\"_blank\" rel=\"noopener noreferrer\">tld4 <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Perform a texture fetch of the 4-texel bilerp footprint.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#texture-instructions-tld4"
            };

        case "trap":
            return {
                "html": "<p>Abort execution and generate an interrupt to the host CPU.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-trap\" target=\"_blank\" rel=\"noopener noreferrer\">trap <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Perform trap operation.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-trap"
            };

        case "txq":
            return {
                "html": "<p>Query an attribute of a texture or sampler. Operand <code class=\"docutils literal notranslate\"><span class=\"pre\">a</span></code> is either a <code class=\"docutils literal notranslate\"><span class=\"pre\">.texref</span></code> or <code class=\"docutils literal notranslate\"><span class=\"pre\">.samplerref</span></code> variable, or a <code class=\"docutils literal notranslate\"><span class=\"pre\">.u64</span></code> register.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#texture-instructions-txq\" target=\"_blank\" rel=\"noopener noreferrer\">txq <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Query texture and sampler attributes.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#texture-instructions-txq"
            };

        case "vadd":
            return {
                "html": "<p>Perform scalar arithmetic operation with optional saturate, and optional secondary arithmetic operation or subword data merge.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#scalar-video-instructions-vadd-vsub-vabsdiff-vmin-vmax\" target=\"_blank\" rel=\"noopener noreferrer\">vadd <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Integer byte/half-word/word addition/subtraction.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#scalar-video-instructions-vadd-vsub-vabsdiff-vmin-vmax"
            };

        case "vadd2":
            return {
                "html": "<p>Two-way SIMD parallel arithmetic operation with secondary operation.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#simd-video-instructions-vadd2-vsub2-vavrg2-vabsdiff2-vmin2-vmax2\" target=\"_blank\" rel=\"noopener noreferrer\">vadd2 <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Integer dual half-word SIMD addition/subtraction.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#simd-video-instructions-vadd2-vsub2-vavrg2-vabsdiff2-vmin2-vmax2"
            };

        case "vadd4":
            return {
                "html": "<p>Four-way SIMD parallel arithmetic operation with secondary operation.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#simd-video-instructions-vadd4-vsub4-vavrg4-vabsdiff4-vmin4-vmax4\" target=\"_blank\" rel=\"noopener noreferrer\">vadd4 <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Integer quad byte SIMD addition/subtraction.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#simd-video-instructions-vadd4-vsub4-vavrg4-vabsdiff4-vmin4-vmax4"
            };

        case "vmad":
            return {
                "html": "<p>Calculate <code class=\"docutils literal notranslate\"><span class=\"pre\">(a*b)</span> <span class=\"pre\">+</span> <span class=\"pre\">c</span></code>, with optional operand negates, <em>plus one</em> mode, and scaling.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#scalar-video-instructions-vmad\" target=\"_blank\" rel=\"noopener noreferrer\">vmad <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Integer byte/half-word/word multiply-accumulate.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#scalar-video-instructions-vmad"
            };

        case "vote":
            return {
                "html": "<p>Performs a reduction of the source predicate across all active threads in a warp. The destination\npredicate value is the same across all threads in the warp.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-vote\" target=\"_blank\" rel=\"noopener noreferrer\">vote <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Vote across thread group.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-vote"
            };

        case "vset":
            return {
                "html": "<p>Compare input values using specified comparison, with optional secondary arithmetic operation or\nsubword data merge.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#scalar-video-instructions-vset\" target=\"_blank\" rel=\"noopener noreferrer\">vset <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Integer byte/half-word/word comparison.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#scalar-video-instructions-vset"
            };

        case "vset2":
            return {
                "html": "<p>Two-way SIMD parallel comparison with secondary operation.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#simd-video-instructions-vset2\" target=\"_blank\" rel=\"noopener noreferrer\">vset2 <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Integer dual half-word SIMD comparison.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#simd-video-instructions-vset2"
            };

        case "vset4":
            return {
                "html": "<p>Four-way SIMD parallel comparison with secondary operation.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#simd-video-instructions-vset4\" target=\"_blank\" rel=\"noopener noreferrer\">vset4 <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Integer quad byte SIMD comparison.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#simd-video-instructions-vset4"
            };

        case "vshl":
            return {
                "html": "<dl class=\"simple\">\n<dt><code class=\"docutils literal notranslate\"><span class=\"pre\">vshl</span></code></dt>\n<dd>\n<p>Shift <code class=\"docutils literal notranslate\"><span class=\"pre\">a</span></code> left by unsigned amount in <code class=\"docutils literal notranslate\"><span class=\"pre\">b</span></code> with optional saturate, and optional secondary\narithmetic operation or subword data merge. Left shift fills with zero.</p>\n</dd>\n<dt><code class=\"docutils literal notranslate\"><span class=\"pre\">vshr</span></code></dt>\n<dd>\n<p>Shift <code class=\"docutils literal notranslate\"><span class=\"pre\">a</span></code> right by unsigned amount in <code class=\"docutils literal notranslate\"><span class=\"pre\">b</span></code> with optional saturate, and optional secondary\narithmetic operation or subword data merge. Signed shift fills with the sign bit, unsigned shift\nfills with zero.</p>\n</dd>\n</dl>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#scalar-video-instructions-vshl-vshr\" target=\"_blank\" rel=\"noopener noreferrer\">vshl <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Integer byte/half-word/word left/right shift.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#scalar-video-instructions-vshl-vshr"
            };

        case "wgmma":
            return {
                "html": "<p>Instruction <code class=\"docutils literal notranslate\"><span class=\"pre\">wgmma.mma_async</span></code> issues a <code class=\"docutils literal notranslate\"><span class=\"pre\">MxNxK</span></code> matrix multiply and accumulate operation, <code class=\"docutils literal notranslate\"><span class=\"pre\">D</span> <span class=\"pre\">=</span>\n<span class=\"pre\">A*B+D</span></code>, where the A matrix is <code class=\"docutils literal notranslate\"><span class=\"pre\">MxK</span></code>, the B matrix is <code class=\"docutils literal notranslate\"><span class=\"pre\">KxN</span></code>, and the D matrix is <code class=\"docutils literal notranslate\"><span class=\"pre\">MxN</span></code>.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-for-sparse-wgmma\" target=\"_blank\" rel=\"noopener noreferrer\">wgmma.mma_async <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>, <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma\" target=\"_blank\" rel=\"noopener noreferrer\">wgmma.mma_async <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>, <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma-sp\" target=\"_blank\" rel=\"noopener noreferrer\">wgmma.mma_async <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>, <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-for-sparse-wgmma\" target=\"_blank\" rel=\"noopener noreferrer\">wgmma.mma_async.sp <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>, <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma-sp\" target=\"_blank\" rel=\"noopener noreferrer\">wgmma.mma_async.sp <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Perform matrix multiply-and-accumulate operation across warpgroup",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-for-sparse-wgmma"
            };

        case "wmma":
            return {
                "html": "<p>Collectively load a matrix across all threads in a warp from the location indicated by address\noperand <code class=\"docutils literal notranslate\"><span class=\"pre\">p</span></code> in the specified state space into destination register <code class=\"docutils literal notranslate\"><span class=\"pre\">r</span></code>.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-wmma\" target=\"_blank\" rel=\"noopener noreferrer\">wmma <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>, <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-wmma-ld\" target=\"_blank\" rel=\"noopener noreferrer\">wmma.load <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>, <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-wmma\" target=\"_blank\" rel=\"noopener noreferrer\">wmma.load,wmma.mma <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>, <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-wmma\" target=\"_blank\" rel=\"noopener noreferrer\">wmma.store <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Multiplicands (A or B):",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-wmma"
            };

        case "xor":
            return {
                "html": "<p>Compute the bit-wise exclusive-or operation for the bits in <code class=\"docutils literal notranslate\"><span class=\"pre\">a</span></code> and <code class=\"docutils literal notranslate\"><span class=\"pre\">b</span></code>.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-xor\" target=\"_blank\" rel=\"noopener noreferrer\">xor <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Bitwise exclusive-OR (inequality).",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-xor"
            };

        case "{}":
            return {
                "html": "<p>The curly braces create a group of instructions, used primarily for defining a function body. The\ncurly braces also provide a mechanism for determining the scope of a variable: any variable declared\nwithin a scope is not available outside the scope.</p>\nFor more information, visit <a href=\"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#control-flow-instructions-curly-braces\" target=\"_blank\" rel=\"noopener noreferrer\">{} <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Instruction grouping.",
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#control-flow-instructions-curly-braces"
            };


        }
    }
    