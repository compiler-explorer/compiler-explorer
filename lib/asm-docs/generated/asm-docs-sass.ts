import {AssemblyInstructionInfo} from '../base.js';

export function getAsmOpcode(opcode: string | undefined): AssemblyInstructionInfo | undefined {
    if (!opcode) return;
    switch (opcode) {
        case "ACQBULK":
            return {
                "html": "Wait for Bulk Release Status Warp State<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Wait for Bulk Release Status Warp State",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "ATOM":
            return {
                "html": "Atomic Operation on Generic Memory<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Atomic Operation on Generic Memory",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "ATOMG":
            return {
                "html": "Atomic Operation on Global Memory<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Atomic Operation on Global Memory",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "ATOMS":
            return {
                "html": "Atomic Operation on Shared Memory<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Atomic Operation on Shared Memory",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "B2R":
            return {
                "html": "Move Barrier To Register<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Move Barrier To Register",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "BAR":
            return {
                "html": "Barrier Synchronization<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Barrier Synchronization",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "BFE":
            return {
                "html": "Bit Field Extract<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Bit Field Extract",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "BFI":
            return {
                "html": "Bit Field Insert<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Bit Field Insert",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "BGMMA":
            return {
                "html": "Bit Matrix Multiply and Accumulate Across Warps<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Bit Matrix Multiply and Accumulate Across Warps",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "BMMA":
            return {
                "html": "Bit Matrix Multiply and Accumulate<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Bit Matrix Multiply and Accumulate",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "BMOV":
            return {
                "html": "Move Convergence Barrier State<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Move Convergence Barrier State",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "BMSK":
            return {
                "html": "Bitfield Mask<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Bitfield Mask",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "BPT":
            return {
                "html": "BreakPoint/Trap<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "BreakPoint/Trap",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "BRA":
            return {
                "html": "Relative Branch<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Relative Branch",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "BREAK":
            return {
                "html": "Break out of the Specified Convergence Barrier<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Break out of the Specified Convergence Barrier",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "BREV":
            return {
                "html": "Bit Reverse<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Bit Reverse",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "BRK":
            return {
                "html": "Break<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Break",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "BRX":
            return {
                "html": "Relative Branch Indirect<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Relative Branch Indirect",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "BRXU":
            return {
                "html": "Relative Branch with Uniform Register Based Offset<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Relative Branch with Uniform Register Based Offset",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "BSSY":
            return {
                "html": "Barrier Set Convergence Synchronization Point<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Barrier Set Convergence Synchronization Point",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "BSYNC":
            return {
                "html": "Synchronize Threads on a Convergence Barrier<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Synchronize Threads on a Convergence Barrier",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "CAL":
            return {
                "html": "Relative Call<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Relative Call",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "CALL":
            return {
                "html": "Call Function<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Call Function",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "CCTL":
            return {
                "html": "Cache Control<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Cache Control",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "CCTLL":
            return {
                "html": "Cache Control<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Cache Control",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "CCTLT":
            return {
                "html": "Texture Cache Control<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Texture Cache Control",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "CGAERRBAR":
            return {
                "html": "CGA Error Barrier<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "CGA Error Barrier",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "CONT":
            return {
                "html": "Continue<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Continue",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "CS2R":
            return {
                "html": "Move Special Register to Register<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Move Special Register to Register",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "CSET":
            return {
                "html": "Test Condition Code And Set<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Test Condition Code And Set",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "CSETP":
            return {
                "html": "Test Condition Code and Set Predicate<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Test Condition Code and Set Predicate",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "DADD":
            return {
                "html": "FP64 Add<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "FP64 Add",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "DEPBAR":
            return {
                "html": "Dependency Barrier<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Dependency Barrier",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "DFMA":
            return {
                "html": "FP64 Fused Mutiply Add<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "FP64 Fused Mutiply Add",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "DMMA":
            return {
                "html": "Matrix Multiply and Accumulate<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Matrix Multiply and Accumulate",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "DMNMX":
            return {
                "html": "FP64 Minimum/Maximum<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "FP64 Minimum/Maximum",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "DMUL":
            return {
                "html": "FP64 Multiply<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "FP64 Multiply",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "DSET":
            return {
                "html": "FP64 Compare And Set<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "FP64 Compare And Set",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "DSETP":
            return {
                "html": "FP64 Compare And Set Predicate<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "FP64 Compare And Set Predicate",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "ELECT":
            return {
                "html": "Elect a Leader Thread<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Elect a Leader Thread",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "ENDCOLLECTIVE":
            return {
                "html": "Reset the MCOLLECTIVE mask<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Reset the MCOLLECTIVE mask",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "ERRBAR":
            return {
                "html": "Error Barrier<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Error Barrier",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "EXIT":
            return {
                "html": "Exit Program<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Exit Program",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "F2F":
            return {
                "html": "Floating Point To Floating Point Conversion<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Floating Point To Floating Point Conversion",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "F2I":
            return {
                "html": "Floating Point To Integer Conversion<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Floating Point To Integer Conversion",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "F2IP":
            return {
                "html": "FP32 Down-Convert to Integer and Pack<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "FP32 Down-Convert to Integer and Pack",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "FADD":
            return {
                "html": "FP32 Add<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "FP32 Add",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "FADD32I":
            return {
                "html": "FP32 Add<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "FP32 Add",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "FCHK":
            return {
                "html": "Floating-point Range Check<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Floating-point Range Check",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "FCMP":
            return {
                "html": "FP32 Compare to Zero and Select Source<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "FP32 Compare to Zero and Select Source",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "FENCE":
            return {
                "html": "Memory Visibility Guarantee for Shared or Global Memory<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Memory Visibility Guarantee for Shared or Global Memory",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "FFMA":
            return {
                "html": "FP32 Fused Multiply and Add<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "FP32 Fused Multiply and Add",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "FFMA32I":
            return {
                "html": "FP32 Fused Multiply and Add<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "FP32 Fused Multiply and Add",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "FLO":
            return {
                "html": "Find Leading One<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Find Leading One",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "FMNMX":
            return {
                "html": "FP32 Minimum/Maximum<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "FP32 Minimum/Maximum",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "FMUL":
            return {
                "html": "FP32 Multiply<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "FP32 Multiply",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "FMUL32I":
            return {
                "html": "FP32 Multiply<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "FP32 Multiply",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "FRND":
            return {
                "html": "Round To Integer<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Round To Integer",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "FSEL":
            return {
                "html": "Floating Point Select<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Floating Point Select",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "FSET":
            return {
                "html": "FP32 Compare And Set<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "FP32 Compare And Set",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "FSETP":
            return {
                "html": "FP32 Compare And Set Predicate<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "FP32 Compare And Set Predicate",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "FSWZADD":
            return {
                "html": "FP32 Swizzle Add<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "FP32 Swizzle Add",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "GETLMEMBASE":
            return {
                "html": "Get Local Memory Base Address<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Get Local Memory Base Address",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "HADD2":
            return {
                "html": "FP16 Add<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "FP16 Add",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "HADD2_32I":
            return {
                "html": "FP16 Add<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "FP16 Add",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "HFMA2":
            return {
                "html": "FP16 Fused Mutiply Add<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "FP16 Fused Mutiply Add",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "HFMA2_32I":
            return {
                "html": "FP16 Fused Mutiply Add<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "FP16 Fused Mutiply Add",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "HGMMA":
            return {
                "html": "Matrix Multiply and Accumulate Across a Warpgroup<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Matrix Multiply and Accumulate Across a Warpgroup",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "HMMA":
            return {
                "html": "Matrix Multiply and Accumulate<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Matrix Multiply and Accumulate",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "HMNMX2":
            return {
                "html": "FP16 Minimum / Maximum<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "FP16 Minimum / Maximum",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "HMUL2":
            return {
                "html": "FP16 Multiply<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "FP16 Multiply",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "HMUL2_32I":
            return {
                "html": "FP16 Multiply<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "FP16 Multiply",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "HSET2":
            return {
                "html": "FP16 Compare And Set<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "FP16 Compare And Set",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "HSETP2":
            return {
                "html": "FP16 Compare And Set Predicate<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "FP16 Compare And Set Predicate",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "I2F":
            return {
                "html": "Integer To Floating Point Conversion<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Integer To Floating Point Conversion",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "I2FP":
            return {
                "html": "Integer to FP32 Convert and Pack<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Integer to FP32 Convert and Pack",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "I2I":
            return {
                "html": "Integer To Integer Conversion<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Integer To Integer Conversion",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "I2IP":
            return {
                "html": "Integer To Integer Conversion and Packing<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Integer To Integer Conversion and Packing",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "IABS":
            return {
                "html": "Integer Absolute Value<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Integer Absolute Value",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "IADD":
            return {
                "html": "Integer Addition<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Integer Addition",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "IADD3":
            return {
                "html": "3-input Integer Addition<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "3-input Integer Addition",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "IADD32I":
            return {
                "html": "Integer Addition<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Integer Addition",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "ICMP":
            return {
                "html": "Integer Compare to Zero and Select Source<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Integer Compare to Zero and Select Source",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "IDP":
            return {
                "html": "Integer Dot Product and Accumulate<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Integer Dot Product and Accumulate",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "IDP4A":
            return {
                "html": "Integer Dot Product and Accumulate<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Integer Dot Product and Accumulate",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "IGMMA":
            return {
                "html": "Integer Matrix Multiply and Accumulate Across a Warpgroup<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Integer Matrix Multiply and Accumulate Across a Warpgroup",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "IMAD":
            return {
                "html": "Integer Multiply And Add<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Integer Multiply And Add",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "IMADSP":
            return {
                "html": "Extracted Integer Multiply And Add.<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Extracted Integer Multiply And Add.",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "IMMA":
            return {
                "html": "Integer Matrix Multiply and Accumulate<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Integer Matrix Multiply and Accumulate",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "IMNMX":
            return {
                "html": "Integer Minimum/Maximum<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Integer Minimum/Maximum",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "IMUL":
            return {
                "html": "Integer Multiply<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Integer Multiply",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "IMUL32I":
            return {
                "html": "Integer Multiply<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Integer Multiply",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "ISCADD":
            return {
                "html": "Scaled Integer Addition<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Scaled Integer Addition",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "ISCADD32I":
            return {
                "html": "Scaled Integer Addition<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Scaled Integer Addition",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "ISET":
            return {
                "html": "Integer Compare And Set<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Integer Compare And Set",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "ISETP":
            return {
                "html": "Integer Compare And Set Predicate<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Integer Compare And Set Predicate",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "JCAL":
            return {
                "html": "Absolute Call<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Absolute Call",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "JMP":
            return {
                "html": "Absolute Jump<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Absolute Jump",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "JMX":
            return {
                "html": "Absolute Jump Indirect<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Absolute Jump Indirect",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "JMXU":
            return {
                "html": "Absolute Jump with Uniform Register Based Offset<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Absolute Jump with Uniform Register Based Offset",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "KILL":
            return {
                "html": "Kill Thread<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Kill Thread",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "LD":
            return {
                "html": "Load from generic Memory<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Load from generic Memory",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "LDC":
            return {
                "html": "Load Constant<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Load Constant",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "LDG":
            return {
                "html": "Load from Global Memory<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Load from Global Memory",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "LDGDEPBAR":
            return {
                "html": "Global Load Dependency Barrier<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Global Load Dependency Barrier",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "LDGMC":
            return {
                "html": "Reducing Load<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Reducing Load",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "LDGSTS":
            return {
                "html": "Asynchronous Global to Shared Memcopy<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Asynchronous Global to Shared Memcopy",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "LDL":
            return {
                "html": "Load within Local Memory Window<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Load within Local Memory Window",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "LDS":
            return {
                "html": "Load within Shared Memory Window<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Load within Shared Memory Window",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "LDSM":
            return {
                "html": "Load Matrix from Shared Memory with Element Size Expansion<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Load Matrix from Shared Memory with Element Size Expansion",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "LEA":
            return {
                "html": "LOAD Effective Address<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "LOAD Effective Address",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "LEPC":
            return {
                "html": "Load Effective PC<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Load Effective PC",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "LOP":
            return {
                "html": "Logic Operation<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Logic Operation",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "LOP3":
            return {
                "html": "Logic Operation<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Logic Operation",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "LOP32I":
            return {
                "html": "Logic Operation<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Logic Operation",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "MATCH":
            return {
                "html": "Match Register Values Across Thread Group<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Match Register Values Across Thread Group",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "MEMBAR":
            return {
                "html": "Memory Barrier<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Memory Barrier",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "MOV":
            return {
                "html": "Move<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Move",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "MOV32I":
            return {
                "html": "Move<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Move",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "MOVM":
            return {
                "html": "Move Matrix with Transposition or Expansion<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Move Matrix with Transposition or Expansion",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "MUFU":
            return {
                "html": "FP32 Multi Function Operation<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "FP32 Multi Function Operation",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "NANOSLEEP":
            return {
                "html": "Suspend Execution<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Suspend Execution",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "NOP":
            return {
                "html": "No Operation<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "No Operation",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "P2R":
            return {
                "html": "Move Predicate Register To Register<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Move Predicate Register To Register",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "PBK":
            return {
                "html": "Pre-Break<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Pre-Break",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "PCNT":
            return {
                "html": "Pre-continue<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Pre-continue",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "PEXIT":
            return {
                "html": "Pre-Exit<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Pre-Exit",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "PLOP3":
            return {
                "html": "Predicate Logic Operation<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Predicate Logic Operation",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "PMTRIG":
            return {
                "html": "Performance Monitor Trigger<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Performance Monitor Trigger",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "POPC":
            return {
                "html": "Population count<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Population count",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "PREEXIT":
            return {
                "html": "Dependent Task Launch Hint<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Dependent Task Launch Hint",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "PRET":
            return {
                "html": "Pre-Return From Subroutine<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Pre-Return From Subroutine",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "PRMT":
            return {
                "html": "Permute Register Pair<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Permute Register Pair",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "PSET":
            return {
                "html": "Combine Predicates and Set<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Combine Predicates and Set",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "PSETP":
            return {
                "html": "Combine Predicates and Set Predicate<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Combine Predicates and Set Predicate",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "QGMMA":
            return {
                "html": "FP8 Matrix Multiply and Accumulate Across a Warpgroup<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "FP8 Matrix Multiply and Accumulate Across a Warpgroup",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "QSPC":
            return {
                "html": "Query Space<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Query Space",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "R2B":
            return {
                "html": "Move Register to Barrier<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Move Register to Barrier",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "R2P":
            return {
                "html": "Move Register To Predicate Register<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Move Register To Predicate Register",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "R2UR":
            return {
                "html": "Move from Vector Register to a Uniform Register<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Move from Vector Register to a Uniform Register",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "RED":
            return {
                "html": "Reduction Operation on Generic Memory<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Reduction Operation on Generic Memory",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "REDAS":
            return {
                "html": "Asynchronous Reduction on Distributed Shared Memory With Explicit Synchronization<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Asynchronous Reduction on Distributed Shared Memory With Explicit Synchronization",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "REDG":
            return {
                "html": "Reduction Operation on Generic Memory<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Reduction Operation on Generic Memory",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "REDUX":
            return {
                "html": "Reduction of a Vector Register into a Uniform Register<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Reduction of a Vector Register into a Uniform Register",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "RET":
            return {
                "html": "Return From Subroutine<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Return From Subroutine",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "RPCMOV":
            return {
                "html": "PC Register Move<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "PC Register Move",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "RRO":
            return {
                "html": "Range Reduction Operator FP<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Range Reduction Operator FP",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "RTT":
            return {
                "html": "Return From Trap<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Return From Trap",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "S2R":
            return {
                "html": "Move Special Register to Register<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Move Special Register to Register",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "S2UR":
            return {
                "html": "Move Special Register to Uniform Register<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Move Special Register to Uniform Register",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "SEL":
            return {
                "html": "Select Source with Predicate<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Select Source with Predicate",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "SETCTAID":
            return {
                "html": "Set CTA ID<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Set CTA ID",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "SETLMEMBASE":
            return {
                "html": "Set Local Memory Base Address<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Set Local Memory Base Address",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "SGXT":
            return {
                "html": "Sign Extend<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Sign Extend",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "SHF":
            return {
                "html": "Funnel Shift<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Funnel Shift",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "SHFL":
            return {
                "html": "Warp Wide Register Shuffle<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Warp Wide Register Shuffle",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "SHL":
            return {
                "html": "Shift Left<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Shift Left",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "SHR":
            return {
                "html": "Shift Right<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Shift Right",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "SSY":
            return {
                "html": "Set Synchronization Point<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Set Synchronization Point",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "ST":
            return {
                "html": "Store to Generic Memory<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Store to Generic Memory",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "STAS":
            return {
                "html": "Asynchronous Store to Distributed Shared Memory With Explicit Synchronization<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Asynchronous Store to Distributed Shared Memory With Explicit Synchronization",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "STG":
            return {
                "html": "Store to Global Memory<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Store to Global Memory",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "STL":
            return {
                "html": "Store within Local or Shared Window<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Store within Local or Shared Window",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "STS":
            return {
                "html": "Store within Local or Shared Window<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Store within Local or Shared Window",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "STSM":
            return {
                "html": "Store Matrix to Shared Memory<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Store Matrix to Shared Memory",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "SUATOM":
            return {
                "html": "Atomic Op on Surface Memory<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Atomic Op on Surface Memory",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "SULD":
            return {
                "html": "Surface Load<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Surface Load",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "SURED":
            return {
                "html": "Reduction Op on Surface Memory<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Reduction Op on Surface Memory",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "SUST":
            return {
                "html": "Surface Store<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Surface Store",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "SYNC":
            return {
                "html": "Converge threads after conditional branch<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Converge threads after conditional branch",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "SYNCS":
            return {
                "html": "Sync Unit<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Sync Unit",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "TEX":
            return {
                "html": "Texture Fetch<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Texture Fetch",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "TEXS":
            return {
                "html": "Texture Fetch with scalar/non-vec4 source/destinations<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Texture Fetch with scalar/non-vec4 source/destinations",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "TLD":
            return {
                "html": "Texture Load<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Texture Load",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "TLD4":
            return {
                "html": "Texture Load 4<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Texture Load 4",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "TLD4S":
            return {
                "html": "Texture Load 4 with scalar/non-vec4 source/destinations<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Texture Load 4 with scalar/non-vec4 source/destinations",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "TLDS":
            return {
                "html": "Texture Load with scalar/non-vec4 source/destinations<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Texture Load with scalar/non-vec4 source/destinations",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "TMML":
            return {
                "html": "Texture MipMap Level<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Texture MipMap Level",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "TXD":
            return {
                "html": "Texture Fetch With Derivatives<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Texture Fetch With Derivatives",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "TXQ":
            return {
                "html": "Texture Query<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Texture Query",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "UBLKCP":
            return {
                "html": "Bulk Data Copy<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Bulk Data Copy",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "UBLKPF":
            return {
                "html": "Bulk Data Prefetch<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Bulk Data Prefetch",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "UBLKRED":
            return {
                "html": "Bulk Data Copy from Shared Memory with Reduction<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Bulk Data Copy from Shared Memory with Reduction",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "UBMSK":
            return {
                "html": "Uniform Bitfield Mask<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Uniform Bitfield Mask",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "UBREV":
            return {
                "html": "Uniform Bit Reverse<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Uniform Bit Reverse",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "UCGABAR_ARV":
            return {
                "html": "CGA Barrier Synchronization<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "CGA Barrier Synchronization",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "UCGABAR_WAIT":
            return {
                "html": "CGA Barrier Synchronization<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "CGA Barrier Synchronization",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "UCLEA":
            return {
                "html": "Load Effective Address for a Constant<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Load Effective Address for a Constant",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "UF2FP":
            return {
                "html": "Uniform FP32 Down-convert and Pack<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Uniform FP32 Down-convert and Pack",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "UFLO":
            return {
                "html": "Uniform Find Leading One<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Uniform Find Leading One",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "UIADD3":
            return {
                "html": "Uniform Integer Addition<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Uniform Integer Addition",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "UIADD3.64":
            return {
                "html": "Uniform Integer Addition<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Uniform Integer Addition",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "UIMAD":
            return {
                "html": "Uniform Integer Multiplication<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Uniform Integer Multiplication",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "UISETP":
            return {
                "html": "Integer Compare and Set Uniform Predicate<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Integer Compare and Set Uniform Predicate",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "ULDC":
            return {
                "html": "Load from Constant Memory into a Uniform Register<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Load from Constant Memory into a Uniform Register",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "ULEA":
            return {
                "html": "Uniform Load Effective Address<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Uniform Load Effective Address",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "ULEPC":
            return {
                "html": "Uniform Load Effective PC<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Uniform Load Effective PC",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "ULOP":
            return {
                "html": "Logic Operation<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Logic Operation",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "ULOP3":
            return {
                "html": "Logic Operation<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Logic Operation",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "ULOP32I":
            return {
                "html": "Logic Operation<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Logic Operation",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "UMOV":
            return {
                "html": "Uniform Move<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Uniform Move",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "UP2UR":
            return {
                "html": "Uniform Predicate to Uniform Register<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Uniform Predicate to Uniform Register",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "UPLOP3":
            return {
                "html": "Uniform Predicate Logic Operation<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Uniform Predicate Logic Operation",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "UPOPC":
            return {
                "html": "Uniform Population Count<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Uniform Population Count",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "UPRMT":
            return {
                "html": "Uniform Byte Permute<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Uniform Byte Permute",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "UPSETP":
            return {
                "html": "Uniform Predicate Logic Operation<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Uniform Predicate Logic Operation",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "UR2UP":
            return {
                "html": "Uniform Register to Uniform Predicate<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Uniform Register to Uniform Predicate",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "USEL":
            return {
                "html": "Uniform Select<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Uniform Select",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "USETMAXREG":
            return {
                "html": "Release, Deallocate and Allocate Registers<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Release, Deallocate and Allocate Registers",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "USGXT":
            return {
                "html": "Uniform Sign Extend<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Uniform Sign Extend",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "USHF":
            return {
                "html": "Uniform Funnel Shift<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Uniform Funnel Shift",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "USHL":
            return {
                "html": "Uniform Left Shift<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Uniform Left Shift",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "USHR":
            return {
                "html": "Uniform Right Shift<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Uniform Right Shift",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "UTMACCTL":
            return {
                "html": "TMA Cache Control<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "TMA Cache Control",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "UTMACMDFLUSH":
            return {
                "html": "TMA Command Flush<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "TMA Command Flush",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "UTMALDG":
            return {
                "html": "Tensor Load from Global to Shared Memory<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Tensor Load from Global to Shared Memory",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "UTMAPF":
            return {
                "html": "Tensor Prefetch<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Tensor Prefetch",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "UTMAREDG":
            return {
                "html": "Tensor Store from Shared to Global Memory with Reduction<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Tensor Store from Shared to Global Memory with Reduction",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "UTMASTG":
            return {
                "html": "Tensor Store from Shared to Global Memory<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Tensor Store from Shared to Global Memory",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "VABSDIFF":
            return {
                "html": "Absolute Difference<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Absolute Difference",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "VABSDIFF4":
            return {
                "html": "Absolute Difference<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Absolute Difference",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "VHMNMX":
            return {
                "html": "SIMD FP16 3-Input Minimum / Maximum<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "SIMD FP16 3-Input Minimum / Maximum",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "VIADD":
            return {
                "html": "SIMD Integer Addition<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "SIMD Integer Addition",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "VIADDMNMX":
            return {
                "html": "SIMD Integer Addition and Fused Min/Max Comparison<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "SIMD Integer Addition and Fused Min/Max Comparison",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "VIMNMX":
            return {
                "html": "SIMD Integer Minimum / Maximum<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "SIMD Integer Minimum / Maximum",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "VIMNMX3":
            return {
                "html": "SIMD Integer 3-Input Minimum / Maximum<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "SIMD Integer 3-Input Minimum / Maximum",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "VOTE":
            return {
                "html": "Vote Across SIMT Thread Group<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Vote Across SIMT Thread Group",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "VOTEU":
            return {
                "html": "Voting across SIMD Thread Group with Results in Uniform Destination<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Voting across SIMD Thread Group with Results in Uniform Destination",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "WARPGROUP":
            return {
                "html": "Warpgroup Synchronization<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Warpgroup Synchronization",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "WARPGROUPSET":
            return {
                "html": "Set Warpgroup Counters<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Set Warpgroup Counters",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "WARPSYNC":
            return {
                "html": "Synchronize Threads in Warp<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Synchronize Threads in Warp",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "XMAD":
            return {
                "html": "Integer Short Multiply Add<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Integer Short Multiply Add",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };

        case "YIELD":
            return {
                "html": "Yield Control<br><br>For more information, visit <a href=\"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14\" target=\"_blank\" rel=\"noopener noreferrer\">CUDA Binary Utilities documentation <sup><small class=\"fas fa-external-link-alt opens-new-window\" title=\"Opens in a new window\"></small></sup></a>.",
                "tooltip": "Yield Control",
                "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
            };


    }
}
