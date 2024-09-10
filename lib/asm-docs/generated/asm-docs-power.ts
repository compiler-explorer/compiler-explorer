import { AssemblyInstructionInfo } from "../base.js";

// Based on the IBM documentation of assembly instructions for AIX 7.3 (https://www.ibm.com/docs/en/aix/7.3?topic=reference-instruction-set).
//
// An automatic generator is available at etc/scripts/docenizers/docenizer-power.py, but it has a lot of quirks and is considered incomplete.
// As such, this was created manually to have a complete documentation of the current baseline ISA.
//
// There are a *lot* of instructions that have been defined in later versions of the Power ISA that IBM hasn't written documentation for yet.
// This includes a lot of convenience "extended" instructions, every AltiVec/VMX instruction, and a myriad of other new instructions.
// If you're up to the significant challenge, feel free to start working through the OpenPOWER ISA manual and start writing proper documentation
// for the many missing instructions that IBM hasn't documented yet. The latest revision is always available at https://openpowerfoundation.org/specifications/isa/.
export function getAsmOpcode(opcode: string | undefined): AssemblyInstructionInfo | undefined {
    if (!opcode) return;
    switch (opcode.toUpperCase()) {
        case "ABS":
        case "ABS.":
        case "ABSO":
        case "ABSO.":
            return {
                "html": `
                    <p>The <strong>abs</strong> instruction places the absolute value of the contents of general-purpose register (GPR) <em>RA</em> into the target GPR <em>RT</em>.</p>
                    <p>If GPR <em>RA</em> contains the most negative number ('8000 0000'), the result of the instruction is the most negative number, and the instruction will set the Overflow bit in the Fixed-Point Exception Register to 1 if the OE bit is set to 1.</p>
                `,
                "tooltip": "Absolute",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-abs-absolute-instruction"
            };
        case "ADD":
        case "ADD.":
        case "ADDO":
        case "ADDO.":
        case "CAX":
        case "CAX.":
        case "CAXO":
        case "CAXO.":
            return {
                "html": `
                    <p>The <strong>add</strong> and <strong>cax</strong> instructions place the sum of the contents of general-purpose register (GPR) <em>RA</em> and GPR <em>RB</em> into the target GPR <em>RT</em>.</p>
                `,
                "tooltip": "Add or Compute Address",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-add-add-cax-compute-address-instruction"
            };
        case "A":
        case "A.":
        case "AO":
        case "AO.":
        case "ADDC":
        case "ADDC.":
        case "ADDCO":
        case "ADDCO.":
            return {
                "html": `
                    <p>The <strong>addc</strong> and <strong>a</strong> instructions place the sum of the contents of general-purpose register (GPR) <em>RA</em> and GPR <em>RB</em> into the target GPR <em>RT</em>.</p>
                `,
                "tooltip": "Add Carrying",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-addc-add-carrying-instruction"
            };
        case "AE":
        case "AE.":
        case "AEO":
        case "AEO.":
        case "ADDE":
        case "ADDE.":
        case "ADDEO":
        case "ADDEO.":
            return {
                "html": `
                    <p>The <strong>adde</strong> and <strong>ae</strong> instructions place the sum of the contents of general-purpose register (GPR) <em>RA</em>, GPR <em>RB</em>, and the Carry bit into the target GPR <em>RT</em>.</p>
                `,
                "tooltip": "Add Extended",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-adde-ae-add-extended-instruction"
            };
        case "ADDI":
        case "CAL":
            return {
                "html": `
                    <p>The <strong>addi</strong> and <strong>cal</strong> instructions place the sum of the contents of general-purpose register (GPR) <em>RA</em> and the 16-bit two's complement integer <em>SI</em> or <em>D</em>, sign-extended to 32 bits, into the target GPR <em>RT</em>. If GPR <em>RA</em> is GPR 0, then <em>SI</em> or <em>D</em> is stored into the target GPR <em>RT</em>.</p>
                    <p>The <strong>addi</strong> and <strong>cal</strong> instructions have one syntax form and do not affect Condition Register Field 0 or the Fixed-Point Exception Register.</p>
                `,
                "tooltip": "Add Immediate or Compute Address Lower",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=is-addi-add-immediate-cal-compute-address-lower-instruction"
            };
        case "AI":
        case "ADDIC":
            return {
                "html": `
                    <p>The <strong>addic</strong> and <strong>ai</strong> instructions place the sum of the contents of general-purpose register (GPR) <em>RA</em> and a 16-bit signed integer, <em>SI</em>, into target GPR <em>RT</em>.</p>
                    <p>The 16-bit integer provided as immediate data is sign-extended to 32 bits prior to carrying out the addition operation.</p>
                    <p>The <strong>addic</strong> and <strong>ai</strong> instructions have one syntax form and can set the Carry bit of the Fixed-Point Exception Register; these instructions never affect Condition Register Field 0.</p>
                `,
                "tooltip": "Add Immediate Carrying",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-addic-ai-add-immediate-carrying-instruction"
            };
        case "AI.":
        case "ADDIC.":
            return {
                "html": `
                    <p>The <strong>addic.</strong> and <strong>ai.</strong> instructions place the sum of the contents of general-purpose register (GPR) <em>RA</em> and a 16-bit signed integer, <em>SI</em>, into the target GPR <em>RT</em>.</p>
                    <p>The 16-bit integer <em>SI</em> provided as immediate data is sign-extended to 32 bits prior to carrying out the addition operation.</p>
                    <p>The <strong>addic.</strong> and <strong>ai.</strong> instructions have one syntax form and can set the Carry Bit of the Fixed-Point Exception Register. These instructions also affect Condition Register Field 0.</p>
                `,
                "tooltip": "Add Immediate Carrying and Record",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-addic-ai-add-immediate-carrying-record-instruction"
            };
        case "ADDIS":
        case "CAU":
            return {
                "html": `
                    <p>The <strong>addis</strong> and <strong>cau</strong> instructions place the sum of the contents of general-purpose register (GPR) <em>RA</em> and the concatenation of a 16-bit unsigned integer, <em>SI</em> or <em>UI,</em> and x'0000' into the target GPR <em>RT</em>. If GPR <em>RA</em> is GPR 0, then the sum of the concatenation of 0, <em>SI</em> or <em>UI</em>, and x'0000' is stored into the target GPR <em>RT</em>.</p>
                    The <strong>cau</strong> instruction has one syntax form. The <strong>addis</strong> instruction has two syntax forms; however, the second form is only valid when the R_TOCU relocation type is used in the <em>D</em> expression. The R_TOCU relocation type can be specified explicitly with the <strong>@u</strong> relocation specifier or implicitly by using a <strong>QualName</strong> parameter with a TE storage-mapping class.</p>
                    <blockquote><strong>Note:</strong> The immediate value for the <strong>cau</strong> instruction is a 16-bit unsigned integer, whereas the immediate value for the <strong>addis</strong> instruction is a 16-bit signed integer. This difference is a result of extending the architecture to 64 bits.</blockquote>
                    <p>The assembler does a 0 to 65535 value-range check for the <em>UI</em> field, and a -32768 to 32767 value-range check for the <em>SI</em> field.</p>
                    <p>To keep the source compatibility of the <strong>addis</strong> and <strong>cau</strong> instructions, the assembler expands the value-range check for the <strong>addis</strong> instruction to -65536 to 65535. The sign bit is ignored and the assembler only ensures that the immediate value fits into 16 bits. This expansion does not affect the behavior of a 32-bit implementation or 32-bit mode in a 64-bit implementation.</p>
                    <p>The <strong>addis</strong> instruction has different semantics in 32-bit mode than it does in 64-bit mode. If bit 32 is set, it propagates through the upper 32 bits of the 64-bit general-purpose register. Use caution when using the <strong>addis</strong> instruction to construct an unsigned integer. The <strong>addis</strong> instruction with an unsigned integer in 32-bit may not be directly ported to 64-bit mode. The code sequence needed to construct an unsigned integer in 64-bit mode is significantly different from that needed in 32-bit mode.</p>
                `,
                "tooltip": "Add Immediate Shifted",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-addis-cau-add-immediate-shifted-instruction"
            };
        case "AME":
        case "AME.":
        case "AMEO":
        case "AMEO.":
        case "ADDME":
        case "ADDME.":
        case "ADDMEO":
        case "ADDMEO.":
            return {
                "html": `
                    <p>The <strong>addme</strong> and <strong>ame</strong> instructions place the sum of the contents of general-purpose register (GPR) <em>RA</em>, the Carry bit of the Fixed-Point Exception Register, and -1 (0xFFFF FFFF<samp>)</samp> into the target GPR <em>RT</em>.</p>
                `,
                "tooltip": "Add to Minus One Extended",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-addme-ame-add-minus-one-extended-instruction"
            };
        case "AZE":
        case "AZE.":
        case "AZEO":
        case "AZEO.":
        case "ADDZE":
        case "ADDZE.":
        case "ADDZEO":
        case "ADDZEO.":
            return {
                "html": `
                    <p>The <strong>addze</strong> and <strong>aze</strong> instructions add the contents of general-purpose register (GPR) <em>RA</em>, the Carry bit, and 0x0000 0000 and place the result into the target GPR <em>RT</em>.</p>
                `,
                "tooltip": "Add to Zero Extended",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-addze-aze-add-zero-extended-instruction"
            };
        case "AND":
        case "AND.":
            return {
                "html": `
                    <p>The <strong>and</strong> instruction logically ANDs the contents of general-purpose register (GPR) <em>RS</em> with the contents of GPR <em>RB</em> and places the result into the target GPR <em>RA</em>.</p>
                `,
                "tooltip": "AND",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-instruction"
            };
        case "ANDC":
        case "ANDC.":
            return {
                "html": `
                    <p>The <strong>andc</strong> instruction logically ANDs the contents of general-purpose register (GPR) <em>RS</em> with the complement of the contents of GPR <em>RB</em> and places the result into GPR <em>RA</em>.</p>
                `,
                "tooltip": "AND with Complement",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-andc-complement-instruction"
            };
        case "ANDI.":
        case "ANDIL.":
            return {
                "html": `
                    <p>The <strong>andi.</strong> and <strong>andil.</strong> instructions logically AND the contents of general-purpose register (GPR) <em>RS</em> with the concatenation of x'0000' and a 16-bit unsigned integer, <em>UI,</em> and place the result in GPR <em>RA</em>.</p>
                    <p>The <strong>andi.</strong> and <strong>andil.</strong> instructions have one syntax form and never affect the Fixed-Point Exception Register. The <strong>andi.</strong> and <strong>andil.</strong> instructions copies the Summary Overflow (SO) bit from the Fixed-Point Exception Register into Condition Register Field 0 and sets one of the Less Than (LT), Greater Than (GT), or Equal To (EQ) bits of Condition Register Field 0.</p>
                `,
                "tooltip": "AND Immediate",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-andi-andil-immediate-instruction"
            };
        case "ANDIS.":
        case "ANDIU.":
            return {
                "html": `
                    <p>The <strong>andis.</strong> and <strong>andiu.</strong> instructions logically AND the contents of general-purpose register (GPR) <em>RS</em> with the concatenation of a 16-bit unsigned integer, <em>UI,</em> and x'0000' and then place the result into the target GPR <em>RA</em>.</p>
                    <p>The <strong>andis.</strong> and <strong>andiu.</strong> instructions have one syntax form and never affect the Fixed-Point Exception Register. The <strong>andis.</strong> and <strong>andiu.</strong> instructions set the Less Than (LT) zero, Greater Than (GT) zero, Equal To (EQ) zero, or Summary Overflow (SO) bit in Condition Register Field 0.</p>
                `,
                "tooltip": "AND Immediate Shifted",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-andis-andiu-immediate-shifted-instruction"
            };
        case "B":
        case "BA":
        case "BL":
        case "BLA":
            return {
                "html": `
                    <p>The <strong>b</strong> instruction branches to an instruction specified by the branch target address. The branch target address is computed one of two ways.</p>
                    <p>Consider the following when using the <strong>b</strong> instruction:</p>
                    <ul>
                        <li>If the Absolute Address bit (AA) is 0, the branch target address is computed by concatenating the 24-bit <em>LI</em> field. This field is calculated by subtracting the address of the instruction from the target address and dividing the result by 4 and b<samp>'</samp>00<samp>'</samp>. The result is then sign-extended to 32 bits and added to the address of this branch instruction.</li>
                        <li>If the AA bit is 1, then the branch target address is the <em>LI</em> field concatenated with b<samp>'</samp>00<samp>'</samp> sign-extended to 32 bits. The <em>LI</em> field is the low-order 26 bits of the target address divided by four.</li>
                    </ul>
                `,
                "tooltip": "Branch",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-b-branch-instruction"
            };
        case "BC":
        case "BCA":
        case "BCL":
        case "BCLA":
            return {
                "html": `
                    <p>The <strong>bc</strong> instruction branches to an instruction specified by the branch target address. The branch target address is computed one of two ways:</p>
                    <ul>
                        <li>If the Absolute Address bit (AA) is 0, then the branch target address is computed by concatenating the 14-bit Branch Displacement (BD) and b'00', sign-extending this to 32 bits, and adding the result to the address of this branch instruction.</li>
                        <li>If the AA is 1, then the branch target address is BD concatenated with b'00' sign-extended to 32 bits.</li>
                    </ul>
                `,
                "tooltip": "Branch Conditional",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-bc-branch-conditional-instruction"
            };
        case "BCC":
        case "BCCL":
        case "BCCTR":
        case "BCCTRL":
            return {
                "html": `<p>The <strong>bcctr</strong> and <strong>bcc</strong> instructions conditionally branch to an instruction specified by the branch target address contained within the Count Register. The branch target address is the concatenation of Count Register bits 0-29 and b'00'.</p>`,
                "tooltip": "Branch Conditional to Count Register",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-bcctr-bcc-branch-conditional-count-register-instruction"
            };
        case "BCR":
        case "BCRL":
        case "BCLR":
        case "BCLRL":
            return {
                "html": `<p>The <strong>bclr</strong> and <strong>bcr</strong> instructions branch to an instruction specified by the branch target address. The branch target address is the concatenation of bits 0-29 of the Link Register and b'00'.</p>`,
                "tooltip": "Branch Conditional Link Register",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-bclr-bcr-branch-conditional-link-register-instruction"
            };
        case "CLCS":
            return {
                "html": `<p>The <strong>clcs</strong> instruction places the cache line size specified by <em>RA</em> into the target general-purpose register (GPR) <em>RT</em>. The value of <em>RA</em> determines the cache line size returned in GPR <em>RT</em>.</p>`,
                "tooltip": "Cache Line Compute Size",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-clcs-cache-line-compute-size-instruction"
            };
        case "CLF":
            return {
                "html": `<p>The <strong>clf</strong> instruction calculates an effective address (EA) by adding the contents of general-purpose register (GPR) <em>RA</em> to the contents of GPR <em>RB</em>. If the <em>RA</em> field is 0, EA is the sum of the contents of <em>RB</em> and 0. If the <em>RA</em> field is not 0 and if the instruction does not cause a data storage interrupt, the result of the operation is placed back into GPR <em>RA</em>.</p>`,
                "tooltip": "Cache Line Flush",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-clf-cache-line-flush-instruction"
            };
        case "CLI":
            return {
                "html": `<p>The <strong>cli</strong> instruction invalidates a line containing the byte addressed in either the data or instruction cache. If <em>RA</em> is not 0, the <strong>cli</strong> instruction calculates an effective address (EA) by adding the contents of general-purpose register (GPR) <em>RA</em> to the contents of GPR <em>RB</em>. If <em>RA</em> is not GPR 0 or the instruction does not cause a Data Storage interrupt, the result of the calculation is placed back into GPR <em>RA</em>.</p>`,
                "tooltip": "Cache Line Invalidate",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-cli-cache-line-invalidate-instruction"
            };
        case "CMP":
            return {
                "html": `
                    <p>The <strong>cmp</strong> instruction compares the contents of general-purpose register (GPR) <em>RA</em> with the contents of GPR <em>RB</em> as signed integers and sets one of the bits in Condition Register Field <em>BF</em>.</p>
                    <p><em>BF</em> can be Condition Register Field 0-7; programmers can specify which Condition Register Field will indicate the result of the operation.</p>
                `,
                "tooltip": "Compare",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-cmp-compare-instruction"
            };
        case "CMPI":
            return {
                "html": `
                    <p>The <strong>cmpi</strong> instruction compares the contents of general-purpose register (GPR) <em>RA</em> and a 16- bit signed integer, <em>S</em>I, as signed integers and sets one of the bits in Condition Register Field <em>BF</em>.</p>
                    <p><em>BF</em> can be Condition Register Field 0-7; programmers can specify which Condition Register Field will indicate the result of the operation.</p>
                `,
                "tooltip": "Compare Immediate",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-cmpi-compare-immediate-instruction"
            };
        case "CMPL":
            return {
                "html": `
                    <p>The <strong>cmpl</strong> instruction compares the contents of general-purpose register (GPR) <em>RA</em> with the contents of GPR <em>RB</em> as unsigned integers and sets one of the bits in Condition Register Field <em>BF</em>.</p>
                    <p><em>BF</em> can be Condition Register Field 0-7; programmers can specify which Condition Register Field will indicate the result of the operation.</p>
                `,
                "tooltip": "Compare Logical",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-cmpl-compare-logical-instruction"
            };
        case "CMPLI":
            return {
                "html": `
                    <p>The <strong>cmpli</strong> instruction compares the contents of general-purpose register (GPR) <em>RA</em> with the concatenation of x'0000' and a 16-bit unsigned integer, <em>UI,</em> as unsigned integers and sets one of the bits in the Condition Register Field <em>BF</em>.</p>
                    <p><em>BF</em> can be Condition Register Field 0-7; programmers can specify which Condition Register Field will indicate the result of the operation.</p>
                `,
                "tooltip": "Compare Logical Immediate",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-cmpli-compare-logical-immediate-instruction"
            };
        case "CNTLZD":
        case "CNTLZD.":
            return {
                "html": `
                    <p>A count of the number of consecutive zero bits, starting at bit 0 (the high-order bit) of register GPR <em>RS</em> is placed into GPR <em>RA</em>. This number ranges from 0 to 64, inclusive.</p>
                    <p>This instruction is defined only for 64-bit implementations. Using it on a 32-bit implementation will cause the system illegal instruction error handler to be invoked.</p>
                `,
                "tooltip": "Count Leading Zeros Double Word",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-cntlzd-count-leading-zeros-double-word-instruction"
            };
        case "CNTLZ":
        case "CNTLZ.":
        case "CNTLZW":
        case "CNTLZW.":
            return {
                "html": `<p>The <strong>cntlzw</strong> and <strong>cntlz</strong> instructions count the number (0 - 32) of consecutive zero bits of the 32 low-order bits of GPR <em>RS</em> and store the result in the target GPR <em>RA</em>.</p>`,
                "tooltip": "Count Leading Zeros Word",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-cntlzw-cntlz-count-leading-zeros-word-instruction"
            };
        case "CRAND":
            return {
                "html": `<p>The <strong>crand</strong> instruction logically ANDs the Condition Register bit specified by <em>BA</em> and the Condition Register bit specified by <em>BB</em> and places the result in the target Condition Register bit specified by <em>BT</em>.</p>`,
                "tooltip": "Condition Register AND",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-crand-condition-register-instruction"
            };
        case "CRANDC":
            return {
                "html": `<p>The <strong>crandc</strong> instruction logically ANDs the Condition Register bit specified in <em>BA</em> and the complement of the Condition Register bit specified by <em>BB</em> and places the result in the target Condition Register bit specified by <em>BT</em>.</p>`,
                "tooltip": "Condition Register AND with Complement",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-crandc-condition-register-complement-instruction"
            };
        case "CREQV":
            return {
                "html": `<p>The <strong>creqv</strong> instruction logically XORs the Condition Register bit specified in <em>BA</em> and the Condition Register bit specified by <em>BB</em> and places the complemented result in the target Condition Register bit specified by <em>BT</em>.</p>`,
                "tooltip": "Condition Register Equivalent",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-creqv-condition-register-equivalent-instruction"
            };
        case "CRNAND":
            return {
                "html": `<p>The <strong>crnand</strong> instruction logically ANDs the Condition Register bit specified by <em>BA</em> and the Condition Register bit specified by <em>BB</em> and places the complemented result in the target Condition Register bit specified by <em>BT</em>.</p>`,
                "tooltip": "Condition Register NAND",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-crnand-condition-register-nand-instruction"
            };
        case "CRNOR":
            return {
                "html": `<p>The <strong>crnor</strong> instruction logically ORs the Condition Register bit specified in <em>BA</em> and the Condition Register bit specified by <em>BB</em> and places the complemented result in the target Condition Register bit specified by <em>BT</em>.</p>`,
                "tooltip": "Condition Register NOR",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-crnor-condition-register-nor-instruction"
            };
        case "CROR":
            return {
                "html": `<p>The <strong>cror</strong> instruction logically ORs the Condition Register bit specified by <em>BA</em> and the Condition Register bit specified by <em>BB</em> and places the result in the target Condition Register bit specified by <em>BT</em>.</p>`,
                "tooltip": "Condition Register OR",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-cror-condition-register-instruction"
            };
        case "CRORC":
            return {
                "html": `<p>The <strong>crorc</strong> instruction logically ORs the Condition Register bit specified by <em>BA</em> and the complement of the Condition Register bit specified by <em>BB</em> and places the result in the target Condition Register bit specified by <em>BT</em>.</p>`,
                "tooltip": "Condition Register OR with Complement",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-crorc-condition-register-complement-instruction"
            };
        case "CRXOR":
            return {
                "html": `<p>The <strong>crxor</strong> instruction logically XORs the Condition Register bit specified by <em>BA</em> and the Condition Register bit specified by <em>BB</em> and places the result in the target Condition Register bit specified by <em>BT</em>.</p>`,
                "tooltip": "Condition Register XOR",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-crxor-condition-register-xor-instruction"
            };
        case "DCBF":
            return {
                "html": `<p>The <strong>dcbf</strong> instruction calculates an effective address (EA) by adding the contents of general-purpose register (GPR) <em>RA</em> to the contents of GPR <em>RB</em>. If the <em>RA</em> field is 0, EA is the sum of the contents of <em>RB</em> and 0. If the cache block containing the target storage locations is in the data cache, it is copied back to main storage, provided it is different than the main storage copy.</p>`,
                "tooltip": "Data Cache Block Flush",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-dcbf-data-cache-block-flush-instruction"
            };
        case "DCBI":
            return {
                "html": `<p>If the contents of general-purpose register (GPR) <em>RA</em> is not 0, the <strong>dcbi</strong> instruction computes an effective address (EA) by adding the contents of GPR <em>RA</em> to the contents of GPR <em>RB</em>. Otherwise, the EA is the content of GPR <em>RB</em>.</p>`,
                "tooltip": "Data Cache Block Invalidate",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-dcbi-data-cache-block-invalidate-instruction"
            };
        case "DCBST":
            return {
                "html": `
                    <p>The <strong>dcbst</strong> instruction causes any modified copy of the block to be copied to main memory. If <em>RA</em> is not 0, the <strong>dcbst</strong> instruction computes an effective address (EA) by adding the contents of general-purpose register (GPR) <em>RA</em> to the contents of GPR <em>RB</em>. Otherwise, the EA is the contents of <em>RB</em>. If the cache block containing the addressed byte is in the data cache and is modified, the block is copied to main memory.</p>
                    <p>The <strong>dcbst</strong> instruction may be used to ensure that the copy of a location in main memory contains the most recent updates. This may be important when sharing memory with an I/O device that does not participate in the coherence protocol. In addition, the <strong>dcbst</strong> instruction can ensure that updates are immediately copied to a graphics frame buffer.</p>
                    <p>Treat the <strong>dcbst</strong> instruction as a load from the addressed byte with respect to address translation and protection.</p>
                `,
                "tooltip": "Data Cache Block Store",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-dcbst-data-cache-block-store-instruction"
            };
        case "DCBT":
            return {
                "html": `
                    <p>The <strong>dcbt</strong> instruction may improve performance by anticipating a load from the addressed byte. The block containing the byte addressed by the effective address (EA) is fetched into the data cache before the block is needed by the program. The program can later perform loads from the block and may not experience the added delay caused by fetching the block into the cache. Executing the <strong>dcbt</strong> instruction does not invoke the system error handler.</p>
                    <p>If general-purpose register (GPR) <em>RA</em> is not 0, the effective address (EA) is the sum of the content of GPR <em>RA</em> and the content of GPR <em>RB</em>. Otherwise, the EA is the content of GPR <em>RB</em>.</p>
                `,
                "tooltip": "Data Cache Block Touch",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-dcbt-data-cache-block-touch-instruction"
            };
        case "DCBTST":
            return {
                "html": `
                    <p>The <strong>dcbtst</strong> instruction improves performance by anticipating a store to the addressed byte. The block containing the byte addressed by the effective address (EA) is fetched into the data cache before the block is needed by the program. The program can later perform stores to the block and may not experience the added delay caused by fetching the block into the cache. Executing the <strong>dcbtst</strong> instruction does not invoke the system error handler.</p>
                    <p>The <strong>dcbtst</strong> instruction calculates an effective address (EA) by adding the contents of general-purpose register (GPR) <em>RA</em> to the contents of GPR <em>RB</em>. If the <em>RA</em> field is 0, EA is the sum of the contents of <em>RB</em> and 0.</p>
                `,
                "tooltip": "Data Cache Block Touch for Store",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-dcbtst-data-cache-block-touch-store-instruction"
            };
        case "DCBZ":
        case "DCLZ":
            return {
                "html": `<p>The <strong>dcbz</strong> and <strong>dclz</strong> instructions work with data cache blocks and data cache lines respectively. If <em>RA</em> is not 0, the <strong>dcbz</strong> and <strong>dclz</strong> instructions compute an effective address (EA) by adding the contents of general-purpose register (GPR) <em>RA</em> to the contents of GPR <em>RB</em>. If GPR <em>RA</em> is 0, the EA is the contents of GPR <em>RB</em>.</p>`,
                "tooltip": "Data Cache Block Set to Zero",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=is-dcbz-dclz-data-cache-block-set-zero-instruction"
            };
        case "DCLST":
            return {
                "html": `<p>The <strong>dclst</strong> instruction adds the contents of general-purpose register (GPR) <em>RA</em> to the contents of GPR <em>RB</em>. It then stores the sum in <em>RA</em> as the effective address (EA) if <em>RA</em> is not 0 and the instruction does not cause a Data Storage interrupt.</p>`,
                "tooltip": "Data Cache Line Store",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-dclst-data-cache-line-store-instruction"
            };
        case "DIV":
        case "DIV.":
        case "DIVO":
        case "DIVO.":
            return {
                "html": `
                    <p>The <strong>div</strong> instruction concatenates the contents of general-purpose register (GPR) <em>RA</em> and the contents of Multiply Quotient (MQ) Register, divides the result by the contents of GPR <em>RB</em>, and stores the result in the target GPR <em>RT</em>. The remainder has the same sign as the dividend, except that a zero quotient or a zero remainder is always positive. The results obey the equation:</p>
                    <pre><code>dividend = (divisor x quotient) + remainder</code></pre>
                    <p>where a <samp>dividend</samp> is the original (RA) || (MQ), <samp>divisor</samp> is the original (<em>RB</em>), <samp>quotient</samp> is the final (<em>RT</em>), and <samp>remainder</samp> is the final (MQ).</p>
                    <p>For the case of <samp>-2**31 P -1</samp>, the MQ Register is set to 0 and <samp>-2**31</samp> is placed in GPR <em>RT</em>. For all other overflows, the contents of MQ, the target GPR <em>RT</em>, and the Condition Register Field 0 (if the Record Bit (Rc) is 1) are undefined.</p>
                `,
                "tooltip": "Divide",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-div-divide-instruction"
            };
        case "DIVD":
        case "DIVD.":
        case "DIVDO":
        case "DIVDO.":
            return {
                "html": `
                    <p>The 64-bit dividend is the contents of <em>RA</em>. The 64-bit divisor is the contents of <em>RB</em>. The 64- bit quotient is placed into <em>RT</em>. The remainder is not supplied as a result.</p>
                    <p>Both the operands and the quotient are interpreted as signed integers. The quotient is the unique signed integer that satisfies the equation-dividend = (quotient * divisor) + r, where 0 &lt;= r &lt; |divisor| if the dividend is non-negative, and -|divisor| &lt; r &lt;=0 if the dividend is negative.</p>
                    <p>If an attempt is made to perform the divisions 0x8000_0000_0000_0000 / -1 or / 0, the contents of <em>RT</em> are undefined, as are the contents of the LT, GT, and EQ bits of the condition register 0 field (if the record bit (Rc) = 1 (the <strong>divd.</strong> or <strong>divdo.</strong> instructions)). In this case, if overflow enable (OE) = 1 then the overflow bit (OV) is set.</p>
                `,
                "tooltip": "Divide Double Word",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-divd-divide-double-word-instruction"
            };
        case "DIVDU":
        case "DIVDU.":
        case "DIVDUO":
        case "DIVDUO.":
            return {
                "html": `
                    <p>The 64-bit dividend is the contents of <em>RA</em>. The 64-bit divisor is the contents of <em>RB</em>. The 64- bit quotient is placed into <em>RT</em>. The remainder is not supplied as a result.</p>
                    <p>Both the operands and the quotient are interpreted as unsigned integers, except that if the record bit (Rc) is set to 1 the first three bits of th condition register 0 (CR0) field are set by signed comparison of the result to zero. The quotient is the unique unsigned integer that satisfies the equation: dividend = (quotient * divisor) + r, where 0 &lt;= r &lt; divisor.</p>
                    <p>If an attempt is made to perform the division (<em>anything</em>) / 0 the contents of <em>RT</em> are undefined, as are the contents of the LT, GT, and EQ bits of the CR0 field (if Rc = 1). In this case, if the overflow enable bit (OE) = 1 then the overflow bit (OV) is set.</p>
                `,
                "tooltip": "Divide Double Word Unsigned",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-divdu-divide-double-word-unsigned-instruction"
            };
        case "DIVS":
        case "DIVS.":
        case "DIVSO":
        case "DIVSO.":
            return {
                "html": `
                    <p>The <strong>divs</strong> instruction divides the contents of general-purpose register (GPR) <em>RA</em> by the contents of GPR <em>RB</em> and stores the result in the target GPR <em>RT</em>. The remainder has the same sign as the dividend, except that a zero quotient or a zero remainder is always positive. The results obey the equation:</p>
                    <pre><code>dividend = (divisor x quotient) + remainder</code></pre>
                    <p>where a <samp>dividend</samp> is the original (<em>RA</em>), <samp>divisor</samp> is the original (<em>RB</em>), <samp>quotient</samp> is the final (<em>RT</em>), and <samp>remainder</samp> is the final (MQ).</p>
                    <p>For the case of <samp>-2**31 P -1</samp>, the MQ Register is set to 0 and <samp>-2**31</samp> is placed in GPR <em>RT</em>. For all other overflows, the contents of MQ, the target GPR <em>RT</em> and the Condition Register Field 0 (if the Record Bit (Rc) is 1) are undefined.</p>
                `,
                "tooltip": "Divide Short",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-divs-divide-short-instruction"
            };
        case "DIVW":
        case "DIVW.":
        case "DIVWO":
        case "DIVWO.":
            return {
                "html": `
                    <p>The <strong>divw</strong> instruction divides the contents of general-purpose register (GPR) <em>RA</em> by the contents of GPR <em>RB</em>, and stores the result in the target GPR <em>RT</em>. The dividend, divisor, and quotient are interpreted as signed integers.</p>
                    <p>For the case of -2**31 / -1, and all other cases that cause overflow, the content of GPR <em>RT</em> is undefined.</p>
                `,
                "tooltip": "Divide Word",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-divw-divide-word-instruction"
            };
        case "DIVWU":
        case "DIVWU.":
        case "DIVWUO":
        case "DIVWUO.":
            return {
                "html": `
                    <p>The <strong>divwu</strong> instruction divides the contents of general-purpose register (GPR) <em>RA</em> by the contents of GPR <em>RB</em>, and stores the result in the target GPR <em>RT</em>. The dividend, divisor, and quotient are interpreted as unsigned integers.</p>
                    <p>For the case of division by 0, the content of GPR <em>RT</em> is undefined.</p>
                    <blockquote><strong>Note:</strong> Although the operation treats the result as an unsigned integer, if Rc is 1, the Less Than (LT) zero, Greater Than (GT) zero, and Equal To (EQ) zero bits of Condition Register Field 0 are set as if the result were interpreted as a signed integer.</blockquote>
                `,
                "tooltip": "Divide Word Unsigned",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-divwu-divide-word-unsigned-instruction"
            };
        case "DOZ":
        case "DOZ.":
        case "DOZO":
        case "DOZO.":
            return {
                "html": `
                    <p>The <strong>doz</strong> instruction adds the complement of the contents of general-purpose register (GPR) <em>RA</em>, 1, and the contents of GPR <em>RB,</em> and stores the result in the target GPR <em>RT</em>.</p>
                    <p>If the value in GPR <em>RA</em> is algebraically greater than the value in GPR <em>RB</em>, then GPR <em>RT</em> is set to 0.</p>
                `,
                "tooltip": "Difference or Zero",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-doz-difference-zero-instruction"
            };
        case "DOZI":
            return {
                "html": `
                    <p>The <strong>dozi</strong> instruction adds the complement of the contents of general-purpose register (GPR) <em>RA</em>, the 16-bit signed integer <em>SI</em>, and 1 and stores the result in the target GPR <em>RT</em>.</p>
                    <p>If the value in GPR <em>RA</em> is algebraically greater than the 16-bit signed value in the <em>SI</em> field, then GPR <em>RT</em> is set to 0.</p>
                `,
                "tooltip": "Difference or Zero Immediate",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-dozi-difference-zero-immediate-instruction"
            };
        case "ECIWX":
            return {
                "html": `
                    <p>The <strong>eciwx</strong> instruction translates EA to a real address, sends the real address to a controller, and places the word returned by the controller in general-purpose register <em>RT</em>. If <em>RA</em> = 0, the EA is the content of <em>RB</em>, otherwise EA is the sum of the content of <em>RA</em> plus the content of <em>RB</em>.</p>
                    <p>If EAR(E) = 1, a load request for the real address corresponding to EA is sent to the controller identified by EAR(RID), bypassing the cache. The word returned by the controller is placed in <em>RT</em>.</p>
                `,
                "tooltip": "External Control In Word Indexed",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-eciwx-external-control-in-word-indexed-instruction"
            };
        case "ECOWX":
            return {
                "html": `
                    <p>The <strong>ecowx</strong> instruction translates EA to a real address and sends the real address and the content of general-purpose register <em>RS</em> to a controller. If <em>RA</em> = 0, the EA is the content of <em>RB</em>, otherwise EA is the sum of the content of <em>RA</em> plus the content of <em>RB</em>.</p>
                    <p>If EAR(E) = 1, a store request for the real address corresponding to EA is sent to the controller identified by EAR(RID), bypassing the cache. The content of <em>RS</em> is sent with the store request.</p>
                `,
                "tooltip": "External Control Out Word Indexed",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-ecowx-external-control-out-word-indexed-instruction"
            };
        case "EIEIO":
            return {
                "html": `<p>The <strong>eieio</strong> instruction provides an ordering function that ensures that all load and store instructions initiated prior to the <strong>eieio</strong> instruction complete in main memory before any loads or stores subsequent to the <strong>eieio</strong> instruction access memory. If the <strong>eieio</strong> instruction is omitted from a program, and the memory locations are unique, the accesses to main storage may be performed in any order.</p>`,
                "tooltip": "Enforce In-Order Execution of I/O",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-eieio-enforce-in-order-execution-io-instruction"
            };
        case "EXTSW":
        case "EXTSW.":
            return {
                "html": `<p>The contents of the low-order 32 bits of general purpose register (GPR) <em>RS</em> are placed into the low-order 32 bits of GPR <em>RA</em>. Bit 32 of GPR <em>RS</em> is used to fill the high-order 32 bits of GPR <em>RA</em>.</p>`,
                "tooltip": "Extend Sign Word",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-extsw-extend-sign-word-instruction"
            };
        case "EQV":
        case "EQV.":
            return {
                "html": `<p>The <strong>eqv</strong> instruction logically XORs the contents of general-purpose register (GPR) <em>RS</em> with the contents of GPR <em>RB</em> and stores the complemented result in the target GPR <em>RA</em>.</p>`,
                "tooltip": "Equivalent",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-eqv-equivalent-instruction"
            };
        case "EXTSB":
        case "EXTSB.":
            return {
                "html": `<p>The <strong>extsb</strong> instruction places bits 24-31 of general-purpose register (GPR) <em>RS</em> into bits 24-31 of GPR <em>RA</em> and copies bit 24 of register <em>RS</em> in bits 0-23 of register <em>RA</em>.</p>`,
                "tooltip": "Extend Sign Byte",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-extsb-extend-sign-byte-instruction"
            };
        case "EXTS":
        case "EXTS.":
        case "EXTSH":
        case "EXTSH.":
            return {
                "html": `<p>The <strong>extsh</strong> and <strong>exts</strong> instructions place bits 16-31 of general-purpose register (GPR) <em>RS</em> into bits 16-31 of GPR <em>RA</em> and copy bit 16 of GPR <em>RS</em> in bits 0-15 of GPR <em>RA</em>.</p>`,
                "tooltip": "Extend Sign Halfword",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-extsh-exts-extend-sign-halfword-instruction"
            };
        case "FABS":
        case "FABS.":
            return {
                "html": `<p>The <strong>fabs</strong> instruction sets bit 0 of floating-point register (FPR) <em>FRB</em> to 0 and places the result into FPR <em>FRT</em>.</p>`,
                "tooltip": "Floating Absolute Value",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-fabs-floating-absolute-value-instruction"
            };
        case "FA":
        case "FA.":
        case "FADD":
        case "FADD.":
        case "FADDS":
        case "FADDS.":
            return {
                "html": `
                    <p>The <strong>fadd</strong> and <strong>fa</strong> instructions add the 64-bit, double-precision floating-point operand in floating-point register (FPR) <em>FRA</em> to the 64-bit, double-precision floating-point operand in FPR <em>FRB</em>.</p>
                    <p>The <strong>fadds</strong> instruction adds the 32-bit single-precision floating-point operand in FPR <em>FRA</em> to the 32-bit single-precision floating-point operand in FPR <em>FRB</em>.</p>
                    <p>The result is rounded under control of the Floating-Point Rounding Control Field <em>RN</em> of the Floating-Point Status and Control Register and is placed in FPR <em>FRT</em>.</p>
                    <p>Addition of two floating-point numbers is based on exponent comparison and addition of the two significands. The exponents of the two operands are compared, and the significand accompanying the smaller exponent is shifted right, with its exponent increased by one for each bit shifted, until the two exponents are equal. The two significands are then added algebraically to form the intermediate sum. All 53 bits in the significand as well as all three guard bits (G, R and X) enter into the computation.</p>
                    <p>The Floating-Point Result Field of the Floating-Point Status and Control Register is set to the class and sign of the result except for Invalid Operation exceptions when the Floating-Point Invalid Operation Exception Enable (VE) bit of the Floating-Point Status and Control Register is set to 1.</p>
                `,
                "tooltip": "Floating Add",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-fadd-fa-floating-add-instruction"
            };
        case "FCFID":
        case "FCFID.":
            return {
                "html": `
                    <p>The 64-bit signed fixed-point operand in floating-point register (FPR) <em>FRB</em> is converted to an infinitely precise floating-point integer. The result of the conversion is rounded to double-precision using the rounding mode specified by FPSCR[RN] and placed into FPR <em>FRT</em>.</p>
                    <p>FPSCR[FPRF] is set to the class and sign of the result. FPSCR[FR] is set if the result is incremented when rounded. FPSCR[FI] is set if the result is inexact.</p>
                `,
                "tooltip": "Floating Convert from Integer Double Word",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=is-fcfid-floating-convert-from-integer-double-word-instruction"
            };
        case "FCMPO":
            return {
                "html": `<p>The <strong>fcmpo</strong> instruction compares the 64-bit, double-precision floating-point operand in floating-point register (FPR) <em>FRA</em> to the 64-bit, double-precision floating-point operand in FPR <em>FRB</em>. The Floating-Point Condition Code Field (FPCC) of the Floating-Point Status and Control Register (FPSCR) is set to reflect the value of the operand FPR <em>FRA</em> with respect to operand FPR <em>FRB</em>. The value <em>BF</em> determines which field in the condition register receives the four FPCC bits.</p>`,
                "tooltip": "Floating Compare Ordered",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-fcmpo-floating-compare-ordered-instruction"
            };
        case "FCMPU":
            return {
                "html": `<p>The <strong>fcmpu</strong> instruction compares the 64-bit double precision floating-point operand in floating-point register (FPR) <em>FRA</em> to the 64-bit double precision floating-point operand in FPR <em>FRB</em>. The Floating-Point Condition Code Field (FPCC) of the Floating-Point Status and Control Register (FPSCR) is set to reflect the value of the operand <em>FRA</em> with respect to operand <em>FRB</em>. The value <em>BF</em> determines which field in the condition register receives the four FPCC bits.</p>`,
                "tooltip": "Floating Compare Unordered",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-fcmpu-floating-compare-unordered-instruction"
            };
        case "FCTID":
        case "FCTID.":
            return {
                "html": `
                    <p>The floating-point operand in floating-point register (FPR) <em>FRB</em> is converted to a 64-bit signed fixed-point integer, using the rounding mode specified by FPSCR[RN], and placed into FPR <em>FRT</em>.</p>
                    <p>If the operand in <em>FRB</em> is greater than 2**63 - 1, then FPR <em>FRT</em> is set to 0x7FFF_FFFF_FFFF_FFFF. If the operand in <em>FRB</em> is less than 2**63 , then FPR <em>FRT</em> is set to 0x8000_0000_0000_0000.</p>
                    <p>Except for enabled invalid operation exceptions, FPSCR[FPRF] is undefined. FPSCR[FR] is set if the result is incremented when rounded. FPSCR[FI] is set if the result is inexact.</p>
                `,
                "tooltip": "Floating Convert to Integer Double Word",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-fctid-floating-convert-integer-double-word-instruction"
            };
        case "FCTIDZ":
        case "FCTIDZ.":
            return {
                "html": `
                    <p>The floating-point operand in floating-point register (FRP) <em>FRB</em> is converted to a 64-bit signed fixed-point integer, using the rounding mode round toward zero, and placed into FPR <em>FRT</em>.</p>
                    <p>If the operand in FPR <em>FRB</em> is greater than 2**63 - 1, then FPR <em>FRT</em> is set to 0x7FFF_FFFF_FFFF_FFFF. If the operand in frB is less than 2**63 , then FPR <em>FRT</em> is set to 0x8000_0000_0000_0000.</p>
                    <p>Except for enabled invalid operation exceptions, FPSCR[FPRF] is undefined. FPSCR[FR] is set if the result is incremented when rounded. FPSCR[FI] is set if the result is inexact.</p>
                `,
                "tooltip": "Floating Convert to Integer Double Word with Round Toward Zero",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=is-fctidz-floating-convert-integer-double-word-round-toward-zero-instruction"
            };
        case "FCIR":
        case "FCIR.":
        case "FCTIW":
        case "FCTIW.":
            return {
                "html": `
                    <p>The <strong>fctiw</strong> and <strong>fcir</strong> instructions convert the floating-point operand in floating-point register (FPR) <em>FRB</em> to a 32-bit signed, fixed-point integer, using the rounding mode specified by Floating-Point Status and Control Register (FPSCR) RN. The result is placed in bits 32-63 of FPR <em>FRT</em>. Bits 0-31 of FPR <em>FRT</em> are undefined.</p>
                    <p>If the operand in FPR <em>FRB</em> is greater than 231 - 1, then the bits 32-63 of FPR <em>FRT</em> are set to 0x7FFF FFFF. If the operand in FPR <em>FRB</em> is less than -231, then the bits 32-63 of FPR <em>FRT</em> are set to 0x8000 0000.</p>
                `,
                "tooltip": "Floating Convert to Integer Word",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-fctiw-fcir-floating-convert-integer-word-instruction"
            };
        case "FCIRZ":
        case "FCIRZ.":
        case "FCTIWZ":
        case "FCTIWZ.":
            return {
                "html": `
                    <p>The <strong>fctiwz</strong> and <strong>fcirz</strong> instructions convert the floating-point operand in floating-point register (FPR) <em>FRB</em> to a 32-bit, signed, fixed-point integer, rounding the operand toward 0. The result is placed in bits 32-63 of FPR <em>FRT</em>. Bits 0-31 of FPR <em>FRT</em> are undefined.</p>
                    <p>If the operand in FPR <em>FRB</em> is greater than 231 - 1, then the bits 32-63 of FPR <em>FRT</em> are set to 0x7FFF FFFF. If the operand in FPR <em>FRB</em> is less than -231, then the bits 32-63 of FPR <em>FRT</em> are set to 0x8000 0000.</p>
                `,
                "tooltip": "Floating Convert to Integer Word with Round to Zero",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=is-fctiwz-fcirz-floating-convert-integer-word-round-zero-instruction"
            };
        case "FD":
        case "FD.":
        case "FDIV":
        case "FDIV.":
        case "FDIVS":
        case "FDIVS.":
            return {
                "html": `
                    <p>The <strong>fdiv</strong> and <strong>fd</strong> instructions divide the 64-bit, double-precision floating-point operand in floating-point register (FPR) <em>FRA</em> by the 64-bit, double-precision floating-point operand in FPR <em>FRB</em>. No remainder is preserved.</p>
                    <p>The <strong>fdivs</strong> instruction divides the 32-bit single-precision floating-point operand in FPR <em>FRA</em> by the 32-bit single-precision floating-point operand in FPR <em>FRB</em>. No remainder is preserved.</p>
                    <p>The result is rounded under control of the Floating-Point Rounding Control Field <em>RN</em> of the Floating-Point Status and Control Register (FPSCR), and is placed in the target FPR <em>FRT</em>.</p>
                    <p>The floating-point division operation is based on exponent subtraction and division of the two significands.</p>
                    <p><strong>Note:</strong> If an operand is a denormalized number, then it is prenormalized before the operation is begun.</p>
                    <p>The Floating-Point Result Flags Field of the Floating-Point Status and Control Register is set to the class and sign of the result, except for Invalid Operation Exceptions, when the Floating-Point Invalid Operation Exception Enable bit is 1.</p>
                `,
                "tooltip": "Floating Divide",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-fdiv-fd-floating-divide-instruction"
            };
        case "FMA":
        case "FMA.":
        case "FMADD":
        case "FMADD.":
        case "FMADDS":
        case "FMADDS.":
            return {
                "html": `
                    <p>The <strong>fmadd</strong> and <strong>fma</strong> instructions multiply the 64-bit, double-precision floating-point operand in floating-point register (FPR) <em>FRA</em> by the 64-bit, double-precision floating-point operand in FPR <em>FRC,</em> and then add the result of this operation to the 64-bit, double-precision floating-point operand in FPR <em>FRB</em>.</p>
                    <p>The <strong>fmadds</strong> instruction multiplies the 32-bit, single-precision floating-point operand in FPR <em>FRA</em> by the 32-bit, single-precision floating-point operand in FPR <em>FRC</em> and adds the result of this operation to the 32-bit, single-precision floating-point operand in FPR <em>FRB</em>.</p>
                    <p>The result is rounded under control of the Floating-Point Rounding Control Field <em>RN</em> of the Floating-Point Status and Control Register and is placed in the target FPR <em>FRT</em>.</p>
                    <p><strong>Note:</strong> If an operand is a denormalized number, then it is prenormalized before the operation is begun.</p>
                    <p>The Floating-Point Result Flags Field of the Floating-Point Status and Control Register is set to the class and sign of the result, except for Invalid Operation Exceptions, when the Floating-Point Invalid Operation Exception Enable bit is 1.</p>
                `,
                "tooltip": "Floating Multiply-Add",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-fmadd-fma-floating-multiply-add-instruction"
            };
        case "FMR":
        case "FMR.":
            return {
                "html": `<p>The <strong>fmr</strong> instruction places the contents of floating-point register (FPR) <em>FRB</em> into the target FPR <em>FRT</em>.</p>`,
                "tooltip": "Floating Move Register",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-fmr-floating-move-register-instruction"
            };
        case "FMS":
        case "FMS.":
        case "FMSUB":
        case "FMSUB.":
        case "FMSUBS":
        case "FMSUBS.":
            return {
                "html": `
                    <p>The <strong>fmsub</strong> and <strong>fms</strong> instructions multiply the 64-bit, double-precision floating-point operand in floating-point register (FPR) <em>FRA</em> by the 64-bit, double-precision floating-point operand in FPR <em>FRC</em> and subtract the 64-bit, double-precision floating-point operand in FPR <em>FRB</em> from the result of the multiplication.</p>
                    <p>The <strong>fmsubs</strong> instruction multiplies the 32-bit, single-precision floating-point operand in FPR <em>FRA</em> by the 32-bit, single-precision floating-point operand in FPR <em>FRC</em> and subtracts the 32-bit, single-precision floating-point operand in FPR <em>FRB</em> from the result of the multiplication.</p>
                    <p>The result is rounded under control of the Floating-Point Rounding Control Field <em>RN</em> of the Floating-Point Status and Control Register and is placed in the target FPR <em>FRT</em>.</p>
                    <p><strong>Note:</strong> If an operand is a denormalized number, then it is prenormalized before the operation is begun.</p>
                    <p>The Floating-Point Result Flags Field of the Floating-Point Status and Control Register is set to the class and sign of the result, except for Invalid Operation Exceptions, when the Floating-Point Invalid Operation Exception Enable bit is 1.</p>
                `,
                "tooltip": "Floating Multiply-Subtract",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-fmsub-fms-floating-multiply-subtract-instruction"
            };
        case "FM":
        case "FM.":
        case "FMUL":
        case "FMUL.":
        case "FMULS":
        case "FMULS.":
            return {
                "html": `
                    <p>The <strong>fmul</strong> and <strong>fm</strong> instructions multiply the 64-bit, double-precision floating-point operand in floating-point register (FPR) <em>FRA</em> by the 64-bit, double-precision floating-point operand in FPR <em>FRC</em>.</p>
                    <p>The <strong>fmuls</strong> instruction multiplies the 32-bit, single-precision floating-point operand in FPR <em>FRA</em> by the 32-bit, single-precision floating-point operand in FPR <em>FRC</em>.</p>
                    <p>The result is rounded under control of the Floating-Point Rounding Control Field <em>RN</em> of the Floating-Point Status and Control Register and is placed in the target FPR <em>FRT</em>.</p>
                    <p>Multiplication of two floating-point numbers is based on exponent addition and multiplication of the two significands.</p>
                    <p><strong>Note:</strong> If an operand is a denormalized number, then it is prenormalized before the operation is begun.</p>
                    <p>The Floating-Point Result Flags Field of the Floating-Point Status and Control Register is set to the class and sign of the result, except for Invalid Operation Exceptions, when the Floating-Point Invalid Operation Exception Enable bit is 1.</p>
                `,
                "tooltip": "Floating Multiply",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-fmul-fm-floating-multiply-instruction"
            };
        case "FNABS":
        case "FNABS.":
            return {
                "html": `<p>The <strong>fnabs</strong> instruction places the negative absolute of the contents of floating-point register (FPR) <em>FRB</em> with bit 0 set to 1 into the target FPR <em>FRT</em>.</p>`,
                "tooltip": "Floating Negative Absolute Value",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-fnabs-floating-negative-absolute-value-instruction"
            };
        case "FNEG":
        case "FNEG.":
            return {
                "html": `<p>The <strong>fneg</strong> instruction places the negated contents of floating-point register <em>FRB</em> into the target FPR <em>FRT</em>.</p>`,
                "tooltip": "Floating Negate",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-fneg-floating-negate-instruction"
            };
        case "FNMA":
        case "FNMA.":
        case "FNMADD":
        case "FNMADD.":
        case "FNMADDS":
        case "FNMADDS.":
            return {
                "html": `
                    <p>The <strong>fnmadd</strong> and <strong>fnma</strong> instructions multiply the 64-bit, double-precision floating-point operand in floating-point register (FPR) <em>FRA</em> by the 64,bit, double-precision floating-point operand in FPR <em>FRC</em>, and add the 64-bit, double-precision floating-point operand in FPR <em>FRB</em> to the result of the multiplication.</p>
                    <p>The <strong>fnmadds</strong> instruction multiplies the 32-bit, single-precision floating-point operand in FPR <em>FRA</em> by the 32-bit, single-precision floating-point operand in FPR <em>FRC</em>, and adds the 32-bit, single-precision floating-point operand in FPR <em>FRB</em> to the result of the multiplication.</p>
                `,
                "tooltip": "Floating Negative Multiply-Add",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-fnmadd-fnma-floating-negative-multiply-add-instruction"
            };
        case "FNMS":
        case "FNMS.":
        case "FNMSUB":
        case "FNMSUB.":
        case "FNMSUBS":
        case "FNMSUBS.":
            return {
                "html": `
                    <p>The <strong>fnms</strong> and <strong>fnmsub</strong> instructions multiply the 64-bit, double-precision floating-point operand in floating-point register (FPR) <em>FRA</em> by the 64,-bit double-precision floating-point operand in FPR <em>FRC</em>, subtract the 64-bit, double-precision floating-point operand in FPR <em>FRB</em> from the result of the multiplication, and place the negated result in the target FPR <em>FRT</em>.</p>
                    <p>The <strong>fnmsubs</strong> instruction multiplies the 32-bit, single-precision floating-point operand in FPR <em>FRA</em> by the 32-bit, single-precision floating-point operand in FPR <em>FRC</em>, subtracts the 32-bit, single-precision floating-point operand in FPR <em>FRB</em> from the result of the multiplication, and places the negated result in the target FPR <em>FRT</em>.</p>
                `,
                "tooltip": "Floating Negative Multiply-Subtract",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-fnmsub-fnms-floating-negative-multiply-subtract-instruction"
            };
        case "FRES":
        case "FRES.":
            return {
                "html": `
                    <p>The <strong>fres</strong> instruction calculates a single-precision estimate of the reciprocal of the 64-bit, double-precision floating-point operand in floating-point register (FPR) <em>FRB</em> and places the result in FPR <em>FRT</em>.</p>
                    <p>The estimate placed into register <em>FRT</em> is correct to a precision of one part in 256 of the reciprocal of <em>FRB</em>. The value placed into <em>FRT</em> may vary between implementations, and between different executions on the same implementation.</p>
                `,
                "tooltip": "Floating Reciprocal Estimate Single",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-fres-floating-reciprocal-estimate-single-instruction"
            };
        case "FRSP":
        case "FRSP.":
            return {
                "html": `
                    <p>The <strong>frsp</strong> instruction rounds the 64-bit, double-precision floating-point operand in floating-point register (FPR) <em>FRB</em> to single precision, using the rounding mode specified by the Floating Rounding Control field of the Floating-Point Status and Control Register, and places the result in the target FPR <em>FRT</em>.</p>
                    <p>The Floating-Point Result Flags Field of the Floating-Point Status and Control Register is set to the class and sign of the result, except for Invalid Operation (SNaN), when Floating-Point Status and Control Register Floating-Point Invalid Operation Exception Enable bit is 1.</p>
                `,
                "tooltip": "Floating Round to Single Precision",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-frsp-floating-round-single-precision-instruction"
            };
        case "FRSQRTE":
        case "FRSQRTE.":
            return {
                "html": `
                    <p>The <strong>frsqrte</strong> instruction computes a double-precision estimate of the reciprocal of the square root of the 64-bit, double-precision floating-point operand in floating-point register (FPR) <em>FRB</em> and places the result in FPR <em>FRT</em>.</p>
                    <p>The estimate placed into register <em>FRT</em> is correct to a precision of one part in 32 of the reciprocal of the square root of <em>FRB</em>. The value placed in <em>FRT</em> may vary between implementations and between different executions on the same implementation.</p>
                `,
                "tooltip": "Floating Reciprocal Square Root Estimate",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-frsqrte-floating-reciprocal-square-root-estimate-instruction"
            };
        case "FSEL":
        case "FSEL.":
            return {
                "html": `<p>The double-precision floating-point operand in floating-point register (FPR) <em>FRA</em> is compared with the value zero. If the value in <em>FRA</em> is greater than or equal to zero, floating point register <em>FRT</em> is set to the contents of floating-point register <em>FRC</em>. If the value in <em>FRA</em> is less than zero or is a NaN, floating point register <em>FRT</em> is set to the contents of floating-point register <em>FRB</em>.The comparison ignores the sign of zero; both +0 and -0 are equal to zero.</p>`,
                "tooltip": "Floating-Point Select",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-fsel-floating-point-select-instruction"
            };
        case "FSQRT":
        case "FSQRT.":
            return {
                "html": `
                    <p>The square root of the operand in floating-point register (FPR) <em>FRB</em> is placed into register FPR <em>FRT</em>.</p>
                    <p>If the most-significant bit of the resultant significand is not a one the result is normalized. The result is rounded to the target precision under control of the floating-point rounding control field RN of the FPSCR and placed into register FPR <em>FRT</em>.</p>
                `,
                "tooltip": "Floating Square Root Double-Precision",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-fsqrt-floating-square-root-double-precision-instruction"
            };
        case "FSQRTS":
        case "FSQRTS.":
            return {
                "html": `
                    <p>The square root of the floating-point operand in floating-point register (FPR) <em>FRB</em> is placed into register FPR <em>FRT</em>.</p>
                    <p>If the most-significant bit of the resultant significand is not a one the result is normalized. The result is rounded to the target precision under control of the floating-point rounding control field RN of the FPSCR and placed into register FPR <em>FRT</em>.</p>
                `,
                "tooltip": "Floating Square Root Single",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-fsqrts-floating-square-root-single-instruction"
            };
        case "FS":
        case "FS.":
        case "FSUB":
        case "FSUB.":
        case "FSUBS":
        case "FSUBS.":
            return {
                "html": `
                    <p>The <strong>fsub</strong> and <strong>fs</strong> instructions subtract the 64-bit, double-precision floating-point operand in floating-point register (FPR) <em>FRB</em> from the 64-bit, double-precision floating-point operand in FPR <em>FRA</em>.</p>
                    <p>The <strong>fsubs</strong> instruction subtracts the 32-bit single-precision floating-point operand in FPR <em>FRB</em> from the 32-bit single-precision floating-point operand in FPR <em>FRA</em>.</p>
                    <p>The result is rounded under control of the Floating-Point Rounding Control Field <em>RN</em> of the Floating-Point Status and Control Register and is placed in the target FPR <em>FRT</em>.</p>
                    <p>The execution of the <strong>fsub</strong> instruction is identical to that of <strong>fadd</strong>, except that the contents of FPR <em>FRB</em> participate in the operation with bit 0 inverted.</p>
                    <p>The execution of the <strong>fs</strong> instruction is identical to that of <strong>fa</strong>, except that the contents of FPR <em>FRB</em> participate in the operation with bit 0 inverted.</p>
                    <p>The Floating-Point Result Flags Field of the Floating-Point Status and Control Register is set to the class and sign of the result, except for Invalid Operation Exceptions, when the Floating-Point Invalid Operation Exception Enable bit is 1.</p>
                `,
                "tooltip": "Floating Subtract",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-fsqrts-floating-square-root-single-instruction"
            };
        case "ICBI":
            return {
                "html": `<p>The <strong>icbi</strong> instruction invalidates a block containing the byte addressed in the instruction cache. If <em>RA</em> is not 0, the <strong>icbi</strong> instruction calculates an effective address (EA) by adding the contents of general-purpose register (GPR) <em>RA</em> to the contents of GPR <em>RB</em>.</p>`,
                "tooltip": "Instruction Cache Block Invalidate",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-icbi-instruction-cache-block-invalidate-instruction"
            };
        case "ICS":
        case "ISYNC":
            return {
                "html": `
                    <p>The <strong>isync</strong> and <strong>ics</strong> instructions cause the processor to refetch any instructions that might have been fetched prior to the <strong>isync</strong> or <strong>ics</strong> instruction.</p>
                    <p>The PowerPC® instruction <strong>isync</strong> causes the processor to wait for all previous instructions to complete. Then any instructions already fetched are discarded and instruction processing continues in the environment established by the previous instructions.</p>
                    <p>The POWER® family instruction <strong>ics</strong> causes the processor to wait for any previous <strong>dcs</strong> instructions to complete. Then any instructions already fetched are discarded and instruction processing continues under the conditions established by the content of the Machine State Register.</p>
                `,
                "tooltip": "Instruction Synchronize",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-isync-ics-instruction-synchronize-instruction"
            };
        case "LBZ":
            return {
                "html": `
                    <p>The <strong>lbz</strong> instruction loads a byte in storage addressed by the effective address (EA) into bits 24-31 of the target general-purpose register (GPR) <em>RT</em> and sets bits 0-23 of GPR <em>RT</em> to 0.</p>
                    <p>If <em>RA</em> is not 0, the EA is the sum of the contents of GPR <em>RA</em> and <em>D</em>, a 16-bit, signed two's complement integer sign-extended to 32 bits. If <em>RA</em> is 0, then the EA is <em>D</em>.</p>
                `,
                "tooltip": "Load Byte and Zero",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-lbz-load-byte-zero-instruction"
            };
        case "LBZU":
            return {
                "html": `
                    <p>The <strong>lbzu</strong> instruction loads a byte in storage addressed by the effective address (EA) into bits 24-31 of the target general-purpose register (GPR) <em>RT</em> and sets bits 0-23 of GPR <em>RT</em> to 0.</p>
                    <p>If <em>RA</em> is not 0, the EA is the sum of the contents of GPR <em>RA</em> and <em>D</em>, a 16-bit signed two's complement integer sign extended to 32 bits. If <em>RA</em> is 0, then the EA is <em>D</em>.</p>
                    <p>If <em>RA</em> does not equal <em>RT</em> and <em>RA</em> does not equal 0, and the storage access does not cause an Alignment interrupt or a Data Storage interrupt, then the EA is stored in GPR <em>RA</em>.</p>
                `,
                "tooltip": "Load Byte and Zero with Update",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-lbzu-load-byte-zero-update-instruction"
            };
        case "LBZUX":
            return {
                "html": `
                    <p>The <strong>lbzux</strong> instruction loads a byte in storage addressed by the effective address (EA) into bits 24-31 of the target general-purpose register (GPR) <em>RT</em> and sets bits 0-23 of GPR <em>RT</em> to 0.</p>
                    <p>If <em>RA</em> is not 0, the EA is the sum of the contents of GPR <em>RA</em> and GPR <em>RB</em>. If <em>RA</em> is 0, then the EA is the contents of <em>RB</em>.</p>
                    <p>If <em>RA</em> does not equal <em>RT</em> and <em>RA</em> does not equal 0, and the storage access does not cause an Alignment interrupt or a Data Storage interrupt, then the EA is stored in GPR <em>RA</em>.</p>
                `,
                "tooltip": "Load Byte and Zero with Update Indexed",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-lbzux-load-byte-zero-update-indexed-instruction"
            };
        case "LBZX":
            return {
                "html": `
                    <p>The <strong>lbzx</strong> instruction loads a byte in storage addressed by the effective address (EA) into bits 24-31 of the target general-purpose register (GPR) <em>RT</em> and sets bits 0-23 of GPR <em>RT</em> to 0.</p>
                    <p>If <em>RA</em> is not 0, the EA is the sum of the contents of GPR <em>RA</em> and GPR <em>RB</em>. If <em>RA</em> is 0, then the EA is <em>D</em>.</p>
                `,
                "tooltip": "Load Byte and Zero Indexed",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-lbzx-load-byte-zero-indexed-instruction"
            };
        case "LD":
            return {
                "html": `
                    <p>The <strong>ld</strong> instruction loads a doubleword in storage from a specified location in memory addressed by the effective address (EA) into the target general-purpose register (GPR) <em>RT</em>.</p>
                    <p>DS is a 14-bit, signed two's complement number, which is sign-extended to 64 bits, and then multiplied by 4 to provide a displacement <em>Disp</em>. If GPR <em>RA</em> is not 0, the EA is the sum of the contents of GPR <em>RA</em> and <em>Disp</em>. If GPR <em>RA</em> is 0, then the EA is <em>Disp</em>.</p>
                `,
                "tooltip": "Load Doubleword",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-ld-load-doubleword-instruction"
            };
        case "LDARX":
            return {
                "html": `
                    <p>The <strong>ldarx</strong> and <strong>stdcx</strong> (<strong>Store Doubleword Conditional Indexed</strong>) instructions are used to perform a read-modify-write operation to storage. If the store operation is performed, the use of the<strong> ldarx</strong> and <strong>stdcx </strong> instructions ensures that no other processor or mechanism changes the target memory location between the time the <strong>ldarx</strong> instruction is run and the time the <strong>stdcx</strong> instruction is completed.</p>
                    <p>If general-purpose register (GPR) <em>RA</em> equals 0, the effective address (EA) is the content of GPR <em>RB</em>. Otherwise, the EA is the sum of the content of GPR <em>RA</em> plus the content of GPR <em>RB</em>.</p>
                    <p>The <strong>ldarx</strong> instruction loads the word from the location in storage that is specified by the EA into the target GPR <em>RT</em>. In addition, a reservation on the memory location is created for use by a subsequent <strong>stwcx.</strong> instruction.</p>
                `,
                "tooltip": "Load Doubleword Reserve Indexed",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-ldarx-load-doubleword-reserve-indexed-instruction"
            };
        case "LDU":
            return {
                "html": `
                    <p>The <strong>ldu</strong> instruction loads a doubleword in storage from a specified location in memory that is addressed by the effective address (EA) into the target GPR <em>RT</em>.</p>
                    <p>DS is a 14-bit, signed two's complement number, which is sign-extended to 64 bits, and then multiplied by 4 to provide a displacement <em>(Disp)</em>. If GPR <em>RA</em> is not 0, the EA is the sum of the contents of GPR <em>RA</em> and <em>Disp</em>.</p>
                    <p>If <em>RA</em> equals 0 or <em>RA</em> equals <em>RT</em>, the instruction form is invalid.</p>
                `,
                "tooltip": "Load Doubleword with Update",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-ldu-load-doubleword-update-instruction"
            };
        case "LDUX":
            return {
                "html": `
                    <p>The effective address (EA) is calculated from the sum of the GPR, <em>RA</em> and <em>RB</em>. A doubleword of data is read from the memory location that is referenced by the EA and placed into GPR <em>RT</em>. GPR <em>RA</em> is updated with the EA.</p>
                    <p>If RA equals 0 or RA equals RD, the instruction form is invalid.</p>
                `,
                "tooltip": "Load Doubleword with Update Indexed",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-ldux-load-doubleword-update-indexed-instruction"
            };
        case "LDX":
            return {
                "html": `
                    <p>The <strong>ldx</strong> instruction loads a doubleword from the specified memory location that is referenced by the effective address (EA) into the GPR <em>RT</em>.</p>
                    <p>If GRP <em>RA</em> is not 0, the effective address (EA) is the sum of the contents of GRPs, <em>RA</em> and <em>RB</em>. Otherwise, the EA is equal to the contents of <em>RB</em>.</p>
                `,
                "tooltip": "Load Doubleword Indexed",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-ldx-load-doubleword-indexed-instruction"
            };
        case "LFD":
            return {
                "html": `
                    <p>The <strong>lfd</strong> instruction loads a doubleword in storage from a specified location in memory addressed by the effective address (EA) into the target floating-point register (FPR) <em>FRT</em>.</p>
                    <p>If general-purpose register (GPR) <em>RA</em> is not 0, the EA is the sum of the contents of GPR <em>RA</em> and <em>D</em>, a 16-bit, signed two's complement integer sign-extended to 32 bits. If GPR <em>RA</em> is 0, then the EA is <em>D</em>.</p>
                `,
                "tooltip": "Load Floating-Point Double",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-lfd-load-floating-point-double-instruction"
            };
        case "LFDU":
            return {
                "html": `
                    <p>The <strong>lfdu</strong> instruction loads a doubleword in storage from a specified location in memory addressed by the effective address (EA) into the target floating-point register (FPR) <em>FRT</em>.</p>
                    <p>If <em>RA</em> is not 0, the EA is the sum of the contents of GPR <em>RA</em> and <em>D</em>, a 16-bit, signed two's complement integer sign-extended to 32 bits. If <em>RA</em> is 0, then the effective address (EA) is <em>D</em>.</p>
                    <p>If <em>RA</em> does not equal 0, and the storage access does not cause an Alignment interrupt or a Data Storage interrupt, then the effective address is stored in GPR <em>RA</em>.</p>
                `,
                "tooltip": "Load Floating-Point Double with Update",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-lfdu-load-floating-point-double-update-instruction"
            };
        case "LFDUX":
            return {
                "html": `
                    <p>The <strong>lfdux</strong> instruction loads a doubleword in storage from a specified location in memory addressed by the effective address (EA) into the target floating-point register (FPR) <em>FRT</em>.</p>
                    <p>If <em>RA</em> is not 0, the EA is the sum of the contents of general-purpose register (GPR) <em>RA</em> and GPR <em>RB</em>. If <em>RA</em> is 0, then the EA is the contents of <em>RB</em>.</p>
                    <p>If <em>RA</em> does not equal 0, and the storage access does not cause an Alignment interrupt or a Data Storage interrupt, then the EA is stored in GPR RA.</p>
                `,
                "tooltip": "Load Floating-Point Double with Update Indexed",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=is-lfdux-load-floating-point-double-update-indexed-instruction"
            };
        case "LFDX":
            return {
                "html": `
                    <p>The <strong>lfdx</strong> instruction loads a doubleword in storage from a specified location in memory addressed by the effective address (EA) into the target floating-point register (FPR) <em>FRT</em>.</p>
                    <p>If <em>RA</em> is not 0, the EA is the sum of the contents of general-purpose register (GPR) <em>RA</em> and GPR <em>RB</em>. If <em>RA</em> is 0, then the EA is the contents of GPR <em>RB</em>.</p>
                `,
                "tooltip": "Load Floating-Point Double-Indexed",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-lfdx-load-floating-point-double-indexed-instruction"
            };
        case "LFQ":
            return {
                "html": `
                    <p>The <strong>lfq</strong> instruction loads the two doublewords from the location in memory specified by the effective address (EA) into two floating-point registers (FPR).</p>
                    <p><em>DS</em> is sign-extended to 30 bits and concatenated on the right with b'00' to form the offset value. If general-purpose register (GPR) <em>RA</em> is 0, the offset value is the EA. If GPR <em>RA</em> is not 0, the offset value is added to GPR <em>RA</em> to generate the EA. The doubleword at the EA is loaded into FPR <em>FRT</em>. If <em>FRT</em> is 31, the doubleword at EA+8 is loaded into FPR 0; otherwise, it is loaded into <em>FRT</em>+1.</p>
                `,
                "tooltip": "Load Floating-Point Quad",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-lfq-load-floating-point-quad-instruction"
            };
        case "LFQU":
            return {
                "html": `
                    <p>The <strong>lfqu</strong> instruction loads the two doublewords from the location in memory specified by the effective address (EA) into two floating-point registers (FPR).</p>
                    <p><em>DS</em> is sign-extended to 30 bits and concatenated on the right with b'00' to form the offset value. If general-purpose register GPR <em>RA</em> is 0, the offset value is the EA. If GPR <em>RA</em> is not 0, the offset value is added to GPR <em>RA</em> to generate the EA. The doubleword at the EA is loaded into FPR <em>FRT</em>. If <em>FRT</em> is 31, the doubleword at EA+8 is loaded into FPR 0; otherwise, it is loaded into <em>FRT</em>+1.</p>
                    <p>If GPR <em>RA</em> is not 0, the EA is placed into GPR <em>RA</em>.</p>
                `,
                "tooltip": "Load Floating-Point Quad with Update",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-lfqu-load-floating-point-quad-update-instruction"
            };
        case "LFQUX":
            return {
                "html": `
                    <p>The <strong>lfqux</strong> instruction loads the two doublewords from the location in memory specified by the effective address (EA) into two floating-point registers (FPR).</p>
                    <p>If general-purpose register (GPR) <em>RA</em> is not 0, the EA is the sum of the contents of GPR <em>RA</em> and GPR <em>RB</em>. If GPR <em>RA</em> is 0, the EA is the contents of GPR <em>RB</em>. The doubleword at the EA is loaded into FPR <em>FRT</em>. If <em>FRT</em> is 31, the doubleword at EA+8 is loaded into FPR 0; otherwise, it is loaded into <em>FRT</em>+1.</p>
                    <p>If GPR <em>RA</em> is not 0, the EA is placed into GPR <em>RA</em>.</p>
                `,
                "tooltip": "Load Floating-Point Quad with Update Indexed",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=is-lfqux-load-floating-point-quad-update-indexed-instruction"
            };
        case "LFQX":
            return {
                "html": `
                    <p>The <strong>lfqx</strong> instruction loads the two doublewords from the location in memory specified by the effective address (EA) into two floating-point registers (FPR).</p>
                    <p>If general-purpose register (GPR) <em>RA</em> is not 0, the EA is the sum of the contents of GPR <em>RA</em> and GPR <em>RB</em>. If GPR <em>RA</em> is 0, the EA is the contents of GPR <em>RB</em>. The doubleword at the EA is loaded into FPR <em>FRT</em>. If <em>FRT</em> is 31, the doubleword at EA+8 is loaded into FPR 0; otherwise, it is loaded into <em>FRT</em>+1.</p>
                `,
                "tooltip": "Load Floating-Point Quad Indexed",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-lfqx-load-floating-point-quad-indexed-instruction"
            };
        case "LFS":
            return {
                "html": `
                    <p>The <strong>lfs</strong> instruction converts a floating-point, single-precision word in storage addressed by the effective address (EA) to a floating-point, double-precision word and loads the result into floating-point register (FPR) <em>FRT</em>.</p>
                    <p>If <em>RA</em> is not 0, the EA is the sum of the contents of GPR <em>RA</em> and <em>D</em>, a 16-bit, signed two's complement integer sign-extended to 32 bits. If <em>RA</em> is 0, then the EA is <em>D</em>.</p>
                `,
                "tooltip": "Load Floating-Point Single",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-lfs-load-floating-point-single-instruction"
            };
        case "LFSU":
            return {
                "html": `
                    <p>The <strong>lfsu</strong> instruction converts a floating-point, single-precision word in storage addressed by the effective address (EA) to floating-point, double-precision word and loads the result into floating-point register (FPR) <em>FRT</em>.</p>
                    <p>If <em>RA</em> is not 0, the EA is the sum of the contents of general-purpose register (GPR) <em>RA</em> and <em>D</em>, a 16-bit signed two's complement integer sign extended to 32 bits. If <em>RA</em> is 0, then the EA is <em>D</em>.</p>
                    <p>If <em>RA</em> does not equal 0 and the storage access does not cause an Alignment interrupt or a Data Storage interrupt, then the EA is stored in GPR <em>RA</em>.</p>
                `,
                "tooltip": "Load Floating-Point Single with Update",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-lfsu-load-floating-point-single-update-instruction"
            };
        case "LFSUX":
            return {
                "html": `
                    <p>The <strong>lfsux</strong> instruction converts a floating-point, single-precision word in storage addressed by the effective address (EA) to floating-point, double-precision word and loads the result into floating-point register (FPR) <em>FRT</em>.</p>
                    <p>If general-purpose register (GPR) <em>RA</em> is not 0, the EA is the sum of the contents of GPR <em>RA</em> and GPR <em>RB</em>. If <em>RA</em> is 0, then the EA is the contents of GPR <em>RB</em>.</p>
                    <p>If GPR <em>RA</em> does not equal 0 and the storage access does not cause an Alignment interrupt or a Data Storage interrupt, then the EA is stored in GPR <em>RA</em>.</p>
                `,
                "tooltip": "Load Floating-Point Single with Update Indexed",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=is-lfsux-load-floating-point-single-update-indexed-instruction"
            };
        case "LFSX":
            return {
                "html": `
                    <p>The <strong>lfsx</strong> instruction converts a floating-point, single-precision word in storage addressed by the effective address (EA) to floating-point, double-precision word and loads the result into floating-point register (FPR) <em>FRT</em>.</p>
                    <p>If general-purpose register (GPR) <em>RA</em> is not 0, the EA is the sum of the contents of GPR <em>RA</em> and GPR <em>RB</em>. If <em>RA</em> is 0, then the EA is the contents of GPR <em>RB</em>.</p>
                `,
                "tooltip": "Load Floating-Point Single Indexed",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-lfsx-load-floating-point-single-indexed-instruction"
            };
        case "LHA":
            return {
                "html": `
                    <p>The <strong>lha</strong> instruction loads a halfword of data from a specified location in memory, addressed by the effective address (EA), into bits 16-31 of the target general-purpose register (GPR) <em>RT</em> and copies bit 0 of the halfword into bits 0-15 of GPR <em>RT</em>.</p>
                    <p>If GPR <em>RA</em> is not 0, the EA is the sum of the contents of GPR <em>RA</em> and <em>D</em>, a 16-bit signed two's complement integer sign extended to 32 bits. If GPR <em>RA</em> is 0, then the EA is <em>D</em>.</p>
                `,
                "tooltip": "Load Half Algebraic",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-lha-load-half-algebraic-instruction"
            };
        case "LHAU":
            return {
                "html": `
                    <p>The <strong>lhau</strong> instruction loads a halfword of data from a specified location in memory, addressed by the effective address (EA), into bits 16-31 of the target general-purpose register (GPR) <em>RT</em> and copies bit 0 of the halfword into bits 0-15 of GPR <em>RT</em>.</p>
                    <p>If GPR <em>RA</em> is not 0, the EA is the sum of the contents of GPR <em>RA</em> and <em>D</em>, a 16-bit, signed two's complement integer sign-extended to 32 bits. If GPR <em>RA</em> is 0, then the EA is <em>D</em>.</p>
                    <p>If <em>RA</em> does not equal <em>RT</em> and <em>RA</em> does not equal 0, and the storage access does not cause an Alignment interrupt or a Data Storage interrupt, then the EA is placed into GPR <em>RA</em>.</p>
                `,
                "tooltip": "Load Half Algebraic with Update",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-lhau-load-half-algebraic-update-instruction"
            };
        case "LHAUX":
            return {
                "html": `
                    <p>The <strong>lhaux</strong> instruction loads a halfword of data from a specified location in memory addressed by the effective address (EA) into bits 16-31 of the target general-purpose register (GPR) <em>RT</em> and copies bit 0 of the halfword into bits 0-15 of GPR <em>RT</em>.</p>
                    <p>If GPR <em>RA</em> is not 0, the EA is the sum of the contents of GPR <em>RA</em> and GPR <em>RB</em>. If GPR <em>RA</em> is 0, then the EA is the contents of GPR <em>RB</em>.</p>
                    <p>If <em>RA</em> does not equal <em>RT</em> and <em>RA</em> does not equal 0, and the storage access does not cause an Alignment interrupt or a Data Storage interrupt, then the EA is placed into GPR <em>RA</em>.</p>
                `,
                "tooltip": "Load Half Algebraic with Update Indexed",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-lhaux-load-half-algebraic-update-indexed-instruction"
            };
        case "LHAX":
            return {
                "html": `
                    <p>The <strong>lhax</strong> instruction loads a halfword of data from a specified location in memory, addressed by the effective address (EA), into bits 16-31 of the target general-purpose register (GPR) <em>RT</em> and copies bit 0 of the halfword into bits 0-15 of GPR <em>RT</em>.</p>
                    <p>If GPR <em>RA</em> is not 0, the EA is the sum of the contents of GPR <em>RA</em> and GPR <em>RB</em>. If GPR <em>RA</em> is 0, then the EA is the contents of GPR <em>RB</em>.</p>
                `,
                "tooltip": "Load Half Algebraic Indexed",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-lhax-load-half-algebraic-indexed-instruction"
            };
        case "LHBRX":
            return {
                "html": `
                    <p>The <strong>lhbrx</strong> instruction loads bits 00-07 and bits 08-15 of the halfword in storage addressed by the effective address (EA) into bits 24-31 and bits 16-23 of general-purpose register (GPR) <em>RT,</em> and sets bits 00-15 of GPR <em>RT</em> to 0.</p>
                    <p>If GPR <em>RA</em> is not 0, the EA is the sum of the contents of GPR <em>RA</em> and GPR <em>RB</em>. If GPR <em>RA</em> is 0, then the EA is the contents of GPR <em>RB</em>.</p>
                `,
                "tooltip": "Load Half Byte-Reverse Indexed",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-lhbrx-load-half-byte-reverse-indexed-instruction"
            };
        case "LHZ":
            return {
                "html": `
                    <p>The <strong>lhz</strong> instruction loads a halfword of data from a specified location in memory, addressed by the effective address (EA), into bits 16-31 of the target general-purpose register (GPR) <em>RT</em> and sets bits 0-15 of GPR <em>RT</em> to 0.</p>
                    <p>If GPR <em>RA</em> is not 0, the EA is the sum of the contents of GPR <em>RA</em> and <em>D</em>, a 16-bit, signed two's complement integer sign-extended to 32 bits. If GPR <em>RA</em> is 0, then the EA is <em>D</em>.</p>
                `,
                "tooltip": "Load Half and Zero",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-lhz-load-half-zero-instruction"
            };
        case "LHZU":
            return {
                "html": `
                    <p>The <strong>lhzu</strong> instruction loads a halfword of data from a specified location in memory, addressed by the effective address (EA), into bits 16-31 of the target general-purpose register (GPR) <em>RT</em> and sets bits 0-15 of GPR <em>RT</em> to 0.</p>
                    <p>If GPR <em>RA</em> is not 0, the EA is the sum of the contents of GPR <em>RA</em> and <em>D</em>, a 16-bit, signed two's complement integer sign-extended to 32 bits. If GPR <em>RA</em> is 0, then the EA is <em>D</em>.</p>
                    <p>If <em>RA</em> does not equal <em>RT</em> and <em>RA</em> does not equal 0, and the storage access does not cause an Alignment interrupt or a Data Storage interrupt, then the EA is placed into GPR <em>RA</em>.</p>
                `,
                "tooltip": "Load Half and Zero with Update",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-lhzu-load-half-zero-update-instruction"
            };
        case "LHZUX":
            return {
                "html": `
                    <p>The <strong>lhzux</strong> instruction loads a halfword of data from a specified location in memory, addressed by the effective address (EA), into bits 16-31 of the target general-purpose register (GPR) <em>RT</em> and sets bits 0-15 of GPR <em>RT</em> to 0.</p>
                    <p>If GPR <em>RA</em> is not 0, the EA is the sum of the contents of GPR <em>RA</em> and GPR <em>RB</em>. If GPR <em>RA</em> is 0, then the EA is the contents of GPR <em>RB</em>.</p>
                    <p>If <em>RA</em> does not equal <em>RT</em> and <em>RA</em> does not equal 0, and the storage access does not cause an Alignment interrupt or a Data Storage interrupt, then the EA is placed into GPR <em>RA</em>.</p>
                `,
                "tooltip": "Load Half and Zero with Update Indexed",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-lhzux-load-half-zero-update-indexed-instruction"
            };
        case "LHZX":
            return {
                "html": `
                    <p>The <strong>lhzx</strong> instruction loads a halfword of data from a specified location in memory, addressed by the effective address (EA), into bits 16-31 of the target general-purpose register (GPR) <em>RT</em> and sets bits 0-15 of GPR <em>RT</em> to 0.</p>
                    <p>If GPR <em>RA</em> is not 0, the EA is the sum of the contents of GPR <em>RA</em> and GPR <em>RB</em>. If GPR <em>RA</em> is 0, then the EA is the contents of GPR <em>RB</em>.</p>
                `,
                "tooltip": "Load Half and Zero Indexed",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-lhzx-load-half-zero-indexed-instruction"
            };
        case "LMW":
        case "LM":
            return {
                "html": `
                    <p>The <strong>lmw</strong> and <strong>lm</strong> instructions load <em>N</em> consecutive words starting at the calculated effective address (EA) into a number of general-purpose registers (GPR), starting at GPR <em>RT</em> and filling all GPRs through GPR 31. <em>N</em> is equal to 32-<em>RT</em> field, the total number of consecutive words that are placed in consecutive registers.</p>
                    <p>If GPR <em>RA</em> is not 0, the EA is the sum of the contents of GPR <em>RA</em> and <em>D</em>. If GPR <em>RA</em> is 0, then the EA is <em>D</em>.</p>
                    <p>Consider the following when using the PowerPC® instruction <strong>lmw</strong>:</p>
                    <ul>
                        <li>If GPR <em>RA</em> or GPR <em>RB</em> is in the range of registers to be loaded or <em>RT</em> = <em>RA</em> = 0, the results are boundedly undefined.</li>
                        <li>The EA must be a multiple of 4. If it is not, the system alignment error handler may be invoked or the results may be boundedly undefined.</li>
                    </ul>
                    <p>For the POWER® family instruction <strong>lm</strong>, if GPR <em>RA</em> is not equal to 0 and GPR <em>RA</em> is in the range to be loaded, then GPR <em>RA</em> is not written to. The data that would have normally been written into <em>RA</em> is discarded and the operation continues normally.</p>
                `,
                "tooltip": "Load Multiple Word",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-lmw-lm-load-multiple-word-instruction"
            };
        case "LQ":
            return {
                "html": `
                    <p>The <strong>lq</strong> instruction loads a quad word in storage from a specified location in memory addressed by the effective address (EA) into the target general-purpose registers (GPRs) <em>RT</em> and <em>RT+1</em>.</p>
                    <p>DQ is a 12-bit, signed two's complement number, which is sign extended to 64 bits and then multiplied by 16 to provide a displacement <em>Disp</em>. If GPR <em>RA</em> is not 0, the EA is the sum of the contents of GPR <em>RA</em> and <em>Disp</em>. If GPR <em>RA</em> is 0, then the EA is <em>Disp</em>.</p>
                `,
                "tooltip": "Load Quad Word",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-lq-load-quad-word-instruction"
            };
        case "LSCBX":
        case "LSCBX.":
            return {
                "html": `
                    <p>The <strong>lscbx</strong> instruction loads <em>N</em> consecutive bytes addressed by effective address (EA) into general-purpose register (GPR) <em>RT</em>, starting with the leftmost byte in register <em>RT</em>, through <em>RT</em> + <em>NR</em> - 1, and wrapping around back through GPR 0, if required, until either a byte match is found with XER16-23 or <em>N</em> bytes have been loaded. If a byte match is found, then that byte is also loaded.</p>
                    <p>If GPR <em>RA</em> is not 0, the EA is the sum of the contents of GPR <em>RA</em> and the address stored in GPR <em>RB</em>. If <em>RA</em> is 0, then EA is the contents of GPR <em>RB</em>.</p>
                    <p>Consider the following when using the <strong>lscbx</strong> instruction:</p>
                    <ul>
                        <li>XER(16-23) contains the byte to be compared.</li>
                        <li>XER(25-31) contains the byte count before the instruction is invoked and the number of bytes loaded after the instruction has completed.</li>
                        <li>If XER(25-31) = 0, GPR <em>RT</em> is not altered.</li>
                        <li><em>N</em> is XER(25-31), which is the number of bytes to load.</li>
                        <li><em>NR</em> is ceiling(<em>N</em>/4), which is the total number of registers required to contain the consecutive bytes.</li>
                    </ul>
                    <p>Bytes are always loaded left to right in the register. In the case when a match was found before <em>N</em> bytes were loaded, the contents of the rightmost bytes not loaded from that register and the contents of all succeeding registers up to and including register <em>RT</em> + <em>NR</em> - 1 are undefined. Also, no reference is made to storage after the matched byte is found. In the case when a match was not found, the contents of the rightmost bytes not loaded from register <em>RT</em> + <em>NR</em> - 1 are undefined.</p>
                    <p>If GPR <em>RA</em> is not 0 and GPRs <em>RA</em> and <em>RB</em> are in the range to be loaded, then GPRs <em>RA</em> and <em>RB</em> are not written to. The data that would have been written into them is discarded, and the operation continues normally. If the byte in XER(16-23) compares with any of the 4 bytes that would have been loaded into GPR <em>RA</em> or <em>RB</em>, but are being discarded for restartability, the EQ bit in the Condition Register and the count returned in XER(25-31) are undefined. The Multiply Quotient (MQ) Register is not affected by this operation.</p>
                `,
                "tooltip": "Load String and Compare Byte Indexed",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-lscbx-load-string-compare-byte-indexed-instruction"
            };
        case "LSWI":
        case "LSI":
            return {
                "html": `
                    <p>The <strong>lswi</strong> and <strong>lsi</strong> instructions load <em>N</em> consecutive bytes in storage addressed by the effective address (EA) into general-purpose register GPR <em>RT,</em> starting with the leftmost byte, through GPR <em>RT</em>+<em>NR</em>-1, and wrapping around back through GPR 0, if required.</p>
                    <p>If GPR <em>RA</em> is not 0, the EA is the contents of GPR <em>RA</em>. If GPR <em>RA</em> is 0, then the EA is 0.</p>
                    <p>Consider the following when using the <strong>lswi</strong> and <strong>lsi</strong> instructions:</p>
                    <ul>
                        <li><em>NB</em> is the byte count.</li>
                        <li><em>RT</em> is the starting general-purpose register.</li>
                        <li><em>N</em> is <em>NB</em>, which is the number of bytes to load. If <em>NB</em> is 0, then <em>N</em> is 32.</li>
                        <li>NR is ceiling(N/4), which is the number of general-purpose registers to receive data.</li>
                    </ul>
                    <p>For the PowerPC® instruction <strong>lswi</strong>, if GPR <em>RA</em> is in the range of registers to be loaded or <em>RT</em> = <em>RA</em> = 0, the instruction form is invalid.</p>
                    <p>Consider the following when using the POWER® family instruction <strong>lsi</strong>:</p>
                    <ul>
                        <li>If GPR <em>RT</em> + <em>NR</em> - 1 is only partially filled on the left, the rightmost bytes of that general-purpose register are set to 0.</li>
                        <li>If GPR <em>RA</em> is in the range to be loaded, and if GPR <em>RA</em> is not equal to 0, then GPR <em>RA</em> is not written into by this instruction. The data that would have been written into it is discarded, and the operation continues normally.</li>
                    </ul>
                `,
                "tooltip": "Load String Word Immediate",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-lswi-lsi-load-string-word-immediate-instruction"
            };
        case "LSWX":
        case "LSX":
            return {
                "html": `
                    <p>The <strong>lswx</strong> and <strong>lsx</strong> instructions load <em>N</em> consecutive bytes in storage addressed by the effective address (EA) into general-purpose register (GPR) <em>RT,</em> starting with the leftmost byte, through GPR <em>RT</em> + <em>NR</em> - 1, and wrapping around back through GPR 0 if required.</p>
                    <p>If GPR <em>RA</em> is not 0, the EA is the sum of the contents of GPR <em>RA</em> and the address stored in GPR <em>RB</em>. If GPR <em>RA</em> is 0, then EA is the contents of GPR <em>RB</em>.</p>
                    <p>Consider the following when using the <strong>lswx</strong> and <strong>lsx</strong> instructions:</p>
                    <ul>
                        <li>XER(25-31) contain the byte count.</li>
                        <li><em>RT</em> is the starting general-purpose register.</li>
                        <li><em>N</em> is XER(25-31), which is the number of bytes to load.</li>
                        <li><em>NR</em> is ceiling(N/4), which is the number of registers to receive data.</li>
                        <li>If XER(25-31) = 0, general-purpose register <em>RT</em> is not altered.</li>
                    </ul>
                    <p>For the PowerPC® instruction <strong>lswx</strong>, if <em>RA</em> or <em>RB</em> is in the range of registers to be loaded or <em>RT</em> = <em>RA</em> = 0, the results are boundedly undefined.</p>
                    <p>Consider the following when using the POWER® family instruction <strong>lsx</strong>:</p>
                    <ul>
                        <li>If GPR <em>RT</em> + <em>NR</em> - 1 is only partially filled on the left, the rightmost bytes of that general-purpose register are set to 0.</li>
                        <li>If GPRs <em>RA</em> and <em>RB</em> are in the range to be loaded, and if GPR <em>RA</em> is not equal to 0, then GPR <em>RA</em> and <em>RB</em> are not written into by this instruction. The data that would have been written into them is discarded, and the operation continues normally.</li>
                    </ul>
                `,
                "tooltip": "Load String Word Indexed",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-lswx-lsx-load-string-word-indexed-instruction"
            };
        case "LWA":
            return {
                "html": `
                    <p>The fullword in storage located at the effective address (EA) is loaded into the low-order 32 bits of the target general purpose register (GRP) <em>RT</em>. The value is then sign-extended to fill the high-order 32 bits of the register.</p>
                    <p>DS is a 14-bit, signed two's complement number, which is sign-extended to 64 bits, and then multiplied by 4 to provide a displacement <em>Disp</em>. If GPR <em>RA</em> is not 0, the EA is the sum of the contents of GPR <em>RA</em> and <em>Disp</em>. If GPR <em>RA</em> is 0, then the EA is <em>Disp</em>.</p>
                `,
                "tooltip": "Load Word Algebraic",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-lwa-load-word-algebraic-instruction"
            };
        case "LWARX":
            return {
                "html": `
                    <p>The <strong>lwarx</strong> and <strong>stwcx.</strong> instructions are primitive, or simple, instructions used to perform a read-modify-write operation to storage. If the store is performed, the use of the <strong>lwarx</strong> and <strong>stwcx.</strong> instructions ensures that no other processor or mechanism has modified the target memory location between the time the <strong>lwarx</strong> instruction is executed and the time the <strong>stwcx.</strong> instruction completes.</p>
                    <p>If general-purpose register (GPR) <em>RA</em> = 0, the effective address (EA) is the content of GPR <em>RB</em>. Otherwise, the EA is the sum of the content of GPR <em>RA</em> plus the content of GPR <em>RB</em>.</p>
                    <p>The <strong>lwarx</strong> instruction loads the word from the location in storage specified by the EA into the target GPR <em>RT</em>. In addition, a reservation on the memory location is created for use by a subsequent <strong>stwcx.</strong> instruction.</p>
                `,
                "tooltip": "Load Word and Reserve Indexed",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-lwarx-load-word-reserve-indexed-instruction"
            };
        case "LWAUX":
            return {
                "html": `
                    <p>The <strong>lwaux</strong> instruction loads a full word of data from storage into the low-order 32 bits of the specified general purpose register and sign extends the data into the high-order 32 bits of the register while updating the address base.</p>
                    <p>The fullword in storage located at the effective address (EA) is loaded into the low-order 32 bits of the target general puspose register (GRP). The value is then sign-extended to fill the high-order 32 bits of the register. The EA is the sum of the contents of GRP <em>RA</em> and GRP <em>RB</em>.</p>
                    <p>If <em>RA</em> = 0 or <em>RA</em> = <em>RT</em>, the instruction form is invalid.</p>
                `,
                "tooltip": "Load Word Algebraic with Update Indexed",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-lwaux-load-word-algebraic-update-indexed-instruction"
            };
        case "LWAX":
            return {
                "html": `
                    <p>The <strong>lwax</strong> instruction loads a fullword of data from storage into the low-order 32 bits of the specified general purpose register and sign-extends the data into the high-order 32 bits of the register.</p>
                    <p>The fullword in storage located at the effective address (EA) is loaded into the low-order 32 bits of the target general puspose register (GRP). The value is then sign-extended to fill the high-order 32 bits of the register.</p>
                    <p>If GRP <em>RA</em> is not 0, the EA is the sum of the contents of GRP <em>RA</em> and <em>B</em>; otherwise, the EA is equal to the contents of <em>RB</em>.</p>
                `,
                "tooltip": "Load Word Algebraic Indexed",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-lwax-load-word-algebraic-indexed-instruction"
            };
        case "LWBRX":
        case "LBRX":
            return {
                "html": `
                    <p>The <strong>lwbrx</strong> and <strong>lbrx</strong> instructions load a byte-reversed word in storage from a specified location in memory addressed by the effective address (EA) into the target general-purpose register (GPR) <em>RT</em>.</p>
                    <p>Consider the following when using the <strong>lwbrx</strong> and <strong>lbrx</strong> instructions:</p>
                    <ul>
                        <li>Bits 00-07 of the word in storage addressed by EA are placed into bits 24-31 of GPR <em>RT</em>.</li>
                        <li>Bits 08-15 of the word in storage addressed by EA are placed into bits 16-23 of GPR <em>RT</em>.</li>
                        <li>Bits 16-23 of the word in storage addressed by EA are placed into bits 08-15 of GPR <em>RT</em>.</li>
                        <li>Bits 24-31 of the word in storage addressed by EA are placed into bits 00-07 of GPR <em>RT</em>.</li>
                    </ul>
                    <p>If GPR <em>RA</em> is not 0, the EA is the sum of the contents of GPR <em>RA</em> and GPR <em>RB</em>. If GPR <em>RA</em> is 0, then the EA is the contents of GPR <em>RB</em>.</p>
                `,
                "tooltip": "Load Word Byte-Reverse Indexed",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=is-lwbrx-lbrx-load-word-byte-reverse-indexed-instruction"
            };
        case "LWZ":
        case "L":
            return {
                "html": `
                    <p>The <strong>lwz and l</strong> instructions load a word in storage from a specified location in memory addressed by the effective address (EA) into the target general-purpose register (GPR) <em>RT</em>.</p>
                    <p>If GPR <em>RA</em> is not 0, the EA is the sum of the contents of GPR <em>RA</em> and <em>D</em>, a 16-bit, signed two's complement integer sign-extended to 32 bits. If GPR <em>RA</em> is 0, then the EA is <em>D</em>.</p>
                `,
                "tooltip": "Load Word and Zero",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-lwz-l-load-word-zero-instruction"
            };
        case "LWZU":
        case "LU":
            return {
                "html": `
                    <p>The <strong>lwzu</strong> and <strong>lu</strong> instructions load a word in storage from a specified location in memory addressed by the effective address (EA) into the target general-purpose register (GPR) <em>RT</em>.</p>
                    <p>If GPR <em>RA</em> is not 0, the EA is the sum of the contents of GPR <em>RA</em> and <em>D</em>, a 16-bit, signed two's complement integer sign-extended to 32 bits. If GPR <em>RA</em> is 0, then the EA is <em>D</em>.</p>
                    <p>If <em>RA</em> does not equal <em>RT</em> and <em>RA</em> doesnot equal 0, and the storage access does not cause an Alignment interruptor a Data Storage interrupt, then the EA is placed into GPR <em>RA</em>.</p>
                `,
                "tooltip": "Load Word with Zero Update",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-lwzu-lu-load-word-zero-update-instruction"
            };
        case "LWZUX":
        case "LUX":
            return {
                "html": `
                    <p>The <strong>lwzux and lux</strong> instructions load a word of data from a specified location in memory, addressed by the effective address (EA), into the target general-purpose register (GPR) <em>RT</em>.</p>
                    <p>If GPR <em>RA</em> is not 0, the EA is the sum of the contents of GPR <em>RA</em> and GPR <em>RB</em>. If GPR <em>RA</em> is 0, then the EA is the contents of GPR <em>RB</em>.</p>
                    <p>If GPR <em>RA</em> does not equal RT and <em>RA</em> does not equal 0, and the storage access does not cause an Alignment interrupt or a Data Storage interrupt, then the EA is placed into GPR <em>RA</em>.</p>
                `,
                "tooltip": "Load Word and Zero with Update Indexed",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=is-lwzux-lux-load-word-zero-update-indexed-instruction"
            };
        case "LWXZ":
        case "LX":
            return {
                "html": `
                    <p>The <strong>lwzx</strong> and <strong>lx</strong> instructions load a word of data from a specified location in memory, addressed by the effective address (EA), into the target general-purpose register (GPR) <em>RT</em>.</p>
                    <p>The <strong>lwzx</strong> and <strong>lx</strong> instructions load a word of data from a specified location in memory, addressed by the effective address (EA), into the target general-purpose register (GPR) <em>RT</em>.</p>
                `,
                "tooltip": "Load Word and Zero Indexed",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-lwzx-lx-load-word-zero-indexed-instruction"
            };
        case "MASKG":
        case "MASKG.":
            return {
                "html": `
                    <p>The <strong>maskg</strong> instruction generates a mask from a starting point defined by bits 27-31 of general-purpose register (GPR) <em>RS</em> to an end point defined by bits 27-31 of GPR <em>RB</em> and stores the mask in GPR <em>RA</em>.</p>
                    <p>Consider the following when using the <strong>maskg</strong> instruction:</p>
                    <ul>
                        <li>If the starting point bit is less than the end point bit + 1, then the bits between and including the starting point and the end point are set to ones. All other bits are set to 0.</li>
                        <li>If the starting point bit is the same as the end point bit + 1, then all 32 bits are set to ones.</li>
                        <li>If the starting point bit is greater than the end point bit + 1, then all of the bits between and including the end point bit + 1 and the starting point bit - 1 are set to zeros. All other bits are set to ones.</li>
                    </ul>
                `,
                "tooltip": "Mask Generate",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-maskg-mask-generate-instruction"
            };
        case "MASKIR":
        case "MASKIR.":
            return {
                "html": `
                    <p>The <strong>maskir</strong> instruction stores the contents of general-purpose register (GPR) <em>RS</em> in GPR <em>RA</em> under control of the bit mask in GPR <em>RB</em>.</p>
                    <p>The value for each bit in the target GPR <em>RA</em> is determined as follows:</p>
                    <ul>
                        <li>If the corresponding bit in the mask GPR <em>RB</em> is 1, then the bit in the target GPR <em>RA</em> is given the value of the corresponding bit in the source GPR <em>RS</em>.</li>
                        <li>If the corresponding bit in the mask GPR <em>RB</em> is 0, then the bit in the target GPR <em>RA</em> is unchanged.</li>
                    </ul>
                `,
                "tooltip": "Mask Insert from Register",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-maskir-mask-insert-from-register-instruction"
            };
        case "MCRF":
            return {
                "html": `<p>The <strong>mcrf</strong> instruction copies the contents of the condition register field specified by <em>BFA</em> into the condition register field specified by <em>BF</em>. All other fields remain unaffected.</p>`,
                "tooltip": "Move Condition Register Field",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-mcrf-move-condition-register-field-instruction"
            };
        case "MCRFS":
            return {
                "html": `
                    <p>The <strong>mcrfs</strong> instruction copies four bits of the Floating-Point Status and Control Register (FPSCR) specified by <em>BFA</em> into Condition Register Field <em>BF</em>. All other Condition Register bits are unchanged.</p>
                    <p>If the field specified by <em>BFA</em> contains reserved or undefined bits, then bits of zero value are supplied for the copy.</p>
                `,
                "tooltip": "Move to Condition Register from FPSCR",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-mcrfs-move-condition-register-from-fpscr-instruction"
            };
        case "MCRXR":
            return {
                "html": `<p>The <strong>mcrxr</strong> copies the contents of Fixed-Point Exception Register Field 0 bits 0-3 into Condition Register Field <em>BF</em> and resets Fixed-Point Exception Register Field 0 to 0.</p>`,
                "tooltip": "Move to Condition Register from XER",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-mcrxr-move-condition-register-from-xer-instruction"
            };
        case "MFCR":
            return {
                "html": `<p>The <strong>mfcr</strong> instruction copies the contents of the Condition Register into target general-purpose register (GPR) <em>RT</em>.</p>`,
                "tooltip": "Move from Condition Register",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-mfcr-move-from-condition-register-instruction"
            };
        case "MFFS":
        case "MFFS.":
            return {
                "html": `<p>The <strong>mffs</strong> instruction places the contents of the Floating-Point Status and Control Register into bits 32-63 of floating-point register (FPR) <em>FRT</em>. The bits 0-31 of floating-point register <em>FRT</em> are undefined.</p>`,
                "tooltip": "Move from FPSCR",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-mffs-move-from-fpscr-instruction"
            };
        case "MFMSR":
            return {
                "html": `<p>The <strong>mfmsr</strong> instruction copies the contentsof the Machine State Register into the target general-purpose register(GPR) <em>RT</em>.</p>`,
                "tooltip": "Move from Machine State Register",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-mfmsr-move-from-machine-state-register-instruction"
            };
        case "MFOCRF":
            return {
                "html": `<p>The <strong>mfocrf</strong> instruction copies the contents of one Condition Register field specified by the field mask FXM into the target general-purpose register (GPR) <em>RT</em>.</p>`,
                "tooltip": "Move from One Condition Register Field",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=is-mfocrf-move-from-one-condition-register-field-instruction"
            };
        case "MFSPR":
            return {
                "html": `<p>The <strong>mfspr</strong> instruction copies the contents of the special-purpose register <em>SPR</em> into target general-purpose register (GPR) <em>RT</em>.</p>`,
                "tooltip": "Move from Special-Purpose Register",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-mfspr-move-from-special-purpose-register-instruction"
            };
        case "MFSR":
            return {
                "html": `<p>The <strong>mfsr</strong> instruction copies the contents of segment register (SR) into target general-purpose register (GPR) <em>RT</em>.</p>`,
                "tooltip": "Move from Segment Register",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-mfsr-move-from-segment-register-instruction"
            };
        case "MFSRI":
            return {
                "html": `<p>The <strong>mfsri</strong> instruction copies the contents of segment register (SR), specified by bits 0-3 of the calculated contents of the general-purpose register (GPR) <em>RA</em>, into GPR <em>RS</em>. If <em>RA</em> is not 0, the specifying bits in GPR <em>RA</em> are calculated by adding the original contents of <em>RA</em> to GPR <em>RB</em> and placing the sum in <em>RA</em>. If <em>RA</em> = <em>RS</em>, the sum is not placed in <em>RA</em>.</p>`,
                "tooltip": "Move from Segment Register Indirect",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-mfsri-move-from-segment-register-indirect-instruction"
            };
        case "MFSRIN":
            return {
                "html": `<p>The <strong>mfsrin</strong> instruction copies the contents of segment register (SR), specified by bits 0-3 of the general-purpose register (GPR) <em>RB</em>, into GPR <em>RT</em>.</p>`,
                "tooltip": "Move from Segment Register Indirect",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-mfsrin-move-from-segment-register-indirect-instruction"
            };
        case "MTCRF":
            return {
                "html": `<p>The <strong>mtcrf</strong> instruction copies the contents of source general-purpose register (GPR) <em>RS</em> into the condition register under the control of field mask <em>FXM</em>.</p>`,
                "tooltip": "Move to Condition Register Fields",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-mtcrf-move-condition-register-fields-instruction"
            };
        case "MTFSB0":
        case "MTFSB0.":
            return {
                "html": `<p>The <strong>mtfsb0</strong> instruction sets the Floating-Point Status and Control Register bit specified by <em>BT</em> to 0.</p>`,
                "tooltip": "Move to FPSCR Bit 0",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-mtfsb0-move-fpscr-bit-0-instruction"
            };
        case "MTFSB1":
        case "MTFSB1.":
            return {
                "html": `<p>The <strong>mtfsb1</strong> instruction sets the Floating-Point Status and Control Register (FPSCR) bit specified by <em>BT</em> to 1.</p>`,
                "tooltip": "Move to FPSCR Bit 1",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-mtfsb1-move-fpscr-bit-1-instruction"
            };
        case "MTFSF":
        case "MTFSF.":
            return {
                "html": `<p>The <strong>mtfsf</strong> instruction copies bits 32-63 of the contents of the floating-point register (FPR) <em>FRB</em> into the Floating-Point Status and Control Register under the control of the field mask specified by <em>FLM</em>.</p>`,
                "tooltip": "Move to FPSCR Fields",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-mtfsf-move-fpscr-fields-instruction"
            };
        case "MTFSFI":
        case "MTFSFI.":
            return {
                "html": `<p>The <strong>mtfsfi</strong> instruction copies the immediate value specified by the <em>I</em> parameter into the Floating-Point Status and Control Register field specified by <em>BF</em>. None of the other fields of the Floating-Point Status and Control Register are affected.</p>`,
                "tooltip": "Move to FPSCR Field Immediate",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-mtfsfi-move-fpscr-field-immediate-instruction"
            };
        case "MTOCRF":
            return {
                "html": `<p>The <strong>mtocrf</strong> instruction copies the contents of source general-purpose register (GPR) <em>RS</em> into the condition register under the control of field mask <em>FXM</em>.</p>`,
                "tooltip": "Move to One Condition Register Field",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-mtocrf-move-one-condition-register-field-instruction"
            };
        case "MTSPR":
            return {
                "html": `<p>The <strong>mtspr</strong> instruction copies the contents of the source general-purpose register <em>RS</em> into the target special-purpose register <em>SPR</em>.</p>`,
                "tooltip": "Move to Special-Purpose Register",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-mtspr-move-special-purpose-register-instruction"
            };
        case "MUL":
        case "MUL.":
        case "MULO":
        case "MULO.":
            return {
                "html": `<p>The <strong>mul</strong> instruction multiplies the contents of general-purpose register (GPR) <em>RA</em> and GPR <em>RB</em>, and stores bits 0-31 of the result in the target GPR <em>RT</em> and bits 32-63 of the result in the MQ Register.</p>`,
                "tooltip": "Multiply",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-mul-multiply-instruction"
            };
        case "MULHD":
        case "MULHD.":
            return {
                "html": `
                    <p>The 64-bit operands are the contents of general purpose registers (GPR) <em>RA</em> and <em>RB</em>. The high-order 64 bits of the 128-bit product of the operands are placed into <em>RT</em>.</p>
                    <p>Both the operands and the product are interpreted as signed integers.</p>
                    <p>This instruction may execute faster on some implementations if <em>RB</em> contains the operand having the smaller absolute value.</p>
                `,
                "tooltip": "Multiply High Double Word",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-mulhd-multiply-high-double-word-instruction"
            };
        case "MULHDU":
        case "MULHDU.":
            return {
                "html": `
                    <p>Both the operands and the product are interpreted as unsigned integers, except that if Rc = 1 (the <strong>mulhw.</strong> instruction) the first three bits of the condition register 0 field are set by signed comparison of the result to zero.</p>
                    <p>The 64-bit operands are the contents of <em>RA</em> and <em>RB</em>. The low-order 64 bits of the 128-bit product of the operands are placed into <em>RT</em>.</p>
                    <p>Other registers altered:</p>
                    <ul>
                        <li>
                            Condition Register (CR0 field):
                            <p>Affected: LT, GT, EQ, SO (if Rc = 1)</p>
                            <p>Note: The setting of CR0 bits LT, GT, and EQ is mode-dependent, and reflects overflow of the 64-bit result.</p>
                        </li>
                    </ul>
                    <p>This instruction may execute faster on some implementations if <em>RB</em> contains the operand having the smaller absolute value.</p>
                `,
                "tooltip": "Multiply High Double Word Unsigned",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-mulhdu-multiply-high-double-word-unsigned-instruction"
            };
        case "MULHW":
        case "MULHW.":
            return {
                "html": `<p>The <strong>mulhw</strong> instruction multiplies the contents of general-purpose register (GPR) <em>RA</em> and GPR <em>RB</em> and places the most significant 32 bits of the 64-bit product in the target GPR <em>RT</em>. Both the operands and the product are interpreted as signed integers.</p>`,
                "tooltip": "Multiply High Word",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-mulhw-multiply-high-word-instruction"
            };
        case "MULHWU":
        case "MULHWU.":
            return {
                "html": `<p>The <strong>mulhwu</strong> instruction multiplies the contents of general-purpose register (GPR) <em>RA</em> and GPR <em>RB</em> and places the most significant 32 bits of the 64-bit product in the target GPR <em>RT</em>. Both the operands and the product are interpreted as unsigned integers.</p>`,
                "tooltip": "Multiply High Word Unsigned",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-mulhwu-multiply-high-word-unsigned-instruction"
            };
        case "MULLD":
        case "MULLD.":
        case "MULLDO":
        case "MULLDO.":
            return {
                "html": `
                    <p>The 64-bit operands are the contents of general purpose registers (GPR) <em>RA</em> and <em>RB</em>. The low-order 64 bits of the 128-bit product of the operands are placed into <em>RT</em>.</p>
                    <p>Both the operands and the product are interpreted as signed integers. The low-order 64 bits of the product are independent of whether the operands are regarded as signed or unsigned 64-bit integers. If OE = 1 (the <strong>mulldo</strong> and <strong>mulldo.</strong> instructions), then OV is set if the product cannot be represented in 64 bits.</p>
                    <p>This instruction may execute faster on some implementations if <em>RB</em> contains the operand having the smaller absolute value.</p>
                `,
                "tooltip": "Multiply Low Double Word",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-mulld-multiply-low-double-word-instruction"
            };
        case "MULLI":
        case "MULI":
            return {
                "html": `<p>The <strong>mulli</strong> and <strong>muli</strong> instructions sign extend the <em>SI</em> field to 32 bits and then multiply the extended value by the contents of general-purpose register (GPR) <em>RA</em>. The least significant 32 bits of the 64-bit product are placed in the target GPR <em>RT</em>.</p>`,
                "tooltip": "Multiply Low Immediate",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-mulli-muli-multiply-low-immediate-instruction"
            };
        case "MULLW":
        case "MULLW.":
        case "MULLWO":
        case "MULLWO.":
        case "MULS":
        case "MULS.":
        case "MULSO":
        case "MULSO.":
            return {
                "html": `<p>The <strong>mullw</strong> and <strong>muls</strong> instructions multiply the contents of general-purpose register (GPR) <em>RA</em> by the contents of GPR <em>RB</em>, and place the least significant 32 bits of the result in the target GPR <em>RT</em>.</p>`,
                "tooltip": "Multiply Low Word",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-mullw-muls-multiply-low-word-instruction"
            };
        case "NAND":
        case "NAND.":
            return {
                "html": `<p>The <strong>nand</strong> instruction logically ANDs the contents of general-purpose register (GPR) <em>RS</em> with the contents of GPR <em>RB</em> and stores the complement of the result in the target GPR <em>RA</em>.</p>`,
                "tooltip": "NAND",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-nand-nand-instruction"
            };
        case "NEG":
        case "NEG.":
        case "NEGO":
        case "NEGO.":
            return {
                "html": `
                    <p>The <strong>neg</strong> instruction adds 1 to the one's complement of the contents of a general-purpose register (GPR) <em>RA</em> and stores the result in GPR <em>RT</em>.</p>
                    <p>If GPR <em>RA</em> contains the most negative number (that is, 0x8000 0000), the result of the instruction is the most negative number and signals the Overflow bit in the Fixed-Point Exception Register if OE is 1.</p>
                `,
                "tooltip": "Negate",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-neg-negate-instruction"
            };
        case "NOR":
        case "NOR.":
            return {
                "html": `<p>The <strong>nor</strong> instruction logically ORs the contents of general-purpose register (GPR) <em>RS</em> with the contents of GPR <em>RB</em> and stores the complemented result in GPR <em>RA</em>.</p>`,
                "tooltip": "NOR",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-nor-nor-instruction"
            };
        case "OR":
        case "OR.":
            return {
                "html": `<p>The <strong>or</strong> instruction logically ORs the contents of general-purpose register (GPR) <em>RS</em> with the contents of GPR <em>RB</em> and stores the result in GPR <em>RA</em>.</p>`,
                "tooltip": "OR",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=is-instruction"
            };
        case "ORC":
        case "ORC.":
            return {
                "html": `<p>The <strong>orc</strong> instruction logically ORs the contents of general-purpose register (GPR) <em>RS</em> with the complement of the contents of GPR <em>RB</em> and stores the result in GPR <em>RA</em>.</p>`,
                "tooltip": "OR with Complement",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-orc-complement-instruction"
            };
        case "ORI":
        case "ORIL":
            return {
                "html": `<p>The <strong>ori</strong> and <strong>oril</strong> instructions logically OR the contents of general-purpose register (GPR) <em>RS</em> with the concatenation of x'0000' and a 16-bit unsigned integer, <em>UI</em>, and place the result in GPR <em>RA</em>.</p>`,
                "tooltip": "OR Immediate",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-ori-oril-immediate-instruction"
            };
        case "ORIS":
        case "ORIU":
            return {
                "html": `<p>The <strong>oris</strong> and <strong>oriu</strong> instructions logically OR the contents of general-purpose register (GPR) <em>RS</em> with the concatenation of a 16-bit unsigned integer, <em>UI</em>, and x'0000' and store the result in GPR <em>RA</em>.</p>`,
                "tooltip": "OR Immediate Shifted",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-oris-oriu-immediate-shifted-instruction"
            };
        case "POPCNTBD":
            return {
                "html": `<p>The <strong>popcntbd</strong> instruction counts the number of one bits in each byte of register <em>RS</em> and places the count in to the corresponding byte of register <em>RA</em>. The number ranges from 0 to 8, inclusive.</p>`,
                "tooltip": "Population Count Byte Doubleword",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-popcntbd-population-count-byte-doubleword-instruction"
            };
        case "RAC":
        case "RAC.":
            return {
                "html": `
                    <p>The <strong>rac</strong> instruction computes an effective address (EA) from the sum of the contents of general-purpose register (GPR) <em>RA</em> and the contents of GPR <em>RB,</em> and expands the EA into a virtual address.</p>
                    <p>If <em>RA</em> is not 0 and if <em>RA</em> is not <em>RT</em>, then the <strong>rac</strong> instruction stores the EA in GPR <em>RA</em>, translates the result into a real address, and stores the real address in GPR <em>RT</em>.</p>
                    <p>Consider the following when using the <strong>rac</strong> instruction:</p>
                    <ul>
                        <li>If GPR <em>RA</em> is 0, then EA is the sum of the contents of GPR <em>RB</em> and 0.</li>
                        <li>EA is expanded into its virtual address and translated into a real address, regardless of whether data translation is enabled.</li>
                        <li>If the translation is successful, the EQ bit in the condition register is set and the real address is placed in GPR <em>RT</em>.</li>
                        <li>If the translation is unsuccessful, the EQ bit is set to 0, and 0 is placed in GPR <em>RT</em>.</li>
                        <li>If the effective address specifies an I/O address, the EQ bit is set to 0, and 0 is placed in GPR <em>RT</em>.</li>
                        <li>The reference bit is set if the real address is not in the Translation Look-Aside buffer (TLB).</li>
                    </ul>
                `,
                "tooltip": "Real Address Compute",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-rac-real-address-compute-instruction"
            };
        case "RFI":
            return {
                "html": `
                    <p>The <strong>rfi</strong> instruction places bits 16-31 of Save Restore Register1 (SRR1) into bits 16-31 of the Machine State Register (MSR), and then begins fetching and processing instructions at the address contained inSave Restore Register0 (SRR0), using the new MSR value.</p>
                    <p>If the Link bit (LK) is set to 1, the contents of the Link Register are undefined.</p>
                `,
                "tooltip": "Return from Interrupt",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-rfi-return-from-interrupt-instruction"
            };
        case "RFID":
            return {
                "html": `<p>Reinitializes the Machine State Register and continues processing after an interrupt.</p>`,
                "tooltip": "Return from Interrupt Double Word",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-rfid-return-from-interrupt-double-word-instruction"
            };
        case "RFSVC":
            return {
                "html": `
                    <p>The <strong>rfsvc</strong> instruction reinitializes the Machine State Register (MSR) and starts processing after a supervisor call. This instruction places bits 16-31 of the Count Register into bits 16-31 of the Machine State Register (MSR), and then begins fetching and processing instructions at the address contained in the Link Register, using the new MSR value.</p>
                    <p>If the Link bit (LK) is set to 1, then the contents of the Link Register are undefined.</p>
                `,
                "tooltip": "Return from SVC",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-rfsvc-return-from-svc-instruction"
            };
        case "RLDCL":
        case "RLDCL.":
            return {
                "html": `<p>The contents of general purpose register (GPR) <em>RS</em> are rotated left the number of bits specified by the operand in the low-order six bits of <em>RB</em>. A mask is generated having 1 bits from bit <em>MB</em> through bit 63 and 0 bits elsewhere. The rotated data is ANDed with the generated mask and the result is placed into <em>RA</em>.</p>`,
                "tooltip": "Rotate Left Double Word then Clear Left",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=is-rldcl-rotate-left-double-word-then-clear-left-instruction"
            };
        case "RLDICL":
        case "RLDICL.":
            return {
                "html": `<p>The contents of rS are rotated left the number of bits specified by operand SH. A mask is generated having 1 bits from bit MB through bit 63 and 0 bits elsewhere. The rotated data is ANDed with the generated mask and the result is placed into rA.</p>`,
                "tooltip": "Rotate Left Double Word Immediate then Clear Left",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=is-rldicl-rotate-left-double-word-immediate-then-clear-left-instruction"
            };
        case "RLDCR":
        case "RLDCR.":
            return {
                "html": `<p>The contents of general purpose register (GPR) <em>RS</em> are rotated left the number of bits specified by the low-order six bits of <em>RB</em>. A mask is generated having 1 bits from bit 0 through bit <em>ME</em> and 0 bits elsewhere. The rotated data is ANDed with the generated mask and the result is placed into <em>RA</em>.</p>`,
                "tooltip": "Rotate Left Double Word then Clear Right",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=is-rldcr-rotate-left-double-word-then-clear-right-instruction"
            };
        case "RLDIC":
        case "RLDIC.":
            return {
                "html": `<p>The contents of general purpose register (GPR) <em>RS</em> are rotated left the number of bits specified by operand <em>SH</em>. A mask is generated having 1 bits from bit <em>MB</em> through bit 63 - <em>SH</em> and 0 bits elsewhere. The rotated data is ANDed with the generated mask and the result is placed into GPR <em>RA</em>.</p>`,
                "tooltip": "Rotate Left Double Word Immediate then Clear",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=is-rldic-rotate-left-double-word-immediate-then-clear-instruction"
            };
        case "RLDICL":
        case "RLDICL.":
            return {
                "html": `<p>The contents of general purpose register <em>RS</em> are rotated left the number of bits specified by operand <em>SH</em>. A mask is generated containing 1 bits from bit <em>MB</em> through bit 63 and 0 bits elsewhere. The rotated data is ANDed with the generated mask and the result is placed into GPR <em>RA</em>.</p>`,
                "tooltip": "Rotate Left Double Word Immediate then Clear Left",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=is-rldicl-rotate-left-double-word-immediate-then-clear-left-instruction-1"
            };
        case "RLDICR":
        case "RLDICR.":
            return {
                "html": `<p>The contents of general purpose register (GPR) <em>RS</em> are rotated left the number of bits specified by operand <em>SH</em>. A mask is generated having 1 bits from bit 0 through bit <em>ME</em> and 0 bits elsewhere. The rotated data is ANDed with the generated mask and the result is placed into GPR <em>RA</em>.</p>`,
                "tooltip": "Rotate Left Double Word Immediate then Clear Right",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=is-rldicr-rotate-left-double-word-immediate-then-clear-right-instruction"
            };
        case "RLDIMI":
        case "RLDIMI.":
            return {
                "html": `<p>The contents of general purpose register (GPR) <em>RS</em> are rotated left the number of bits specified by operand <em>SH</em>. A mask is generated having 1 bits from bit <em>MB</em> through bit 63 - <em>SH</em> and 0 bits elsewhere. The rotated data is inserted into <em>RA</em> under control of the generated mask.</p>`,
                "tooltip": "Rotate Left Double Word Immediate then Mask Insert",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=is-rldimi-rotate-left-double-word-immediate-then-mask-insert-instruction"
            };
        case "RLMI":
        case "RLMI.":
            return {
                "html": `<p>The <strong>rlmi</strong> instruction rotates the contents of the source general-purpose register (GPR) <em>RS</em> to the left by the number of bits specified by bits 27-31 of GPR <em>RB</em> and then stores the rotated data in GPR <em>RA</em> under control of a 32-bit generated mask defined by the values in Mask Begin (<em>MB</em>) and Mask End (<em>ME</em>).</p>`,
                "tooltip": "Rotate Left Then Mask Insert",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-rlmi-rotate-left-then-mask-insert-instruction"
            };
        case "RLIMI":
        case "RLIMI.":
        case "RLWIMI":
        case "RLWIMI.":
            return {
                "html": `<p>The <strong>rlwimi</strong> and <strong>rlimi</strong> instructions rotate left the contents of the source general-purpose register (GPR) <em>RS</em> by the number of bits by the <em>SH</em> parameter and then store the rotated data in GPR <em>RA</em> under control of a 32-bit generated mask defined by the values in Mask Begin (<em>MB</em>) and Mask End (<em>ME</em>). If a mask bit is 1, the instructions place the associated bit of rotated data in GPR <em>RA</em>; if a mask bit is 0, the GPR <em>RA</em> bit remains unchanged.</p>`,
                "tooltip": "Rotate Left Word Immediate Then Mask Insert",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=is-rlwimi-rlimi-rotate-left-word-immediate-then-mask-insert-instruction"
            };
        case "RLINM":
        case "RLINM.":
        case "RLWINM":
        case "RLWINM.":
            return {
                "html": `<p>The <strong>rlwinm</strong> and <strong>rlinm</strong> instructions rotate left the contents of the source general-purpose register (GPR) <em>RS</em> by the number of bits specified by the <em>SH</em> parameter, logically AND the rotated data with a 32-bit generated mask defined by the values in Mask Begin (<em>MB</em>) and Mask End (<em>ME</em>), and store the result in GPR <em>RA</em>.</p>`,
                "tooltip": "Rotate Left Word Immediate Then AND with Mask",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=is-rlwinm-rlinm-rotate-left-word-immediate-then-mask-instruction"
            };
        case "RLNM":
        case "RLNM.":
        case "RLWNM":
        case "RLWNM.":
            return {
                "html": `<p>The <strong>rlwnm</strong> and <strong>rlnm</strong> instructions rotate the contents of the source general-purpose register (GPR) <em>RS</em> to the left by the number of bits specified by bits 27-31 of GPR <em>RB</em>, logically AND the rotated data with a 32-bit generated mask defined by the values in Mask Begin (<em>MB</em>) and Mask End (<em>ME</em>), and store the result in GPR <em>RA</em>.</p>`,
                "tooltip": "Rotate Left Word Then AND with Mask",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=is-rlwnm-rlnm-rotate-left-word-then-mask-instruction"
            };
        case "RRIB":
        case "RRIB.":
            return {
                "html": `<p>The <strong>rrib</strong> instruction rotates bit 0 of the source general-purpose register (GPR) <em>RS</em> to the right by the number of bits specified by bits 27-31 of GPR <em>RB</em> and then stores the rotated bit in GPR <em>RA</em>.</p>`,
                "tooltip": "Rotate Right and Insert Bit",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-rrib-rotate-right-insert-bit-instruction"
            };
        case "SC":
            return {
                "html": `
                    <p>The <strong>sc</strong> instruction causes a system call interrupt. The effective address (EA) of the instruction following the <strong>sc</strong> instruction is placed into the Save Restore Register 0 (SRR0). Bits 0, 5-9, and 16-31 of the Machine State Register (MSR) are placed into the corresponding bits of Save Restore Register 1 (SRR1). Bits 1-4 and 10-15 of SRR1 are set to undefined values.</p>
                    <p>The <strong>sc</strong> instruction serves as both a basic and an extended mnemonic. In the extended form, the <em>LEV</em> field is omitted and assumed to be 0.</p>
                `,
                "tooltip": "System Call",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-sc-system-call-instruction"
            };
        case "SCV":
            return {
                "html": `<p>The <strong>scv</strong> instruction causes a system call interrupt. The effective address (EA) of the instruction following the <strong>scv</strong> instruction is placed into the Link Register. Bits 0-32, 37-41, and 48-63 of the Machine State Register (MSR) are placed into the corresponding bits of Count Register. Bits 33-36 and 42-47 of the Count Register are set to undefined values.</p>`,
                "tooltip": "System Call Vectored",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-scv-system-call-vectored-instruction"
            };
        case "SI":
            return {
                "html": `<p>The <strong>si</strong> instruction subtracts the 16-bit signed integer specified by the <em>SINT</em> parameter from the contents of general-purpose register (GPR) <em>RA</em> and stores the result in the target GPR <em>RT</em>. This instruction has the same effect as the <strong>ai</strong> instruction used with a negative <em>SINT</em> value. The assembler negates <em>SINT</em> and places this value (<em>SI</em>) in the machine instruction: <pre><code>ai RT,RA,-SINT</code></pre></p>`,
                "tooltip": "Subtract Immediate",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-si-subtract-immediate-instruction"
            };
        case "SI.":
            return {
                "html": `<p>The <strong>si.</strong> instruction subtracts the 16-bit signed integer specified by the <em>SINT</em> parameter from the contents of general-purpose register (GPR) <em>RA</em> and stores the result into the target GPR <em>RT</em>. This instruction has the same effect as the <strong>ai.</strong> instruction used with a negative <em>SINT</em>. The assembler negates <em>SINT</em> and places this value (<em>SI</em>) in the machine instruction: <pre>code>ai. RT,RA,-SINT</code></pre></p>`,
                "tooltip": "Subtract Immediate and Record",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-si-subtract-immediate-record-instruction"
            };
        case "SLD":
        case "SLD.":
            return {
                "html": `<p>The contents of general purpose register (GPR) <em>RS</em> are shifted left the number of bits specified by the low-order seven bits of GPR <em>RB</em>. Bits shifted out of position 0 are lost. Zeros are supplied to the vacated positions on the right. The result is placed into GPR <em>RA</em>. Shift amounts from 64 to 127 give a zero result.</p>`,
                "tooltip": "Shift Left Double Word",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-sld-shift-left-double-word-instruction"
            };
        case "SLE":
        case "SLE.":
            return {
                "html": `<p>The <strong>sle</strong> instruction rotates the contents of the source general-purpose register (GPR) <em>RS</em> to the left by <em>N</em> bits, where <em>N</em> is the shift amount specified in bits 27-31 of GPR <em>RB</em>. The instruction also stores the rotated word in the MQ Register and the logical AND of the rotated word and the generated mask in GPR <em>RA</em>. The mask consists of 32 minus <em>N</em> ones followed by <em>N</em> zeros.</p>`,
                "tooltip": "Shift Left Extended",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-sle-shift-left-extended-instruction"
            };
        case "SLEQ":
        case "SLEQ.":
            return {
                "html": `<p>The <strong>sleq</strong> instruction rotates the contents of the source general-purpose register (GPR) <em>RS</em> left <em>N</em> bits, where <em>N</em> is the shift amount specified in bits 27-31 of GPR <em>RB</em>. The instruction merges the rotated word with the contents of the MQ Register under control of a mask, and stores the rotated word in the MQ Register and merged word in GPR <em>RA</em>. The mask consists of 32 minus <em>N</em> ones followed by <em>N</em> zeros.</p>`,
                "tooltip": "Shift Left Extended with MQ",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-sleq-shift-left-extended-mq-instruction"
            };
        case "SLIQ":
        case "SLIQ.":
            return {
                "html": `<p>The <strong>sliq</strong> instruction rotates the contents of the source general-purpose register (GPR) <em>RS</em> to the left by <em>N</em> bits, where <em>N</em> is the shift amount specified by <em>SH</em>. The instruction stores the rotated word in the MQ Register and the logical AND of the rotated word and places the generated mask in GPR <em>RA</em>. The mask consists of 32 minus <em>N</em> ones followed by <em>N</em> zeros.</p>`,
                "tooltip": "Shift Left Immediate with MQ",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-sliq-shift-left-immediate-mq-instruction"
            };
        case "SLLIQ":
        case "SLLIQ.":
            return {
                "html": `<p>The <strong>slliq</strong> instruction rotates the contents of the source general-purpose register (GPR) <em>RS</em> to the left by <em>N</em> bits, where <em>N</em> is the shift amount specified in <em>SH</em>, merges the result with the contents of the MQ Register, and stores the rotated word in the MQ Register and the final result in GPR <em>RA</em>. The mask consists of 32 minus <em>N</em> ones followed by <em>N</em> zeros.</p>`,
                "tooltip": "Shift Left Long Immediate with MQ",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-slliq-shift-left-long-immediate-mq-instruction"
            };
        case "SLLQ":
        case "SLLQ.":
            return {
                "html": `
                    <p>The <strong>sllq</strong> instruction rotates the contents of the source general-purpose register (GPR) <em>RS</em> to the left by <em>N</em> bits, where <em>N</em> is the shift amount specified in bits 27-31 of GPR <em>RB</em>. The merge depends on the value of bit 26 in GPR <em>RB</em>.</p>
                    <p>Consider the following when using the <strong>sllq</strong> instruction:</p>
                    <ul>
                        <li>If bit 26 of GPR <em>RB</em> is 0, then a mask of <em>N</em> zeros followed by 32 minus <em>N</em> ones is generated. The rotated word is then merged with the contents of the MQ Register under the control of this generated mask.</li>
                        <li>If bit 26 of GPR <em>RB</em> is 1, then a mask of <em>N</em> ones followed by 32 minus <em>N</em> zeros is generated. A word of zeros is then merged with the contents of the MQ Register under the control of this generated mask.</li>
                    </ul>
                    <p>The resulting merged word is stored in GPR <em>RA</em>. The MQ Register is not altered.</p>
                `,
                "tooltip": "Shift Left Long with MQ",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-sllq-shift-left-long-mq-instruction"
            };
        case "SLQ":
        case "SLQ.":
            return {
                "html": `
                    <p>The <strong>slq</strong> instruction rotates the contents of the source general-purpose register (GPR) <em>RS</em> to the left by <em>N</em> bits, where <em>N</em> is the shift amount specified in bits 27-31 of GPR <em>RB</em>, and stores the rotated word in the MQ Register. The mask depends on bit 26 of GPR <em>RB</em>.</p>
                    <p>Consider the following when using the <strong>slq</strong> instruction:</p>
                    <ul>
                        <li>If bit 26 of GPR <em>RB</em> is 0, then a mask of 32 minus <em>N</em> ones followed by <em>N</em> zeros is generated.</li>
                        <li>If bit 26 of GPR <em>RB</em> is 1, then a mask of all zeros is generated.</em>
                    </ul>
                    <p>This instruction then stores the logical AND of the rotated word and the generated mask in GPR <em>RA</em>.</p>
                `,
                "tooltip": "Shift Left with MQ",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-slq-shift-left-mq-instruction"
            };
        case "SL":
        case "SL.":
        case "SLW":
        case "SLW.":
            return {
                "html": `
                    <p>The <strong>slw</strong> and <strong>sl</strong> instructions rotate the contents of the source general-purpose register (GPR) <em>RS</em> to the left <em>N</em> bits, where <em>N</em> is the shift amount specified in bits 27-31 of GPR <em>RB</em>, and store the logical AND of the rotated word and the generated mask in GPR <em>RA</em>.</p>
                    <p>Consider the following when using the <strong>slw</strong> and <strong>sl</strong> instructions:</p>
                    <ul>
                        <li>If bit 26 of GPR <em>RB</em> is 0, then a mask of 32-<em>N</em> ones followed by <em>N</em> zeros is generated.</li>
                        <li>If bit 26 of GPR <em>RB</em> is 1, then a mask of all zeros is generated.</li>
                    </ul>
                `,
                "tooltip": "Shift Left Word",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-slw-sl-shift-left-word-instruction"
            };
        case "SRAD":
        case "SRAD.":
            return {
                "html": `
                    <p>Algebraically shift the contents of a general purpose register right by the number of bits specified by the contents of another general purpose register. Place the result of the operation in another general purpose register.</p>
                    <p>The contents of general purpose register (GPR) <em>RS</em> are shifted right the number of bits specified by the low-order seven bits of GPR <em>RB</em>. Bits shifted out of position 63 are lost. Bit 0 of GPR <em>RS</em> is replicated to fill the vacated positions on the left. The result is placed into GRP <em>RA</em>. XER[CA] is set if GPR <em>RS</em> is negative and any 1 bits are shifted out of position 63; otherwise XER[CA] is cleared. A shift amount of zero causes GRP <em>RA</em> to be set equal to GPR <em>RS</em>, and XER[CA] to be cleared. Shift amounts from 64 to 127 give a result of 64 sign bits in GRP <em>RA</em>, and cause XER[CA] to receive the sign bit of GPR <em>RS</em>.</p>
                    <p>Note that the <strong>srad</strong> instruction, followed by addze, can by used to divide quickly by 2**n. The setting of the CA bit, by <strong>srad</strong>, is independent of mode.</p>
                `,
                "tooltip": "Shift Right Algebraic Double Word",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-srad-shift-right-algebraic-double-word-instruction"
            };
        case "SRADI":
        case "SRADI.":
            return {
                "html": `
                    <p>Algebraically shift the contents of a general purpose register right by the number of bits specified by the immediate value. Place the result of the operation in another general purpose register.</p>
                    <p>The contents of general purpose register (GPR) <em>RS</em> are shifted right <em>SH</em> bits. Bits shifted out of position 63 are lost. Bit 0 of GPR <em>RS</em> is replicated to fill the vacated positions on the left. The result is placed into GPR <em>RA</em>. XER[CA] is set if GPR <em>RS</em> is negative and any 1 bits are shifted out of position 63; otherwise XER[CA] is cleared. A shift amount of zero causes GPR <em>RA</em> to be set equal to GPR <em>RS</em>, and XER[CA] to be cleared.</p>
                    <p>Note that the <strong>sradi</strong> instruction, followed by addze, can by used to divide quickly by 2**n. The setting of the CA bit, by <strong>sradi</strong>, is independent of mode.</p>
                `,
                "tooltip": "Shift Right Algebraic Double Word Immediate",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=is-sradi-shift-right-algebraic-double-word-immediate-instruction"
            };
        case "SRAIQ":
        case "SRAIQ.":
            return {
                "html": `
                    <p>The <strong>sraiq</strong> instruction rotates the contents of the source general-purpose register (GPR) <em>RS</em> to the left by 32 minus <em>N</em> bits, where <em>N</em> is the shift amount specified by <em>SH</em>, merges the rotated data with a word of 32 sign bits from GPR <em>RS</em> under control of a generated mask, and stores the rotated word in the MQ Register and the merged result in GPR <em>RA</em>. A word of 32 sign bits is generated by taking the sign bit of a GPR and repeating it 32 times to make a fullword. This word can be either 0x0000 0000 or 0xFFFF FFFF depending on the value of the GPR. The mask consists of <em>N</em> zeros followed by 32 minus <em>N</em> ones.</p>
                    <p>This instruction then ANDs the rotated data with the complement of the generated mask, ORs the 32-bit result together, and ANDs the bit result with bit 0 of GPR <em>RS</em> to produce the Carry bit (CA).</p>
                `,
                "tooltip": "Shift Right Algebraic Immediate with MQ",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-sraiq-shift-right-algebraic-immediate-mq-instruction"
            };
        case "SRAQ":
        case "SRAQ.":
            return {
                "html": `
                    <p>The <strong>sraq</strong> instruction rotates the contents of the source general-purpose register (GPR) <em>RS</em> to the left by 32 minus <em>N</em> bits, where <em>N</em> is the shift amount specified in bits 27-31 of GPR <em>RB</em>. The instruction then merges the rotated data with a word of 32 sign bits from GPR <em>RS</em> under control of a generated mask and stores the merged word in GPR <em>RA</em>. The rotated word is stored in the MQ Register. The mask depends on the value of bit 26 in GPR <em>RB</em>.</p>
                    <p>Consider the following when using the <strong>sraq</strong> instruction:</p>
                    <ul>
                        <li>If bit 26 of GPR <em>RB</em> is 0, then a mask of <em>N</em> zeros followed by 32 minus <em>N</em> ones is generated.</li>
                        <li>If bit 26 of GPR <em>RB</em> is 1, then a mask of all zeros is generated.</li>
                    </ul>
                    <p>A word of 32 sign bits is generated by taking the sign bit of a GPR and repeating it 32 times to make a full word. This word can be either 0x0000 0000 or 0xFFFF FFFF depending on the value of the GPR.</p>
                    <p>This instruction then ANDs the rotated data with the complement of the generated mask, ORs the 32-bit result together, and ANDs the bit result with bit 0 of GPR <em>RS</em> to produce the Carry bit (CA).</p>
                `,
                "tooltip": "Shift Right Algebraic with MQ",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-sraq-shift-right-algebraic-mq-instruction"
            };
        case "SRA":
        case "SRA.":
        case "SRAW":
        case "SRAW.":
            return {
                "html": `
                    <p>The <strong>sraw</strong> and <strong>sra</strong> instructions rotate the contents of the source general-purpose register (GPR) <em>RS</em> to the left by 32 minus <em>N</em> bits, where <em>N</em> is the shift amount specified in bits 27-31 of GPR <em>RB</em>, and merge the rotated word with a word of 32 sign bits from GPR <em>RS</em> under control of a generated mask. A word of 32 sign bits is generated by taking the sign bit of a GPR and repeating it 32 times to make a full word. This word can be either 0x0000 0000 or 0xFFFF FFFF depending on the value of the GPR.</p>
                    <p>The mask depends on the value of bit 26 in GPR <em>RB</em>.</p>
                    <p>Consider the following when using the <strong>sraw</strong> and <strong>sra</strong> instructions:</p>
                    <ul>
                        <li>If bit 26 of GPR <em>RB</em> is zero, then a mask of <em>N</em> zeros followed by 32 minus <em>N</em> ones is generated.</li>
                        <li>If bit 26 of GPR <em>RB</em> is one, then a mask of all zeros is generated.</li>
                    </ul>
                    <p>The merged word is placed in GPR <em>RA</em>. The <strong>sraw</strong> and <strong>sra</strong> instructions then AND the rotated data with the complement of the generated mask, OR the 32-bit result together, and AND the bit result with bit 0 of GPR <em>RS</em> to produce the Carry bit (CA).</p>
                `,
                "tooltip": "Shift Right Algebraic Word",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-sraw-sra-shift-right-algebraic-word-instruction"
            };
        case "SRAI":
        case "SRAI.":
        case "SRAWI":
        case "SRAWI.":
            return {
                "html": `
                    <p>The <strong>srawi</strong> and <strong>srai</strong> instructions rotate the contents of the source general-purpose register (GPR) <em>RS</em> to the left by 32 minus <em>N</em> bits, where <em>N</em> is the shift amount specified by <em>SH</em>, merge the rotated data with a word of 32 sign bits from GPR <em>RS</em> under control of a generated mask, and store the merged result in GPR <em>RA</em>. A word of 32 sign bits is generated by taking the sign bit of a GPR and repeating it 32 times to make a full word. This word can be either 0x0000 0000 or 0xFFFF FFFF depending on the value of the GPR. The mask consists of <em>N</em> zeros followed by 32 minus <em>N</em> ones.</p>
                    <p>The <strong>srawi</strong> and <strong>srai</strong> instructions then AND the rotated data with the complement of the generated mask, OR the 32-bit result together, and AND the bit result with bit 0 of GPR <em>RS</em> to produce the Carry bit (CA).</p>
                `,
                "tooltip": "Shift Right Algebraic Word Immediate",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=is-srawi-srai-shift-right-algebraic-word-immediate-instruction"
            };
        case "SRD":
        case "SRD.":
            return {
                "html": `
                    <p>The <strong>srd</strong> instruction shifts the contents of a general purpose register right by the number of bits specified by the contents of another general purpose register.</p>
                    <p>The contents of general purpose register (GPR) <em>RS</em> are shifted right the number of bits specified by the low-order seven bits of GPR <em>RB</em>. Bits shifted out of position 63 are lost. Zeros are supplied to the vacated positions on the left. The result is placed into GRP <em>RA</em>. Shift amounts from 64 to 127 give a zero result.</p>
                `,
                "tooltip": "Shift Right Double Word",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-srd-shift-right-double-word-instruction"
            };
        case "SRE":
        case "SRE.":
            return {
                "html": `<p>The <strong>sre</strong> instruction rotates the contents of the source general-purpose register (GPR) <em>RS</em> to the left by 32 minus <em>N</em> bits, where <em>N</em> is the shift amount specified in bits 27-31 of GPR <em>RB</em>, and stores the rotated word in the MQ Register and the logical AND of the rotated word and a generated mask in GPR <em>RA</em>. The mask consists of <em>N</em> zeros followed by 32 minus <em>N</em> ones.</p>`,
                "tooltip": "Shift Right Extended",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-sre-shift-right-extended-instruction"
            };
        case "SREA":
        case "SREA.":
            return {
                "html": `
                    <p>The <strong>srea</strong> instruction rotates the contents of the source general-purpose register (GPR) <em>RS</em> to the left by 32 minus <em>N</em> bits, where <em>N</em> is the shift amount specified in bits 27-31 of GPR <em>RB</em>, stores the rotated word in the MQ Register, and merges the rotated word and a word of 32 sign bits from GPR <em>RS</em> under control of a generated mask. A word of 32 sign bits is generated by taking the sign bit of a general-purpose register and repeating it 32 times to make a full word. This word can be either 0x0000 0000 or 0xFFFF FFFF depending on the value of the general-purpose register. The mask consists of <em>N</em> zeros followed by 32 minus <em>N</em> ones. The merged word is stored in GPR <em>RA</em>.</p>
                    <p>This instruction then ANDs the rotated data with the complement of the generated mask, ORs together the 32-bit result, and ANDs the bit result with bit 0 of GPR <em>RS</em> to produce the Carry bit (CA).</p>
                `,
                "tooltip": "Shift Right Extended Algebraic",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-srea-shift-right-extended-algebraic-instruction"
            };
        case "SREQ":
        case "SREQ.":
            return {
                "html": `<p>The <strong>sreq</strong> instruction rotates the contents of the source general-purpose register (GPR) <em>RS</em> to the left by 32 minus <em>N</em> bits, where <em>N</em> is the shift amount specified in bits 27-31 of GPR <em>RB</em>, merges the rotated word with the contents of the MQ Register under a generated mask, and stores the rotated word in the MQ Register and the merged word in GPR <em>RA</em>. The mask consists of <em>N</em> zeros followed by 32 minus <em>N</em> ones.</p>`,
                "tooltip": "Shift Right Extended with MQ",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-sreq-shift-right-extended-mq-instruction"
            };
        case "SRIQ":
        case "SRIQ.":
            return {
                "html": `<p>The <strong>sriq</strong> instruction rotates the contents of the source general-purpose register (GPR) <em>RS</em> to the left 32 minus <em>N</em> bits, where <em>N</em> is the shift amount specified by <em>SH</em>, and stores the rotated word in the MQ Register, and the logical AND of the rotated word and the generated mask in GPR <em>RA</em>. The mask consists of <em>N</em> zeros followed by 32 minus <em>N</em> ones.</p>`,
                "tooltip": "Shift Right Immediate with MQ",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-sriq-shift-right-immediate-mq-instruction"
            };
        case "SRLIQ":
        case "SRLIQ.":
            return {
                "html": `<p>The <strong>srliq</strong> instruction rotates the contents of the source general-purpose register (GPR) <em>RS</em> to the left by 32 minus <em>N</em> bits, where <em>N</em> is the shift amount specified by <em>SH</em>, merges the result with the contents of the MQ Register under control of a generated mask, and stores the rotated word in the MQ Register and the merged result in GPR <em>RA</em>. The mask consists of <em>N</em> zeros followed by 32 minus <em>N</em> ones.</p>`,
                "tooltip": "Shift Right Long Immediate with MQ",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-srliq-shift-right-long-immediate-mq-instruction"
            };
        case "SRLQ":
        case "SRLQ.":
            return {
                "html": `
                    <p>The <strong>srlq</strong> instruction rotates the contents of the source general-purpose register (GPR) <em>RS</em> to the left 32 minus <em>N</em> bits, where <em>N</em> is the shift amount specified in bits 27-31 of GPR <em>RB</em>. The merge depends on the value of bit 26 in GPR <em>RB</em>.</p>
                    <p>Consider the following when using the <strong>srlq</strong> instruction:</p>
                    <ul>
                        <li>If bit 26 of GPR <em>RB</em> is 0, then a mask of <em>N</em> zeros followed by 32 minus <em>N</em> ones is generated. The rotated word is then merged with the contents of the MQ Register under control of this generated mask.</li>
                        <li>If bit 26 of GPR <em>RB</em> is 1, then a mask of <em>N</em> ones followed by 32 minus <em>N</em> zeros is generated. A word of zeros is then merged with the contents of the MQ Register under control of this generated mask.</li>
                    </ul>
                    <p>The merged word is stored in GPR <em>RA</em>. The MQ Register is not altered.</p>
                `,
                "tooltip": "Shift Right Long with MQ",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-srlq-shift-right-long-mq-instruction"
            };
        case "SRQ":
        case "SRQ.":
            return {
                "html": `
                    <p>The <strong>srq</strong> instruction rotates the contents of the source general-purpose register (GPR) <em>RS</em> to the left by 32 minus <em>N</em> bits, where <em>N</em> is the shift amount specified in bits 27-31 of GPR <em>RB</em>, and stores the rotated word in the MQ Register. The mask depends on bit 26 of GPR <em>RB</em>.</p>
                    <p id="idalangref_srq_instrs__a247dd0c694melh">Consider the following when using the <strong>srq</strong> instruction:</p>
                    <ul>
                        <li>If bit 26 of GPR <em>RB</em> is 0, then a mask of <em>N</em> zeros followed by 32 minus <em>N</em> ones is generated.</li>
                        <li>If bit 26 of GPR <em>RB</em> is 1, then a mask of all zeros is generated.</li>
                    </ul>
                    <p id="idalangref_srq_instrs__a247dd0c824melh">This instruction then stores the logical AND of the rotated word and the generated mask in GPR <em>RA</em>.</p>
                `,
                "tooltip": "Shift Right with MQ",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-srq-shift-right-mq-instruction"
            };
        case "SR":
        case "SR.":
        case "SRW":
        case "SRW.":
            return {
                "html": `
                    <p>The <strong>srw</strong> and <strong>sr</strong> instructions rotate the contents of the source general-purpose register (GPR) <em>RS</em> to the left by 32 minus <em>N</em> bits, where <em>N</em> is the shift amount specified in bits 27-31 of GPR <em>RB</em>, and store the logical AND of the rotated word and the generated mask in GPR <em>RA</em>.</p>
                    <p>Consider the following when using the <strong>srw</strong> and <strong>sr</strong> instructions:</p>
                    <ul>
                        <li>If bit 26 of GPR <em>RB</em> is 0, then a mask of <em>N</em> zeros followed by 32 - <em>N</em> ones is generated.</li>
                        <li>If bit 26 of GPR <em>RB</em> is 1, then a mask of all zeros is generated.</li>
                    </ul>
                `,
                "tooltip": "Shift Right Word",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-srw-sr-shift-right-word-instruction"
            };
        case "STB":
            return {
                "html": `
                    <p>The <strong>stb</strong> instruction stores bits 24-31 of general-purpose register (GPR) <em>RS</em> into a byte of storage addressed by the effective address (EA).</p>
                    <p>If GPR <em>RA</em> is not 0, the EA is the sum of the contents of GPR <em>RA</em> and <em>D</em>, a 16-bit signed two's complement integer sign-extended to 32 bits. If GPR <em>RA</em> is 0, then the EA is <em>D</em>.</p>
                `,
                "tooltip": "Store Byte",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-stb-store-byte-instruction"
            };
        case "STBU":
            return {
                "html": `
                    <p>The <strong>stbu</strong> instruction stores bits 24-31 of the source general-purpose register (GPR) <em>RS</em> into the byte in storage addressed by the effective address (EA).</p>
                    <p>If GPR <em>RA</em> is not 0, the EA is the sum of the contents of GPR <em>RA</em> and <em>D</em>, a 16-bit signed two's complement integer sign-extended to 32 bits. If GPR <em>RA</em> is 0, then the EA is <em>D</em>.</p>
                    <p>If <em>RA</em> does not equal 0 and the storage access does not cause an Alignment Interrupt, then the EA is stored in GPR <em>RA</em>.</p>
                `,
                "tooltip": "Store Byte with Update",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-stbu-store-byte-update-instruction"
            };
        case "STBUX":
            return {
                "html": `
                    <p>The <strong>stbux</strong> instruction stores bits 24-31 of the source general-purpose register (GPR) <em>RS</em> into the byte in storage addressed by the effective address (EA).</p>
                    <p>If GPR <em>RA</em> is not 0, the EA is the sum of the contents of GPR <em>RA</em> and the contents of GPR <em>RB</em>. If <em>RA</em> is 0, then the EA is the contents of GPR <em>RB</em>.</p>
                    <p>If GPR <em>RA</em> does not equal 0 and the storage access does not cause an Alignment Interrupt, then the EA is stored in GPR <em>RA</em>.</p>
                `,
                "tooltip": "Store Byte with Update Indexed",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-stbux-store-byte-update-indexed-instruction"
            };
        case "STBX":
            return {
                "html": `
                    <p>The <strong>stbx</strong> instruction stores bits 24-31 from general-purpose register (GPR) <em>RS</em> into a byte of storage addressed by the effective address (EA). The contents of GPR <em>RS</em> are unchanged.</p>
                    <p>If GPR <em>RA</em> is not 0, the EA is the sum of the contents of GPR <em>RA</em> and the contents of GPR <em>RB</em>. If GPR <em>RA</em> is 0, then the EA is the contents of GPR <em>RB</em>.</p>
                `,
                "tooltip": "Store Byte Indexed",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-stbx-store-byte-indexed-instruction"
            };
        case "STD":
            return {
                "html": `
                    <p>The <strong>std</strong> instruction stores a doubleword in storage from the source general-purpose register (GPR) <em>RS</em> into the specified location in memory referenced by the effective address (EA).</p>
                    <p>DS is a 14-bit, signed two's complement number, which is sign-extended to 64 bits, and then multiplied by 4 to provide a displacement <em>Disp</em>. If GPR <em>RA</em> is not 0, the EA is the sum of the contents of GPR <em>RA</em> and <em>Disp</em>. If GPR <em>RA</em> is 0, then the EA is <em>Disp</em>.</p>
                `,
                "tooltip": "Store Double Word",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-std-store-double-word-instruction"
            };
        case "STDCX.":
            return {
                "html": `<p>The <strong>stdcx.</strong> instruction conditionally stores the contents of a general purpose register into a storage location, based upon an existing reservation.</p>`,
                "tooltip": "Store Double Word Condition Indexed",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-stdcx-store-double-word-conditional-indexed-instruction"
            };
        case "STDU":
            return {
                "html": `
                    <p>The <strong>stdu</strong> instruction stores a doubleword in storage from the source general-purpose register (GPR) <em>RS</em> into the specified location in memory referenced by the effective address (EA).</p>
                    <p>DS is a 14-bit, signed two's complement number, which is sign-extended to 64 bits, and then multiplied by 4 to provide a displacement <em>Disp</em>. If GPR <em>RA</em> is not 0, the EA is the sum of the contents of GPR <em>RA</em> and <em>Disp</em>. If GPR <em>RA</em> is 0, then the EA is <em>Disp</em>.</p>
                    <p>If GPR <em>RA</em> = 0, the instruction form is invalid.</p>
                `,
                "tooltip": "Store Double Word with Update",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-stdu-store-double-word-update-instruction"
            };
        case "STDUX":
            return {
                "html": `
                    <p>The <strong>stdux</strong> instruction stores a doubleword in storage from the source general-purpose register (GPR) <em>RS</em> into the location in storage specified by the effective address (EA).</p>
                    <p>The EA is the sum of the contents of GPR <em>RA</em> and <em>RB</em>. GRP <em>RA</em> is updated with the EA.</p>
                    <p>If rA = 0, the instruction form is invalid.</p>
                `,
                "tooltip": "Store Double Word with Update Indexed",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-stdux-store-double-word-update-indexed-instruction"
            };
        case "STDX":
            return {
                "html": `
                    <p>The <strong>stdx</strong> instruction stores a doubleword in storage from the source general-purpose register (GPR) <em>RS</em> into the location in storage specified by the effective address (EA).</p>
                    <p>If GPR <em>RA</em> is not 0, the EA is the sum of the contents of GPR <em>RA</em> and <em>RB</em>. If GPR <em>RA</em> is 0, then the EA is <em>RB</em>.</p>
                `,
                "tooltip": "Store Double Word Indexed",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-stdx-store-double-word-indexed-instruction"
            };
        case "STFD":
            return {
                "html": `
                    <p>The <strong>stfd</strong> instruction stores the contents of floating-point register (FPR) <em>FRS</em> into the doubleword storage addressed by the effective address (EA).</p>
                    <p>If general-purpose register (GPR) <em>RA</em> is not 0, the EA is the sum of the contents of GPR <em>RA</em> and <em>D</em>. The sum is a 16-bit signed two's complement integer sign-extended to 32 bits. If GPR <em>RA</em> is 0, then the EA is <em>D</em>.</p>
                `,
                "tooltip": "Store Floating-Point Double",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-stfd-store-floating-point-double-instruction"
            };
        case "STFDU":
            return {
                "html": `
                    <p>The <strong>stfdu</strong> instruction stores the contents of floating-point register (FPR) <em>FRS</em> into the doubleword storage addressed by the effective address (EA).</p>
                    <p>If general-purpose register (GPR) <em>RA</em> is not 0, the EA is the sum of the contents of GPR <em>RA</em> and <em>D</em>. The sum is a 16-bit signed two's complement integer sign-extended to 32 bits. If GPR <em>RA</em> is 0, then the EA is <em>D</em>.</p>
                    <p>If GPR <em>RA</em> does not equal 0 and the storage access does not cause Alignment Interrupt or a Data Storage Interrupt, then the EA is stored in GPR <em>RA</em>.</p>
                `,
                "tooltip": "Store Floating-Point Double with Update",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-stfdu-store-floating-point-double-update-instruction"
            };
        case "STFDUX":
            return {
                "html": `
                    <p>The <strong>stfdux</strong> instruction stores the contents of floating-point register (FPR) <em>FRS</em> into the doubleword storage addressed by the effective address (EA).</p>
                    <p>If general-purpose register (GPR) <em>RA</em> is not 0, the EA is the sum of the contents of GPRs <em>RA</em> and <em>RB</em>. If GPR <em>RA</em> is 0, then the EA is the contents of GPR <em>RB</em>.</p>
                    <p>If GPR <em>RA</em> does not equal 0 and the storage access does not cause Alignment Interrupt or a Data Storage Interrupt, then the EA is stored in GPR <em>RA</em>.</p>
                `,
                "tooltip": "Store Floating-Point Double with Update Indexed",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=is-stfdux-store-floating-point-double-update-indexed-instruction"
            };
        case "STFDX":
            return {
                "html": `
                    <p>The <strong>stfdx</strong> instruction stores the contents of floating-point register (FPR) <em>FRS</em> into the doubleword storage addressed by the effective address (EA).</p>
                    <p>If general-purpose register (GPR) <em>RA</em> is not 0, the EA is the sum of the contents of GPRs <em>RA</em> and <em>RB</em>. If GPR <em>RA</em> is 0, then the EA is the contents of GPR <em>RB</em>.</p>
                `,
                "tooltip": "Store Floating-Point Double Indexed",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-stfdx-store-floating-point-double-indexed-instruction"
            };
        case "STFIWX":
            return {
                "html": `
                    <p>The <strong>stfifx</strong> instruction stores the contents of the low-order 32 bits of floating-point register (FPR) <em>FRS</em>,without conversion, into the word storage addressed by the effective address (EA).</p>
                    <p>If general-purpose register (GPR) <em>RA</em> is not 0, the EA is the sum of the contents of GPRs <em>RA</em> and <em>RB</em>. If GPR <em>RA</em> is 0, then the EA is the contents of GPR <em>RB</em>.</p>
                `,
                "tooltip": "Store Floating-Point as Integer Word Indexed",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=is-stfiwx-store-floating-point-as-integer-word-indexed"
            };
        case "STFQ":
            return {
                "html": `
                    <p>The <strong>stfq</strong> instruction stores in memory the contents of two consecutive floating-point registers (FPR) at the location specified by the effective address (EA).</p>
                    <p><em>DS</em> is sign-extended to 30 bits and concatenated on the right with b'00' to form the offset value. If general-purpose register (GPR) <em>RA</em> is 0, the offset value is the EA. If GPR <em>RA</em> is not 0, the offset value is added to GPR <em>RA</em> to generate the EA. The contents of FPR <em>FRS</em> is stored into the doubleword of storage at the EA. If FPR <em>FRS</em> is 31, then the contents of FPR 0 is stored into the doubleword at EA+8; otherwise, the contents of <em>FRS</em>+1 are stored into the doubleword at EA+8.</p>
                `,
                "tooltip": "Store Floating-Point Quad",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-stfq-store-floating-point-quad-instruction"
            };
        case "STFQU":
            return {
                "html": `
                    <p>The <strong>stfqu</strong> instruction stores in memory the contents of two consecutive floating-point registers (FPR) at the location specified by the effective address (EA).</p>
                    <p><em>DS</em> is sign-extended to 30 bits and concatenated on the right with b'00' to form the offset value. If general-purpose register (GPR) <em>RA</em> is 0, the offset value is the EA. If GPR <em>RA</em> is not 0, the offset value is added to GPR <em>RA</em> to generate the EA. The contents of FPR <em>FRS</em> is stored into the doubleword of storage at the EA. If FPR <em>FRS</em> is 31, then the contents of FPR 0 is stored into the doubleword at EA+8; otherwise, the contents of <em>FRS</em>+1 is stored into the doubleword at EA+8.</p>
                    <p>If GPR <em>RA</em> is not 0, the EA is placed into GPR <em>RA</em>.</p>
                `,
                "tooltip": "Store Floating-Point Quad with Update",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-stfqu-store-floating-point-quad-update-instruction"
            };
        case "STFQUX":
            return {
                "html": `
                    <p>The <strong>stfqux</strong> instruction stores in memory the contents of two consecutive floating-point registers (FPR) at the location specified by the effective address (EA).</p>
                    <p>If general-purpose register (GPR) <em>RA</em> is not 0, the EA is the sum of the contents of GPR <em>RA</em> and GPR <em>RB</em>. If GPR <em>RA</em> is 0, the EA is the contents of GPR <em>RB</em>. The contents of FPR <em>FRS</em> is stored into the doubleword of storage at the EA. If FPR <em>FRS</em> is 31, then the contents of FPR 0 is stored into the doubleword at EA+8; otherwise, the contents of <em>FRS</em>+1 is stored into the doubleword at EA+8.</p>
                    <p>If GPR <em>RA</em> is not 0, the EA is placed into GPR <em>RA</em>.</p>
                `,
                "tooltip": "Store Floating-Point Quad with Update Indexed",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=is-stfqux-store-floating-point-quad-update-indexed-instruction"
            };
        case "STFQX":
            return {
                "html": `
                    <p>The <strong>stfqx</strong> instruction stores in memory the contents of floating-point register (FPR) <em>FRS</em> at the location specified by the effective address (EA).</p>
                    <p>If general-purpose register (GPR) <em>RA</em> is not 0, the EA is the sum of the contents of GPR <em>RA</em> and GPR <em>RB</em>. If GPR <em>RA</em> is 0, the EA is the contents of GPR <em>RB</em>. The contents of FPR <em>FRS</em> is stored into the doubleword of storage at the EA. If FPR <em>FRS</em> is 31, then the contents of FPR 0 is stored into the doubleword at EA+8; otherwise, the contents of <em>FRS</em>+1 is stored into the doubleword at EA+8.</p>
                `,
                "tooltip": "Store Floating-Point Quad Indexed",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-stfqx-store-floating-point-quad-indexed-instruction"
            };
        case "STFS":
            return {
                "html": `
                    <p>The <strong>stfs</strong> instruction converts the contents of floating-point register (FPR) <em>FRS</em> to single-precision and stores the result into the word of storage addressed by the effective address (EA).</p>
                    <p>If general-purpose register (GPR) <em>RA</em> is not 0, the EA is the sum of the contents of GPR <em>RA</em> and <em>D</em>, a 16-bit signed two's complement integer sign-extended to 32 bits. If GPR <em>RA</em> is 0, then the EA is <em>D</em>.</p>
                `,
                "tooltip": "Store Floating-Point Single",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-stfs-store-floating-point-single-instruction"
            };
        case "STFSU":
            return {
                "html": `
                    <p>The <strong>stfsu</strong> instruction converts the contents of floating-point register (FPR) <em>FRS</em> to single-precision and stores the result into the word of storage addressed by the effective address (EA).</p>
                    <p>If general-purpose register (GPR) <em>RA</em> is not 0, the EA is the sum of the contents of GPR <em>RA</em> and <em>D</em>, a 16-bit signed two's complement integer sign-extended to 32 bits. If GPR <em>RA</em> is 0, then the EA is <em>D</em>.</p>
                    <p>If GPR <em>RA</em> does not equal 0 and the storage access does not cause Alignment Interrupt or Data Storage Interrupt, then the EA is stored in GPR <em>RA</em>.</p>
                `,
                "tooltip": "Store Floating-Point Single with Update",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-stfsu-store-floating-point-single-update-instruction"
            };
        case "STFSUX":
            return {
                "html": `
                    <p>The <strong>stfsux</strong> instruction converts the contents of floating-point register (FPR) <em>FRS</em> to single-precision and stores the result into the word of storage addressed by the effective address (EA).</p>
                    <p>If general-purpose register (GPR) <em>RA</em> is not 0, the EA is the sum of the contents of GPR <em>RA</em> and GPR <em>RB</em>. If GPR <em>RA</em> is 0, then the EA is the contents of GPR <em>RB</em>.</p>
                    <p>If GPR <em>RA</em> does not equal 0 and the storage access does not cause Alignment Interrupt or Data Storage Interrupt, then the EA is stored in GPR <em>RA</em>.</p>
                `,
                "tooltip": "Store Floating-Point Single with Update Indexed",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=is-stfsux-store-floating-point-single-update-indexed-instruction"
            };
        case "STFSX":
            return {
                "html": `
                    <p>The <strong>stfsx</strong> instruction converts the contents of floating-point register (FPR) <em>FRS</em> to single-precision and stores the result into the word of storage addressed by the effective address (EA).</p>
                    <p>If general-purpose register (GPR) <em>RA</em> is not 0, the EA is the sum of the contents of GPR <em>RA</em> and GPR <em>RB</em>. If GPR <em>RA</em> is 0, then the EA is the contents of GPR <em>RB</em>.</p>
                `,
                "tooltip": "Store Floating-Point Single Indexed",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-stfsx-store-floating-point-single-indexed-instruction"
            };
        case "STH":
            return {
                "html": `
                    <p>The <strong>sth</strong> instruction stores bits 16-31 of general-purpose register (GPR) <em>RS</em> into the halfword of storage addressed by the effective address (EA).</p>
                    <p>If GPR <em>RA</em> is not 0, the EA is the sum of the contents of GPR <em>RA</em> and <em>D</em>, a 16-bit signed two's complement integer sign-extended to 32 bits. If GPR <em>RA</em> is 0, then the EA is <em>D</em>.</p>
                `,
                "tooltip": "Store Half",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-sth-store-half-instruction"
            };
        case "STHBRX":
            return {
                "html": `
                    <p>The <strong>sthbrx</strong> instruction stores bits 16-31 of general-purpose register (GPR) <em>RS</em> into the halfword of storage addressed by the effective address (EA).</p>
                    <p>Consider the following when using the <strong>sthbrx</strong> instruction:</p>
                    <ul>
                        <li>Bits 24-31 of GPR <em>RS</em> are stored into bits 00-07 of the halfword in storage addressed by EA.</li>
                        <li>Bits 16-23 of GPR <em>RS</em> are stored into bits 08-15 of the word in storage addressed by EA.</li>
                    </ul>
                    <p>If GPR <em>RA</em> is not 0, the EA is the sum of the contents of GPR <em>RA</em> and GPR <em>RB</em>. If GPR <em>RA</em> is 0, then the EA is the contents of GPR <em>RB</em>.</p>
                `,
                "tooltip": "Store Half Byte-Reverse Indexed",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-sthbrx-store-half-byte-reverse-indexed-instruction"
            };
        case "STHU":
            return {
                "html": `
                    <p>The <strong>sthu</strong> instruction stores bits 16-31 of general-purpose register (GPR) <em>RS</em> into the halfword of storage addressed by the effective address (EA).</p>
                    <p>If GPR <em>RA</em> is not 0, the EA is the sum of the contents of GPR <em>RA</em> and <em>D</em>, a 16-bit signed two's complement integer sign-extended to 32 bits. If GPR <em>RA</em> is 0, then the EA is <em>D</em>.</p>
                    <p>If GPR <em>RA</em> does not equal 0 and the storage access does not cause an Alignment Interrupt or a Data Storage Interrupt, then the EA is placed into GPR <em>RA</em>.</p>
                `,
                "tooltip": "Store Half with Update",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-sthu-store-half-update-instruction"
            };
        case "STHUX":
            return {
                "html": `
                    <p>The <strong>sthux</strong> instruction stores bits 16-31 of general-purpose register (GPR) <em>RS</em> into the halfword of storage addressed by the effective address (EA).</p>
                    <p>If GPR <em>RA</em> is not 0, the EA is the sum of the contents of GPR <em>RA</em> and GPR <em>RB</em>. If GPR <em>RA</em> is 0, then the EA is the contents of GPR <em>RB</em>.</p>
                    <p>If GPR <em>RA</em> does not equal 0 and the storage access does not cause an Alignment Interrupt or a Data Storage Interrupt, then the EA is placed into register GPR <em>RA</em>.</p>
                `,
                "tooltip": "Store Half with Update Indexed",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-sthux-store-half-update-indexed-instruction"
            };
        case "STHX":
            return {
                "html": `
                    <p>The <strong>sthx</strong> instruction stores bits 16-31 of general-purpose register (GPR) <em>RS</em> into the halfword of storage addressed by the effective address (EA).</p>
                    <p>If GPR <em>RA</em> is not 0, the EA is the sum of the contents of GPR <em>RA</em> and GPR <em>RB</em>. If GPR <em>RA</em> is 0, then the EA is the contents of GPR <em>RB</em>.</p>
                `,
                "tooltip": "Store Half Indexed",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-sthx-store-half-indexed-instruction"
            };
        case "STM":
        case "STMW":
            return {
                "html": `
                    <p>The <strong>stmw</strong> and <strong>stm</strong> instructions store <em>N</em> consecutive words from general-purpose register (GPR)<em> RS</em> through GPR 31. Storage starts at the effective address (EA). <em>N</em> is a register number equal to 32 minus <em>RS</em>.</p>
                    <p>If GPR <em>RA</em> is not 0, the EA is the sum of the contents of GPR <em>RA</em> and <em>D</em>. The sum is a 16-bit signed two's complement integer sign-extended to 32 bits. If GPR <em>RA</em> is 0, then the EA is <em>D</em>.</p>
                `,
                "tooltip": "Store Multiple Word",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-stmw-stm-store-multiple-word-instruction"
            };
        case "STQ":
            return {
                "html": `
                    <p>The <strong>stq</strong> instruction stores a quad-word in storage from the source general-purpose registers (GPR) <em>RS</em> and <em>RS+1</em> into the specified location in memory referenced by the effective address (EA).</p>
                    <p>DS is a 14-bit, signed two's complement number, which is sign-extended to 64 bits, and then multiplied by 4 to provide a displacement <em>Disp</em>. If GPR <em>RA</em> is not 0, the EA is the sum of the contents of GPR <em>RA</em> and <em>Disp</em>. If GPR <em>RA</em> is 0, then the EA is <em>Disp</em>.</p>
                `,
                "tooltip": "Store Quad Word",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-stq-store-quad-word-instruction"
            };
        case "STSI":
        case "STSWI":
            return {
                "html": `
                    <p>The <strong>stswi</strong> and <strong>stsi</strong> instructions store <em>N</em> consecutive bytes starting with the leftmost byte in general-purpose register (GPR) <em>RS</em> at the effective address (EA) from GPR <em>RS</em> through GPR <em>RS</em> + <em>NR</em> - 1.</p>
                    <p>If GPR <em>RA</em> is not 0, the EA is the contents of GPR <em>RA</em>. If <em>RA</em> is 0, then the EA is 0.</p>
                `,
                "tooltip": "Store String Word Immediate",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-stswi-stsi-store-string-word-immediate-instruction"
            };
        case "STSX":
        case "STSWX":
            return {
                "html": `
                    <p>The <strong>stswx</strong> and <strong>stsx</strong> instructions store <em>N</em> consecutive bytes starting with the leftmost byte in register <em>RS</em> at the effective address (EA) from general-purpose register (GPR) <em>RS</em> through GPR <em>RS</em> + <em>NR</em> - 1.</p>
                    <p>If GPR <em>RA</em> is not 0, the EA is the sum of the contents of GPR <em>RA</em> and the contents of GPR <em>RB</em>. If GPR <em>RA</em> is 0, then EA is the contents of GPR <em>RB</em>.</p>
                `,
                "tooltip": "Store String Word Indexed",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-stswx-stsx-store-string-word-indexed-instruction"
            };
        case "ST":
        case "STW":
            return {
                "html": `
                    <p>The <strong>stw</strong> and <strong>st</strong> instructions store a word from general-purpose register (GPR) <em>RS</em> into a word of storage addressed by the effective address (EA).</p>
                    <p>If GPR <em>RA</em> is not 0, the EA is the sum of the contents of GPR <em>RA</em> and <em>D</em>, a 16-bit signed two's complement integer sign-extended to 32 bits. If GPR <em>RA</em> is 0, then the EA is <em>D</em>.</p>
                `,
                "tooltip": "Store",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-stw-st-store-instruction"
            };
        case "STBRX":
        case "STWBRX":
            return {
                "html": `<p>The <strong>stwbrx</strong> and <strong>stbrx</strong> instructions store a byte-reversed word from general-purpose register (GPR) <em>RS</em> into a word of storage addressed by the effective address (EA).</p>`,
                "tooltip": "Store Word Byte-Reverse Indexed",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=is-stwbrx-stbrx-store-word-byte-reverse-indexed-instruction"
            };
        case "STWCX.":
            return {
                "html": `<p>The <strong>stwcx.</strong> and <strong>lwarx</strong> instructions are primitive, or simple, instructions used to perform a read-modify-write operation to storage. If the store is performed, the use of the <strong>stwcx.</strong> and <strong>lwarx</strong> instructions ensures that no other processor or mechanism has modified the target memory location between the time the <strong>lwarx</strong> instruction is executed and the time the <strong>stwcx.</strong> instruction completes.</p>`,
                "tooltip": "Store Word Conditional Indexed",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-stwcx-store-word-conditional-indexed-instruction"
            };
        case "STU":
        case "STWU":
            return {
                "html": `
                    <p>The <strong>stwu</strong> and <strong>stu</strong> instructions store the contents of general-purpose register (GPR) <em>RS</em> into the word of storage addressed by the effective address (EA).</p>
                    <p>If GPR <em>RA</em> is not 0, the EA is the sum of the contents of GPR <em>RA</em> and <em>D</em>, a 16-bit signed two's complement integer sign-extended to 32 bits. If GPR <em>RA</em> is 0, then the EA is <em>D</em>.</p>
                    <p>If GPR <em>RA</em> is not 0 and the storage access does not cause an Alignment Interrupt or a Data Storage Interrupt, then EA is placed into GPR <em>RA</em>.</p>
                `,
                "tooltip": "Store Word with Update",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-stwu-stu-store-word-update-instruction"
            };
        case "STUX":
        case "STWUX":
            return {
                "html": `
                    <p>The <strong>stwux</strong> and <strong>stux</strong> instructions store the contents of general-purpose register (GPR) <em>RS</em> into the word of storage addressed by the effective address (EA).</p>
                    <p>If GPR <em>RA</em> is not 0, the EA is the sum of the contents of GPR <em>RA</em> and GPR <em>RB</em>. If GPR <em>RA</em> is 0, then the EA is the contents of GPR <em>RB</em>.</p>
                    <p>If GPR <em>RA</em> is not 0 and the storage access does not cause an Alignment Interrupt or a Data Storage Interrupt, then the EA is placed into GPR <em>RA</em>.</p>
                `,
                "tooltip": "Store Word with Update Indexed",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-stwux-stux-store-word-update-indexed-instruction"
            };
        case "STX":
        case "STWX":
            return {
                "html": `
                    <p>The <strong>stwx</strong> and <strong>stx</strong> instructions store the contents of general-purpose register (GPR) <em>RS</em> into the word of storage addressed by the effective address (EA).</p>
                    <p>If GPR <em>RA</em> is not 0, the EA is the sum of the contents of GPR <em>RA</em> and GPR <em>RB</em>. If GPR <em>RA</em> is 0, then the EA is the contents of GPR <em>RB</em>.</p>
                `,
                "tooltip": "Store Word Indexed",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-stwx-stx-store-word-indexed-instruction"
            };
        case "SUBF":
        case "SUBF.":
        case "SUBFO":
        case "SUBFO.":
            return {
                "html": `<p>The <strong>subf</strong> instruction adds the ones complement of the contents of general-purpose register (GPR) <em>RA</em> and 1 to the contents of GPR <em>RB</em> and stores the result in the target GPR <em>RT</em>.</p>`,
                "tooltip": "Subtract From",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-subf-subtract-from-instruction"
            };
        case "SF":
        case "SF.":
        case "SFO":
        case "SFO.":
        case "SUBFC":
        case "SUBFC.":
        case "SUBFCO":
        case "SUBFCO.":
            return {
                "html": `<p>The <strong>subfc</strong> and <strong>sf</strong> instructions add the ones complement of the contents of general-purpose register (GPR) <em>RA</em> and 1 to the contents of GPR <em>RB</em> and stores the result in the target GPR <em>RT</em>.</p>`,
                "tooltip": "Subtract from Carrying",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-subfc-sf-subtract-from-carrying-instruction"
            };
        case "SFE":
        case "SFE.":
        case "SFEO":
        case "SFEO.":
        case "SUBFE":
        case "SUBFE.":
        case "SUBFEO":
        case "SUBFEO.":
            return {
                "html": `<p>The <strong>subfe</strong> and <strong>sfe</strong> instructions add the value of the Fixed-Point Exception Register Carry bit, the contents of general-purpose register (GPR) <em>RB</em>, and the one's complement of the contents of GPR <em>RA</em> and store the result in the target GPR <em>RT</em>.</p>`,
                "tooltip": "Subtract from Extended",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-subfe-sfe-subtract-from-extended-instruction"
            };
        case "SFI":
        case "SUBFIC":
            return {
                "html": `<p>The <strong>subfic</strong> and <strong>sfi</strong> instructions add the one's complement of the contents of general-purpose register (GPR) <em>RA</em>, 1, and a 16-bit signed integer <em>SI</em>. The result is placed in the target GPR <em>RT</em>.</p>`,
                "tooltip": "Subtract from Immediate Carrying",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-subfic-sfi-subtract-from-immediate-carrying-instruction"
            };
        case "SFME":
        case "SFME.":
        case "SFMEO":
        case "SFMEO.":
        case "SUBFME":
        case "SUBFME.":
        case "SUBFMEO":
        case "SUBFMEO.":
            return {
                "html": `<p>The <strong>subfme</strong> and <strong>sfme</strong> instructions add the one's complement of the contents of general-purpose register(GPR) <em>RA</em>, the Carry Bit of the Fixed-Point Exception Register, and x<samp>'</samp>FFFFFFFF<samp>'</samp> and place the result in the target GPR <em>RT</em>.</p>`,
                "tooltip": "Subtract from Minus One Extended",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=is-subfme-sfme-subtract-from-minus-one-extended-instruction"
            };
        case "SFZE":
        case "SFZE.":
        case "SFZEO":
        case "SFZEO.":
        case "SUBFZE":
        case "SUBFZE.":
        case "SUBFZEO":
        case "SUBFZEO.":
            return {
                "html": `<p>The <strong>subfze</strong> and <strong>sfze</strong> instructions add the one's complement of the contents of general-purpose register (GPR) <em>RA</em>, the Carry bit of the Fixed-Point Exception Register, and x'00000000' and store the result in the target GPR <em>RT</em>.</p>`,
                "tooltip": "Subtract from Zero Extended",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-subfze-sfze-subtract-from-zero-extended-instruction"
            };
        case "SVC":
        case "SVCL":
        case "SVCA":
        case "SVCLA":
            return {
                "html": `<p>The <strong>svc</strong> instruction generates a supervisor call interrupt and places bits 16-31 of the <strong>svc</strong> instruction into bits 0-15 of the Count Register (CR) and bits 16-31 of the Machine State Register (MSR) into bits 16-31 of the CR.</p>`,
                "tooltip": "Supervisor Call",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-svc-supervisor-call-instruction"
            };
        case "SYNC":
        case "DCS":
            return {
                "html": `
                    <p>The PowerPC® instruction, <strong>sync</strong>, provides an ordering function that ensures that all instructions initiated prior to the <strong>sync</strong> instruction complete, and that no subsequent instructions initiate until after the <strong>sync</strong> instruction completes. When the <strong>sync</strong> instruction completes, all storage accesses initiated prior to the <strong>sync</strong> instruction are complete.</p>
                    <p>The POWER® family instruction, <strong>dcs</strong>, causes the processor to wait until all data cache lines being written or scheduled for writing to main memory have finished writing.</p>
                `,
                "tooltip": "Synchronize or Data Cache Synchronize",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-sync-synchronize-dcs-data-cache-synchronize-instruction"
            };
        case "TD":
            return {
                "html": `
                    <p>The <strong>td</strong> instruction generates a program interrupt when a specific condition is true.</p>
                    <p>The contents of general-purpose register (GPR) <em>RA</em> are compared with the contents of GPR <em>RB</em>. If any bit in the TO field is set and its corresponding condition is met by the result of the comparison, then a trap-type program interrupt is generated.</p>
                `,
                "tooltip": "Trap Double Word",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-td-trap-double-word-instruction"
            };
        case "TDI":
            return {
                "html": `
                    <p>The <strong>tdi</strong> instruction generates a program interrupt when a specific condition is true.</p>
                    <p>The contents of general-purpose register <em>RA</em> are compared with the sign-extended value of the SIMM field. If any bit in the <em>TO</em> field is set and its corresponding condition is met by the result of the comparison, then the system trap handler is invoked.</p>
                `,
                "tooltip": "Trap Double Word Immediate",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-tdi-trap-double-word-immediate-instruction"
            };
        case "TLBI":
        case "TLBIE":
            return {
                "html": `
                    <p>The PowerPC® instruction <strong>tlbie</strong> searches the Translation Look-Aside Buffer (TLB) for an entry corresponding to the effective address (EA). The search is done regardless of the setting of Machine State Register (MSR) Instruction Relocate bit or the MSR Data Relocate bit. The search uses a portion of the EA including the least significant bits, and ignores the content of the Segment Registers. Entries that satisfy the search criteria are made invalid so will not be used to translate subsequent storage accesses.</p>
                    <p>The POWER® family instruction <strong>tlbi</strong> expands the EA to its virtual address and invalidates any information in the TLB for the virtual address, regardless of the setting of MSR Instruction Relocate bit or the MSR Data Relocate bit. The EA is placed into the general-purpose register (GPR) <em>RA</em>.</p>
                `,
                "tooltip": "Translation Look-Aside Buffer Invalidate Entry",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=is-tlbie-tlbi-translation-look-aside-buffer-invalidate-entry-instruction"
            };
        case "TLBLD":
            return {
                "html": `<p>The <strong>tlbld</strong> instruction loads the data Translation Look-Aside Buffer (TLB) entry to assist a TLB reload function performed in software on the PowerPC 603 RISC Microprocessor.</p>`,
                "tooltip": "Load Data TLB Entry",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-tlbld-load-data-tlb-entry-instruction"
            };
        case "TLBLI":
            return {
                "html": `<p>The <strong>tlbli</strong> instruction loads the instruction Translation Look-Aside Buffer (TLB) entry to assist a TLB reload function performed in software on the PowerPC 603 RISC Microprocessor.</p>`,
                "tooltip": "Load Instruction TLB Entry",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-tlbli-load-instruction-tlb-entry-instruction"
            };
        case "TLBSYNC":
            return {
                "html": `<p>The <strong>tlbsync</strong> instruction ensures that a <strong>tlbie</strong> and <strong>tlbia</strong> instruction executed by one processor has completed on all other processors.</p>`,
                "tooltip": "Translation Look-Aside Buffer Synchronize",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-tlbsync-translation-look-aside-buffer-synchronize-instruction"
            };
        case "T":
        case "TW":
            return {
                "html": `<p>The <strong>tw</strong> and <strong>t</strong> instructions compare the contents of general-purpose register (GPR) <em>RA</em> with the contents of GPR <em>RB</em>, AND the compared results with <em>TO</em>, and generate a trap-type Program Interrupt if the result is not 0.</p>`,
                "tooltip": "Trap Word",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-tw-t-trap-word-instruction"
            };
        case "TI":
        case "TWI":
            return {
                "html": `<p>The <strong>twi</strong> and <strong>ti</strong> instructions compare the contents of general-purpose register (GPR) <em>RA</em> with the sign extended <em>SI</em> field, AND the compared results with <em>TO</em>, and generate a trap-type program interrupt if the result is not 0.</p>`,
                "tooltip": "Trap Word Immediate",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-twi-ti-trap-word-immediate-instruction"
            };
        case "XOR":
        case "XOR.":
            return {
                "html": `<p>The <strong>xor</strong> instruction XORs the contents of general-purpose register (GPR) <em>RS</em> with the contents of GPR <em>RB</em> and stores the result in GPR <em>RA</em>.</p>`,
                "tooltip": "XOR",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-xor-xor-instruction"
            };
        case "XORI":
        case "XORIL":
            return {
                "html": `<p>The <strong>xori</strong> and <strong>xoril</strong> instructions XOR the contents of general-purpose register (GPR) <em>RS</em> with the concatenation of x'0000' and a 16-bit unsigned integer <em>UI</em> and store the result in GPR <em>RA</em>.</p>`,
                "tooltip": "XOR Immediate",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-xori-xoril-xor-immediate-instruction"
            };
        case "XORIS":
        case "XORIU":
            return {
                "html": `<p>The <strong>xoris</strong> and <strong>xoriu</strong> instructions XOR the contents of general-purpose register (GPR) <em>RS</em> with the concatenation of a 16-bit unsigned integer <em>UI</em> and 0x'0000' and store the result in GPR <em>RA</em>.</p>`,
                "tooltip": "XOR Immediate Shift",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-xoris-xoriu-xor-immediate-shift-instruction"
            };
    }
};
