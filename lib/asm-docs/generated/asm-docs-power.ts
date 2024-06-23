import { AssemblyInstructionInfo } from "../base.js";

// Based on the IBM documentation of assembly instructions for AIX 7.3 (https://www.ibm.com/docs/en/aix/7.3?topic=reference-instruction-set).
// An automatic generator is available at etc/scripts/docenizers/docenizer-power.py, but it has a lot of quirks and is considered incomplete.
// This is because IBM renders their documentation pages with React, which makes it impossible to do scraping without Selenium.
// However, what's worse is that some of the pages have slightly different layouts and formats, which makes automated processing awful.
// As such, this was created manually to have a complete documentation of the current ISA.
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
    }
};
