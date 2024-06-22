import { AssemblyInstructionInfo } from "../base.js";

// Based on the IBM documentation of assembly instructions for AIX 7.3 (https://www.ibm.com/docs/en/aix/7.3?topic=reference-instruction-set).
// An automatic generator is available at etc/scripts/docenizers/docenizer-power.py, but it has a lot of quirks and is considered incomplete.
// However, IBM renders their documentation pages with React, which makes it impossible to do scraping without Selenium.
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
                    <p>The <strong>abs</strong> instruction has four syntax forms. Each syntax form has a different effect on Condition Register Field 0 and the Fixed-Point Exception Register.</p>
                    <p>The four syntax forms of the <strong>abs</strong> instruction always affect the Carry bit (CA) in the Fixed-Point Exception Register. If the syntax form sets the Overflow Exception (OE) bit to 1, the instruction affects the Summary Overflow (SO) and Overflow (OV) bits in the Fixed-Point Exception Register. If the syntax form sets the Record (Rc) bit to 1, the instruction affects the Less Than (LT) zero, Greater Than (GT) zero, Equal To (EQ) zero, and Summary Overflow (SO) bits in Condition Register Field 0.</p>
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
                    <p>The <strong>add</strong> and <strong>cax</strong> instructions have four syntax forms. Each syntax form has a different effect on Condition Register Field 0 and the Fixed-Point Exception Register.</p>
                    <p>The four syntax forms of the <strong>add</strong> instruction and the four syntax forms of the <strong>cax</strong> instruction never affect the Carry bit (CA) in the Fixed-Point Exception Register. If the syntax form sets the Overflow Exception (OE) bit to 1, the instruction affects the Summary Overflow (SO) and Overflow (OV) bits in the Fixed-Point Exception Register. If the syntax form sets the Record (Rc) bit to 1, the instruction affects the Less Than (LT) zero, Greater Than (GT) zero, Equal To (EQ) zero, and Summary Overflow (SO) bits in Condition Register Field 0.</p>
                `,
                "tooltip": "Add or Compute Address",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-add-add-cax-compute-address-instruction"
            };
        case "ADDC":
        case "ADDC.":
        case "ADDCO":
        case "ADDCO.":
        case "A":
        case "A.":
        case "AO":
        case "AO.":
            return {
                "html": `
                    <p>The <strong>addc</strong> and <strong>a</strong> instructions place the sum of the contents of general-purpose register (GPR) <em>RA</em> and GPR <em>RB</em> into the target GPR <em>RT</em>.</p>
                    <p>The <strong>addc</strong> instruction has four syntax forms. Each syntax form has a different effect on Condition Register Field 0 and the Fixed-Point Exception Register.</p>
                    <p>The <strong>a</strong> instruction has four syntax forms. Each syntax form has a different effect on Condition Register Field 0 and the Fixed-Point Exception Register.</p>
                    <p>The four syntax forms of the <strong>addc</strong> instruction and the four syntax forms of the <strong>a</strong> instruction always affect the Carry bit (CA) in the Fixed-Point Exception Register. If the syntax form sets the Overflow Exception (OE) bit to 1, the instruction affects the Summary Overflow (SO) and Overflow (OV) bits in the Fixed-Point Exception Register. If the syntax form sets the Record (Rc) bit to 1, the instruction affects the Less Than (LT) zero, Greater Than (GT) zero, Equal To (EQ) zero, and Summary Overflow (SO) bits in Condition Register Field 0.</p>
                `,
                "tooltip": "Add Carrying",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-addc-add-carrying-instruction"
            };
        case "ADDE":
        case "ADDE.":
        case "ADDEO":
        case "ADDEO.":
        case "AE":
        case "AE.":
        case "AEO":
        case "AEO.":
            return {
                "html": `
                    <p>The <strong>adde</strong> and <strong>ae</strong> instructions place the sum of the contents of general-purpose register (GPR) <em>RA</em>, GPR <em>RB</em>, and the Carry bit into the target GPR <em>RT</em>.</p>
                    <p>The <strong>adde</strong> instruction has four syntax forms. Each syntax form has a different effect on Condition Register Field 0 and the Fixed-Point Exception Register.</p>
                    <p>The <strong>ae</strong> instruction has four syntax forms. Each syntax form has a different effect on Condition Register Field 0 and the Fixed-Point Exception Register.</p>
                    <p>The four syntax forms of the <strong>adde</strong> instruction and the four syntax forms of the <strong>ae</strong> instruction always affect the Carry bit (CA) in the Fixed-Point Exception Register. If the syntax form sets the Overflow Exception (OE) bit to 1, the instruction affects the Summary Overflow (SO) and Overflow (OV) bits in the Fixed-Point Exception Register. If the syntax form sets the Record (Rc) bit to 1, the instruction affects the Less Than (LT) zero, Greater Than (GT) zero, Equal To (EQ) zero, and Summary Overflow (SO) bits in Condition Register Field 0.</p>
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
        case "ADDIC":
        case "AI":
            return {
                "html": `
                    <p>The <strong>addic</strong> and <strong>ai</strong> instructions place the sum of the contents of general-purpose register (GPR) <em>RA</em> and a 16-bit signed integer, <em>SI</em>, into target GPR <em>RT</em>.</p>
                    <p>The 16-bit integer provided as immediate data is sign-extended to 32 bits prior to carrying out the addition operation.</p>
                    <p>The <strong>addic</strong> and <strong>ai</strong> instructions have one syntax form and can set the Carry bit of the Fixed-Point Exception Register; these instructions never affect Condition Register Field 0.</p>
                `,
                "tooltip": "Add Immediate Carrying",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-addic-ai-add-immediate-carrying-instruction"
            };
        case "ADDIC.":
        case "AI.":
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
                    <p>The <strong>addis</strong> and <strong>cau</strong> instructions do not affect Condition Register Field 0 or the Fixed-Point Exception Register.  The <strong>cau</strong> instruction has one syntax form. The <strong>addis</strong> instruction has two syntax forms; however, the second form is only valid when the R_TOCU relocation type is used in the <em>D</em> expression. The R_TOCU relocation type can be specified explicitly with the <strong>@u</strong> relocation specifier or implicitly by using a <strong>QualName</strong> parameter with a TE storage-mapping class.</p>
                    <blockquote><span>“</span><strong>Note:</strong> The immediate value for the <strong>cau</strong> instruction is a 16-bit unsigned integer, whereas the immediate value for the <strong>addis</strong> instruction is a 16-bit signed integer. This difference is a result of extending the architecture to 64 bits.<span>”</span></blockquote>
                    <p>The assembler does a 0 to 65535 value-range check for the <em>UI</em> field, and a -32768 to 32767 value-range check for the <em>SI</em> field.</p>
                    <p>To keep the source compatibility of the <strong>addis</strong> and <strong>cau</strong> instructions, the assembler expands the value-range check for the <strong>addis</strong> instruction to -65536 to 65535. The sign bit is ignored and the assembler only ensures that the immediate value fits into 16 bits. This expansion does not affect the behavior of a 32-bit implementation or 32-bit mode in a 64-bit implementation.</p>
                    <p>The <strong>addis</strong> instruction has different semantics in 32-bit mode than it does in 64-bit mode. If bit 32 is set, it propagates through the upper 32 bits of the 64-bit general-purpose register. Use caution when using the <strong>addis</strong> instruction to construct an unsigned integer. The <strong>addis</strong> instruction with an unsigned integer in 32-bit may not be directly ported to 64-bit mode. The code sequence needed to construct an unsigned integer in 64-bit mode is significantly different from that needed in 32-bit mode.</p>
                `,
                "tooltip": "Add Immediate Shifted",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-addis-cau-add-immediate-shifted-instruction"
            };
        case "ADDME":
        case "ADDME.":
        case "ADDMEO":
        case "ADDMEO.":
        case "AME":
        case "AME.":
        case "AMEO":
        case "AMEO.":
            return {
                "html": `
                    <p>The <strong>addme</strong> and <strong>ame</strong> instructions place the sum of the contents of general-purpose register (GPR) <em>RA</em>, the Carry bit of the Fixed-Point Exception Register, and -1 (0xFFFF FFFF<samp>)</samp> into the target GPR <em>RT</em>.</p>
                    <p>The <strong>addme</strong> instruction has four syntax forms. Each syntax form has a different effect on Condition Register Field 0 and the Fixed-Point Exception Register.</p>
                    <p>The <strong>ame</strong> instruction has four syntax forms. Each syntax form has a different effect on Condition Register Field 0 and the Fixed-Point Exception Register.</p>
                    <p>The four syntax forms of the <strong>addme</strong> instruction and the four syntax forms of the <strong>ame</strong> instruction always affect the Carry bit (CA) in the Fixed-Point Exception Register. If the syntax form sets the Overflow Exception (OE) bit to 1, the instruction affects the Summary Overflow (SO) and Overflow (OV) bits in the Fixed-Point Exception Register. If the syntax form sets the Record (Rc) bit to 1, the instruction affects the Less Than (LT) zero, Greater Than (GT) zero, Equal To (EQ) zero, and Summary Overflow (SO) bits in Condition Register Field 0.</p>
                `,
                "tooltip": "Add to Minus One Extended",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-addme-ame-add-minus-one-extended-instruction"
            };
        case "ADDZE":
        case "ADDZE.":
        case "ADDZEO":
        case "ADDZEO.":
        case "AZE":
        case "AZE.":
        case "AZEO":
        case "AZEO.":
            return {
                "html": `
                    <p>The <strong>addze</strong> and <strong>aze</strong> instructions add the contents of general-purpose register (GPR) <em>RA</em>, the Carry bit, and 0x0000 0000 and place the result into the target GPR <em>RT</em>.</p>
                    <p>The <strong>addze</strong> instruction has four syntax forms. Each syntax form has a different effect on Condition Register Field 0 and the Fixed-Point Exception Register.</p>
                    <p>The <strong>aze</strong> instruction has four syntax forms. Each syntax form has a different effect on Condition Register Field 0 and the Fixed-Point Exception Register.</p>
                    <p>The four syntax forms of the <strong>addze</strong> instruction and the four syntax forms of the <strong>aze</strong> instruction always affect the Carry bit (CA) in the Fixed-Point Exception Register. If the syntax form sets the Overflow Exception (OE) bit to 1, the instruction affects the Summary Overflow (SO) and Overflow (OV) bits in the Fixed-Point Exception Register. If the syntax form sets the Record (Rc) bit to 1, the instruction affects the Less Than (LT) zero, Greater Than (GT) zero, Equal To (EQ) zero, and Summary Overflow (SO) bits in Condition Register Field 0.</p>
                `,
                "tooltip": "Add to Zero Extended",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-addze-aze-add-zero-extended-instruction"
            };
        case "AND":
        case "AND.":
            return {
                "html": `
                    <p>The <strong>and</strong> instruction logically ANDs the contents of general-purpose register (GPR) <em>RS</em> with the contents of GPR <em>RB</em> and places the result into the target GPR <em>RA</em>.</p>
                    <p>The <strong>and</strong> instruction has two syntax forms. Each syntax form has a different effect on Condition Register Field 0.</p>
                    <p>The two syntax forms of the <strong>and</strong> instruction never affect the Fixed-Point Exception Register. If the syntax form sets the Record (Rc) bit to 1, the instruction affects the Less Than (LT) zero, Greater Than (GT) zero, Equal To (EQ) zero, and Summary Overflow (SO) bits in Condition Register Field 0.</p>
                `,
                "tooltip": "AND",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-instruction"
            };
        case "ANDC":
        case "ANDC.":
            return {
                "html": `
                    <p>The <strong>andc</strong> instruction logically ANDs the contents of general-purpose register (GPR) <em>RS</em> with the complement of the contents of GPR <em>RB</em> and places the result into GPR <em>RA</em>.</p>
                    <p>The <strong>andc</strong> instruction has two syntax forms. Each syntax form has a different effect on Condition Register Field 0.</p>
                    <p>The two syntax forms of the <strong>andc</strong> instruction never affect the Fixed-Point Exception Register. If the syntax form sets the Record (Rc) bit to 1, the instruction affects the Less Than (LT) zero, Greater Than (GT) zero, Equal To (EQ) zero, and Summary Overflow (SO) bits in Condition Register Field 0.</p>
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
        case "BCCTR":
        case "BCCTRL":
        case "BCC":
        case "BCCL":
            return {
                "html": `<p>The <strong>bcctr</strong> and <strong>bcc</strong> instructions conditionally branch to an instruction specified by the branch target address contained within the Count Register. The branch target address is the concatenation of Count Register bits 0-29 and b'00'.</p>`,
                "tooltip": "Branch Conditional to Count Register",
                "url": "https://www.ibm.com/docs/en/aix/7.3?topic=set-bcctr-bcc-branch-conditional-count-register-instruction"
            };
        case "BCLR":
        case "BCLRL":
        case "BCR":
        case "BCRL":
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
        case "CNTLZW":
        case "CNTLZW.":
        case "CNTLZ":
        case "CNTLZ.":
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
    }
};
