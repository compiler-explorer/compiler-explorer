import { getAsmOpcode } from '../asm-docs-6502';
import { BaseAssemblyDocumentationHandler } from '../base-assembly-documentation-handler';

export class Mos6502DocumentationHandler extends BaseAssemblyDocumentationHandler {
    getInstructionInformation(instruction) {
        return getAsmOpcode(instruction) || null;
    }
}
