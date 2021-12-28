import { getAsmOpcode } from '../asm-docs-avr';
import { BaseAssemblyDocumentationHandler } from '../base-assembly-documentation-handler';

export class AVRDocumentationHandler extends BaseAssemblyDocumentationHandler {
    getInstructionInformation(instruction) {
        return getAsmOpcode(instruction) || null;
    }
}
