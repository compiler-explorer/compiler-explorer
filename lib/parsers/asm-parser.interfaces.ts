import {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';

export interface IAsmParser {
    process(asm: string, filters: ParseFiltersAndOutputOptions);
}
