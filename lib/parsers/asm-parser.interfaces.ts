import {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces';

export interface IAsmParser {
    process(asm: string, filters: ParseFiltersAndOutputOptions);
}
