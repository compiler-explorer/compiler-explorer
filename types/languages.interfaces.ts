export interface Language {
    // Id of language. Added programmatically based on CELanguages key
    id: string;
    // UI display name of the language
    name: string;
    // Monaco Editor language ID (Selects which language Monaco will use to highlight the code)
    monaco: string;
    // Usual extensions associated with the language. First one is used as file input etx
    extensions: string[];
    // Different ways in which we can also refer to this language
    alias: string[];
    // Format API name to use (See https://godbolt.org/api/formats)
    formatter: string | null;
    // Whether there's at least 1 compiler in this language that supportsExecute
    supportsExecute: boolean | null;
}
