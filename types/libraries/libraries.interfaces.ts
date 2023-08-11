export type LibraryVersion = {
    name?: string;
    staticliblink: string[];
    alias?: string;
    version?: string;
    dependencies: string[];
    liblink: string[];
    libpath: string[];
    path: string[];
    options: string[];
    packagedheaders?: boolean;
};

export type Library = {
    id: string;
    versions: Record<string, LibraryVersion>;
    name?: string;
};

export type SelectedLibraryVersion = {
    id: string;
    version: string;
};
