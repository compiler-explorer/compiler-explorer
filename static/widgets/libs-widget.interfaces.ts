export interface StateLib {
    id?: string;
    name?: string;
    ver?: string;
    version?: string;
}

export interface WidgetState {
    libs?: StateLib[];
}
