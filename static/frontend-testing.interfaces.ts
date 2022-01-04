
export interface ICETestable {
    readonly description: string;
    run(): Promise<void>;
};

export interface ICEFrontendTesting {
    add(test: ICETestable);
    run(testToRun: string): Promise<void>;
}
