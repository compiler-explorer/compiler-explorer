
export interface ITestable {
    readonly description: string;
    run(): Promise<void>;
}

export interface IFrontendTesting {
    add(test: ITestable);
    getAllTestNames(): string[];
    run(testToRun: string): Promise<void>;
}
