
export interface ICETestable {
    run();
};

export interface ICEFrontendTestResult {
    desc: string;
    success: Boolean;
    error?: string;
}

export interface ICEFrontendTesting {
    add(test: ICETestable);
    run();
}