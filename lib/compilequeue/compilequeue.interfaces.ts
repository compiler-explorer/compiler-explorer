export type CompileMessage = {
    dt: Date;
    req: any;
    requestId?: string;
};

export type CompileQueueResult = {
    requestId: string;
    dt: Date;
    body: string;
    status: number;
    headers: Record<string, string>;
};

export interface ICompileQueue {
    pop(): Promise<CompileMessage | undefined>;
    push(message: CompileMessage): Promise<string | undefined>;

    pushResult(result: CompileQueueResult): Promise<void>;
    popResult(requestId: string): Promise<CompileQueueResult | undefined>;
}

export const FakeRequestProperties = ['url', 'method', 'originalUrl', 'baseUrl', 'params', 'query', 'body', 'headers'];

export class FakeRequest {
    public url;
    public method;
    public originalUrl;
    public baseUrl;
    public params;
    public query;
    public body;
    public headers;

    constructor(msg: any) {
        for (const k in msg) {
            this[k] = msg[k];
        }
    }

    is(contentType) {
        return this.headers['content-type'].includes(contentType);
    }

    accepts(arr) {
        // if Accept is 'application/json, text/javascript, *.*', then 'json' is more important than 'text', so return 'json' in that situation
        const accepted = this.headers['accept'].split(', ');
        for (const acc of accepted) {
            for (const totest of arr) {
                if (acc.includes(totest)) {
                    return totest;
                }
            }
        }
        return 'text';
    }
}

export class FakeResponse {
    public result: CompileQueueResult;
    public endHook: () => Promise<void> = async () => {};

    constructor(requestId: string) {
        this.result = {
            requestId: requestId,
            dt: new Date(),
            body: '',
            status: 200,
            headers: {},
        };
    }

    send(msg) {
        this.write(msg);

        this.endHook();

        return this;
    }

    status(stat) {
        this.result.status = stat;
        return this;
    }

    sendStatus(stat) {
        this.result.status = stat;
        return this.end();
    }

    end(msg?) {
        if (msg) {
            this.write(msg);
        }

        this.endHook();

        return this;
    }

    set(k, v) {
        this.result.headers[k] = v;
        return this;
    }

    write(msg) {
        if (typeof msg === 'string') {
            this.result.body += msg;
        } else {
            this.result.body += JSON.stringify(msg);
            this.set('content-type', 'application/json');
        }

        return this;
    }
}
