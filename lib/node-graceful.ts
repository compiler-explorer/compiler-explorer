// +----------------------------------------------------------------------+
// | node-graceful v3 (https://github.com/mrbar42/node-graceful)      |
// | Graceful process exit manager.                                       |
// |----------------------------------------------------------------------|

export type GracefulListener = (
    signal: string,
    details?: object,
) => void | any | Promise<any> | Promise<void> | Promise<Error>;

export type GracefulSubscription = () => void;

export class Graceful {
    private static DEADLY_SIGNALS = ['SIGTERM', 'SIGINT', 'SIGBREAK', 'SIGHUP'];

    public static exitOnDouble = true;
    public static timeout = 30000;

    private static _captureExceptions = false;
    private static _captureRejections = false;
    private static listeners: GracefulListener[] = [];
    private static isRegistered = false;
    private static isExiting = false;
    private static exceptionListener = (event: any) => {
        process.exitCode = 1;
        Graceful.onDeadlyEvent('uncaughtException', event);
    };
    private static rejectionListener = (event: any) => {
        process.exitCode = 1;
        Graceful.onDeadlyEvent('unhandledRejection', event);
    };
    private static signalsListeners: {[signal: string]: (event: any) => void} = {};

    public static get captureExceptions() {
        return Graceful._captureExceptions;
    }

    public static set captureExceptions(newValue: boolean) {
        if (Graceful._captureExceptions === newValue) return;
        Graceful._captureExceptions = newValue;

        if (Graceful._captureExceptions) {
            process.on('uncaughtException' as any, Graceful.exceptionListener);
        } else {
            process.removeListener('uncaughtException', Graceful.exceptionListener);
        }
    }

    public static get captureRejections() {
        return Graceful._captureRejections;
    }

    public static set captureRejections(newValue: boolean) {
        if (Graceful._captureRejections === newValue) return;
        Graceful._captureRejections = newValue;

        if (Graceful._captureRejections) {
            process.on('unhandledRejection' as any, Graceful.rejectionListener);
        } else {
            process.removeListener('unhandledRejection', Graceful.rejectionListener);
        }
    }

    public static on(signal: 'exit', listener: GracefulListener): GracefulSubscription {
        if (signal !== 'exit') throw new Error("Only supports 'exit' signal");

        Graceful.listeners.push(listener);

        Graceful.updateRegistration();
        return () => Graceful.off('exit', listener);
    }

    public static off(signal: 'exit', listener: GracefulListener) {
        if (signal !== 'exit') throw new Error("Only supports 'exit' signal");

        const index = Graceful.listeners.indexOf(listener);
        if (index !== -1) Graceful.listeners.splice(index, 1);

        Graceful.updateRegistration();
    }

    public static clear() {
        Graceful.listeners.splice(0, Infinity);
        Graceful.updateRegistration();
    }

    public static exit(code?: number | string, signal = 'SIGTERM') {
        const exitSignal = typeof code === 'string' ? code : signal;

        if (typeof code === 'number') {
            process.exitCode = code;
        }

        Graceful.onDeadlyEvent(exitSignal, {reason: 'Manual call to Graceful.exit()'});
    }

    private static onDeadlyEvent(signal: string, details?: object) {
        // console.log(signal, details);
        if (Graceful.isExiting) {
            if (Graceful.exitOnDouble) Graceful.killProcess(true);
            return;
        }

        const listeners = Graceful.listeners.slice(0);

        Graceful.isExiting = true;

        let completedListeners = 0;
        const done = () => {
            completedListeners++;
            if (completedListeners === listeners.length) {
                Graceful.killProcess(false);
            }
        };

        if (Number(Graceful.timeout)) {
            const timeoutRef = setTimeout(() => Graceful.killProcess(true), Graceful.timeout);
            if (timeoutRef && timeoutRef.unref) timeoutRef.unref();
        }

        for (const listener of listeners) {
            Graceful.invokeListener(listener, done, signal, details);
        }
    }

    private static invokeListener(listener: GracefulListener, done: () => void, signal: string, details?: object) {
        let invoked = false;
        const listenerDone = () => {
            if (!invoked) {
                invoked = true;
                done();
            }
        };

        const retVal: any = listener(signal, details);
        // allow returning a promise
        if (retVal && typeof retVal.then === 'function') {
            retVal.then(listenerDone, listenerDone);
        } else {
            listenerDone();
        }
    }

    private static updateRegistration() {
        if (Graceful.listeners.length && !Graceful.isRegistered) {
            for (const deadlySignal of Graceful.DEADLY_SIGNALS) {
                const listener = () => Graceful.onDeadlyEvent(deadlySignal);
                Graceful.signalsListeners[deadlySignal] = listener;
                process.on(deadlySignal as any, listener);
            }
            Graceful.isRegistered = true;
        } else if (!Graceful.listeners.length && Graceful.isRegistered) {
            for (const deadlySignal of Graceful.DEADLY_SIGNALS) {
                const listener = Graceful.signalsListeners[deadlySignal];
                if (listener) {
                    process.removeListener(deadlySignal, listener);
                    delete Graceful.signalsListeners[deadlySignal];
                }
            }
            Graceful.isRegistered = false;
        }
    }

    private static killProcess(force: boolean) {
        process.exit(process.exitCode || (force ? 1 : 0));
    }
}
