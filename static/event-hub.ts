// Copyright (c) 2022, Compiler Explorer Authors
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright notice,
//       this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

import GoldenLayout from 'golden-layout';
import * as Sentry from '@sentry/browser';

import {Hub} from './hub';

export type EventHubCallback<T extends unknown[]> = (...args: T) => void;

export interface DependencyProxies<T1 extends unknown[], T2 extends unknown[] = T1> {
    dependencyProxy: EventHubCallback<T1>;
    dependentProxy: EventHubCallback<T2>;
}

export interface Event<T extends unknown[], C = any> {
    evt: string;
    fn: EventHubCallback<T>;
    ctx?: C;
}

/**
 * Event Hub which is enclosed inside a Hub, allowing for dependent calls and
 * deferred execution based on the parent Hub.
 */
export class EventHub {
    private readonly hub: Hub;
    private readonly layoutEventHub: GoldenLayout.EventEmitter;
    private subscriptions: Event<any>[] = [];

    public constructor(hub: any, layoutEventHub: GoldenLayout.EventEmitter) {
        this.hub = hub;
        this.layoutEventHub = layoutEventHub;
    }

    /**
     * Emit an event to the layout event hub.
     *
     * Events are deferred by the parent hub during initialization to allow all
     * components to install their listeners before the emission is performed.
     * This fixes some ordering issues.
     */
    public emit(event: string, ...args: unknown[]): void {
        if (this.hub.deferred) {
            this.hub.deferredEmissions.push([event, ...args]);
        } else {
            this.layoutEventHub.emit(event, ...args);
        }
    }

    /** Attach a listener to the layout event hub. */
    public on<T extends unknown[], C = any>(event: string, callback: EventHubCallback<T>, context?: C): void {
        this.layoutEventHub.on(event, callback, context);
        this.subscriptions.push({evt: event, fn: callback, ctx: context});
    }

    /** Remove all listeners from the layout event hub. */
    public unsubscribe(): void {
        for (const subscription of this.subscriptions) {
            try {
                this.layoutEventHub.off(subscription.evt, subscription.fn, subscription.ctx);
            } catch (e) {
                Sentry.captureMessage(`Can not unsubscribe from ${subscription.evt.toString()}`);
                Sentry.captureException(e);
            }
        }
        this.subscriptions = [];
    }

    public mediateDependentCalls<T1 extends unknown[], T2 extends unknown[] = T1>(
        dependent: EventHubCallback<any>,
        dependency: EventHubCallback<any>
    ): DependencyProxies<T1, T2> {
        let hasDependencyExecuted = false;
        let lastDependentArguments: unknown[] | null = null;
        const dependencyProxy = function (this: unknown, ...args: unknown[]) {
            dependency.apply(this, args);
            hasDependencyExecuted = true;
            if (lastDependentArguments) {
                dependent.apply(this, lastDependentArguments);
                lastDependentArguments = null;
            }
        };
        const dependentProxy = function (this: unknown, ...args: unknown[]) {
            if (hasDependencyExecuted) {
                dependent.apply(this, args);
            } else {
                lastDependentArguments = args;
            }
        };
        return {dependencyProxy, dependentProxy};
    }
}
