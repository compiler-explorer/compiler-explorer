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

/* eslint-disable @typescript-eslint/adjacent-overload-signatures -- nicer format */

import { SiteSettings } from './settings';

/**
 * Exhaustive enumeration of all possible events the Hub's event emitter
 * can emit.
 */
export type EventKeys =
    | 'settingsChange'
    | 'requestSettings'
    | 'modifySettings'

    | 'astViewOpened'
    ;

/**
 * A mapping from event name to a tuple of the arguments that the event expects
 * to be passed to the event emitter's `emit` method.
 */
export interface EventHubArguments extends Record<EventKeys, [...args: any]> {
    settingsChange: [SiteSettings];
    requestSettings: [];
    modifySettings: [Partial<SiteSettings>];

    astViewOpened: [number];
}

/**
 * A mapping from event name to the type the callback function that should be
 * passed to the event emitter's `on` method.
 */
export interface EventHubListeners extends Record<EventKeys, (...args: any) => any> {
    settingsChange: (settings: SiteSettings) => void;
    requestSettings: () => void;
    modifySettings: (settings: SiteSettings) => void;

    astViewOpened: (id: number) => void;
}

export interface EventHubEvents {
    on: <K extends EventKeys>(event: K, listener: EventHubListeners[K]) => void;
    emit: <K extends EventKeys>(event: K, ...args: EventHubArguments[K]) => void;
}
