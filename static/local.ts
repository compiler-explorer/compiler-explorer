// Copyright (c) 2021, Compiler Explorer Authors
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

import {options} from './options.js';

const prefix = options.localStoragePrefix ?? '';

export interface Storage {
    get<T>(key: string, ifNotPresent: T): string | T;

    set(key: string, value: string): boolean;

    remove(key: string);
}

class LocalOnlyStorage implements Storage {
    get<T>(key: string, ifNotPresent: T): string | T {
        try {
            return window.localStorage.getItem(prefix + key) ?? ifNotPresent;
        } catch (e) {
            // Swallow up any security exceptions...
            return ifNotPresent;
        }
    }

    remove(key: string) {
        try {
            window.localStorage.removeItem(prefix + key);
        } catch (e) {
            // Swallow up any security exceptions...
        }
    }

    set(key: string, value: string): boolean {
        try {
            window.localStorage.setItem(prefix + key, value);
            return true;
        } catch (e) {
            // Swallow up any security exceptions...
        }
        return false;
    }
}

export const localStorage = new LocalOnlyStorage();

class SessionThenLocalStorage implements Storage {
    get<T>(key: string, ifNotPresent: T): string | T {
        try {
            const sessionValue = window.sessionStorage.getItem(prefix + key);
            if (sessionValue !== null) return sessionValue;
        } catch (e) {
            // Swallow up any security exceptions...
        }
        return localStorage.get<T>(key, ifNotPresent);
    }

    remove(key: string) {
        this.removeSession(key);
        localStorage.remove(key);
    }

    private removeSession(key: string) {
        try {
            window.sessionStorage.removeItem(prefix + key);
        } catch (e) {
            // Swallow up any security exceptions...
        }
    }

    private setSession(key: string, value: string): boolean {
        try {
            window.sessionStorage.setItem(prefix + key, value);
            return true;
        } catch (e) {
            // Swallow up any security exceptions...
        }
        return false;
    }

    set(key: string, value: string): boolean {
        const setBySession = this.setSession(key, value);
        const setByLocal = localStorage.set(key, value);
        return setBySession || setByLocal;
    }
}

export const sessionThenLocalStorage = new SessionThenLocalStorage();
