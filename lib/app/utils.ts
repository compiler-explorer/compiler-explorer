// Copyright (c) 2025, Compiler Explorer Authors
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

/**
 * Parses a command line option into a number.
 */
export function parseNumberForOptions(value: string): number {
    // Ensure string contains only digits (and optional leading minus sign)
    if (!/^-?\d+$/.test(value)) {
        throw new Error(`Invalid number: "${value}"`);
    }

    const parsedValue = Number.parseInt(value, 10);
    if (Number.isNaN(parsedValue)) {
        throw new Error(`Invalid number: "${value}"`);
    }
    return parsedValue;
}

/**
 * Determines whether the app is running in development mode.
 */
export function isDevMode(): boolean {
    return process.env.NODE_ENV !== 'production';
}

/**
 * Gets the appropriate favicon filename based on the environment.
 * @param isDevModeValue - Whether the app is running in development mode
 * @param env - The environment names array
 */
export function getFaviconFilename(isDevModeValue: boolean, env?: string[]): string {
    if (isDevModeValue) {
        return 'favicon-dev.ico';
    }
    if (env?.includes('beta')) {
        return 'favicon-beta.ico';
    }
    if (env?.includes('staging')) {
        return 'favicon-staging.ico';
    }
    return 'favicon.ico';
}

/**
 * Measures event loop lag to monitor server performance.
 * Used to detect when the server is under heavy load or not responding quickly.
 * @param delayMs - The delay in milliseconds to measure against
 * @returns The lag in milliseconds
 */
export function measureEventLoopLag(delayMs: number): Promise<number> {
    return new Promise<number>(resolve => {
        const start = process.hrtime.bigint();
        setTimeout(() => {
            const elapsed = process.hrtime.bigint() - start;
            const delta = elapsed - BigInt(delayMs * 1000000);
            return resolve(Number(delta) / 1000000);
        }, delayMs);
    });
}
