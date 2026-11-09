// Copyright (c) 2026, Compiler Explorer Authors
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

async function fetchJsonOnce<T>(url: string, cache: RequestCache): Promise<T> {
    const response = await fetch(url, {headers: {Accept: 'application/json'}, cache});
    const body = await response.text();
    const describe = () =>
        `status ${response.status}, ${body.length} bytes, ` +
        `content-type ${response.headers.get('content-type')}, x-cache ${response.headers.get('x-cache')}, ` +
        `starts with ${JSON.stringify(body.slice(0, 40))}`;
    if (!response.ok) throw new Error(`Failed to fetch ${url} (${describe()})`);
    try {
        return JSON.parse(body);
    } catch (e) {
        throw new Error(`Invalid JSON from ${url} (${describe()}): ${e}`);
    }
}

/**
 * Fetches and parses a JSON API response, retrying once with the browser cache bypassed. Errors describe the
 * response actually received, since empty bodies and cached non-JSON responses have both been seen in the wild.
 */
export async function fetchJson<T>(url: string): Promise<T> {
    try {
        return await fetchJsonOnce<T>(url, 'default');
    } catch {
        return await fetchJsonOnce<T>(url, 'reload');
    }
}
