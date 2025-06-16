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

import {execSync} from 'node:child_process';
import * as fs from 'node:fs';
import * as net from 'node:net';
import * as os from 'node:os';
import * as path from 'node:path';
import {describe, expect, it} from 'vitest';

describe('HeaptrackWrapper FD behavior tests', () => {
    it('should verify that net.Socket takes ownership of file descriptors', async () => {
        // Skip on Windows as pipes work differently there
        if (process.platform === 'win32') {
            return;
        }

        const tmpDir = os.tmpdir();
        const pipePath = path.join(tmpDir, `test_pipe_${process.pid}_${Date.now()}`);

        try {
            // Create a named pipe
            execSync(`mkfifo "${pipePath}"`);

            // Open the pipe with O_NONBLOCK (like HeaptrackWrapper does)
            const O_NONBLOCK = 0x800; // Linux O_NONBLOCK
            const O_RDWR = fs.constants.O_RDWR;

            const fd = fs.openSync(pipePath, O_RDWR | O_NONBLOCK);

            // Verify FD is valid before creating socket
            expect(() => fs.fstatSync(fd)).not.toThrow();

            // Create a net.Socket with the pipe FD (like HeaptrackWrapper)
            const socket = new net.Socket({fd: fd, readable: true, writable: true});

            // Set up a promise to wait for socket close
            const socketClosed = new Promise<void>(resolve => {
                socket.on('close', resolve);
            });

            // Destroy the socket
            socket.destroy();

            // Wait for socket to close
            await socketClosed;

            // Verify that the FD has been closed by the socket
            // Attempting to use the FD should throw EBADF
            expect(() => fs.fstatSync(fd)).toThrow(/EBADF/);

            // Attempting to manually close should also fail with EBADF
            // This confirms the socket already closed it
            await expect(
                new Promise((resolve, reject) => {
                    fs.close(fd, err => {
                        if (err) reject(err);
                        else resolve(true);
                    });
                }),
            ).rejects.toThrow(/EBADF/);
        } finally {
            // Clean up the pipe
            try {
                fs.unlinkSync(pipePath);
            } catch {
                // Ignore cleanup errors
            }
        }
    });

    it('should verify net.Socket closes FDs on destroy - critical for HeaptrackWrapper', async () => {
        // Skip on Windows as pipes work differently there
        if (process.platform === 'win32') {
            return;
        }

        // This test verifies the critical assumption that HeaptrackWrapper relies on:
        // When a net.Socket is created with an FD and then destroyed, it closes that FD.
        // If this behavior changes in future Node.js versions, HeaptrackWrapper will break.

        return new Promise<void>((resolve, reject) => {
            // Create a pair of connected sockets
            const server = net.createServer(socket => {
                // Get the raw FD from the socket
                const fd = (socket as any)._handle?.fd;

                if (!fd) {
                    // If we can't get the FD, skip the test but log a warning
                    console.warn('Could not access socket._handle.fd - Node.js internals may have changed');
                    server.close();
                    resolve();
                    return;
                }

                // Verify FD is valid before destroying the socket
                try {
                    fs.fstatSync(fd);
                } catch (err) {
                    reject(new Error('FD should be valid before socket destroy'));
                    return;
                }

                // Now destroy the socket - this is what HeaptrackWrapper does
                socket.on('close', () => {
                    // After destroy, the FD should be closed
                    try {
                        fs.fstatSync(fd);
                        reject(
                            new Error(
                                'FD should be closed after socket.destroy() - HeaptrackWrapper assumption violated!',
                            ),
                        );
                    } catch (err: any) {
                        if (err.code === 'EBADF') {
                            // Good! The FD was closed as expected
                            server.close();
                            resolve();
                        } else {
                            reject(err);
                        }
                    }
                });

                socket.destroy();
            });

            server.on('error', reject);

            server.listen(0, () => {
                const client = net.connect((server.address() as net.AddressInfo).port);
                client.on('error', () => {
                    // Ignore client errors - we're destroying the socket anyway
                });
            });
        });
    });
});
