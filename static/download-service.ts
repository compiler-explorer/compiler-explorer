// Copyright (c) 2023, Compiler Explorer Authors
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

import {FiledataPair} from '../types/compilation/compilation.interfaces';

export type SourceAndFiles = {
    files: FiledataPair[];
    source: string;
};

export type FileToDownload = {
    url: string;
    filename: string;
    contents?: string;
    downloadFailed?: boolean;
};

export class IncludeDownloads {
    private toDownload: FileToDownload[] = [];
    private downloadPromises: Promise<FileToDownload>[] = [];

    private async doDownload(download): Promise<FileToDownload> {
        try {
            const response = await fetch(download.url);
            if (response.status >= 400) {
                download.downloadFailed = true;
            } else {
                download.contents = await response.text();
            }
        } catch {
            // exceptions happens on for example dns errors
            download.downloadFailed = true;
        }
        return download;
    }

    private async startDownload(download) {
        this.downloadPromises.push(this.doDownload(download));
    }

    private getFilenameFromUrl(url: string): string {
        const jsurl = new URL(url);
        const urlpath = jsurl.pathname;
        return jsurl.host + urlpath;
    }

    include(url: string): FileToDownload {
        let found = this.toDownload.find(prev => prev.url === url);
        if (!found) {
            try {
                const filename = this.getFilenameFromUrl(url);
                found = {
                    url: url,
                    filename: filename,
                };
                this.toDownload.push(found);
                this.startDownload(found);
            } catch (err: any) {
                return {
                    url: 'Unknown url',
                    filename: err.message,
                    downloadFailed: true,
                };
            }
        }
        return found;
    }

    async allDownloads(): Promise<FileToDownload[]> {
        return Promise.all(this.downloadPromises);
    }

    async allDownloadsAsFileDataPairs(): Promise<FiledataPair[]> {
        const downloads = await Promise.all(this.downloadPromises);

        return downloads
            .filter(file => !file.downloadFailed)
            .map(file => {
                return {
                    filename: file.filename,
                    contents: file.contents || '',
                };
            });
    }
}
