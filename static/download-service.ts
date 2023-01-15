
import $ from 'jquery';

import {FiledataPair} from '../types/compilation/compilation.interfaces';

export type SourceAndFiles = {
    files: FiledataPair[];
    source: string;
};

export type FileToDownload = {
    url: string;
    filename: string;
    contents?: string;
};

export class IncludeDownloads {
    private toDownload: FileToDownload[] = [];
    private downloadPromises: Promise<FileToDownload>[] = [];

    private async startDownload(download) {
        this.downloadPromises.push(
            new Promise(resolve => {
                $.get(download.url, data => {
                    download.contents = data;
                    resolve(download);
                });
            })
        );
    }

    private getFilenameFromUrl(url) {
        const jsurl = new URL(url);
        const urlpath = jsurl.pathname;
        return jsurl.host + urlpath;
    }

    include(url) {
        let found = this.toDownload.find(prev => prev.url === url);
        if (!found) {
            const filename = this.getFilenameFromUrl(url);
            found = {
                url: url,
                filename: filename,
            };
            this.toDownload.push(found);
            this.startDownload(found);
        }
        return found;
    }

    async allDownloads(): Promise<FileToDownload[]> {
        return Promise.all(this.downloadPromises);
    }

    async allDownloadsAsFileDataPairs(): Promise<FiledataPair[]> {
        const downloads = await Promise.all(this.downloadPromises);

        return downloads.map(file => {
            return {
                filename: file.filename,
                contents: file.contents || '',
            };
        });
    }
}
