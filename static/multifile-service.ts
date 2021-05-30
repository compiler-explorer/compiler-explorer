import _ from 'underscore';
var options = require('./options');
var languages = options.languages;

export class File {
    fileId: number;
    isIncluded: boolean;
    isOpen: boolean;
    isMainSource: boolean;
    filename: string;
    content: string;
    editorId: number;
    langId: string;
}

export class FiledataPair {
    filename: string;
    contents: string;
}

export class MultifileService {
    private files: Array<File>;
    private compilerLanguageId: string;
    private isCMakeProject: boolean;
    private eventHub: any;
    private hub: any;
    private newFileId: number;
    private alertSystem: any;

    constructor(hub, eventHub, alertSystem, state) {
        this.eventHub = eventHub;
        this.hub = hub;
        this.alertSystem = alertSystem;

        this.isCMakeProject = state.isCMakeProject || false;
        this.compilerLanguageId = state.compilerLanguageId || '';
        this.files = state.files || [];
        this.newFileId = state.newFileId || 1;
    }

    public getState() {
        return {
            isCMakeProject: this.isCMakeProject,
            compilerLanguageId: this.compilerLanguageId,
            files: this.files,
            newFileId: this.newFileId,
        };
    }

    public setAsCMakeProject() {
        this.compilerLanguageId = 'c++';
        this.isCMakeProject = true;
    }

    public getFileContents(file: File) {
        if (file.isOpen) {
            const editor = this.hub.getEditorById(file.editorId);
            if (editor) {
                return editor.getSource();
            } else {
                file.isOpen = false;
                file.editorId = -1;
            }
        } else {
            return file.content;
        }
    }

    public isEditorPartOfProject(editorId: Number) {
        var found = _.find(this.files, (file: File) => {
            return (file.isIncluded) && file.isOpen && (editorId === file.editorId);
        });

        return !!found;
    }

    public getFileByFileId(fileId: Number) {
        return _.find(this.files, (file: File) => {
            return file.fileId === fileId;
        });
    }

    public setAsMainSource(mainFileId: Number) {
        for (let file of this.files) {
            file.isMainSource = false;
        }

        var mainfile = this.getFileByFileId(mainFileId);
        mainfile.isMainSource = true;
    }

    // public getEditorIdForMainsource() {
    //     let mainFile: File = null;
    //     if (this.isCMakeProject) {
    //         mainFile = _.find(this.files, (file: File) => {
    //             return file.isIncluded && (file.filename === 'example.cpp');
    //         });

    //         if (mainFile) return mainFile.editorId;
    //     } else {
    //         mainFile = _.find(this.files, (file: File) => {
    //             return file.isMainSource && file.isIncluded;
    //         });

    //         if (mainFile) return mainFile.editorId;
    //     }

    //     return false;
    // }

    public getFiles(): Array<FiledataPair> {
        var filtered = _.filter(this.files, (file: File) => {
            return !file.isMainSource && file.isIncluded;
        });

        return _.map(filtered, (file: File) => {
            return {
                filename: file.filename,
                contents: this.getFileContents(file),
            };
        });
    }

    public getMainSource(): string {
        var mainFile = _.find(this.files, (file: File) => {
            return file.isMainSource && file.isIncluded;
        });

        if (mainFile) {
            return this.getFileContents(mainFile);
        } else {
            return '';
        }
    }

    public getFileByEditorId(editorId: number): File {
        return _.find(this.files, (file: File) => {
            return file.editorId === editorId;
        });
    }

    public getEditorIdByFilename(filename: string): number {
        const file: File = _.find(this.files, (file: File) => {
            return file.isIncluded && (file.filename === filename);
        });

        return (file && file.editorId > 0) ? file.editorId : null;
    }

    public addFileForEditorId(editorId) {
        const file: File = {
            fileId: this.newFileId,
            isIncluded: false,
            isOpen: true,
            isMainSource: false,
            filename: '',
            content: '',
            editorId: editorId,
            langId: '',
        };

        this.newFileId++;
        this.files.push(file);
    }

    public removeFileByFileId(fileId: number): File {
        const file: File = this.getFileByFileId(fileId);

        this.files = this.files.filter((obj: File) => obj.fileId !== fileId);

        return file;
    }

    public async excludeByFileId(fileId: number): Promise<void> {
        const file: File = this.getFileByFileId(fileId);
        file.isIncluded = false;
    }

    public async includeByFileId(fileId: number): Promise<void> {
        const file: File = this.getFileByFileId(fileId);
        file.isIncluded = true;

        if (file.filename === '') {
            const isRenamed = await this.renameFile(fileId);
            if (isRenamed) this.includeByFileId(fileId);
        } else {
            file.isIncluded = true;

            if (file.filename === 'CMakeLists.txt') {
                this.setAsCMakeProject();
                this.setAsMainSource(fileId);
            }
        }
    }

    public forEachOpenFile(callback: (File) => void) {
        for (const file of this.files) {
            if (file.isOpen && file.editorId > 0) {
                callback(file);
            }
        }
    }

    public forEachFile(callback: (File) => void) {
        for (const file of this.files) {
            callback(file);
        }
    }

    private getSuggestedFilename(file: File, editor: any): string {
        let suggestedFilename = file.filename;
        if (file.filename === '') {
            let langId: string = file.langId;
            if (editor) {
                langId = editor.currentLanguage.id;
                if (editor.customPaneName) {
                    suggestedFilename = editor.customPaneName;
                }
            }

            if (!suggestedFilename) {
                const lang = languages[langId];
                const ext0 = lang.extensions[0];
                suggestedFilename = 'example' + ext0;
            }
        }

        return suggestedFilename;
    }

    public async renameFile(fileId: number): Promise<boolean> {
        var file = this.getFileByFileId(fileId);

        let editor: any = null;
        if (file.isOpen && file.editorId > 0) {
            editor = this.hub.getEditorById(file.editorId);
        }

        let suggestedFilename = this.getSuggestedFilename(file, editor);

        return new Promise((resolve) => {
            this.alertSystem.enterSomething('Rename file', 'Please enter a filename', suggestedFilename, {
                yes: (value) => {
                    file.filename = value;

                    if (editor) {
                        editor.setCustomPaneName(file.filename);
                    }

                    resolve(true);
                },
                no: () => {
                    resolve(false);
                }
            });
        });
    }
}
