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

    public getLanguageId() {
        return this.compilerLanguageId;
    }

    public isCompatibleWithCMake(): boolean {
        if (this.compilerLanguageId !== 'c++' && this.compilerLanguageId !== 'c') {
            return false;
        } else {
            return true;
        }
    }

    public setLanguageId(id: string) {
        this.compilerLanguageId = id;
        // if (!this.isCompatibleWithCMake()) {
        //     this.isCMakeProject = false;
        // }
    }

    public isACMakeProject(): boolean {
        return this.isCompatibleWithCMake() && this.isCMakeProject;
    }

    public setAsCMakeProject(yes: boolean) {
        if (yes) {
            this.isCMakeProject = true;
        } else {
            this.isCMakeProject = false;
        }
    }

    private checkFileEditor(file: File) {
        if (file && file.editorId > 0) {
            const editor = this.hub.getEditorById(file.editorId);
            if (!editor) {
                file.isOpen = false;
                file.editorId = -1;
            }
        }
    }

    public getFileContents(file: File) {
        this.checkFileEditor(file);

        if (file.isOpen) {
            const editor = this.hub.getEditorById(file.editorId);
            return editor.getSource();
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
        const file = _.find(this.files, (file: File) => {
            return file.fileId === fileId;
        });

        this.checkFileEditor(file);

        return file;
    }

    public setAsMainSource(mainFileId: Number) {
        for (let file of this.files) {
            file.isMainSource = false;
        }

        var mainfile = this.getFileByFileId(mainFileId);
        mainfile.isMainSource = true;
    }

    private isValidFile(file: File): boolean {
        return (file.editorId > 0) || !!file.filename;
    }

    private filterOutNonsense() {
        this.files = _.filter(this.files, (file: File) => this.isValidFile(file));
    }

    public getFiles(): Array<FiledataPair> {
        this.filterOutNonsense();

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

    private isMainSourceFile(file: File): boolean {
        if (this.isCMakeProject) {
            if (file.filename === this.getDefaultMainCMakeFilename()) {
                this.setAsMainSource(file.fileId);
            }
        } else if (!file.isMainSource) {
            if (file.filename === this.getDefaultMainSourceFilename(this.compilerLanguageId)) {
                this.setAsMainSource(file.fileId);
            }
        }

        return file.isMainSource
    }

    public getMainSource(): string {
        var mainFile = _.find(this.files, (file: File) => {
            return file.isIncluded && this.isMainSourceFile(file);
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

    public getMainSourceEditorId(): number {
        const file: File = _.find(this.files, (file: File) => {
            return file.isIncluded && this.isMainSourceFile(file);
        });

        this.checkFileEditor(file);

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
            if (isRenamed) {
                this.includeByFileId(fileId);
            } else {
                file.isIncluded = false;
            }
        } else {
            file.isIncluded = true;
        }

        return;
    }

    public async includeByEditorId(editorId: number): Promise<void> {
        const file: File = this.getFileByEditorId(editorId);

        return this.includeByFileId(file.fileId);
    }

    public forEachOpenFile(callback: (File) => void) {
        this.filterOutNonsense();

        for (const file of this.files) {
            if (file.isOpen && file.editorId > 0) {
                callback(file);
            }
        }
    }

    public forEachFile(callback: (File) => void) {
        this.filterOutNonsense();

        for (const file of this.files) {
            callback(file);
        }
    }

    private getDefaultMainCMakeFilename() {
        return 'CMakeLists.txt';
    }

    private getDefaultMainSourceFilename(langId) {
        const lang = languages[langId];
        const ext0 = lang.extensions[0];
        return 'example' + ext0;
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
                if (langId === 'cmake') { 
                    suggestedFilename = this.getDefaultMainCMakeFilename();
                } else {
                    suggestedFilename = this.getDefaultMainSourceFilename(langId);
                }
            }
        }

        return suggestedFilename;
    }

    private fileExists(filename: string, excludeFile: File): boolean {
        return !!_.find(this.files, (file: File) => {
            return (file !== excludeFile) && (file.filename === filename);
        });
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
                    if (value !== '' && value[0] !== '/') {
                        if (!this.fileExists(value, file)) {
                            file.filename = value;

                            if (editor) {
                                editor.setCustomPaneName(file.filename);
                            }
    
                            resolve(true);
                        } else {
                            this.alertSystem.alert('Rename file', 'Filename already exists');
                            resolve(false);
                        }
                    } else {
                        this.alertSystem.alert('Rename file', 'Filename cannot be empty or start with a "/"');
                        resolve(false);
                    }
                },
                no: () => {
                    resolve(false);
                }
            });
        });
    }

    public async renameFileByEditorId(editorId: number): Promise<boolean> {
        var file = this.getFileByEditorId(editorId);

        return this.renameFile(file.fileId);
    }
}
