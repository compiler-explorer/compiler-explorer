import _ from 'underscore';
import path from 'path';
var options = require('./options');
var languages = options.languages;
var JSZip = require('jszip');

export class MultifileFile {
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

export class MultifileServiceState {
    isCMakeProject: boolean;
    compilerLanguageId: string;
    files: MultifileFile[];
    newFileId: number;
}

export class MultifileService {
    private files: Array<MultifileFile>;
    private compilerLanguageId: string;
    private isCMakeProject: boolean;
    private hub: any;
    private newFileId: number;
    private alertSystem: any;
    private validExtraFilenameExtensions: string[];
    private defaultLangIdUnknownExt: string;
    private cmakeLangId: string;
    private cmakeMainSourceFilename: string;
    private maxFilesize: number;

    constructor(hub, alertSystem, state: MultifileServiceState) {
        this.hub = hub;
        this.alertSystem = alertSystem;

        this.isCMakeProject = state.isCMakeProject || false;
        this.compilerLanguageId = state.compilerLanguageId || '';
        this.files = state.files || [];
        this.newFileId = state.newFileId || 1;

        this.validExtraFilenameExtensions = ['.txt', '.md', '.rst', '.sh', '.cmake', '.in'];
        this.defaultLangIdUnknownExt = 'c++';
        this.cmakeLangId = 'cmake';
        this.cmakeMainSourceFilename = 'CMakeLists.txt';
        this.maxFilesize = 1024000;
    }

    private isHiddenFile(filename: string): boolean {
        return (filename.length > 0 && filename[0] === '.');
    }

    private isValidFilename(filename: string): boolean {
        if (this.isHiddenFile(filename)) return false;

        const filenameExt = path.extname(filename);
        if (this.validExtraFilenameExtensions.includes(filenameExt)) {
            return true;
        }

        return _.any(languages, (lang) => {
            return lang.extensions.includes(filenameExt);
        });
    }

    private isCMakeFile(filename: string): boolean {
        const filenameExt = path.extname(filename);
        if (filenameExt === '.cmake' || filenameExt === '.in') {
            return true;
        }

        return path.basename(filename) === this.cmakeMainSourceFilename;
    }

    private getLanguageIdFromFilename(filename: string): string {
        const filenameExt = path.extname(filename);

        const possibleLang = _.filter(languages, (lang) => {
            return lang.extensions.includes(filenameExt);
        });

        if (possibleLang.length > 0) {
            return possibleLang[0].id;
        }

        if (this.isCMakeFile(filename)) {
            return this.cmakeLangId;
        }

        return this.defaultLangIdUnknownExt;
    }

    public async loadProjectFromFile(f, callback) {
        this.files = [];
        this.newFileId = 1;

        const zipFilename = path.basename(f.name, '.zip');
        const mainSourcefilename = this.getDefaultMainCMakeFilename();

        const zip = await JSZip.loadAsync(f);

        zip.forEach(async (relativePath, zipEntry) => {
            if (!zipEntry.dir) {
                let removeFromName = 0;
                if (relativePath.indexOf(zipFilename) === 0) {
                    removeFromName = zipFilename.length + 1;
                }

                const properName = relativePath.substring(removeFromName);
                if (!this.isValidFilename(properName)) {
                    return;
                }

                let content = await zip.file(zipEntry.name).async("string");
                if (content.length > this.maxFilesize) {
                    return;
                }

                // remove utf8-bom characters
                content = content.replace(/^(\ufeff)/, '');

                const file: MultifileFile = {
                    fileId: this.newFileId,
                    filename: properName,
                    isIncluded: true,
                    isOpen: false,
                    editorId: -1,
                    isMainSource: properName === mainSourcefilename,
                    content: content,
                    langId: this.getLanguageIdFromFilename(properName),
                };

                this.addFile(file);
                callback(file);
            }
        });
    }

    public async saveProjectToZipfile(callback: (any) => void) {
        const zip = new JSZip();

        this.forEachFile((file: MultifileFile) => {
            if (file.isIncluded) {
                zip.file(file.filename, this.getFileContents(file));
            }
        })

        zip.generateAsync({type:"blob"}).then((blob) => {
            callback(blob);
        }, (err) => {
            throw err;
        });
    }

    public getState(): MultifileServiceState {
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

    private checkFileEditor(file: MultifileFile) {
        if (file && file.editorId > 0) {
            const editor = this.hub.getEditorById(file.editorId);
            if (!editor) {
                file.isOpen = false;
                file.editorId = -1;
            }
        }
    }

    public getFileContents(file: MultifileFile) {
        this.checkFileEditor(file);

        if (file.isOpen) {
            const editor = this.hub.getEditorById(file.editorId);
            return editor.getSource();
        } else {
            return file.content;
        }
    }

    public isEditorPartOfProject(editorId: Number) {
        var found = _.find(this.files, (file: MultifileFile) => {
            return (file.isIncluded) && file.isOpen && (editorId === file.editorId);
        });

        return !!found;
    }

    public getFileByFileId(fileId: Number) {
        const file = _.find(this.files, (file: MultifileFile) => {
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

    private isValidFile(file: MultifileFile): boolean {
        return (file.editorId > 0) || !!file.filename;
    }

    private filterOutNonsense() {
        this.files = _.filter(this.files, (file: MultifileFile) => this.isValidFile(file));
    }

    public getFiles(): Array<FiledataPair> {
        this.filterOutNonsense();

        var filtered = _.filter(this.files, (file: MultifileFile) => {
            return !file.isMainSource && file.isIncluded;
        });

        return _.map(filtered, (file: MultifileFile) => {
            return {
                filename: file.filename,
                contents: this.getFileContents(file),
            };
        });
    }

    private isMainSourceFile(file: MultifileFile): boolean {
        if (this.isCMakeProject) {
            if (file.filename === this.getDefaultMainCMakeFilename()) {
                this.setAsMainSource(file.fileId);
            } else {
                return false;
            }
        } else {
            if (file.filename === this.getDefaultMainSourceFilename(this.compilerLanguageId)) {
                this.setAsMainSource(file.fileId);
            } else {
                return false;
            }
        }

        return file.isMainSource;
    }

    public getMainSource(): string {
        var mainFile = _.find(this.files, (file: MultifileFile) => {
            return file.isIncluded && this.isMainSourceFile(file);
        });

        if (mainFile) {
            return this.getFileContents(mainFile);
        } else {
            return '';
        }
    }

    public getFileByEditorId(editorId: number): MultifileFile {
        return _.find(this.files, (file: MultifileFile) => {
            return file.editorId === editorId;
        });
    }

    public getEditorIdByFilename(filename: string): number {
        const file: MultifileFile = _.find(this.files, (file: MultifileFile) => {
            return file.isIncluded && (file.filename === filename);
        });

        return (file && file.editorId > 0) ? file.editorId : null;
    }

    public getMainSourceEditorId(): number {
        const file: MultifileFile = _.find(this.files, (file: MultifileFile) => {
            return file.isIncluded && this.isMainSourceFile(file);
        });

        this.checkFileEditor(file);

        return (file && file.editorId > 0) ? file.editorId : null;
    }

    private addFile(file: MultifileFile) {
        this.newFileId++;
        this.files.push(file);
    }

    public addFileForEditorId(editorId: number) {
        const file: MultifileFile = {
            fileId: this.newFileId,
            isIncluded: false,
            isOpen: true,
            isMainSource: false,
            filename: '',
            content: '',
            editorId: editorId,
            langId: '',
        };

        this.addFile(file);
    }

    public removeFileByFileId(fileId: number): MultifileFile {
        const file: MultifileFile = this.getFileByFileId(fileId);

        this.files = this.files.filter((obj: MultifileFile) => obj.fileId !== fileId);

        return file;
    }

    public async excludeByFileId(fileId: number): Promise<void> {
        const file: MultifileFile = this.getFileByFileId(fileId);
        file.isIncluded = false;
    }

    public async includeByFileId(fileId: number): Promise<void> {
        const file: MultifileFile = this.getFileByFileId(fileId);
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
        const file: MultifileFile = this.getFileByEditorId(editorId);

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
        return this.cmakeMainSourceFilename;
    }

    private getDefaultMainSourceFilename(langId) {
        const lang = languages[langId];
        const ext0 = lang.extensions[0];
        return 'example' + ext0;
    }

    private getSuggestedFilename(file: MultifileFile, editor: any): string {
        let suggestedFilename = file.filename;
        if (file.filename === '') {
            let langId: string = file.langId;
            if (editor) {
                langId = editor.currentLanguage.id;
                if (editor.filename) {
                    suggestedFilename = editor.filename;
                }
            }

            if (!suggestedFilename) {
                if (langId === this.cmakeLangId) {
                    suggestedFilename = this.getDefaultMainCMakeFilename();
                } else {
                    suggestedFilename = this.getDefaultMainSourceFilename(langId);
                }
            }
        }

        return suggestedFilename;
    }

    private fileExists(filename: string, excludeFile: MultifileFile): boolean {
        return !!_.find(this.files, (file: MultifileFile) => {
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
                                editor.setFilename(file.filename);
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
