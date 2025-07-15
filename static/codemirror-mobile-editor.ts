// CodeMirror 6 implementation for mobile devices in Compiler Explorer
import {EditorView, basicSetup} from 'codemirror';
import {EditorState, Compartment} from '@codemirror/state';
import {cpp} from '@codemirror/lang-cpp';
import {javascript} from '@codemirror/lang-javascript';
import {python} from '@codemirror/lang-python';
import {rust} from '@codemirror/lang-rust';
import {go} from '@codemirror/lang-go';
import {java} from '@codemirror/lang-java';
import {Diagnostic, linter, lintGutter} from '@codemirror/lint';

import {ICompilerExplorerEditor, CompilerError, EditorOptions} from './editor-abstraction';

interface LanguageExtension {
    name: string;
    extension: () => any;
}

// Language mappings for CodeMirror 6
const LANGUAGE_EXTENSIONS: Record<string, LanguageExtension> = {
    'cpp': {name: 'C++', extension: cpp},
    'c': {name: 'C', extension: cpp}, // Use C++ for C (close enough)
    'cxx': {name: 'C++', extension: cpp},
    'cc': {name: 'C++', extension: cpp},
    'javascript': {name: 'JavaScript', extension: javascript},
    'js': {name: 'JavaScript', extension: javascript},
    'typescript': {name: 'TypeScript', extension: javascript}, // Use JS for TS
    'ts': {name: 'TypeScript', extension: javascript},
    'python': {name: 'Python', extension: python},
    'py': {name: 'Python', extension: python},
    'rust': {name: 'Rust', extension: rust},
    'rs': {name: 'Rust', extension: rust},
    'go': {name: 'Go', extension: go},
    'java': {name: 'Java', extension: java},
};

export class CodeMirrorMobileEditor implements ICompilerExplorerEditor {
    private view: EditorView;
    private container: HTMLElement;
    private languageCompartment: Compartment;
    private lintCompartment: Compartment;
    private currentLanguage: string;
    private changeCallbacks: (() => void)[] = [];
    private disposed = false;
    private currentErrors: CompilerError[] = [];

    constructor(container: HTMLElement, options: EditorOptions = {}) {
        this.container = container;
        this.currentLanguage = options.language || 'cpp';
        
        // Create compartments for dynamic reconfiguration
        this.languageCompartment = new Compartment();
        this.lintCompartment = new Compartment();

        // Create initial state
        const initialState = EditorState.create({
            doc: '',
            extensions: [
                basicSetup,
                this.languageCompartment.of(this.getLanguageExtension(this.currentLanguage)),
                this.lintCompartment.of([]),
                EditorView.updateListener.of(this.createUpdateListener()),
                EditorView.theme({
                    '&': {
                        fontSize: options.fontSize ? `${options.fontSize}px` : '14px',
                        height: '100%',
                    },
                    '.cm-content': {
                        padding: '10px',
                        fontFamily: 'Monaco, Menlo, "Ubuntu Mono", monospace',
                        minHeight: '200px',
                    },
                    '.cm-focused': {
                        outline: 'none',
                    },
                    '.cm-editor': {
                        height: '100%',
                    },
                    '.cm-scroller': {
                        fontFamily: 'Monaco, Menlo, "Ubuntu Mono", monospace',
                    },
                    // Error styling
                    '.cm-diagnostic-error': {
                        borderLeft: '3px solid #ff5555',
                        backgroundColor: 'rgba(255, 85, 85, 0.1)',
                    },
                    '.cm-diagnostic-warning': {
                        borderLeft: '3px solid #ffaa00',
                        backgroundColor: 'rgba(255, 170, 0, 0.1)',
                    },
                    '.cm-diagnostic-info': {
                        borderLeft: '3px solid #5555ff',
                        backgroundColor: 'rgba(85, 85, 255, 0.1)',
                    },
                }),
                // Enable readonly if specified
                ...(options.readOnly ? [EditorState.readOnly.of(true)] : []),
            ],
        });

        // Create the editor view
        this.view = new EditorView({
            state: initialState,
            parent: container,
        });

        console.log('CodeMirrorMobileEditor created with language:', this.currentLanguage);
    }

    private getLanguageExtension(languageId: string): any {
        const langConfig = LANGUAGE_EXTENSIONS[languageId.toLowerCase()];
        if (langConfig) {
            return langConfig.extension();
        }
        // Default to C++ if language not found
        return cpp();
    }

    private createUpdateListener() {
        return (update: any) => {
            if (update.docChanged) {
                // Notify change callbacks
                this.notifyChange();
            }
        };
    }

    private notifyChange(): void {
        if (this.disposed) return;
        this.changeCallbacks.forEach(callback => {
            try {
                callback();
            } catch (error) {
                console.error('Error in change callback:', error);
            }
        });
    }

    // ICompilerExplorerEditor implementation
    getValue(): string {
        if (this.disposed) throw new Error('Editor disposed');
        return this.view.state.doc.toString();
    }

    setValue(value: string): void {
        if (this.disposed) throw new Error('Editor disposed');
        
        const transaction = this.view.state.update({
            changes: {
                from: 0,
                to: this.view.state.doc.length,
                insert: value,
            },
        });
        
        this.view.dispatch(transaction);
    }

    setLanguage(language: string): void {
        if (this.disposed) throw new Error('Editor disposed');
        
        const normalizedLanguage = language.toLowerCase();
        if (this.currentLanguage === normalizedLanguage) return;
        
        this.currentLanguage = normalizedLanguage;
        
        // Update language extension
        const languageExtension = this.getLanguageExtension(normalizedLanguage);
        this.view.dispatch({
            effects: this.languageCompartment.reconfigure(languageExtension),
        });
        
        console.log('CodeMirror language changed to:', normalizedLanguage);
    }

    setErrors(errors: CompilerError[]): void {
        if (this.disposed) throw new Error('Editor disposed');
        
        this.currentErrors = [...errors];
        
        // Convert CompilerError[] to CodeMirror Diagnostic[]
        const diagnostics: Diagnostic[] = errors.map(error => {
            // Convert 1-based line numbers to 0-based positions
            const line = Math.max(0, error.line - 1);
            const doc = this.view.state.doc;
            const lineInfo = doc.line(Math.min(line + 1, doc.lines));
            const from = lineInfo.from + Math.max(0, error.column - 1);
            const to = Math.min(from + 1, lineInfo.to); // At least one character
            
            return {
                from,
                to,
                severity: error.severity,
                message: error.message,
            };
        });
        
        // Create linter that returns our diagnostics
        const errorLinter = linter(() => diagnostics);
        
        // Update lint compartment
        this.view.dispatch({
            effects: this.lintCompartment.reconfigure([
                errorLinter,
                lintGutter(),
            ]),
        });
    }

    onChange(callback: () => void): void {
        if (this.disposed) throw new Error('Editor disposed');
        this.changeCallbacks.push(callback);
    }

    dispose(): void {
        if (this.disposed) return;
        
        this.disposed = true;
        this.changeCallbacks = [];
        this.currentErrors = [];
        
        // Destroy CodeMirror view
        this.view.destroy();
        
        // Clear container
        this.container.innerHTML = '';
        
        console.log('CodeMirrorMobileEditor disposed');
    }

    // Additional methods for enhanced functionality
    focus(): void {
        if (!this.disposed) {
            this.view.focus();
        }
    }

    layout(): void {
        // CodeMirror 6 handles layout automatically, but we can force a refresh
        if (!this.disposed) {
            this.view.requestMeasure();
        }
    }

    getLanguage(): string {
        return this.currentLanguage;
    }

    getErrors(): CompilerError[] {
        return [...this.currentErrors];
    }

    isDisposed(): boolean {
        return this.disposed;
    }

    // Get the underlying CodeMirror view for advanced usage
    getCodeMirrorView(): EditorView | null {
        return this.disposed ? null : this.view;
    }
}