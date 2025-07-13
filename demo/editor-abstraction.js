// Browser-compatible version of editor abstraction for demo

// Mock editor implementation that creates a visual editor
class MockEditor {
    constructor(container, options = {}) {
        this.container = container;
        this.content = '';
        this.language = options.language || 'cpp';
        this.errors = [];
        this.changeCallbacks = [];
        this.disposed = false;
        
        this.createEditor();
    }

    createEditor() {
        // Create a visual textarea-based editor
        this.container.innerHTML = `
            <div class="mock-editor">
                <div class="editor-toolbar">
                    <span class="language-indicator">Language: ${this.language}</span>
                    <div class="error-count" id="error-count">No errors</div>
                </div>
                <textarea 
                    id="editor-content" 
                    placeholder="Enter your code here..."
                    style="width: 100%; height: 300px; font-family: monospace; font-size: 14px; border: 1px solid #ccc; padding: 10px;"
                ></textarea>
                <div class="editor-errors" id="editor-errors"></div>
            </div>
        `;

        this.textarea = this.container.querySelector('#editor-content');
        this.errorCountElement = this.container.querySelector('#error-count');
        this.errorsElement = this.container.querySelector('#editor-errors');
        this.languageIndicator = this.container.querySelector('.language-indicator');

        // Setup event listeners
        this.textarea.addEventListener('input', () => {
            const oldContent = this.content;
            this.content = this.textarea.value;
            if (oldContent !== this.content) {
                this.notifyChange();
            }
        });
    }

    getValue() {
        if (this.disposed) throw new Error('Editor disposed');
        return this.content;
    }

    setValue(value) {
        if (this.disposed) throw new Error('Editor disposed');
        const oldContent = this.content;
        this.content = value;
        if (this.textarea) {
            this.textarea.value = value;
        }
        if (oldContent !== value) {
            this.notifyChange();
        }
    }

    setLanguage(language) {
        if (this.disposed) throw new Error('Editor disposed');
        this.language = language;
        if (this.languageIndicator) {
            this.languageIndicator.textContent = `Language: ${language}`;
        }
    }

    setErrors(errors) {
        if (this.disposed) throw new Error('Editor disposed');
        this.errors = [...errors];
        this.updateErrorDisplay();
    }

    onChange(callback) {
        if (this.disposed) throw new Error('Editor disposed');
        this.changeCallbacks.push(callback);
    }

    dispose() {
        this.disposed = true;
        this.changeCallbacks = [];
        this.errors = [];
        this.content = '';
        if (this.container) {
            this.container.innerHTML = '<div>Editor disposed</div>';
        }
    }

    updateErrorDisplay() {
        if (!this.errorCountElement || !this.errorsElement) return;
        
        if (this.errors.length === 0) {
            this.errorCountElement.textContent = 'No errors';
            this.errorCountElement.style.color = 'green';
            this.errorsElement.innerHTML = '';
        } else {
            this.errorCountElement.textContent = `${this.errors.length} error(s)`;
            this.errorCountElement.style.color = 'red';
            
            const errorHtml = this.errors.map(error => 
                `<div class="error-item" style="padding: 5px; margin: 2px 0; background: #ffebee; border-left: 3px solid red;">
                    <strong>Line ${error.line}, Column ${error.column}:</strong> ${error.message} (${error.severity})
                </div>`
            ).join('');
            
            this.errorsElement.innerHTML = errorHtml;
        }
    }

    notifyChange() {
        this.changeCallbacks.forEach(callback => {
            try {
                callback();
            } catch (e) {
                console.error('Error in change callback:', e);
            }
        });
    }

    // Test helpers for demo
    getLanguage() { return this.language; }
    getErrors() { return [...this.errors]; }
    isDisposed() { return this.disposed; }
}

// Monaco Editor Wrapper
class MonacoEditorWrapper {
    constructor(container, options = {}) {
        this.container = container;
        this.language = options.language || 'cpp';
        this.errors = [];
        this.changeCallbacks = [];
        this.disposed = false;
        
        this.createEditor();
    }

    async createEditor() {
        try {
            // Create Monaco editor
            this.editor = monaco.editor.create(this.container, {
                value: '',
                language: this.getMonacoLanguage(this.language),
                theme: 'vs',
                automaticLayout: true,
                minimap: { enabled: false },
                scrollBeyondLastLine: false,
                fontSize: 14
            });

            // Setup change listener
            this.editor.onDidChangeModelContent(() => {
                this.notifyChange();
            });

            this.updateErrorDisplay();
        } catch (error) {
            console.error('Failed to create Monaco editor:', error);
            // Fallback to mock editor
            this.fallbackEditor = new MockEditor(this.container, { language: this.language });
        }
    }

    getMonacoLanguage(lang) {
        const langMap = {
            'cpp': 'cpp',
            'c': 'c',
            'javascript': 'javascript',
            'python': 'python',
            'rust': 'rust',
            'go': 'go'
        };
        return langMap[lang] || 'plaintext';
    }

    getValue() {
        if (this.disposed) throw new Error('Editor disposed');
        if (this.fallbackEditor) return this.fallbackEditor.getValue();
        return this.editor ? this.editor.getValue() : '';
    }

    setValue(value) {
        if (this.disposed) throw new Error('Editor disposed');
        if (this.fallbackEditor) return this.fallbackEditor.setValue(value);
        if (this.editor) {
            this.editor.setValue(value);
        }
    }

    setLanguage(language) {
        if (this.disposed) throw new Error('Editor disposed');
        this.language = language;
        if (this.fallbackEditor) return this.fallbackEditor.setLanguage(language);
        if (this.editor) {
            const model = this.editor.getModel();
            if (model) {
                monaco.editor.setModelLanguage(model, this.getMonacoLanguage(language));
            }
        }
    }

    setErrors(errors) {
        if (this.disposed) throw new Error('Editor disposed');
        this.errors = [...errors];
        if (this.fallbackEditor) return this.fallbackEditor.setErrors(errors);
        this.updateErrorDisplay();
    }

    onChange(callback) {
        if (this.disposed) throw new Error('Editor disposed');
        this.changeCallbacks.push(callback);
        if (this.fallbackEditor) return this.fallbackEditor.onChange(callback);
    }

    dispose() {
        this.disposed = true;
        this.changeCallbacks = [];
        this.errors = [];
        if (this.fallbackEditor) {
            this.fallbackEditor.dispose();
        } else if (this.editor) {
            this.editor.dispose();
        }
        if (this.container) {
            this.container.innerHTML = '<div>Monaco Editor disposed</div>';
        }
    }

    updateErrorDisplay() {
        if (!this.editor) return;
        
        // Convert errors to Monaco markers
        const markers = this.errors.map(error => ({
            startLineNumber: error.line,
            startColumn: error.column,
            endLineNumber: error.line,
            endColumn: error.column + 10,
            message: error.message,
            severity: error.severity === 'error' ? monaco.MarkerSeverity.Error : 
                     error.severity === 'warning' ? monaco.MarkerSeverity.Warning : 
                     monaco.MarkerSeverity.Info
        }));

        const model = this.editor.getModel();
        if (model) {
            monaco.editor.setModelMarkers(model, 'compiler-errors', markers);
        }
    }

    notifyChange() {
        this.changeCallbacks.forEach(callback => {
            try {
                callback();
            } catch (e) {
                console.error('Error in change callback:', e);
            }
        });
    }

    // Test helpers
    getLanguage() { return this.language; }
    getErrors() { return [...this.errors]; }
    isDisposed() { return this.disposed; }
}

// CodeMirror Editor Wrapper
class CodeMirrorEditorWrapper {
    constructor(container, options = {}) {
        this.container = container;
        this.language = options.language || 'cpp';
        this.errors = [];
        this.changeCallbacks = [];
        this.disposed = false;
        
        this.createEditor();
    }

    async createEditor() {
        try {
            // Wait for CodeMirror 6 to load
            while (!window.cm6Loaded || !window.CodeMirror6) {
                await new Promise(resolve => setTimeout(resolve, 100));
            }

            const { EditorView, EditorState } = window.CodeMirror6;
            
            // Create minimal editor without any extensions that cause conflicts
            this.view = new EditorView({
                state: EditorState.create({
                    doc: '',
                    extensions: [
                        EditorView.updateListener.of(update => {
                            if (update.docChanged) {
                                this.notifyChange();
                            }
                        }),
                        EditorView.theme({
                            '&': { height: '300px', border: '1px solid #ddd' },
                            '.cm-content': { padding: '10px', fontFamily: 'monospace', fontSize: '14px' },
                            '.cm-focused': { outline: 'none' },
                            '.cm-editor': { height: '100%' },
                            '.cm-scroller': { fontFamily: 'inherit' }
                        })
                    ]
                }),
                parent: this.container
            });

            this.updateErrorDisplay();
            console.log('CodeMirror 6 created successfully');
        } catch (error) {
            console.error('Failed to create CodeMirror editor:', error);
            // Fallback to mock editor
            this.fallbackEditor = new MockEditor(this.container, { language: this.language });
        }
    }

    getLanguageExtension(lang, langs) {
        // For demo simplicity, return empty array (no language-specific extensions)
        // In a full implementation, this would load language support
        return [];
    }

    getValue() {
        if (this.disposed) throw new Error('Editor disposed');
        if (this.fallbackEditor) return this.fallbackEditor.getValue();
        return this.view ? this.view.state.doc.toString() : '';
    }

    setValue(value) {
        if (this.disposed) throw new Error('Editor disposed');
        if (this.fallbackEditor) return this.fallbackEditor.setValue(value);
        if (this.view) {
            this.view.dispatch({
                changes: {
                    from: 0,
                    to: this.view.state.doc.length,
                    insert: value
                }
            });
        }
    }

    setLanguage(language) {
        if (this.disposed) throw new Error('Editor disposed');
        this.language = language;
        if (this.fallbackEditor) return this.fallbackEditor.setLanguage(language);
        // Note: Dynamic language switching in CodeMirror requires compartments
        // For demo purposes, we'll just track the language
    }

    setErrors(errors) {
        if (this.disposed) throw new Error('Editor disposed');
        this.errors = [...errors];
        if (this.fallbackEditor) return this.fallbackEditor.setErrors(errors);
        this.updateErrorDisplay();
    }

    onChange(callback) {
        if (this.disposed) throw new Error('Editor disposed');
        this.changeCallbacks.push(callback);
        if (this.fallbackEditor) return this.fallbackEditor.onChange(callback);
    }

    dispose() {
        this.disposed = true;
        this.changeCallbacks = [];
        this.errors = [];
        if (this.fallbackEditor) {
            this.fallbackEditor.dispose();
        } else if (this.view) {
            this.view.destroy();
        }
        if (this.container) {
            this.container.innerHTML = '<div>CodeMirror Editor disposed</div>';
        }
    }

    updateErrorDisplay() {
        // For actual CodeMirror, errors would be shown as decorations
        // For demo purposes, just log them
        console.log('CodeMirror errors:', this.errors);
    }

    notifyChange() {
        this.changeCallbacks.forEach(callback => {
            try {
                callback();
            } catch (e) {
                console.error('Error in change callback:', e);
            }
        });
    }

    // Test helpers
    getLanguage() { return this.language; }
    getErrors() { return [...this.errors]; }
    isDisposed() { return this.disposed; }
}

// Factory function for creating editors
function createEditor(container, options = {}) {
    const editorType = options.editorType || 'mock';
    
    switch (editorType) {
        case 'monaco':
            return new MonacoEditorWrapper(container, options);
        case 'codemirror':
            return new CodeMirrorEditorWrapper(container, options);
        case 'mock':
        default:
            return new MockEditor(container, options);
    }
}

// Export for browser use
window.EditorAbstraction = {
    MockEditor,
    MonacoEditorWrapper,
    CodeMirrorEditorWrapper,
    createEditor
};