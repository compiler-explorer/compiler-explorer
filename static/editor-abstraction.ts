// Editor abstraction interface for Compiler Explorer
// Provides unified API for Monaco and CodeMirror editors

export interface CompilerError {
    line: number;
    column: number;
    message: string;
    severity: 'error' | 'warning' | 'info';
}

export interface EditorOptions {
    language?: string;
    readOnly?: boolean;
    theme?: string;
    fontSize?: number;
}

export interface ICompilerExplorerEditor {
    // Core content operations
    getValue(): string;
    setValue(value: string): void;

    // Language and configuration
    setLanguage(language: string): void;

    // Error display
    setErrors(errors: CompilerError[]): void;

    // Change notifications
    onChange(callback: () => void): void;

    // Cleanup
    dispose(): void;
}

// Mock editor implementation for testing
export class MockEditor implements ICompilerExplorerEditor {
    private content = '';
    private language = 'cpp';
    private errors: CompilerError[] = [];
    private changeCallbacks: (() => void)[] = [];
    private disposed = false;

    getValue(): string {
        if (this.disposed) throw new Error('Editor disposed');
        return this.content;
    }

    setValue(value: string): void {
        if (this.disposed) throw new Error('Editor disposed');
        const oldContent = this.content;
        this.content = value;
        if (oldContent !== value) {
            this.notifyChange();
        }
    }

    setLanguage(language: string): void {
        if (this.disposed) throw new Error('Editor disposed');
        this.language = language;
    }

    setErrors(errors: CompilerError[]): void {
        if (this.disposed) throw new Error('Editor disposed');
        this.errors = [...errors];
    }

    onChange(callback: () => void): void {
        if (this.disposed) throw new Error('Editor disposed');
        this.changeCallbacks.push(callback);
    }

    dispose(): void {
        this.disposed = true;
        this.changeCallbacks = [];
        this.errors = [];
        this.content = '';
    }

    // Test helpers
    getLanguage(): string {
        return this.language;
    }
    getErrors(): CompilerError[] {
        return [...this.errors];
    }
    isDisposed(): boolean {
        return this.disposed;
    }

    private notifyChange(): void {
        this.changeCallbacks.forEach(callback => callback());
    }
}

// Factory function for creating editors
export function createEditor(container: HTMLElement, options: EditorOptions): ICompilerExplorerEditor {
    // For now, just return mock editor for testing
    // Later this will detect mobile and return appropriate implementation
    return new MockEditor();
}
