import {beforeEach, describe, expect, it} from 'vitest';
import {type CompilerError, MockEditor, createEditor} from '../static/editor-abstraction.js';

describe('Editor Abstraction Layer', () => {
    let editor: MockEditor;

    beforeEach(() => {
        editor = new MockEditor();
    });

    describe('Basic Content Operations', () => {
        it('should initialize with empty content', () => {
            expect(editor.getValue()).toBe('');
        });

        it('should set and get content correctly', () => {
            const testContent = 'int main() { return 0; }';
            editor.setValue(testContent);
            expect(editor.getValue()).toBe(testContent);
        });

        it('should handle empty string content', () => {
            editor.setValue('some content');
            editor.setValue('');
            expect(editor.getValue()).toBe('');
        });

        it('should handle multiline content', () => {
            const multilineContent = `#include <iostream>
int main() {
    std::cout << "Hello World" << std::endl;
    return 0;
}`;
            editor.setValue(multilineContent);
            expect(editor.getValue()).toBe(multilineContent);
        });
    });

    describe('Change Notifications', () => {
        it('should notify on content change', () => {
            let changeCount = 0;
            editor.onChange(() => changeCount++);

            editor.setValue('first change');
            expect(changeCount).toBe(1);

            editor.setValue('second change');
            expect(changeCount).toBe(2);
        });

        it('should not notify when setting same content', () => {
            let changeCount = 0;
            editor.setValue('initial content');
            editor.onChange(() => changeCount++);

            editor.setValue('initial content');
            expect(changeCount).toBe(0);
        });

        it('should support multiple change listeners', () => {
            let count1 = 0;
            let count2 = 0;
            editor.onChange(() => count1++);
            editor.onChange(() => count2++);

            editor.setValue('test content');
            expect(count1).toBe(1);
            expect(count2).toBe(1);
        });
    });

    describe('Language Support', () => {
        it('should set language', () => {
            editor.setLanguage('javascript');
            expect(editor.getLanguage()).toBe('javascript');
        });

        it('should initialize with cpp language', () => {
            expect(editor.getLanguage()).toBe('cpp');
        });
    });

    describe('Error Display', () => {
        it('should set and retrieve errors', () => {
            const errors: CompilerError[] = [
                {line: 1, column: 5, message: 'Syntax error', severity: 'error'},
                {line: 3, column: 10, message: 'Warning', severity: 'warning'},
            ];

            editor.setErrors(errors);
            const retrievedErrors = editor.getErrors();

            expect(retrievedErrors).toHaveLength(2);
            expect(retrievedErrors[0].message).toBe('Syntax error');
            expect(retrievedErrors[1].severity).toBe('warning');
        });

        it('should handle empty error list', () => {
            editor.setErrors([]);
            expect(editor.getErrors()).toHaveLength(0);
        });

        it('should replace previous errors', () => {
            const firstErrors: CompilerError[] = [{line: 1, column: 1, message: 'Error 1', severity: 'error'}];
            const secondErrors: CompilerError[] = [{line: 2, column: 2, message: 'Error 2', severity: 'warning'}];

            editor.setErrors(firstErrors);
            editor.setErrors(secondErrors);

            const errors = editor.getErrors();
            expect(errors).toHaveLength(1);
            expect(errors[0].message).toBe('Error 2');
        });
    });

    describe('Disposal', () => {
        it('should dispose cleanly', () => {
            editor.setValue('test content');
            editor.onChange(() => {});
            editor.setErrors([{line: 1, column: 1, message: 'Test', severity: 'error'}]);

            editor.dispose();
            expect(editor.isDisposed()).toBe(true);
        });

        it('should throw error when using disposed editor', () => {
            editor.dispose();

            expect(() => editor.getValue()).toThrow('Editor disposed');
            expect(() => editor.setValue('test')).toThrow('Editor disposed');
            expect(() => editor.setLanguage('js')).toThrow('Editor disposed');
            expect(() => editor.setErrors([])).toThrow('Editor disposed');
            expect(() => editor.onChange(() => {})).toThrow('Editor disposed');
        });
    });

    describe('Factory Function', () => {
        it('should create editor via factory', () => {
            // Skip DOM test if document is not available (backend testing)
            if (typeof document === 'undefined') {
                const factoryEditor = createEditor(null as any, {language: 'cpp'});
                expect(factoryEditor).toBeInstanceOf(MockEditor);
                return;
            }

            const container = document.createElement('div');
            const factoryEditor = createEditor(container, {language: 'cpp'});

            expect(factoryEditor).toBeInstanceOf(MockEditor);
            expect(typeof factoryEditor.getValue).toBe('function');
            expect(typeof factoryEditor.setValue).toBe('function');
        });
    });

    describe('Editor Contract Compliance', () => {
        it('should implement all required interface methods', () => {
            const requiredMethods = ['getValue', 'setValue', 'setLanguage', 'setErrors', 'onChange', 'dispose'];

            requiredMethods.forEach(method => {
                expect(typeof editor[method]).toBe('function');
            });
        });

        it('should maintain API consistency across operations', () => {
            // Simulate typical usage pattern
            editor.setLanguage('cpp');
            editor.setValue('#include <iostream>');

            let changeTriggered = false;
            editor.onChange(() => {
                changeTriggered = true;
            });

            editor.setValue('int main() {}');
            expect(changeTriggered).toBe(true);
            expect(editor.getValue()).toBe('int main() {}');

            editor.setErrors([{line: 1, column: 1, message: 'Test error', severity: 'error'}]);
            expect(editor.getErrors()).toHaveLength(1);
        });
    });
});
