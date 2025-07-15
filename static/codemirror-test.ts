// CodeMirror 6 test page entry point
import {EditorView, basicSetup} from 'codemirror';
import {cpp} from '@codemirror/lang-cpp';
// import {createEditor, createEditorSync} from './editor-abstraction';
import {CodeMirrorMobileEditor} from './codemirror-mobile-editor';

console.log('CodeMirror test script loaded, imports successful');
console.log('EditorView:', EditorView);
console.log('basicSetup:', basicSetup);
console.log('cpp:', cpp);

function initializeCodeMirrorTest() {
    console.log('CodeMirror 6 test page loaded - DOM ready');
    
    // Test 1: Mock Editor
    const mockContainer = document.getElementById('mock-editor');
    if (mockContainer) {
        const mockTextarea = document.createElement('textarea');
        mockTextarea.value = `// Mock Editor (textarea-based)
#include <iostream>
int main() {
    std::cout << "Hello from Mock Editor!" << std::endl;
    return 0;
}`;
        mockTextarea.style.cssText = 'width: 100%; height: 250px; font-family: monospace; font-size: 14px; border: 1px solid #ddd; padding: 10px;';
        mockContainer.appendChild(mockTextarea);
    }
    
    // Test 2: CodeMirror 6
    const cmContainer = document.getElementById('codemirror-editor');
    if (cmContainer) {
        try {
            // Exact official example
            new EditorView({
                parent: cmContainer,
                doc: `#include <iostream>
int main() {
    std::cout << "Hello World!" << std::endl;
    return 0;
}`,
                extensions: [basicSetup, cpp()]
            });
            
            console.log('CodeMirror 6 editor created successfully');
            
        } catch (error) {
            console.error('Failed to create CodeMirror editor:', error);
        }
    }
    
    // Test 3: CodeMirror Mobile Editor Class
    const mobileContainer = document.getElementById('mobile-editor');
    if (mobileContainer) {
        try {
            console.log('Creating CodeMirrorMobileEditor...');
            
            const mobileEditor = new CodeMirrorMobileEditor(mobileContainer, {
                language: 'cpp',
                fontSize: 14,
            });
            
            mobileEditor.setValue(`// CodeMirror Mobile Editor Test
#include <iostream>
#include <vector>

int main() {
    std::vector<int> numbers = {1, 2, 3, 4, 5};
    
    for (const auto& num : numbers) {
        std::cout << "Number: " << num << std::endl;
    }
    
    return 0;
}`);

            // Test language switching
            setTimeout(() => {
                console.log('Testing language switch to JavaScript...');
                mobileEditor.setLanguage('javascript');
                mobileEditor.setValue(`// JavaScript test
function fibonacci(n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

console.log('Fibonacci(10):', fibonacci(10));`);
            }, 2000);

            // Test error display
            setTimeout(() => {
                console.log('Testing error display...');
                mobileEditor.setErrors([
                    {
                        line: 3,
                        column: 10,
                        message: 'Example compilation error',
                        severity: 'error'
                    },
                    {
                        line: 8,
                        column: 5,
                        message: 'Example warning',
                        severity: 'warning'
                    }
                ]);
            }, 4000);

            // Test change callback
            mobileEditor.onChange(() => {
                console.log('Mobile editor content changed, new length:', mobileEditor.getValue().length);
            });
            
            console.log('CodeMirrorMobileEditor created successfully');
        } catch (error) {
            console.error('Failed to create CodeMirrorMobileEditor:', error);
        }
    }
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeCodeMirrorTest);
} else {
    initializeCodeMirrorTest();
}