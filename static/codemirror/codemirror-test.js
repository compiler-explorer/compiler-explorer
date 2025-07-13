// Import CodeMirror from our bundled modules
import {EditorView, basicSetup} from '/codemirror/codemirror.bundle.js';
import {EditorState} from '/codemirror/state.bundle.js';

window.addEventListener('DOMContentLoaded', () => {
    console.log('CodeMirror 6 test page loaded');
    
    // Test 1: Mock Editor (for comparison)
    const mockContainer = document.getElementById('mock-editor');
    const mockTextarea = document.createElement('textarea');
    mockTextarea.value = `// Mock Editor (textarea-based)
#include <iostream>
int main() {
    std::cout << "Hello from Mock Editor!" << std::endl;
    return 0;
}`;
    mockTextarea.style.cssText = 'width: 100%; height: 250px; font-family: monospace; font-size: 14px; border: 1px solid #ddd; padding: 10px;';
    mockContainer.appendChild(mockTextarea);
    
    // Test 2: CodeMirror 6
    const cmContainer = document.getElementById('codemirror-editor');
    try {
        const view = new EditorView({
            state: EditorState.create({
                doc: `// CodeMirror 6 Editor (real transaction-based)
#include <iostream>
int main() {
    std::cout << "Hello from CodeMirror 6!" << std::endl;
    return 0;
}`,
                extensions: [
                    basicSetup,
                    EditorView.theme({
                        '&': { height: '250px', border: '1px solid #ddd' },
                        '.cm-content': { padding: '10px', fontFamily: 'monospace', fontSize: '14px' },
                        '.cm-focused': { outline: 'none' }
                    }),
                    EditorView.updateListener.of(update => {
                        if (update.docChanged) {
                            console.log('CodeMirror content changed');
                            document.getElementById('cm-output').textContent = 
                                'Content: ' + view.state.doc.toString().substring(0, 50) + '...';
                        }
                    })
                ]
            }),
            parent: cmContainer
        });
        
        // Test the abstraction API
        document.getElementById('test-setValue').onclick = () => {
            view.dispatch({
                changes: {
                    from: 0,
                    to: view.state.doc.length,
                    insert: 'int main() { return 42; }'
                }
            });
        };
        
        document.getElementById('test-getValue').onclick = () => {
            const content = view.state.doc.toString();
            document.getElementById('cm-output').textContent = 'Content: ' + content;
        };
        
        console.log('CodeMirror 6 editor created successfully');
        document.getElementById('cm-status').textContent = 'CodeMirror 6 loaded successfully!';
        document.getElementById('cm-status').style.color = 'green';
        
    } catch (error) {
        console.error('Failed to create CodeMirror editor:', error);
        document.getElementById('cm-status').textContent = 'CodeMirror 6 failed to load: ' + error.message;
        document.getElementById('cm-status').style.color = 'red';
    }
});