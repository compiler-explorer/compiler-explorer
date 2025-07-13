// CodeMirror 6 test page entry point
import {EditorView} from '@codemirror/view';
import {EditorState} from '@codemirror/state';
import {syntaxHighlighting, HighlightStyle} from '@codemirror/language';
import {tags} from '@lezer/highlight';
import {cpp} from '@codemirror/lang-cpp';

console.log('CodeMirror test script loaded, imports successful');
console.log('EditorView:', EditorView);
console.log('EditorState:', EditorState);
console.log('C++ language support:', cpp);
console.log('C++ language function result:', cpp());
console.log('Syntax highlighting:', syntaxHighlighting);
console.log('Highlight tags:', tags);
console.log('C++ language support:', cpp);

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
            console.log('Creating CodeMirror editor...');
            
            const cppExtension = cpp();
            console.log('C++ extension created:', cppExtension);
            console.log('C++ extension type:', typeof cppExtension);
            console.log('C++ extension constructor:', cppExtension.constructor?.name);
            
            // Create a custom highlight style to avoid version conflicts
            const customHighlightStyle = HighlightStyle.define([
                { tag: tags.comment, color: '#008000' },
                { tag: tags.keyword, color: '#0000FF' },
                { tag: tags.string, color: '#800080' },
                { tag: tags.number, color: '#FF0000' },
                { tag: tags.variableName, color: '#000000' },
                { tag: tags.typeName, color: '#008080' },
                { tag: tags.operator, color: '#000000' },
                { tag: tags.bracket, color: '#000000' }
            ]);
            
            console.log('Custom highlight style created:', customHighlightStyle);
            
            const highlightExtension = syntaxHighlighting(customHighlightStyle);
            console.log('Highlight extension created:', highlightExtension);
            
            const extensions = [
                cppExtension, // Add C++ language support first
                highlightExtension, // Add syntax highlighting
                EditorView.lineWrapping,
                EditorView.theme({
                    '&': { height: '250px', border: '1px solid #ddd' },
                    '.cm-content': { padding: '10px', fontFamily: 'monospace', fontSize: '14px' },
                    '.cm-focused': { outline: 'none' }
                }),
                EditorView.updateListener.of(update => {
                    if (update.docChanged) {
                        console.log('CodeMirror content changed');
                        const outputEl = document.getElementById('cm-output');
                        if (outputEl) {
                            outputEl.textContent = 
                                'Content: ' + view.state.doc.toString().substring(0, 50) + '...';
                        }
                    }
                })
            ];
            
            console.log('Extensions array:', extensions);
            console.log('Extensions length:', extensions.length);
            extensions.forEach((ext, i) => {
                console.log(`Extension ${i}:`, ext, typeof ext);
            });
            
            const state = EditorState.create({
                doc: `// CodeMirror 6 Editor (real transaction-based)
#include <iostream>
int main() {
    std::cout << "Hello from CodeMirror 6!" << std::endl;
    return 0;
}`,
                extensions: extensions
            });
            
            console.log('EditorState created:', state);
            console.log('State language data:', state.languageDataAt('language', 0));
            
            const view = new EditorView({
                state: state,
                parent: cmContainer
            });
            
            console.log('EditorView created:', view);
            
            // Check if syntax highlighting is working
            setTimeout(() => {
                console.log('=== Checking syntax highlighting after render ===');
                const cmContent = cmContainer.querySelector('.cm-content');
                if (cmContent) {
                    console.log('CM content element found:', cmContent);
                    const highlightedElements = cmContent.querySelectorAll('[class*="cm-"]');
                    console.log('Highlighted elements found:', highlightedElements.length);
                    Array.from(highlightedElements).forEach((el, i) => {
                        if (i < 5) { // Log first 5 elements
                            console.log(`Highlighted element ${i}:`, el.className, el.textContent);
                        }
                    });
                    
                    // Check for specific C++ tokens
                    const includeElements = cmContent.querySelectorAll('[class*="keyword"], [class*="include"]');
                    console.log('Include/keyword elements:', includeElements.length);
                    
                    const commentElements = cmContent.querySelectorAll('[class*="comment"]');
                    console.log('Comment elements:', commentElements.length);
                } else {
                    console.log('CM content element not found');
                }
            }, 100);
            
            // Test the abstraction API
            const setValueBtn = document.getElementById('test-setValue');
            if (setValueBtn) {
                setValueBtn.onclick = () => {
                    view.dispatch({
                        changes: {
                            from: 0,
                            to: view.state.doc.length,
                            insert: 'int main() { return 42; }'
                        }
                    });
                };
            }
            
            const getValueBtn = document.getElementById('test-getValue');
            if (getValueBtn) {
                getValueBtn.onclick = () => {
                    const content = view.state.doc.toString();
                    const outputEl = document.getElementById('cm-output');
                    if (outputEl) {
                        outputEl.textContent = 'Content: ' + content;
                    }
                };
            }
            
            console.log('CodeMirror 6 editor created successfully');
            const statusEl = document.getElementById('cm-status');
            if (statusEl) {
                statusEl.textContent = 'CodeMirror 6 loaded successfully with C++ syntax highlighting!';
                statusEl.style.color = 'green';
            }
            
        } catch (error) {
            console.error('Failed to create CodeMirror editor:', error);
            const statusEl = document.getElementById('cm-status');
            if (statusEl) {
                statusEl.textContent = 'CodeMirror 6 failed to load: ' + (error as Error).message;
                statusEl.style.color = 'red';
            }
        }
    }
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeCodeMirrorTest);
} else {
    initializeCodeMirrorTest();
}