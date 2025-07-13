# Mobile Editor Investigation and Implementation Plan

## Executive Summary

Microsoft's Monaco editor lacks official mobile support, resulting in a degraded read-only experience for mobile users of Compiler Explorer. This document investigates alternatives and proposes implementing CodeMirror 6 as a mobile-specific editor to enable full editing capabilities on mobile devices.

## Current State

### Monaco on Mobile
- **Current Implementation**: Monaco is used everywhere but forced into read-only mode on mobile
- **Mobile Detection**: Uses `window.compilerExplorerOptions.mobileViewer` flag
- **Workarounds**: Hidden UI elements, disabled editing, context menu fixes
- **User Experience**: Partially working but practically unusable

### Technical Details
- Editor implementation: `/static/panes/editor.ts`
- Mobile-specific code sections:
  - Read-only mode enforcement (line 208): `readOnly: !!options.readOnly || this.legacyReadOnly || window.compilerExplorerOptions?.mobileViewer`
  - Textarea hiding (lines 295-297): `$(this.domRoot.find('.monaco-placeholder textarea')).hide()`
  - Context menu workarounds (lines 380-388): Manual context menu hiding on cursor selection changes

**Verified Monaco Usage Patterns in Codebase:**
- **Editor initialization without initial content**: Most common pattern - create editor first, set content after (verified in `editor.ts:203-216`, `compiler.ts:372-392`)
- **Initial content during creation**: Only `tool-input-view.ts:64-72` uses `value: ''` property in Monaco config
- **Post-creation content setting**: Main editor uses conditional initialization - `setSource(state.source)` if available, otherwise `updateEditorCode()` (verified in `editor.ts:131-135`)
- **Content loading via updateEditorCode**: Loads language example or previous source via `setSource()` method (verified in `editor.ts:1905-1910`)
- Direct `setValue()` calls in output panes (verified in `ir-view.ts:350`, `ast-view.ts:227`, `compiler.ts:1531`)
- Model `setValue()` for model-based operations (verified in `ir-view.ts:359`)
- **Diff editor model setting**: Uses `setModel({original, modified})` pattern (verified in `diff.ts:230`)
- Complex `pushEditOperations()` for main editor with undo history preservation (verified in `editor.ts:1106-1151`)
- Sequential operations pattern with setTimeout workarounds (verified in `editor.ts:1142-1147`)
- Settings changes with multiple `updateOptions()` calls (verified in `editor.ts:1224-1270`)
- Language changes with multiple sequential operations (verified in `editor.ts:1841-1869`)

## Alternative Editors Evaluation

### CodeMirror 6
**Technical Pros:**
- **Mobile-first architecture**: Built with touch device support as primary goal, uses native browser selection/editing APIs
- **Modular design**: Small core (~1MB) with feature-specific extensions, allows tree-shaking for minimal bundles
- **Transaction-based state management**: Immutable state with atomic updates, better for complex state synchronization
- **Touch API integration**: Leverages platform-native touch handling instead of custom implementations

**Technical Cons:**
- **API migration complexity**: Requires converting from direct method calls to transaction-based patterns  
- **Mobile platform limitations**: Historical iOS touch selection issues ([2021 discussion](https://discuss.codemirror.net/t/touch-on-ios-iphone-not-working-in-codemirror-6/3345)) have been largely resolved through multiple fixes (2022-2024). Some Android selection boundary problems ([documented](https://github.com/codemirror/dev/issues/645)) persist as browser/OS limitations rather than editor architecture issues

### CodeJar
**Technical Summary:** Ultra-lightweight (2KB) but lacks essential features for Compiler Explorer (limited syntax highlighting, insufficient API)

### Ace Editor  
**Technical Summary:** Mature desktop-focused editor with documented mobile rendering issues and poor touch support

## Recommended Solution: CodeMirror 6 for Mobile

### Feature Scope for Mobile

#### Technical Implementation Scope
**Core Features:**
- Full editing capability, syntax highlighting, auto-indentation, bracket matching
- Compilation integration with error markers and result display
- Find/replace, language switching, basic settings

**Excluded Features:**
- Vim mode, line linking, advanced decorations, code actions, multi-pane layouts
- *Rationale: Touch interface constraints and mobile screen limitations*

### Implementation Architecture

```typescript
// Unified editor interface
interface ICompilerExplorerEditor {
  getValue(): string;
  setValue(value: string): void;
  setLanguage(language: string): void;
  setErrors(errors: CompilerError[]): void;
  onChange(callback: () => void): void;
  dispose(): void;
}

// Editor factory
function createEditor(container: HTMLElement, options: EditorOptions): ICompilerExplorerEditor {
  if (isMobileDevice()) {
    return new CodeMirrorMobileEditor(container, options);
  }
  return new MonacoDesktopEditor(container, options);
}
```

### Mobile UI Design

**Technical Requirements:**
- Full-screen editor with slide-up output panel, minimal toolbar
- Touch-optimized targets and swipe gesture handling
- Single-column responsive layout with essential-only UI elements

## Implementation Phases

### Phase 1: Foundation (Week 1-2)
- Add CodeMirror 6 dependency, create editor abstraction interface, implement basic CodeMirrorMobileEditor class, set up mobile bundle build process

### Phase 2: Core Features (Week 3-4)  
- Implement syntax highlighting for major languages, add compilation integration, implement error display system, create mobile-optimized UI components

### Phase 3: Language Support (Week 5-6)
- Port essential language definitions from Monarch to Lezer, test language switching, ensure feature parity for compilation

### Phase 4: Testing & Optimization (Week 7-8)
- Mobile device testing (iOS, Android), performance optimization, accessibility testing, beta user feedback

### Phase 5: Production Rollout
- Feature flag implementation, A/B testing infrastructure, monitoring and metrics, documentation updates

## Technical Implementation Details

### Text Setting API Comparison

Understanding how text is set programmatically in both editors is crucial for implementation.

#### Monaco Editor (Current Implementation)

Monaco provides multiple straightforward methods for setting text:

**Simple Content Update:**
```typescript
// Direct method - simplest approach
editor.setValue("new content");

// Via model - when working with models
editor.getModel()?.setValue("new content");
```

**Advanced Content Update (Preserves Undo History):**
```typescript
// From Compiler Explorer's current implementation in static/panes/editor.ts:1106-1151
updateSource(newSource: string): void {
    const operation = {
        range: this.editor.getModel()?.getFullModelRange(),
        forceMoveMarkers: true,
        text: newSource,
    };
    
    const viewState = this.editor.saveViewState();
    this.editor.pushUndoStop();
    
    this.editor.getModel()?.pushEditOperations(
        viewState?.cursorState ?? null, 
        [operation], 
        () => null
    );
}
```

**Getting Content:**
```typescript
// Get current content
const content = editor.getValue();
// or
const content = editor.getModel()?.getValue();
```

#### CodeMirror 6 (Proposed Implementation)

CodeMirror 6 uses a transaction-based system instead of direct methods. **Verified**: The CodeMirror 6 architecture follows a "Redux-esque" pattern where "the state and view are totally separate, and changes are applied using 'transactions'" with immutable state and unidirectional data flow [Source: CodeMirror System Guide]:

**Simple Content Update:**
```typescript
// Replace entire document
view.dispatch({
    changes: { 
        from: 0, 
        to: view.state.doc.length, 
        insert: "new content" 
    }
});
```
*Verified: [CodeMirror Reference Manual - EditorView.dispatch](https://codemirror.net/docs/ref/#view.EditorView.dispatch) and [Migration Guide](https://codemirror.net/docs/migration/) - "cm.setValue(text) → cm.dispatch({ changes: {from: 0, to: cm.state.doc.length, insert: text} })"*

**Advanced Content Update with State Management:**
```typescript
// More sophisticated approach for maintaining state
const transaction = view.state.update({
    changes: { 
        from: 0, 
        to: view.state.doc.length, 
        insert: newSource 
    },
    // Preserve selection if needed
    selection: view.state.selection,
    // Add to undo history
    userEvent: "input.paste"
});

view.dispatch(transaction);
```
*Verified: [CodeMirror Reference Manual - EditorState.update](https://codemirror.net/docs/ref/#state.EditorState.update) and [Transaction userEvent documentation](https://discuss.codemirror.net/t/more-granular-userevent-categorization-for-transactions/6137)*

**Getting Content:**
```typescript
// Get current content
const content = view.state.doc.toString();
```
*Verified: [CodeMirror Migration Guide](https://codemirror.net/docs/migration/) - "cm.getValue() → cm.state.doc.toString()" and [CodeMirror Examples](https://codemirror.net/examples/change/)*

**Editor Initialization with Content:**
```typescript
// CodeMirror 6 can set initial content during creation
const view = new EditorView({
    doc: "initial content here",  // Content set during initialization
    extensions: [/* extensions */],
    parent: container
});

// Or create empty and set content after
const view = new EditorView({
    extensions: [/* extensions */],
    parent: container
});
view.dispatch({
    changes: { from: 0, to: 0, insert: "initial content" }
});
```
*Verified: [CodeMirror Tutorials](https://davidmyers.dev/blog/how-to-build-a-code-editor-with-codemirror-6-and-typescript/introduction) - "doc property accepts either a string or a Text object" and [CodeMirror Guide](https://www.raresportan.com/how-to-make-a-code-editor-with-codemirror6/)*

#### Implementation Wrapper for Consistency

To maintain the same API regardless of which editor is used:

```typescript
interface ICompilerExplorerEditor {
    setValue(content: string): void;
    getValue(): string;
    setLanguage(language: string): void;
    setErrors(errors: CompilerError[]): void;
    onChange(callback: () => void): void;
    dispose(): void;
}

class MonacoEditorWrapper implements ICompilerExplorerEditor {
    constructor(private editor: monaco.editor.IStandaloneCodeEditor) {}
    
    setValue(content: string): void {
        this.editor.setValue(content);
    }
    
    getValue(): string {
        return this.editor.getValue();
    }
    
    // ... other method implementations
}

class CodeMirrorEditorWrapper implements ICompilerExplorerEditor {
    constructor(private view: EditorView) {}
    
    setValue(content: string): void {
        this.view.dispatch({
            changes: { 
                from: 0, 
                to: this.view.state.doc.length, 
                insert: content 
            }
        });
    }
    
    getValue(): string {
        return this.view.state.doc.toString();
    }
    
    // ... other method implementations
}
```

#### Key API Differences

1. **Monaco**: Direct methods (`setValue`/`getValue`) - simpler and more intuitive
2. **CodeMirror 6**: Transaction-based system - more powerful but requires understanding the state/dispatch pattern
3. **Monaco**: Built-in undo history management with `pushEditOperations`
4. **CodeMirror 6**: Transaction metadata for undo/redo control
5. **Monaco**: Multiple ways to achieve the same result
6. **CodeMirror 6**: Single consistent transaction-based approach

### Current Monaco Usage Patterns

Based on analysis of the codebase, Monaco text is currently set using:

1. **Direct setValue()** - For simple content updates in output panes (`static/panes/ir-view.ts:350`, `static/panes/ast-view.ts:227`)
2. **Model setValue()** - When working with models directly (`static/panes/ir-view.ts:359`)
3. **pushEditOperations()** - For the main editor to preserve undo history (`static/panes/editor.ts:1106-1151`)
4. **Initial content setting** - During editor initialization (`static/panes/editor.ts:131-135`)
5. **Conditional initialization** - Main editor pattern: create editor without content, then set via `setSource(state.source)` or `updateEditorCode()` (`editor.ts:131-135`)

### Sequential Operations That Need Event-Driven Conversion

CodeMirror's transaction-based system means that sequential Monaco operations that rely on immediate state changes will need to be converted to event-driven patterns. Here are the key problem areas:

#### 1. Content Updates with Follow-up Operations

**Current Monaco Pattern (`editor.ts:1106-1151`):**
```typescript
updateSource(newSource: string): void {
    const operation = { range: this.editor.getModel()?.getFullModelRange(), forceMoveMarkers: true, text: newSource };
    
    const viewState = this.editor.saveViewState();
    this.editor.pushUndoStop();                    // 1. Add undo stop
    this.editor.getModel()?.pushEditOperations(...); // 2. Apply edit operation
    this.numberUsedLines();                        // 3. Update line numbering
    
    // Timing workaround with setTimeout
    setTimeout(() => {
        if (this.selection) {
            this.editor.setSelection(this.selection);           // 4. Set selection
            this.editor.revealLinesInCenter(...);               // 5. Reveal selection
        }
    }, 500);
}
```

**CodeMirror Event-Driven Solution:**
```typescript
updateSource(newSource: string): void {
    // Single transaction with all state changes
    const transaction = this.view.state.update({
        changes: { from: 0, to: this.view.state.doc.length, insert: newSource },
        selection: this.selection ? EditorSelection.single(this.selection.from, this.selection.to) : undefined,
        scrollIntoView: true
    });
    
    this.view.dispatch(transaction);
    
    // Listen for transaction completion
    this.view.requestMeasure({
        read: () => {
            this.numberUsedLines(); // Now safe to update line numbering
        }
    });
}
```
*Verified: [EditorSelection.single documentation](https://codemirror.net/docs/ref/#state.EditorSelection^single) and [requestMeasure documentation](https://codemirror.net/docs/ref/#view.EditorView.requestMeasure)*

#### 2. Language Change Cascading Operations

**Current Monaco Pattern (`editor.ts:1841-1869`):**
```typescript
onLanguageChange(newLangId: LanguageKey, firstTime?: boolean): void {
    // Sequential operations without proper synchronization
    this.updateEditorCode();                    // 1. Update source code
    
    const editorModel = this.editor.getModel();
    if (editorModel && this.currentLanguage)
        monaco.editor.setModelLanguage(editorModel, this.currentLanguage.monaco); // 2. Change language
    
    this.isCpp.set(this.currentLanguage?.id === 'c++');     // 3-4. Update context keys
    this.isClean.set(this.currentLanguage?.id === 'clean');
    this.updateLanguageTooltip();              // 5. Update UI
    this.updateTitle();                         // 6. Update UI
    this.updateState();                         // 7. Update state
    this.eventHub.emit('languageChange', this.id, newLangId); // 8. Broadcast change
    this.decorations = {};                      // 9. Clear decorations
    
    if (!firstTime) {
        this.maybeEmitChange(true);             // 10. Emit changes
        this.requestCompilation();              // 11. Trigger compilation
    }
}
```

**CodeMirror Event-Driven Solution:**
```typescript
onLanguageChange(newLangId: LanguageKey, firstTime?: boolean): void {
    // Create a single transaction with all language-related changes
    const newLanguageCompartment = this.getLanguageCompartment(newLangId);
    
    const transaction = this.view.state.update({
        effects: [
            this.languageCompartment.reconfigure(newLanguageCompartment),
            // Clear decorations as part of the transaction
            this.decorationCompartment.reconfigure([])
        ]
    });
    
    this.view.dispatch(transaction);
    
    // Use update listener for follow-up actions
    this.scheduleLanguageChangeFollowup(newLangId, firstTime);
}

private scheduleLanguageChangeFollowup(newLangId: LanguageKey, firstTime?: boolean): void {
    // These operations happen after the language change is complete
    this.updateLanguageTooltip();
    this.updateTitle();
    this.updateState();
    this.eventHub.emit('languageChange', this.id, newLangId);
    
    if (!firstTime) {
        this.maybeEmitChange(true);
        this.requestCompilation();
    }
}
```
*Verified: [Compartment.reconfigure documentation](https://codemirror.net/docs/ref/#state.Compartment.reconfigure) and [Configuration Example](https://codemirror.net/examples/config/) - Shows dynamic language switching with compartments*

#### 3. Settings Changes with Multiple Editor Updates

**Current Monaco Pattern (`editor.ts:1224-1270`):**
```typescript
onSettingsChange(newSettings: SiteSettings): void {
    // Multiple editor option updates in sequence
    this.editor.updateOptions({                     // 1. Update multiple editor options
        autoIndent: this.settings.autoIndent ? 'advanced' : 'none',
        autoClosingBrackets: this.settings.autoCloseBrackets ? 'always' : 'never',
        // ... many more options
    });
    
    // Conditional vim operations
    if (after.useVim && !before.useVim) {
        this.enableVim();                           // 2. Enable vim (multiple operations)
    } else if (!after.useVim && before.useVim) {
        this.disableVim();                          // 3. Disable vim (multiple operations)
    }
    
    this.editor.getModel()?.updateOptions({         // 4. Update model options
        tabSize: this.settings.tabWidth,
        indentSize: this.settings.tabWidth,
        insertSpaces: this.settings.useSpaces,
    });
    
    this.numberUsedLines();                         // 5. Recalculate line colors
}
```

**CodeMirror Event-Driven Solution:**
```typescript
onSettingsChange(newSettings: SiteSettings): void {
    const effects: StateEffect<any>[] = [];
    
    // Batch all setting changes into a single transaction
    if (newSettings.tabWidth !== this.settings.tabWidth) {
        effects.push(this.indentUnitCompartment.reconfigure(
            indentUnit.of(" ".repeat(newSettings.tabWidth))
        ));
    }
    
    if (newSettings.useVim !== this.settings.useVim) {
        effects.push(this.vimCompartment.reconfigure(
            newSettings.useVim ? vim() : []
        ));
    }
    
    // Apply all changes in one transaction
    this.view.dispatch({
        effects: effects
    });
    
    // Schedule follow-up operations
    this.view.requestMeasure({
        read: () => {
            this.numberUsedLines();
        }
    });
}
```
*Verified: [StateEffect documentation](https://codemirror.net/docs/ref/#state.StateEffect) and [System Guide](https://codemirror.net/docs/guide/) - "State effects can be used to represent additional effects associated with a transaction"*

#### 4. Assembly Display with Multiple Operations

**Current Monaco Pattern (compiler.ts:1491-1599):**
```typescript
setAssembly(result: Partial<CompilationResult>, filteredCount = 0) {
    const editorModel = this.editor.getModel();
    if (editorModel) {
        monaco.editor.setModelLanguage(editorModel, result.languageId);  // 1. Set language
    }
    
    editorModel?.setValue(msg);                     // 2. Set content
    
    if (this.previousScroll) {
        this.editor.setScrollTop(this.previousScroll); // 3. Set scroll position
    }
    
    if (this.selection) {
        this.editor.setSelection(this.selection);               // 4. Set selection
        this.editor.revealLinesInCenter(...);                   // 5. Reveal selection
    }
    
    this.updateDecorations();                       // 6. Update decorations
}
```

**CodeMirror Event-Driven Solution:**
```typescript
setAssembly(result: Partial<CompilationResult>, filteredCount = 0) {
    // Single comprehensive transaction
    const effects: StateEffect<any>[] = [];
    
    if (result.languageId) {
        effects.push(this.languageCompartment.reconfigure(
            this.getLanguageSupport(result.languageId)
        ));
    }
    
    const transaction = this.view.state.update({
        changes: { from: 0, to: this.view.state.doc.length, insert: msg },
        selection: this.selection ? EditorSelection.single(this.selection.from, this.selection.to) : undefined,
        scrollIntoView: true,
        effects: effects
    });
    
    this.view.dispatch(transaction);
    
    // Decorations update after transaction completes
    this.view.requestMeasure({
        read: () => {
            this.updateDecorations();
        }
    });
}
```

#### Key Conversion Principles:

1. **Batch Operations**: Use single transactions instead of sequential operations
2. **Use requestMeasure()**: For operations that need DOM measurements
3. **Effects System**: Use StateEffect for configuration changes
4. **Event Listeners**: Use update listeners for follow-up actions
5. **Avoid setTimeout**: Replace timing hacks with proper event-driven patterns

#### Operations That Must Become Async:

1. **Selection after content change** - Need to wait for layout
2. **Scroll position after content** - Need to wait for DOM update
3. **Decoration updates** - Need to wait for syntax highlighting
4. **Line numbering calculations** - Need to wait for content measurement
5. **Language-dependent UI updates** - Need to wait for language change completion

This represents a significant architectural change from Monaco's immediate operation model to CodeMirror's transaction-based, eventually consistent model.

## Technical Considerations

**Bundle Optimization:** Dynamic imports, separate mobile bundle, CDN dependencies  
**Performance:** Lazy language mode loading, optimized initial render, minimal re-renders  
**Browser Support:** Modern mobile browsers (iOS Safari 14+, Chrome/Android WebView 80+)  
**Migration Strategy:** Desktop Monaco unchanged, wrapper pattern for API compatibility, gradual rollout with rollback capability

## Success Metrics

**Technical KPIs:** Mobile edit completion rate, time to interactive, bundle size impact, memory usage  
**User Metrics:** Session duration, compilation attempts, beta feedback, support ticket reduction

## Risk Assessment

**Language Support:** Prioritize top 10 languages, incremental additions  
**Performance:** Progressive enhancement, feature detection for low-end devices  
**Maintenance:** Shared abstraction layer, automated testing for dual-editor codebase

## Mobile Testing and Validation

### **Live CodeMirror 6 Mobile Test Sites**

The following sites can be used to validate CodeMirror 6's mobile functionality on actual devices:

#### **Primary Test Sites:**
- **Official CodeMirror Try Page**: https://www.codemirror.net/try/ - Official playground for testing custom configurations
- **CodeMirror Main Site**: https://www.codemirror.net/ - Live editor on homepage for immediate testing
- **CodeSandbox Demo**: https://codesandbox.io/s/codemirror-6-demo-pl8dc - Full-featured implementation
- **Observable Playground**: https://observablehq.com/@andy0130tw/codemirror-6-playground - Interactive testing environment
- **VizHub Real-time Demo**: https://vizhub.io/ - Production usage with mobile-desktop sync

#### **Mobile Test Checklist:**
**Basic Functionality:**
- Touch cursor placement, text selection (double-tap, drag handles), typing/editing, copy/paste context menus, smooth scrolling

**Advanced Features:**
- Auto-completion, syntax highlighting, bracket matching, find/replace functionality

**Mobile-Specific:**
- Native selection handles, zoom behavior, orientation changes, virtual keyboard interaction

**Quick Validation Steps:**
1. Visit https://www.codemirror.net/try/
2. Tap in editor (cursor appears), type code (smooth text input)
3. Double-tap word (selection with handles), drag selection (extends/contracts)
4. Long press (context menu), test copy/paste (system clipboard integration)

## Technical Conclusion

CodeMirror 6's mobile-first architecture and transaction-based state management provide the technical foundation needed to transform Compiler Explorer's mobile experience from read-only to fully functional editing environment.

## Sources and References

### CodeMirror 6 Mobile Support
- [Replit - A New Code Editor for Mobile](https://blog.replit.com/codemirror-mobile) - Documents the 70% retention improvement
- [Replit - Ace, CodeMirror, and Monaco: A Comparison](https://blog.replit.com/code-editors) - Comprehensive editor comparison including mobile capabilities
- [CodeMirror 6 Migration Guide](https://codemirror.net/docs/migration/) - Official migration documentation
- [CodeMirror System Guide](https://codemirror.net/docs/guide/) - Transaction-based architecture documentation

### CodeMirror 6 Architecture and Performance
- [CodeMirror Bundling Example](https://codemirror.net/examples/bundle/) - Bundle size optimization techniques
- [CodeMirror 6 Transaction System Discussion](https://discuss.codemirror.net/t/dispatching-transactionspec-or-transaction/7891) - Community discussion on transaction architecture
- [Sourcegraph - Migrating from Monaco to CodeMirror](https://sourcegraph.com/blog/migrating-monaco-codemirror) - Real-world migration experience

### Mobile Editor Comparisons
- [StackShare - CodeMirror vs Monaco Editor](https://stackshare.io/stackups/codemirror-vs-monaco-editor) - Feature comparison
- [npm trends - Editor popularity](https://npmtrends.com/ace-code-editor-vs-codemirror-vs-monaco-editor) - Usage statistics
- [DEV Community - Monaco vs CodeMirror in React](https://dev.to/suraj975/monaco-vs-codemirror-in-react-5kf) - Developer experiences

### Monaco Editor Mobile Limitations
- [GitHub Issue - Monaco mobile support](https://github.com/Microsoft/monaco-editor/issues/19) - Official stance on mobile support
- [CodeMirror vs Monaco Editor comparison](https://stackshare.io/stackups/codemirror-vs-monaco-editor) - Community feedback on mobile capabilities

### CodeMirror 6 Mobile Issues and Solutions
- [iOS Touch Issues Discussion](https://discuss.codemirror.net/t/touch-on-ios-iphone-not-working-in-codemirror-6/3345) - iOS Safari selection API limitations
- [Android Touch Selection Issue](https://github.com/codemirror/dev/issues/645) - Android Chrome selection boundary problems
- [Selection Handles Issue](https://github.com/codemirror/dev/issues/466) - Android selection handles disappearing (resolved)
- [iOS/iPadOS Text Selection Bug](https://github.com/codemirror/dev/issues/804) - Safari shadow root selection issues

**Important Note**: These are primarily browser/OS platform limitations rather than CodeMirror architectural issues. Most iOS touch selection issues from 2021 have been resolved through active development (2022-2024). Some Android problems have been resolved in recent updates. Monaco Editor has no mobile support and is completely unusable, whereas CodeMirror 6 is functional with minimal remaining edge-case limitations.

### Technical Examples Verification
All CodeMirror 6 code examples in this document have been verified against official documentation:
- [CodeMirror Reference Manual](https://codemirror.net/docs/ref/) - Complete API reference
- [CodeMirror Examples](https://codemirror.net/examples/) - Official code examples
- [CodeMirror System Guide](https://codemirror.net/docs/guide/) - Architecture and design patterns
- [CodeMirror Migration Guide](https://codemirror.net/docs/migration/) - CodeMirror 5 to 6 migration patterns

---

*Document created: [Date]*  
*Last updated: [Date]*  
*Author: [Your name]*