# Bootstrap 5 Migration Plan

This document outlines the step-by-step process for migrating Compiler Explorer from Bootstrap 4 to Bootstrap 5.3.5. The
migration will be completed incrementally to allow for testing between steps.

## Migration Strategy

We'll break down the migration into smaller, testable chunks rather than making all changes at once. This approach
allows for:

- Easier identification of issues
- Progressive testing by project maintainers
- Minimizing disruption to the codebase

**Progress tracking will be maintained in this document.**

## Phase 1: Dependency Updates and Basic Setup

- [x] Update package.json with Bootstrap 5.3.5
- [x] Add @popperjs/core dependency (replacing Popper.js)
- [x] Update Tom Select theme from bootstrap4 to bootstrap5
- [x] Update main import statements where Bootstrap is initialized
- [x] Update webpack configuration if needed for Bootstrap 5 compatibility
- [x] Verify the application still builds and runs with basic functionality

### Notes for Human Testers (Phase 1)
- At this phase, the application will have console errors related to Bootstrap component initialization
- Tom Select dropdowns (like compiler picker) may have styling differences with the new bootstrap5 theme
- Initial page load should still work, but dropdown functionality will be broken
- Modal dialogs like Settings and Sharing may not function correctly
- The primary layout and basic display of panes should still function

## Phase 2: Global CSS Class Migration

- [x] Update directional utility classes (ml/mr → ms/me)
    - [x] Search and replace in .pug templates
    - [x] Search and replace in .scss files
    - [x] Search and replace in JavaScript/TypeScript files that generate HTML
- [x] Update floating utility classes (float-left/right → float-start/end)
- [x] Update text alignment classes (text-left/right → text-start/end)
- [x] Update other renamed classes (badge-pill → rounded-pill, etc.)
- [x] Test and verify styling changes

### Notes for Human Testers (Phase 2)
- Look for proper spacing and margin alignment in the UI
- Specific components to check:
  - Compiler output area: Check the spacing in compiler.pug (short-compiler-name, compile-info spans)
  - Navbar spacing: The navbar items should maintain proper spacing (ms/me instead of ml/mr)
  - Code editor components: Check the button and icon alignment in codeEditor.pug
  - Tree view components: The tree.pug file had utility class changes
  - Alert messages (widgets/alert.ts): Check that toast messages appear with correct alignment
  - Compiler picker (compiler-picker.ts and compiler-picker-popup.ts): Check dropdown spacing
  - Rounded badge display in menu-policies.pug (now using rounded-pill instead of badge-pill)
- The float-end class replaces float-right in the index.pug file's copy buttons
- Any LTR/RTL layout impacts should be especially checked for correct directionality

## Phase 3: HTML Attribute Updates

- [x] Update data attributes across the codebase
    - [x] data-toggle → data-bs-toggle
    - [x] data-target → data-bs-target
    - [x] data-dismiss → data-bs-dismiss
    - [x] data-ride → data-bs-ride (not used in codebase)
    - [x] data-spy → data-bs-spy (not used in codebase)
    - [x] Other data attributes as needed
- [x] Test components to ensure they function correctly with new attributes

### Notes for Human Testers (Phase 3)
- This phase should restore basic functionality of several Bootstrap components
- Specific components to check:
  - Dropdowns: All dropdown menus throughout the application should open/close properly
  - Modal dialogs: Settings, Sharing, and other modal dialogs should open/close correctly
  - Tooltips: Hover tooltips (using data-bs-toggle="tooltip") should display properly
  - Popovers: Any popovers used in the UI should function correctly
  - Collapse components: Any collapsible sections should toggle properly
  - Tabs: Any tabbed interfaces should switch between tabs correctly
- Watch for console errors related to Bootstrap component initialization
- Some JavaScript component initialization may still be broken until Phase 4 is completed

## Phase 4: JavaScript API Compatibility Layer

- [x] Create a temporary Bootstrap compatibility utility module to abstract component initialization
  - [x] Implement a hybrid approach with `bootstrap-utils.ts` as a compatibility layer
  - [x] Mark clearly as temporary code to be removed after migration is complete
  - [x] Define methods for each component type (Modal, Dropdown, Toast, etc.)
- [x] Update component initialization in key files:
  - [x] widgets/alert.ts (modals and toasts)
  - [x] sharing.ts (modals, tooltips, and dropdowns)
  - [x] compiler-picker-popup.ts (modals)
  - [x] load-save.ts (modals)
  - [x] Other files with Bootstrap component initialization:
    - [x] **Modal Initialization**:
      - [x] widgets/site-templates-widget.ts
      - [x] widgets/runtime-tools.ts
      - [x] widgets/compiler-overrides.ts
      - [x] widgets/timing-info-widget.ts
      - [x] widgets/history-widget.ts
      - [x] widgets/libs-widget.ts
      - [x] main.ts
    - [x] **Dropdown Handling**:
      - [x] panes/tree.ts
      - [x] panes/compiler.ts
      - [x] panes/editor.ts
    - [x] **Popover Handling**:
      - [x] main.ts
      - [x] widgets/compiler-version-info.ts
      - [x] panes/executor.ts
      - [x] panes/conformance-view.ts
      - [x] panes/cfg-view.ts
- [x] Test the compatibility layer with basic components

### Notes for Human Testers (Phase 4)
- This is one of the most critical phases as it involves creating a compatibility layer for the JavaScript API
- A new utility file will be created for component initialization that abstracts Bootstrap 5's new approach
- Key differences to watch for:
  - Modal initialization and events: Bootstrap 5 uses a completely different event system
  - Dropdown initialization: Works with data attributes but requires `data-bs-toggle` instead of `data-toggle`
  - Toast components: The API has changed significantly
  - Popover/Tooltip initialization: API changes but still support data attributes with proper prefixes
- Components to thoroughly test:
  - Alert dialogs (widgets/alert.ts): Check all types of alerts (info, warning, error)
  - Sharing functionality (sharing.ts): The share modal should work properly
  - Compiler picker popup (compiler-picker-popup.ts): Should display and function correctly
  - All dropdown menus: Should open and close properly
  - All tooltips and popovers: Should display correctly on hover/click
- Watch for event handling issues where Bootstrap 4 events no longer exist or are renamed

### Key Learnings From Implementation
- **Data Attributes Still Work Without JavaScript Initialization**: Despite some documentation suggesting otherwise, Bootstrap 5 components with data attributes (like tabs) still work without explicit JavaScript initialization. The key is using the correct `data-bs-*` prefix.
- **Close Button Implementation Completely Changed**: Bootstrap 4 used `.close` class with a `&times;` entity inside a span, while Bootstrap 5 uses `.btn-close` class with a background image and no inner content.
- **Easy to Miss Data Attributes**: Initial migration scripts may miss data attributes in template files and JavaScript. Double-check all files for remaining `data-toggle`, `data-placement`, etc., attributes.
- **Tab Navigation Issues**: The tab navigation problems were fixed by simply updating data attributes, not by adding JavaScript initialization.
- **jQuery Plugin Methods Removal**: jQuery methods like `.popover()` need to be replaced with code that uses the Bootstrap 5 API through a compatibility layer.
- **Don't Mix Data Attributes and JavaScript Modal Creation**: When creating modals via JavaScript (e.g., for dynamically loaded content), don't include `data-bs-toggle="modal"` on the trigger element unless you also add a matching `data-bs-target` attribute pointing to a valid modal element.
- **Modal Events Changed Significantly**: Bootstrap 5 modal events need to be attached directly to the native DOM element rather than jQuery objects, and the event parameter type is different. For proper typing, import the `Modal` type from bootstrap and use `Modal.Event` type.
- **Tooltip API Changed**: The global `window.bootstrap.Tooltip` reference no longer exists. Import the `Tooltip` class directly from bootstrap instead.
- **Input Group Structure Simplified**: Bootstrap 5 removed the need for `.input-group-prepend` and `.input-group-append` wrapper divs. Buttons and other controls can now be direct children of the `.input-group` container. This simplifies the markup but requires template updates.
- **TomSelect Widget Integration**: Bootstrap 5's switch from CSS triangles to SVG background images for dropdowns caused issues with TomSelect. Adding back custom CSS for dropdown arrows was necessary to maintain correct appearance.
- **Btn-block Removed**: Bootstrap 5 removed the `.btn-block` class. Instead, the recommended approach is to wrap buttons in a container with `.d-grid` and use standard `.btn` classes. This affects any full-width buttons in the application.

## Phase 5: Component Migration (By Component Type)

### Modal Component Migration
- [x] Update modal implementation in alert.ts
- [x] Update modal usage in compiler-picker-popup.ts
- [x] Update modal handling in load-save.ts
- [x] Update modal event handling in sharing.ts
- [ ] Test modal functionality thoroughly

### Dropdown Component Migration
- [x] Update dropdown handling in sharing.ts
- [x] Update dropdown usage in compiler.ts, editor.ts, etc.
- [ ] Test dropdown functionality thoroughly

### Toast/Alert Component Migration
- [x] Update toast implementation in alert.ts
- [x] Update toast styling in explorer.scss
- [ ] Test toast notifications and alerts

### Popover/Tooltip Migration
- [x] Update tooltip initialization in sharing.ts
- [x] Update popover usage in compiler.ts, executor.ts, editor.ts, etc.
- [ ] Test popover and tooltip functionality thoroughly

### Card Component Updates
- [ ] Review card usage and update to Bootstrap 5 standards
- [ ] Replace any card-deck implementations with grid system
- [ ] Test card layouts

### Collapse Component Updates
- [ ] Update any collapse component implementations
- [ ] Test collapse functionality

### Button Group Updates
- [ ] Review button group implementations
- [ ] Update to Bootstrap 5 standards
- [ ] Test button group functionality

## Phase 6: Form System Updates

- [ ] Update form control classes to Bootstrap 5 standards
- [ ] Update input group markup and classes
- [ ] Update checkbox/radio markup to Bootstrap 5 standards
- [ ] Update form validation classes and markup
- [ ] Consider implementing floating labels where appropriate (new in Bootstrap 5)
- [ ] Test form functionality and appearance

## Phase 7: Navbar Structure Updates

- [ ] Update navbar structure in templates to match Bootstrap 5 requirements
- [ ] Review custom navbar styling in explorer.scss
- [ ] Test responsive behavior of navbar
- [ ] Ensure mobile menu functionality works correctly
- [ ] Consider implementing offcanvas for mobile navigation (new in Bootstrap 5)

## Phase 8: SCSS Variables and Theming

- [ ] Review any custom SCSS that extends Bootstrap functionality
- [ ] Update any custom themes to use Bootstrap 5 variables
- [ ] Check z-index variable changes in Bootstrap 5
- [ ] Test theme switching functionality

## Phase 9: Accessibility Improvements

- [ ] Review ARIA attributes in custom component implementations
- [ ] Leverage Bootstrap 5's improved accessibility features
- [ ] Test with screen readers and keyboard navigation
- [ ] Ensure color contrast meets accessibility guidelines

## Phase 10: Final Testing and Refinement

- [ ] Comprehensive testing across different viewports
- [ ] Cross-browser testing
- [ ] Fix any styling issues or inconsistencies
- [ ] Performance testing (Bootstrap 5 should be more performant)
- [ ] Ensure no regressions in functionality

## Phase 11: Documentation Update

- [ ] Update any documentation that references Bootstrap components
- [ ] Document custom component implementations
- [ ] Note any deprecated features or changes in functionality

## Phase 12: Optional jQuery Removal and Cleanup (Future Work)

- [ ] Create plan for jQuery removal (if desired)
- [ ] Identify non-Bootstrap jQuery usage that would need refactoring
- [ ] Remove the temporary `bootstrap-utils.ts` compatibility layer
  - [ ] Replace all uses with direct Bootstrap 5 API calls
  - [ ] Document the native Bootstrap 5 API for future reference
- [ ] Note: This would be a separate effort after the Bootstrap migration is stable

## Notes for Implementation

1. **Make minimal changes** in each step to allow for easier testing and troubleshooting
2. **Test thoroughly** after each phase before moving to the next
3. **Document issues** encountered during migration for future reference
4. **Focus on accessibility** to ensure the site remains accessible throughout changes
5. **Maintain browser compatibility** with all currently supported browsers
6. **Consider performance implications** of the changes
7. **NEVER mark any issue as fixed in this document** until you have explicit confirmation from the reviewer that the issue is completely resolved
8. **NEVER commit changes** until you have explicit confirmation that the fix works correctly

## Technical References

- [Bootstrap 5 Migration Guide](https://getbootstrap.com/docs/5.0/migration/)
- [Bootstrap 5 Components Documentation](https://getbootstrap.com/docs/5.3/components/)
- [Bootstrap 5 Utilities Documentation](https://getbootstrap.com/docs/5.3/utilities/)
- [Bootstrap 5 Forms Documentation](https://getbootstrap.com/docs/5.3/forms/overview/)
- [Popper v2 Documentation](https://popper.js.org/docs/v2/)

## Current Progress

### Completed Phases
- ✅ **Phase 1: Dependency Updates and Basic Setup**
  - Updated Bootstrap from 4.6.2 to 5.3.5
  - Replaced popper.js with @popperjs/core
  - Updated Tom Select theme from bootstrap4 to bootstrap5
  - Updated import statements and built successfully with webpack

- ✅ **Phase 2: Global CSS Class Migration**
  - Updated directional utility classes (ml/mr → ms/me)
  - Updated floating utility classes (float-left/right → float-start/end)
  - Updated text alignment classes (text-left/right → text-start/end)
  - Updated other renamed classes (badge-pill → rounded-pill, etc.)
  - Verified styling changes and fixed layout issues

- ✅ **Phase 3: HTML Attribute Updates**
  - Updated all data attributes to use Bootstrap 5 naming convention with bs- prefix
  - Verified data attributes in templates and JavaScript code
  - Fixed tab navigation and other component initialization

- ✅ **Phase 4: JavaScript API Compatibility Layer**
  - Created `bootstrap-utils.ts` compatibility layer
  - Updated all key component initialization code:
    - Modal initialization in multiple widgets
    - Dropdown handling in panes and components
    - Popover/tooltip implementation with proper disposal
  - Fixed all critical functionality issues:
    - Share dialog functionality
    - Sponsors window functionality
    - Tab navigation in modals
    - TomSelect dropdown styling

### Current Phase
- 🔄 **Phase 5: Component Migration**
  - Beginning systematic updates to each component type
  - Using the compatibility layer consistently
  - Testing each component thoroughly

### Next Steps
- Complete remaining phases systematically
- Conduct comprehensive testing using the Final Testing Checklist
- Review responsive behavior on mobile devices
- Plan for further improvements to IDE tree view

## Migration Status Summary

### Completed Fixes

1. **UI Layout & Display Issues**
   - ✓ Font dropdown styling fixed
   - ✓ Templates view proportions fixed (min-width added to columns and modal)
   - ✓ Dialog appearance fixed (updated close buttons to use `.btn-close`)
   - ✓ Dropdown positioning fixed (updated to `.dropdown-menu-end`)
   - ✓ TomSelect dropdown arrows fixed (custom CSS implementation)
   - ✓ IDE mode border styling improved (temporarily with `.list-group-flush`)
   - ✓ Sponsors window styling fixed (replaced `.btn-block` with `.d-grid` approach)

2. **Navigation & Functional Issues**
   - ✓ Tab navigation fixed (updated data attributes to `data-bs-toggle="tab"`)
   - ✓ Share dialog functionality fixed (proper Bootstrap 5 modal initialization)
   - ✓ Sponsors modal error fixed (removed conflicting data attributes)
   - ✓ Share dropdown tooltip conflict fixed (moved tooltip to parent element)

### Remaining Tasks

1. **Continue Implementation**
   - Replace remaining jQuery plugin methods with Bootstrap 5 API equivalents
   - Use `BootstrapUtils` compatibility layer consistently throughout codebase
   - Conduct thorough testing using the Final Testing Checklist
   - Check input group appearance and functionality across all components
   - Verify responsive behavior on mobile devices
   - Plan for a more comprehensive redesign of the IDE tree view in a future phase

2. **Technical Notes**
   - IDE mode: Current fix is temporary; a redesign using card components should be considered
   - Share dropdown: Bootstrap 5 doesn't allow multiple components on the same DOM element
   - Input groups: Bootstrap 5 no longer requires `.input-group-prepend` and `.input-group-append` wrappers

## Final Testing Checklist

Before considering the Bootstrap 5 migration complete, the following areas should be thoroughly tested to ensure proper functionality and appearance:

### UI Components and Controls
- **Dropdowns**
  - All dropdown menus (especially on the right side of the screen)
  - Font size dropdown
  - Compiler picker dropdown and popout functionality
  - Popular arguments dropdown
  - TomSelect dropdowns in all contexts

- **Input Groups**
  - Search and filter inputs with buttons
  - Compiler options inputs
  - Input groups with multiple buttons

- **Buttons and Button Groups**
  - Button alignment and spacing
  - Button groups in toolbars
  - Icon buttons with tooltips

### Specialized Views
- **Conformance View**
  - Compiler selectors and options
  - Results display

- **Tree View (IDE Mode)**
  - Tree structure and file display
  - Right-click menus and dropdowns
  - File manipulation controls

- **Visualization Components**
  - CFG view rendering and controls
  - Opt pipeline viewer
  - AST view

- **Sponsor popup**
  - Sponsor list display
  - Modal dialog appearance and functionality

### Modals and Dialogs
- **Settings Dialog**
  - All tabs and sections
  - Form controls within settings

- **Share Dialog**
  - Link generation
  - Copy to clipboard functionality
  - Tooltips (e.g., "Copied to clipboard" messages)

- **Load/Save Dialog**
  - Local storage interaction
  - File list display

### Library Management
- **Library Views**
  - Library selection and version dropdowns
  - Library information display

### Responsive Behavior
- **Mobile View**
  - Test at various viewport sizes
  - Verify mobile menu functionality
  - Check input group stacking behavior


This plan will be updated as progress is made, with each completed step marked accordingly.
