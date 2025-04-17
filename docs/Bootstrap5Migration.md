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
- [ ] Test and verify styling changes

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
- [ ] Test components to ensure they function correctly with new attributes

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
  - [ ] Other files with Bootstrap component initialization:
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
- [ ] Test the compatibility layer with basic components

### Notes for Human Testers (Phase 4)
- This is one of the most critical phases as it involves creating a compatibility layer for the JavaScript API
- A new utility file will be created for component initialization that abstracts Bootstrap 5's new approach
- Key differences to watch for:
  - Modal initialization and events: Bootstrap 5 uses a completely different event system
  - Dropdown initialization: Now requires explicit instantiation via JavaScript
  - Toast components: The API has changed significantly
  - Popover/Tooltip initialization: These now need explicit initialization
- Components to thoroughly test:
  - Alert dialogs (widgets/alert.ts): Check all types of alerts (info, warning, error)
  - Sharing functionality (sharing.ts): The share modal should work properly
  - Compiler picker popup (compiler-picker-popup.ts): Should display and function correctly
  - All dropdown menus: Should open and close properly
  - All tooltips and popovers: Should display correctly on hover/click
- Watch for event handling issues where Bootstrap 4 events no longer exist or are renamed

## Phase 5: Component Migration (By Component Type)

### Modal Component Migration
- [ ] Update modal implementation in alert.ts
- [ ] Update modal usage in compiler-picker-popup.ts
- [ ] Update modal handling in load-save.ts
- [ ] Update modal event handling in sharing.ts
- [ ] Test modal functionality thoroughly

### Dropdown Component Migration
- [ ] Update dropdown handling in sharing.ts
- [ ] Update dropdown usage in compiler.ts, editor.ts, etc.
- [ ] Test dropdown functionality

### Toast/Alert Component Migration
- [ ] Update toast implementation in alert.ts
- [ ] Update toast styling in explorer.scss
- [ ] Test toast notifications and alerts

### Popover/Tooltip Migration
- [ ] Update tooltip initialization in sharing.ts
- [ ] Update popover usage in compiler.ts, executor.ts, editor.ts, etc.
- [ ] Test popover and tooltip functionality

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

## Technical References

- [Bootstrap 5 Migration Guide](https://getbootstrap.com/docs/5.0/migration/)
- [Bootstrap 5 Components Documentation](https://getbootstrap.com/docs/5.3/components/)
- [Bootstrap 5 Utilities Documentation](https://getbootstrap.com/docs/5.3/utilities/)
- [Bootstrap 5 Forms Documentation](https://getbootstrap.com/docs/5.3/forms/overview/)
- [Popper v2 Documentation](https://popper.js.org/docs/v2/)

## Current Progress

- Initial planning document created
- Phase 1 completed:
  - Updated Bootstrap from 4.6.2 to 5.3.5
  - Replaced popper.js with @popperjs/core
  - Updated Tom Select theme from bootstrap4 to bootstrap5
  - Updated import statements in main.ts and noscript.ts
  - Successfully built with webpack
  - Basic functionality verified (with expected issues due to pending component updates)
- Phase 2 largely completed:
  - Updated directional utility classes (ml/mr → ms/me)
  - Updated floating utility classes (float-left/right → float-start/end)
  - Updated text alignment classes (text-left/right → text-start/end)
  - Updated other renamed classes (badge-pill → rounded-pill, etc.)
  - Testing and verification of styling changes pending
- Phase 3 completed:
  - Updated data attributes to use Bootstrap 5 naming convention with bs- prefix:
    - data-toggle → data-bs-toggle
    - data-target → data-bs-target
    - data-dismiss → data-bs-dismiss
    - data-placement → data-bs-placement
  - Identified that data-ride and data-spy attributes are not used in the codebase
  - Fixed additional data attribute in widget JavaScript code
  - Testing component functionality pending
- Phase 4 largely completed:
  - Created temporary `bootstrap-utils.ts` compatibility layer to abstract Bootstrap 5 APIs
  - Updated key files to use the compatibility layer:
    - Updated alert.ts to use BootstrapUtils for modals and toasts
    - Updated sharing.ts to use BootstrapUtils for tooltips, modals, and dropdowns
    - Updated compiler-picker-popup.ts to use BootstrapUtils for modals
    - Updated load-save.ts to use BootstrapUtils for modals
  - Fixed initial jQuery popover() compatibility issues:
    - Updated editor.ts to use Bootstrap 5 popover API
    - Updated compiler.ts to use bootstrap-utils for popover operations
    - Updated libs-widget.ts to use proper Bootstrap 5 patterns for popovers
    - Fixed popover disposal and hiding in various components
  - Created plan to eventually remove compatibility layer in Phase 12
  - Further component updates and testing pending

## Known Issues To Investigate and Fix

The following issues need to be addressed as part of the ongoing Bootstrap 5 migration:

1. **UI Layout & Display Issues**
   - The font dropdown is broken (it appears too small)
   - Templates view is very long and thin
   - ~~All dialogs look weird, likely related to tab issues~~ ✓ Fixed
   - The "other" dropdown clips off the right hand side of the page
   - TomSelect dropdowns are missing the dropdown arrow, and the "pop out" button isn't styled correctly
   - "IDE mode" has unwanted border lines around everything
   
   *Dialog appearance was fixed by updating close buttons from Bootstrap 4's `.close` class with `&times;` to Bootstrap 5's `.btn-close` class which uses a background image.*

2. **Navigation Issues**
   - ~~Clicking on tabs in any dialog (load/save, browser settings) causes page reloads~~ ✓ Fixed
   - ~~URL changes to include fragment identifiers (e.g., `http://localhost:10240/#site-behaviour`)~~ ✓ Fixed
   - ~~Internal `<a href="...">` elements are being treated as real links instead of switching tabs~~ ✓ Fixed
   - ~~Need to replace with proper Bootstrap 5 tab navigation~~ ✓ Fixed
   
   *Tab navigation was fixed by updating `data-toggle="tab"` to `data-bs-toggle="tab"` in modal templates. The data attribute prefix change was missed in some Pug templates during Phase 3.*

3. **Functional Issues**
   - Share dialog is unpopulated (only shows "Loading...")
   - The sponsors view generates `Uncaught TypeError: Cannot read properties of undefined (reading 'backdrop')` in `initializeBackdrop` inside `model.js`

4. **Next Actions**
   - Continue replacing jQuery plugin methods with Bootstrap 5 API equivalents
   - Use `BootstrapUtils` compatibility layer consistently throughout codebase
   - ~~Fix tab navigation in modals (high priority)~~ ✓ Fixed
   - Address UI layout issues in dropdowns and dialogs
   - Fix TomSelect styling and dropdown arrows
   - Investigate IDE mode border styling issues
   - ~~Investigate remaining dialog appearance issues~~ ✓ Fixed

This plan will be updated as progress is made, with each completed step marked accordingly.