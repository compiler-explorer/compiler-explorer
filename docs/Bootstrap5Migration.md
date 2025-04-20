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
- **jQuery Plugin Methods Removal**: jQuery methods like `.popover()` and `.dropdown('toggle')` need to be replaced with code that uses the Bootstrap 5 API through a compatibility layer. Always use `BootstrapUtils` helper methods rather than direct jQuery plugin calls.
- **Grid and Form Class Renaming**: Bootstrap 5 renamed several core classes, such as changing `.form-row` to `.row`. This can cause subtle template selector issues in code that relies on these class names.
- **Don't Mix Data Attributes and JavaScript Modal Creation**: When creating modals via JavaScript (e.g., for dynamically loaded content), don't include `data-bs-toggle="modal"` on the trigger element unless you also add a matching `data-bs-target` attribute pointing to a valid modal element.
- **Modal Events Changed Significantly**: Bootstrap 5 modal events need to be attached directly to the native DOM element rather than jQuery objects, and the event parameter type is different. For proper typing, import the `Modal` type from bootstrap and use `Modal.Event` type.
- **Tooltip API Changed**: The global `window.bootstrap.Tooltip` reference no longer exists. Import the `Tooltip` class directly from bootstrap instead.
- **Input Group Structure Simplified**: Bootstrap 5 removed the need for `.input-group-prepend` and `.input-group-append` wrapper divs. Buttons and other controls can now be direct children of the `.input-group` container. This simplifies the markup but requires template updates.
- **TomSelect Widget Integration**: Bootstrap 5's switch from CSS triangles to SVG background images for dropdowns caused issues with TomSelect. Adding back custom CSS for dropdown arrows was necessary to maintain correct appearance.
- **Btn-block Removed**: Bootstrap 5 removed the `.btn-block` class. Instead, the recommended approach is to wrap buttons in a container with `.d-grid` and use standard `.btn` classes. This affects any full-width buttons in the application.
- **Element Selection for Components**: When working with Bootstrap 5 components, prefer passing CSS selectors to `BootstrapUtils` methods rather than jQuery objects, as this provides more consistent behavior.

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
- [x] Review card usage and update to Bootstrap 5 standards
- [x] ~~Replace any card-deck implementations with grid system~~ (Not needed - card-deck not used in codebase)
- [ ] Test card layouts, especially tab navigation within cards

### Collapse Component Updates
- [x] ~~Update any collapse component implementations~~ (Not needed - minimal collapse usage in codebase)
- [x] Test collapse functionality (limited to navbar hamburger menu on mobile)

### Button Group Updates
- [x] Review button group implementations
- [x] Update to Bootstrap 5 standards (no changes needed - Bootstrap 5 maintains same button group classes)
- [ ] Test button group functionality in toolbars and dropdown menus


## Phase 6: Form System Updates

- [x] Update form control classes to Bootstrap 5 standards
- [x] Update input group markup and classes
- [x] Update checkbox/radio markup to Bootstrap 5 standards
- [x] Update form validation classes and markup
- [x] ~~Consider implementing floating labels where appropriate (new in Bootstrap 5)~~ (Not needed for existing form usage)
- [x] Test form functionality and appearance

## Phase 7: Navbar Structure Updates

- [x] Update navbar structure in templates to match Bootstrap 5 requirements
- [x] Review custom navbar styling in explorer.scss
- [x] Test responsive behavior of navbar
- [x] Ensure mobile menu functionality works correctly
- [x] ~~Consider implementing offcanvas for mobile navigation (new in Bootstrap 5)~~ (Standard navbar collapse is sufficient for current needs)

## Phase 8: SCSS Variables and Theming

- [x] Review any custom SCSS that extends Bootstrap functionality
- [x] Update any custom themes to use Bootstrap 5 variables
- [x] Check z-index variable changes in Bootstrap 5
- [x] Add navbar container padding fix for proper alignment
- [ ] Test theme switching functionality

## Phase 9: Accessibility Improvements

- [x] Review ARIA attributes in custom component implementations
- [x] Leverage Bootstrap 5's improved accessibility features
- [x] Add ARIA labels and live regions for dynamic content
- [x] Enhance form controls with proper accessibility attributes
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
- [ ] Investigate and fix modal accessibility warnings
  - [ ] Address the warning: "Blocked aria-hidden on an element because its descendant retained focus"
  - [ ] Update modal template markup to leverage Bootstrap 5.3's built-in support for the `inert` attribute
  - [ ] Ensure proper focus management in modals for improved accessibility
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

- ✅ **Phase 5: Component Migration**
  - Updated each component type systematically
  - Used the compatibility layer consistently
  - Button group functions preserved with minimal changes
  - Card component updated to Bootstrap 5 standards
  - Collapse components required minimal changes

- ✅ **Phase 6: Form System Updates**
  - Updated form control classes to Bootstrap 5 standards
  - Replaced .form-group with .mb-3 for spacing
  - Updated checkbox implementation to .form-check pattern
  - Updated select elements to use .form-select
  - Simplified input groups by removing unnecessary wrappers
  - Improved accessibility with proper label-input associations

- ✅ **Phase 7: Navbar Structure Updates**
  - Added container-fluid wrapper for proper responsive behavior
  - Updated navbar classes and structure to Bootstrap 5 standards
  - Improved spacing and alignment for better mobile experience
  - Updated badge classes from badge-primary to bg-primary
  - Tested responsive behavior in mobile view

- ✅ **Phase 8: SCSS Variables and Theming**
  - Reviewed custom SCSS for Bootstrap compatibility
  - Confirmed z-index variables don't conflict
  - Added navbar container padding fix for proper alignment
  - Verified CSS variables for theme support
  
- ✅ **Phase 9: Accessibility Improvements**
  - Updated sr-only class to visually-hidden
  - Added proper aria-live regions for dynamic content
  - Enhanced form controls with aria-label attributes
  - Added role attributes for semantic meaning
  - Improved button accessibility with descriptive labels

### Current Phase
- 🔄 **Phase 10: Final Testing and Refinement**
  - Beginning comprehensive testing across viewports
  - Planning cross-browser compatibility checks

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

3. **Bugs to Fix (From Beta Testing)**
   - The X to close the community/alert notes is harder to see in dark mode than before
   - History view is broken (empty when clicking radio buttons)
   - TomSelect dropdowns for compilers are excessively long (both in executor view and normal view)
   - Default text/placeholder text is too dark, making it hard to read (especially "Compiler options")
   - ✓ Conformance view's "add compiler" functionality is broken (fixed: template selector was looking for `.form-row` which changed to `.row` in Bootstrap 5)
   - Dropdown in the library menu has changed color (possibly acceptable)
   - Layout has changed slightly in the library menu
   - The "popout" on the TomSelect compiler dropdown is misaligned
   - ✓ Need to check for more instances of old Bootstrap v4 code patterns (fixed: replaced `dropdown('toggle')` in main.ts with `BootstrapUtils.getDropdownInstance()` and `.toggle()`)
   - Check Sentry for additional errors on the beta site

## Final Testing Checklist

Before considering the Bootstrap 5 migration complete, the following areas should be thoroughly tested to ensure proper functionality and appearance:

### Modal Components
- **Settings Modal**
  - Open and close using the "Settings" option in the "More" dropdown
  - Test tab navigation between all tab sections (Colouring, Site behaviour, etc.)
  - Verify form controls within settings (checkboxes, selects, inputs)
  - Check that the close button works
  - Verify proper modal appearance/styling in both light and dark themes

- **Share Modal**
  - Open and close using the "Share" button
  - Verify the URL is generated correctly
  - Test the copy button
  - Check that social sharing buttons display correctly
  - Verify proper styling in both light and dark themes
  - Test "Copied to clipboard" tooltip functionality

- **Load/Save Modal**
  - Open and close using "Save" or "Load" options
  - Test tab navigation between sections (Examples, Browser-local storage, etc.)
  - Verify save functionality to browser storage
  - Test loading from browser storage
  - Check proper styling/layout in both light and dark themes

- **Compiler Picker Modal**
  - Open using the popout button next to a compiler selector
  - Test filter functionality (architecture, compiler type, search)
  - Verify compiler selection works
  - Check proper styling in both light and dark themes

- **Other Modals**
  - Test confirmation dialogs (alert.ts)
  - Test library selection modal
  - Test compiler overrides modal
  - Test runtime tools modal
  - Test templates modal
  - Verify proper styling in both light and dark themes

### Dropdown Components
- **Main Navigation Dropdowns**
  - Test "More" dropdown menu (all items work and have proper styling)
  - Test "Other" dropdown menu (all items work and have proper styling)
  - Verify dropdowns are properly positioned (not clipped)
  - Test on different screen sizes to ensure responsive behavior

- **Compiler Option Dropdowns**
  - Test filter dropdowns in compiler pane
  - Test "Add new..." dropdown
  - Test "Add tool..." dropdown
  - Test popular arguments dropdown
  - Verify proper positioning, especially for dropdowns at the right edge of the screen

- **Editor Dropdowns**
  - Test language selector dropdown
  - Test font size dropdown
  - Verify proper styling and positioning

- **TomSelect Dropdowns**
  - Test compiler selectors
  - Test library version selectors
  - Verify dropdown arrows appear correctly
  - Verify dropdown items are styled correctly

### Toast/Alert Components
- **Alert Notifications**
  - Trigger various notifications (info, warning, error)
  - Verify proper styling
  - Test auto-dismiss functionality
  - Check that close button works
  - Test stacking behavior of multiple notifications

- **Alert Dialogs**
  - Test info/warning/error alert dialogs (using the Alert class)
  - Verify proper styling and positioning
  - Check button functionality within dialogs

### Popover/Tooltip Components
- **Tooltips**
  - Hover over various buttons with tooltips (toolbar buttons, share button, etc.)
  - Verify tooltip text appears correctly
  - Check tooltip positioning (above/below/left/right of trigger)
  - Verify proper styling in both light and dark themes

- **Popovers**
  - Trigger popovers on compiler info
  - Check popover content displays correctly
  - Verify popover positioning
  - Test dismissal by clicking outside
  - Verify proper styling in both light and dark themes

### Card Components
- Check card styling in modals (Settings, Load/Save, etc.)
- Verify tab navigation within card headers
- Test card body content layout
- Check responsive behavior on different screen sizes

### Button Group Components
- **Toolbar Button Groups**
  - Test button groups in compiler pane toolbar
  - Test button groups in editor pane toolbar
  - Verify proper alignment and styling
  - Check dropdown buttons within button groups

- **Other Button Groups**
  - Test font size button group
  - Test bottom bar button groups
  - Verify proper styling in both light and dark themes

### Collapse Components
- Test mobile view hamburger menu
- Verify menu expands/collapses correctly
- Check that all menu items are accessible in collapsed mode

### Specialized Views
- **Conformance View**
  - Test compiler selectors and options
  - Verify results display correctly

- **Tree View (IDE Mode)**
  - Check tree structure and file display
  - Test right-click menus and dropdowns
  - Verify file manipulation controls

- **Visualization Components**
  - Test CFG view rendering and controls
  - Check opt pipeline viewer
  - Verify AST view

- **Sponsor Window**
  - Check sponsor list display
  - Verify modal dialog appearance and functionality

### Form Components
- Verify form control styling (inputs, selects, checkboxes)
- Test input groups with buttons
- Check validation states

### Responsive Behavior
- Test at various viewport sizes
- Verify mobile menu functionality
- Check input group stacking behavior

### Additional Specific Tests (from Beta Feedback)
- **Alert/Note Close Button**
  - Verify the X to close community notes is visible in dark mode
  - Check contrast and visibility in all themes

- **History View**
  - Verify the history view populates correctly when clicking radio buttons
  - Test all history functions (load, delete, etc.)

- **TomSelect Dropdowns**
  - Check the length of compiler dropdowns in executor view
  - Check the length of compiler dropdowns in normal view
  - Verify dropdown sizes are appropriate and consistent
  - Check that the popout button alignment is correct on all dropdowns

- **Placeholder Text**
  - Check visibility and contrast of placeholder text in all input fields
  - Specifically test "Compiler options" field visibility
  - Verify that all placeholder text is readable in both light and dark themes

- **Conformance View**
  - Test the "add compiler" functionality in conformance view
  - Verify that compilers can be added and removed
  - Check that conformance testing works end-to-end

- **Library Menu**
  - Compare library menu appearance with production site
  - Check dropdown colors and layout
  - Verify all library functionality works correctly

- **General Testing**
  - Check pixel-perfect alignment of elements compared to production site
  - Inspect for old Bootstrap 4 patterns in JavaScript (like `dropdown('toggle')`)
  - Test all components for unexpected behavioral differences

This plan will be updated as progress is made, with each completed step marked accordingly.
