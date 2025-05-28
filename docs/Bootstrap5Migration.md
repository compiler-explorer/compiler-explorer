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

## Phases & Current Progress

### Phase 1: Dependency Updates and Basic Setup ✅

- [x] Update package.json with Bootstrap 5.3.5
- [x] Add @popperjs/core dependency (replacing Popper.js)
- [x] Update Tom Select theme from bootstrap4 to bootstrap5
- [x] Update main import statements where Bootstrap is initialized
- [x] Update webpack configuration if needed for Bootstrap 5 compatibility
- [x] Verify the application still builds and runs with basic functionality

#### Notes for Human Testers (Phase 1)

- At this phase, the application will have console errors related to Bootstrap component initialization
- Tom Select dropdowns (like compiler picker) may have styling differences with the new bootstrap5 theme
- Initial page load should still work, but dropdown functionality will be broken
- Modal dialogs like Settings and Sharing may not function correctly
- The primary layout and basic display of panes should still function

### Phase 2: Global CSS Class Migration ✅

- [x] Update directional utility classes (ml/mr → ms/me)
    - [x] Search and replace in .pug templates
    - [x] Search and replace in .scss files
    - [x] Search and replace in JavaScript/TypeScript files that generate HTML
- [x] Update floating utility classes (float-left/right → float-start/end)
- [x] Update text alignment classes (text-left/right → text-start/end)
- [x] Update other renamed classes (badge-pill → rounded-pill, etc.)
- [x] Test and verify styling changes

#### Notes for Human Testers (Phase 2)

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

### Phase 3: HTML Attribute Updates ✅

- [x] Update data attributes across the codebase
    - [x] data-toggle → data-bs-toggle
    - [x] data-target → data-bs-target
    - [x] data-dismiss → data-bs-dismiss
    - [x] data-ride → data-bs-ride (not used in codebase)
    - [x] data-spy → data-bs-spy (not used in codebase)
    - [x] Other data attributes as needed
- [x] Test components to ensure they function correctly with new attributes

#### Notes for Human Testers (Phase 3)

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

### Phase 4: JavaScript API Compatibility Layer ✅

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

#### Notes for Human Testers (Phase 4)

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

### Phase 5: Component Migration (By Component Type) ✅

#### Modal Component Migration

- [x] Update modal implementation in alert.ts
- [x] Update modal usage in compiler-picker-popup.ts
- [x] Update modal handling in load-save.ts
- [x] Update modal event handling in sharing.ts
- [x] Test modal functionality thoroughly

#### Dropdown Component Migration

- [x] Update dropdown handling in sharing.ts
- [x] Update dropdown usage in compiler.ts, editor.ts, etc.
- [x] Test dropdown functionality thoroughly

#### Toast/Alert Component Migration

- [x] Update toast implementation in alert.ts
- [x] Update toast styling in explorer.scss
- [x] Test toast notifications and alerts

#### Popover/Tooltip Migration

- [x] Update tooltip initialization in sharing.ts
- [x] Update popover usage in compiler.ts, executor.ts, editor.ts, etc.
- [x] Test popover and tooltip functionality thoroughly

#### Card Component Updates

- [x] Review card usage and update to Bootstrap 5 standards
- [x] ~~Replace any card-deck implementations with grid system~~ (Not needed - card-deck not used in codebase)
- [x] Test card layouts, especially tab navigation within cards

#### Collapse Component Updates

- [x] ~~Update any collapse component implementations~~ (Not needed - minimal collapse usage in codebase)
- [x] Test collapse functionality (limited to navbar hamburger menu on mobile)

#### Button Group Updates

- [x] Review button group implementations
- [x] Update to Bootstrap 5 standards (no changes needed - Bootstrap 5 maintains same button group classes)
- [x] Test button group functionality in toolbars and dropdown menus

### Phase 6: Form System Updates ✅

- [x] Update form control classes to Bootstrap 5 standards
- [x] Update input group markup and classes
- [x] Update checkbox/radio markup to Bootstrap 5 standards
- [x] Update form validation classes and markup
- [x] ~~Consider implementing floating labels where appropriate (new in Bootstrap 5)~~ (Not needed for existing form
  usage)
- [x] Test form functionality and appearance

### Phase 7: Navbar Structure Updates ✅

- [x] Update navbar structure in templates to match Bootstrap 5 requirements
- [x] Review custom navbar styling in explorer.scss
- [x] Test responsive behavior of navbar
- [x] Ensure mobile menu functionality works correctly
- [x] ~~Consider implementing offcanvas for mobile navigation (new in Bootstrap 5)~~ (Standard navbar collapse is
  sufficient for current needs)

### Phase 8: SCSS Variables and Theming ✅

- [x] Review any custom SCSS that extends Bootstrap functionality
- [x] Update any custom themes to use Bootstrap 5 variables
- [x] Check z-index variable changes in Bootstrap 5
- [x] Add navbar container padding fix for proper alignment
- [x] Test theme switching functionality

### Phase 9: Accessibility Improvements ✅

- [x] Review ARIA attributes in custom component implementations
- [x] Leverage Bootstrap 5's improved accessibility features
- [x] Add ARIA labels and live regions for dynamic content
- [x] Enhance form controls with proper accessibility attributes
- [ ] ~~Test with screen readers and keyboard navigation~~ (left for future work)
- [ ] ~~Ensure color contrast meets accessibility guidelines~~ (left for future work)

### Phase 10: Final Testing and Refinement ✅

- [x] ~~Comprehensive testing across different viewports~~ cursory testing with a few viewports
- [x] Cross-browser testing (at least; looked in FireFox and we're good)
- [x] Fix any styling issues or inconsistencies
- [x] ~~Performance testing (Bootstrap 5 should be more performant)~~ (don't care; site is fine)
- [x] Ensure no regressions in functionality

## Key Learnings From Implementation

These insights were gathered during the migration process and may be helpful for future reference:

- **Data Attributes Still Work Without JavaScript Initialization**: Despite some documentation suggesting otherwise,
  Bootstrap 5 components with data attributes (like tabs) still work without explicit JavaScript initialization. The key
  is using the correct `data-bs-*` prefix.
- **Close Button Implementation Completely Changed**: Bootstrap 4 used `.close` class with a `&times;` entity inside a
  span, while Bootstrap 5 uses `.btn-close` class with a background image and no inner content.
- **Tab Navigation Issues**: The tab navigation problems were fixed by simply updating data attributes, not by adding
  JavaScript initialization.
- **jQuery Plugin Methods Removal**: jQuery methods like `.popover()` and `.dropdown('toggle')` need to be replaced with
  code that uses the Bootstrap 5 API through a compatibility layer. Always use `BootstrapUtils` helper methods rather
  than direct jQuery plugin calls.
- **Grid and Form Class Renaming**: Bootstrap 5 renamed several core classes, such as changing `.form-row` to `.row`.
  This can cause subtle template selector issues in code that relies on these class names.
- **Don't Mix Data Attributes and JavaScript Modal Creation**: When creating modals via JavaScript (e.g., for
  dynamically loaded content), don't include `data-bs-toggle="modal"` on the trigger element unless you also add a
  matching `data-bs-target` attribute pointing to a valid modal element.
- **Modal Events Changed Significantly**: Bootstrap 5 modal events need to be attached directly to the native DOM
  element rather than jQuery objects, and the event parameter type is different. For proper typing, import the `Modal`
  type from bootstrap and use `Modal.Event` type.
- **jQuery Event Binding vs Native DOM Events**: Bootstrap 5 requires native DOM event binding instead of jQuery's
  `.on()` method. Replace `$(selector).on('shown.bs.modal', handler)` with
  `domElement.addEventListener('shown.bs.modal', handler)`. This is particularly important for modal events like '
  shown.bs.modal'.
- **Tooltip API Changed**: The global `window.bootstrap.Tooltip` reference no longer exists. Import the `Tooltip` class
  directly from bootstrap instead.
- **Input Group Structure Simplified**: Bootstrap 5 removed the need for `.input-group-prepend` and
  `.input-group-append` wrapper divs. Buttons and other controls can now be direct children of the `.input-group`
  container. This simplifies the markup but requires template updates.
- **TomSelect Widget Integration**: Bootstrap 5's switch from CSS triangles to SVG background images for dropdowns
  caused issues with TomSelect. Adding back custom CSS for dropdown arrows was necessary to maintain correct appearance.
- **Btn-block Removed**: Bootstrap 5 removed the `.btn-block` class. Instead, the recommended approach is to wrap
  buttons in a container with `.d-grid` and use standard `.btn` classes. This affects any full-width buttons in the
  application.
- **Element Selection for Components**: When working with Bootstrap 5 components, prefer passing CSS selectors to
  `BootstrapUtils` methods rather than jQuery objects, as this provides more consistent behavior.

## Fixed Issues & Completed Work

### UI Layout & Display Issues

- [x] Font dropdown styling fixed
- [x] Templates view proportions fixed (min-width added to columns and modal)
- [x] Dialog appearance fixed (updated close buttons to use `.btn-close`)
- [x] Dropdown positioning fixed (updated to `.dropdown-menu-end`)
- [x] TomSelect dropdown arrows fixed (custom CSS implementation)
- [x] IDE mode border styling improved (temporarily with `.list-group-flush`)
- [x] Sponsors window styling fixed (replaced `.btn-block` with `.d-grid` approach)
- [x] The X to close the community/alert notes is harder to see in dark mode than before (fixed: added
  `filter: invert(100%)` to make btn-close buttons visible in dark themes)
- [x] TomSelect dropdowns for compilers are excessively long (both in executor view and normal view) (fixed manually)
- [x] Default text/placeholder text is too dark, making it hard to read (especially "Compiler options") (fixed manually)
- [x] Dropdown in the library menu has changed color (fixed: updated `.custom-select` to `.form-select` in theme files)
- [x] ~~Layout has changed slightly in the library menu~~ (decided it looks better now)
- [x] The "popout" on the TomSelect compiler dropdown is misaligned (fixed: updated styling for TomSelect components)
- [x] Compiler combobox rounding overlaps left border by 1 pixel (fixed: overrode CSS variables to reset Bootstrap 5's
  negative margin)
- [x] Diff view - changing left/right side compiler/window turns combobox to a white background (fixed: removed
  form-select class to avoid transparent background)
- [x] The popular arguments dropdown at the right of the options isn't properly aligned (fixed: updated dropdown styling
  in compiler.pug)
- [x] Long compiler names wrap instead of widening the dropdown (fixed: improved styling for TomSelect dropdowns)

### Navigation & Functional Issues

- [x] Tab navigation fixed (updated data attributes to `data-bs-toggle="tab"`)
- [x] Share dialog functionality fixed (proper Bootstrap 5 modal initialization)
- [x] Sponsors modal error fixed (removed conflicting data attributes)
- [x] Share dropdown tooltip conflict fixed (moved tooltip to parent element)
- [x] History view is broken (empty when clicking radio buttons) (fixed: updated modal event binding from jQuery's
  `.on('shown.bs.modal')` to native DOM `addEventListener('shown.bs.modal')`)
- [x] Conformance view's "add compiler" functionality is broken (fixed: template selector was looking for `.form-row`
  which changed to `.row` in Bootstrap 5)
- [x] Need to check for more instances of old Bootstrap v4 code patterns (fixed: replaced `dropdown('toggle')` in
  main.ts with `BootstrapUtils.getDropdownInstance()` and `.toggle()`)
- [x] Runtime tools window is broken - doesn't save settings anymore (fixed: updated modal hide event handling with
  setElementEventHandler)
- [x] Emulation functionality is broken due to modal issues (fixed: replaced direct .modal() calls with
  BootstrapUtils.showModal)

### Code Structure Improvements

- [x] Custom classes in runtime tools selection (`.custom-runtimetool`) and overrides selection (`.custom-override`) -
  removed as they were superfluous
- [x] `.form-row` still used in theme files (dark-theme.scss, one-dark-theme.scss, pink-theme.scss) - replaced with
  standard `.row`
- [x] Border directional properties in explorer.scss updated for better RTL support - added `border-inline-start` and
  border radius logical properties with appropriate fallbacks for older browsers
- [x] Input group structures verified - all instances of the deprecated `.input-group-prepend` and `.input-group-append`
  have already been updated to use Bootstrap 5's simplified approach
- [x] Toast header close button styling verified - explorer.scss already uses `.btn-close` consistently for toast
  components
- [x] Event handlers verified - history-widget.ts and sharing.ts are correctly using native DOM addEventListener methods
  with the appropriate Bootstrap 5 event names

## Future Work

### Phase 11: Documentation Update ✅

- [x] Update any documentation that references Bootstrap components
- [x] Document custom component implementations
- [x] Note any deprecated features or changes in functionality

### Phase 12: Optional jQuery Removal and Cleanup

- [x] ~~Create a plan for jQuery removal (if desired)~~ (tracked in [issue #7600](https://github.com/compiler-explorer/compiler-explorer/issues/7600))
- [x] ~~Identify non-Bootstrap jQuery usage that would need refactoring~~ (tracked in [issue #7600](https://github.com/compiler-explorer/compiler-explorer/issues/7600))
- [x] ~~Remove the temporary `bootstrap-utils.ts` compatibility layer~~ (Decision: Keep this utility for the foreseeable future as it provides valuable functionality for jQuery-Bootstrap 5 integration)
    - [x] ~~Replace all uses with direct Bootstrap 5 API calls~~ (Not necessary - updated documentation to indicate direct API usage when possible)
    - [x] ~~Document the native Bootstrap 5 API for future reference~~ (Added documentation in the utilities themselves)
- [ ] ~~Investigate and fix modal accessibility warnings~~ (tracked in [issue #7602](https://github.com/compiler-explorer/compiler-explorer/issues/7602))
    - [ ] ~~Address the warning: "Blocked aria-hidden on an element because its descendant retained focus"~~ (part of issue #7602)
    - [ ] ~~Update modal template markup to leverage Bootstrap 5.3's built-in support for the `inert` attribute~~ (part of issue #7602)
    - [ ] ~~Ensure proper focus management in modals for improved accessibility~~ (part of issue #7602)

### Additional Pending Issues

- [ ] Check Sentry for additional errors on the live site
- [ ] ~~Investigate the "focus" selected check boxes in the settings view. They're very light when focused, in particular in pink theme. I couldn't work out how to fix this, but it seemed minor.~~ (tracked in [issue #7603](https://github.com/compiler-explorer/compiler-explorer/issues/7603))
- [ ] ~~The "pop out" div that's attached to the compiler picker doesn't work on the conformance view: this was broken before. Essentially the z-order means it's drawn behind the lower conformance compilers and `z-index` can't fix it. Needs a rethink of how this is done.~~ (tracked in [issue #7604](https://github.com/compiler-explorer/compiler-explorer/issues/7604))
- [x] ~~File tracking issues for anything on this list we don't complete.~~ (completed - all issues have been tracked)

## Custom Component Implementations Reference

This section provides documentation for the custom Bootstrap component implementations developed during the migration.

### BootstrapUtils (bootstrap-utils.ts)

The `bootstrap-utils.ts` file serves as a temporary compatibility layer between jQuery-based Bootstrap 4 code and the new Bootstrap 5 JavaScript API. It provides methods for initializing and controlling various Bootstrap components.

#### Modal Component

**Methods:**
- `showModal(selector)`: Displays a modal using a CSS selector or jQuery object
- `hideModal(selector)`: Hides a modal using a CSS selector or jQuery object
- `setModalHiddenHandler(selector, handler)`: Sets a handler for the 'hidden.bs.modal' event
- `setModalShownHandler(selector, handler)`: Sets a handler for the 'shown.bs.modal' event
- `getModalInstance(selector)`: Gets the Bootstrap Modal instance for a given element

**Key Changes from Bootstrap 4:**
- Modal events like 'shown.bs.modal' now need native DOM event listeners
- Modal objects are obtained using `bootstrap.Modal.getInstance()` or `new bootstrap.Modal()`
- jQuery's `.modal('show')` is replaced with `modalInstance.show()`

#### Dropdown Component

**Methods:**
- `getDropdownInstance(selector)`: Gets the Bootstrap Dropdown instance for a given element
- `initializeAllDropdowns()`: Initializes all dropdowns on the page

**Key Changes from Bootstrap 4:**
- Dropdown toggling with jQuery's `.dropdown('toggle')` is replaced with `dropdownInstance.toggle()`
- Data attributes changed from `data-toggle="dropdown"` to `data-bs-toggle="dropdown"`

#### Tooltip Component

**Methods:**
- `createTooltip(element, options)`: Creates a tooltip instance on an element
- `initializeAllTooltips(selector, options)`: Initializes all tooltips matching a selector

**Key Changes from Bootstrap 4:**
- Tooltips must be explicitly initialized with `new bootstrap.Tooltip()`
- Global tooltip access changed from `$.fn.tooltip` to direct import from bootstrap
- Data attribute changed from `data-toggle="tooltip"` to `data-bs-toggle="tooltip"`

#### Popover Component

**Methods:**
- `createPopover(element, options)`: Creates a popover instance on an element
- `initializeAllPopovers(selector, options)`: Initializes all popovers matching a selector

**Key Changes from Bootstrap 4:**
- Popovers must be explicitly initialized with `new bootstrap.Popover()`
- Data attribute changed from `data-toggle="popover"` to `data-bs-toggle="popover"`

#### Tab Component

**Methods:**
- `activateTab(selector)`: Activates a specific tab

**Key Changes from Bootstrap 4:**
- Tab activation changed from `$(selector).tab('show')` to `tabInstance.show()`
- Data attribute changed from `data-toggle="tab"` to `data-bs-toggle="tab"`

#### Toast Component

**Methods:**
- `createToast(element, options)`: Creates a toast instance on an element
- `showToast(selector)`: Shows a toast using a CSS selector or jQuery object

**Key Changes from Bootstrap 4:**
- Toasts must be explicitly initialized with `new bootstrap.Toast()`
- Toast show/hide methods are direct methods on the toast instance

### Event Handling

**Key Changes:**
- jQuery event methods (`.on()`, `.off()`, etc.) should be replaced with native DOM methods
- Event registration for Bootstrap components now requires direct DOM element access
- Event types include the `bs` prefix (e.g., `shown.bs.modal`)

### CSS Class Changes

Several Bootstrap CSS classes were renamed in version 5:

- Directional classes: `ml-*` → `ms-*`, `mr-*` → `me-*`, etc.
- Floating classes: `float-left` → `float-start`, `float-right` → `float-end`
- Text alignment: `text-left` → `text-start`, `text-right` → `text-end`
- Form classes: `custom-select` → `form-select`, `form-row` → `row`
- Close button: `.close` → `.btn-close` (with completely different HTML structure)
- Dropdown alignment: `.dropdown-menu-right` → `.dropdown-menu-end`
- Full-width buttons: `.btn-block` → Container with `.d-grid`

### Special Integration Notes

**TomSelect Integration:**
- Bootstrap 5 changed dropdown indicators from CSS triangles to SVG backgrounds
- Custom CSS was needed in explorer.scss to fix dropdown arrow appearance
- The bootstrap5 theme for TomSelect needed additional styling fixes

**Modal Focus Management:**
- Bootstrap 5 has stricter focus management in modals
- Focus must be properly handled when modals are shown/hidden
- The `inert` attribute is now supported for better accessibility

## Deprecated Features

The following Bootstrap 4 features were deprecated or removed in Bootstrap 5:

1. **jQuery Dependency**: Bootstrap 5 no longer requires jQuery
2. **jQuery Plugin Methods**: Methods like `.modal('show')` no longer exist
3. **Global Bootstrap Access**: No more `$.fn.modal` or similar
4. **Card Decks**: Replaced by grid system
5. **Form Row**: `.form-row` replaced by standard `.row`
6. **Input Group Prepend/Append**: Wrappers removed, children can be direct
7. **Button Block**: `.btn-block` removed in favor of `.d-grid`
8. **Close Class**: `.close` replaced by `.btn-close` with different HTML structure

## Final Testing Checklist

Before considering the Bootstrap 5 migration complete, a comprehensive UI testing checklist was created and used to
verify functionality. This checklist has been completed with all tests passing. The tests cover all major UI components
that could be affected by the Bootstrap migration.

The checklist included:

- Modal dialogs (Settings, Share, Load/Save, etc.)
- Dropdown components (navigation, compiler options, TomSelect)
- Toast/Alert components
- Popovers and tooltips
- Card, button group, and form components
- Specialized views (Conformance, Tree, Visualization)
- Responsive behavior

A permanent version of this UI testing checklist has been created as a separate document and can be used for testing
future UI changes or upgrades: [UI Testing Checklist](TestingTheUi.md)

## Notes for Implementation

1. **Make minimal changes** in each step to allow for easier testing and troubleshooting
2. **Test thoroughly** after each phase before moving to the next
3. **Document issues** encountered during migration for future reference
4. **Focus on accessibility** to ensure the site remains accessible throughout changes
5. **Maintain browser compatibility** with all currently supported browsers
6. **Consider performance implications** of the changes
7. **NEVER mark any issue as fixed in this document** until you have explicit confirmation from the reviewer that the
   issue is completely resolved
8. **NEVER commit changes** until you have explicit confirmation that the fix works correctly

## Technical References

- [Bootstrap 5 Migration Guide](https://getbootstrap.com/docs/5.0/migration/)
- [Bootstrap 5 Components Documentation](https://getbootstrap.com/docs/5.3/components/)
- [Bootstrap 5 Utilities Documentation](https://getbootstrap.com/docs/5.3/utilities/)
- [Bootstrap 5 Forms Documentation](https://getbootstrap.com/docs/5.3/forms/overview/)
- [Popper v2 Documentation](https://popper.js.org/docs/v2/)