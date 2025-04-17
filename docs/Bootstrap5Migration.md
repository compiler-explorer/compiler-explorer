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

## Phase 2: Global CSS Class Migration

- [ ] Update directional utility classes (ml/mr → ms/me)
    - [ ] Search and replace in .pug templates
    - [ ] Search and replace in .scss files
    - [ ] Search and replace in JavaScript/TypeScript files that generate HTML
- [ ] Update floating utility classes (float-left/right → float-start/end)
- [ ] Update text alignment classes (text-left/right → text-start/end)
- [ ] Update other renamed classes (badge-pill → rounded-pill, etc.)
- [ ] Test and verify styling changes

## Phase 3: HTML Attribute Updates

- [ ] Update data attributes across the codebase
    - [ ] data-toggle → data-bs-toggle
    - [ ] data-target → data-bs-target
    - [ ] data-dismiss → data-bs-dismiss
    - [ ] data-ride → data-bs-ride
    - [ ] data-spy → data-bs-spy
    - [ ] Other data attributes as needed
- [ ] Test components to ensure they function correctly with new attributes

## Phase 4: JavaScript API Compatibility Layer

- [ ] Create a Bootstrap compatibility utility module to abstract component initialization
  - [ ] This will help transition from jQuery-based initialization to native JS
  - [ ] It will also make future jQuery removal easier if desired
- [ ] Define methods for each component type (Modal, Dropdown, Toast, etc.)
- [ ] Implement both jQuery and native JS paths depending on configuration
- [ ] Test the compatibility layer with basic components

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

## Phase 12: Optional jQuery Removal (Future Work)

- [ ] Create plan for jQuery removal (if desired)
- [ ] Identify non-Bootstrap jQuery usage that would need refactoring
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

This plan will be updated as progress is made, with each completed step marked accordingly.