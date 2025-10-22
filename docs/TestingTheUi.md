# UI Testing Checklist for Compiler Explorer

This document provides a checklist for testing the Compiler Explorer UI components. Use this checklist for
major UI changes, framework updates, or when implementing significant new features.

## Modal Components

### Settings Modal

- Open and close using the "Settings" option in the "More" dropdown
- Test tab navigation between all tab sections (Colouring, Site behaviour, etc.)
- Verify form controls within settings (checkboxes, selects, inputs)
- Check that the close button works
- Verify proper modal appearance/styling in both light and dark themes

### Share Modal

- Open and close using the "Share" button
- Verify the URL is generated correctly
- Test the copy button
- Check that social sharing buttons display correctly
- Verify proper styling in both light and dark themes
- Check the `<iframe>` of embed is correct HTML, including its various options
- Check the inner link of the iframe for embed works still (that is, embed mode)
- Test "Copied to clipboard" tooltip functionality

### Load/Save Modal

- Open and close using "Save" or "Load" options
- Test tab navigation between sections (Examples, Browser-local storage, etc.)
- Verify save functionality to browser storage
- Test loading from browser storage
- Check proper styling/layout in both light and dark themes

### Compiler Picker Modal

- Open using the popout button next to a compiler selector
- Test filter functionality (architecture, compiler type, search)
- Verify compiler selection works
- Check proper styling in both light and dark themes

### Other Modals

- Test confirmation dialogs (alert.ts)
- Test library selection modal
- Test compiler overrides modal
- Test runtime tools modal
- Test templates modal
- Verify proper styling in both light and dark themes

## Dropdown Components

### Main Navigation Dropdowns

- Test "More" dropdown menu (all items work and have proper styling)
- Test "Other" dropdown menu (all items work and have proper styling)
- Verify dropdowns are properly positioned (not clipped)
- Test on different screen sizes to ensure responsive behavior

### Compiler Option Dropdowns

- Test filter dropdowns in compiler pane
- Test "Add new..." dropdown
- Test "Add tool..." dropdown
- Test popular arguments dropdown
- Verify proper positioning, especially for dropdowns at the right edge of the screen

### Editor Dropdowns

- Test language selector dropdown
- Test font size dropdown
- Verify proper styling and positioning

### TomSelect Dropdowns

- Test compiler selectors
- Test library version selectors
- Verify dropdown arrows appear correctly
- Verify dropdown items are styled correctly

## Toast/Alert Components

### Alert Notifications

- Trigger various notifications (info, warning, error)
- Verify proper styling
- Test auto-dismiss functionality
- Check that close button works
- Test stacking behavior of multiple notifications

### Alert Dialogs

- Test info/warning/error alert dialogs (using the Alert class)
- Verify proper styling and positioning
- Check button functionality within dialogs

## Popover/Tooltip Components

### Tooltips

- Hover over various buttons with tooltips (toolbar buttons, share button, etc.)
- Verify tooltip text appears correctly
- Check tooltip positioning (above/below/left/right of trigger)
- Verify proper styling in both light and dark themes

### Popovers

- Trigger popovers on compiler info
- Check popover content displays correctly
- Verify popover positioning
- Test dismissal by clicking outside
- Verify proper styling in both light and dark themes

## Card Components

- Check card styling in modals (Settings, Load/Save, etc.)
- Verify tab navigation within card headers
- Test card body content layout
- Check responsive behavior on different screen sizes

## Button Group Components

### Toolbar Button Groups

- Test button groups in compiler pane toolbar
- Test button groups in editor pane toolbar
- Verify proper alignment and styling
- Check dropdown buttons within button groups

### Other Button Groups

- Test font size button group
- Test bottom bar button groups
- Verify proper styling in both light and dark themes

## Collapse Components

- Test mobile view hamburger menu
- Verify menu expands/collapses correctly
- Check that all menu items are accessible in collapsed mode

## Specialized Views

### Conformance View

- Test compiler selectors and options
- Verify results display correctly
- Test the "add compiler" functionality
- Verify that compilers can be added and removed
- Check that conformance testing works end-to-end

### Tree View (IDE Mode)

- Check tree structure and file display
- Test right-click menus and dropdowns
- Verify file manipulation controls

### Visualization Components

- Test CFG view rendering and controls
- Check opt pipeline viewer
- Verify AST view

### Sponsor Window

- Check sponsor list display
- Verify modal dialog appearance and functionality

## Form Components

- Verify form control styling (inputs, selects, checkboxes)
- Test input groups with buttons
- Check validation states

## Responsive Behavior

- Test at various viewport sizes
- Verify mobile menu functionality
- Check input group stacking behavior

## Runtime Tool Integration

### Runtime Tools

- Open the runtime tools window from compiler pane
- Change settings and click outside the modal to close
- Verify settings are properly saved
- Test with multiple runtime tool options
- Verify event handling properly handles modal opening/closing

### Emulation Features

- Test BBC emulation by clicking emulator links
- Check Z80 emulation features (e.g. https://godbolt.org/z/qnE7jhnvc)
- Verify emulator modals open properly
- Test interaction between emulator windows and the main interface

## Diff View

- Test changing compilers in both left and right panes
- Verify backgrounds remain themed correctly in dark mode
- Check that the diff view layout is correct (no excessive height)
- Confirm that input groups and buttons are properly sized
- Test different diff view types (Assembly, Compiler output, etc.)

## TomSelect and Input Components

### Compiler Selection Dropdowns

- Verify long compiler names display properly without excessive wrapping
- Check that dropdowns expand to fit compiler names rather than wrapping text
- Test the flex-grow behavior of dropdown elements
- Check the alignment of the popout button on all dropdowns
- Verify border colors in dark themes are appropriate

### Placeholder Text

- Check visibility and contrast of placeholder text in all input fields
- Specifically test "Compiler options" field visibility
- Verify that all placeholder text is readable in both light and dark themes

## Library Components

### Library Menu

- Check dropdown colors and layout
- Verify all library functionality works correctly
- Test adding and removing libraries
- Check library version selection

## History View

- Verify the history view populates correctly when clicking radio buttons
- Test all history functions (load, delete, etc.)

## General Testing

- Check pixel-perfect alignment of elements (compare with live site)
- Test all components for unexpected behavioral differences
- Verify theme switching works correctly for all components
- Test cross-browser compatibility (at least Firefox and Chrome)
- Check accessibility features (tab navigation, screen reader support)
