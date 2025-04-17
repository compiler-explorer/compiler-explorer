// Copyright (c) 2025, Compiler Explorer Authors
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright notice,
//       this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

/**
 * TEMPORARY COMPATIBILITY LAYER
 *
 * This module provides utilities to help transition from Bootstrap 4's jQuery-based API
 * to Bootstrap 5's vanilla JavaScript API. This is intended as a temporary solution
 * during the migration from Bootstrap 4 to 5 and should be removed once the migration
 * is complete.
 *
 * The goal is to minimize changes throughout the codebase by centralizing the Bootstrap
 * API changes in this file, while still allowing for gradual migration to direct API calls.
 *
 * @deprecated This module should be removed after the Bootstrap 5 migration is complete.
 */

import $ from 'jquery';

// Import Bootstrap JS
import 'bootstrap';

// Type definitions for Bootstrap 5 objects
declare global {
    interface Window {
        bootstrap: {
            Modal: ModalClass;
            Dropdown: DropdownClass;
            Toast: ToastClass;
            Tooltip: TooltipClass;
            Popover: PopoverClass;
            Tab: TabClass;
            Collapse: CollapseClass;
        };
    }

    interface ModalClass {
        new (element: HTMLElement, options?: any): ModalInstance;
        getInstance(element: HTMLElement): ModalInstance | null;
    }

    interface ModalInstance {
        show(): void;
        hide(): void;
        toggle(): void;
        dispose(): void;
    }

    interface DropdownClass {
        new (element: HTMLElement, options?: any): DropdownInstance;
        getInstance(element: HTMLElement): DropdownInstance | null;
    }

    interface DropdownInstance {
        show(): void;
        hide(): void;
        toggle(): void;
        dispose(): void;
    }

    interface ToastClass {
        new (element: HTMLElement, options?: any): ToastInstance;
        getInstance(element: HTMLElement): ToastInstance | null;
    }

    interface ToastInstance {
        show(): void;
        hide(): void;
        dispose(): void;
    }

    interface TooltipClass {
        new (element: HTMLElement, options?: any): TooltipInstance;
        getInstance(element: HTMLElement): TooltipInstance | null;
    }

    interface TooltipInstance {
        show(): void;
        hide(): void;
        toggle(): void;
        dispose(): void;
    }

    interface PopoverClass {
        new (element: HTMLElement, options?: any): PopoverInstance;
        getInstance(element: HTMLElement): PopoverInstance | null;
    }

    interface PopoverInstance {
        show(): void;
        hide(): void;
        toggle(): void;
        dispose(): void;
        update(): void;
    }

    interface TabClass {
        new (element: HTMLElement): TabInstance;
        getInstance(element: HTMLElement): TabInstance | null;
    }

    interface TabInstance {
        show(): void;
        dispose(): void;
    }

    interface CollapseClass {
        new (element: HTMLElement, options?: any): CollapseInstance;
        getInstance(element: HTMLElement): CollapseInstance | null;
    }

    interface CollapseInstance {
        show(): void;
        hide(): void;
        toggle(): void;
        dispose(): void;
    }
}

export class BootstrapUtils {
    /**
     * Initialize a modal
     * @param elementOrSelector Element or selector for the modal
     * @param options Modal options
     * @returns Modal instance
     * @throws Error if the element cannot be found
     */
    static initModal(elementOrSelector: string | HTMLElement | JQuery, options?: any): ModalInstance {
        const element = BootstrapUtils.getElement(elementOrSelector);
        if (!element) throw new Error(`Failed to find element for modal: ${elementOrSelector}`);

        return new window.bootstrap.Modal(element, options);
    }

    /**
     * Initialize a modal if the element exists, returning null otherwise
     * @param elementOrSelector Element or selector for the modal
     * @param options Modal options
     * @returns Modal instance or null if the element cannot be found
     */
    static initModalIfExists(elementOrSelector: string | HTMLElement | JQuery, options?: any): ModalInstance | null {
        const element = BootstrapUtils.getElement(elementOrSelector);
        if (!element) return null;

        return new window.bootstrap.Modal(element, options);
    }

    /**
     * Get an existing modal instance for an element
     * @param elementOrSelector Element or selector for the modal
     * @returns Existing modal instance or null if not found
     */
    static getModalInstance(elementOrSelector: string | HTMLElement | JQuery): ModalInstance | null {
        const element = BootstrapUtils.getElement(elementOrSelector);
        if (!element) return null;

        return window.bootstrap.Modal.getInstance(element);
    }

    /**
     * Show a modal
     * @param elementOrSelector Element or selector for the modal
     */
    static showModal(elementOrSelector: string | HTMLElement | JQuery): void {
        const element = BootstrapUtils.getElement(elementOrSelector);
        if (!element) return;

        const modal = window.bootstrap.Modal.getInstance(element) || new window.bootstrap.Modal(element);
        modal.show();
    }

    /**
     * Hide a modal
     * @param elementOrSelector Element or selector for the modal
     */
    static hideModal(elementOrSelector: string | HTMLElement | JQuery): void {
        const element = BootstrapUtils.getElement(elementOrSelector);
        if (!element) return;

        const modal = window.bootstrap.Modal.getInstance(element);
        if (modal) modal.hide();
    }

    /**
     * Initialize a toast
     * @param elementOrSelector Element or selector for the toast
     * @param options Toast options
     * @returns Toast instance
     */
    static initToast(elementOrSelector: string | HTMLElement | JQuery, options?: any): ToastInstance {
        const element = BootstrapUtils.getElement(elementOrSelector);
        if (!element) throw new Error(`Failed to find element for toast: ${elementOrSelector}`);

        return new window.bootstrap.Toast(element, options);
    }

    /**
     * Initialize a toast if the element exists
     * @param elementOrSelector Element or selector for the toast
     * @param options Toast options
     * @returns Toast instance or null if element doesn't exist
     */
    static initToastIfExists(elementOrSelector: string | HTMLElement | JQuery, options?: any): ToastInstance | null {
        const element = BootstrapUtils.getElement(elementOrSelector);
        if (!element) return null;

        return new window.bootstrap.Toast(element, options);
    }

    /**
     * Show a toast
     * @param elementOrSelector Element or selector for the toast
     */
    static showToast(elementOrSelector: string | HTMLElement | JQuery): void {
        const element = BootstrapUtils.getElement(elementOrSelector);
        if (!element) return;

        const toast = window.bootstrap.Toast.getInstance(element) || new window.bootstrap.Toast(element);
        toast.show();
    }

    /**
     * Hide a toast
     * @param elementOrSelector Element or selector for the toast
     */
    static hideToast(elementOrSelector: string | HTMLElement | JQuery): void {
        const element = BootstrapUtils.getElement(elementOrSelector);
        if (!element) return;

        const toast = window.bootstrap.Toast.getInstance(element);
        if (toast) toast.hide();
    }

    /**
     * Initialize a dropdown
     * @param elementOrSelector Element or selector for the dropdown
     * @param options Dropdown options
     * @returns Dropdown instance
     * @throws Error if the element cannot be found
     */
    static initDropdown(elementOrSelector: string | HTMLElement | JQuery, options?: any): DropdownInstance {
        const element = BootstrapUtils.getElement(elementOrSelector);
        if (!element) throw new Error(`Failed to find element for dropdown: ${elementOrSelector}`);

        return new window.bootstrap.Dropdown(element, options);
    }

    /**
     * Initialize a dropdown if the element exists, returning null otherwise
     * @param elementOrSelector Element or selector for the dropdown
     * @param options Dropdown options
     * @returns Dropdown instance or null if the element cannot be found
     */
    static initDropdownIfExists(
        elementOrSelector: string | HTMLElement | JQuery,
        options?: any,
    ): DropdownInstance | null {
        const element = BootstrapUtils.getElement(elementOrSelector);
        if (!element) return null;

        return new window.bootstrap.Dropdown(element, options);
    }

    /**
     * Get an existing dropdown instance for an element
     * @param elementOrSelector Element or selector for the dropdown
     * @returns Existing dropdown instance or null if not found
     */
    static getDropdownInstance(elementOrSelector: string | HTMLElement | JQuery): DropdownInstance | null {
        const element = BootstrapUtils.getElement(elementOrSelector);
        if (!element) return null;

        return window.bootstrap.Dropdown.getInstance(element);
    }

    /**
     * Show a dropdown
     * @param elementOrSelector Element or selector for the dropdown
     */
    static showDropdown(elementOrSelector: string | HTMLElement | JQuery): void {
        const element = BootstrapUtils.getElement(elementOrSelector);
        if (!element) return;

        const dropdown = window.bootstrap.Dropdown.getInstance(element) || new window.bootstrap.Dropdown(element);
        dropdown.show();
    }

    /**
     * Hide a dropdown
     * @param elementOrSelector Element or selector for the dropdown
     */
    static hideDropdown(elementOrSelector: string | HTMLElement | JQuery): void {
        const element = BootstrapUtils.getElement(elementOrSelector);
        if (!element) return;

        const dropdown = window.bootstrap.Dropdown.getInstance(element);
        if (dropdown) dropdown.hide();
    }

    /**
     * Initialize a tooltip
     * @param elementOrSelector Element or selector for the tooltip
     * @param options Tooltip options
     * @returns Tooltip instance
     */
    static initTooltip(elementOrSelector: string | HTMLElement | JQuery, options?: any): TooltipInstance {
        const element = BootstrapUtils.getElement(elementOrSelector);
        if (!element) throw new Error(`Failed to find element for tooltip: ${elementOrSelector}`);

        return new window.bootstrap.Tooltip(element, options);
    }

    /**
     * Initialize a tooltip if the element exists
     * @param elementOrSelector Element or selector for the tooltip
     * @param options Tooltip options
     * @returns Tooltip instance or null if element doesn't exist
     */
    static initTooltipIfExists(
        elementOrSelector: string | HTMLElement | JQuery,
        options?: any,
    ): TooltipInstance | null {
        const element = BootstrapUtils.getElement(elementOrSelector);
        if (!element) return null;

        return new window.bootstrap.Tooltip(element, options);
    }

    /**
     * Initialize a popover
     * @param elementOrSelector Element or selector for the popover
     * @param options Popover options
     * @returns Popover instance
     * @throws Error if the element cannot be found
     */
    static initPopover(elementOrSelector: string | HTMLElement | JQuery, options?: any): PopoverInstance {
        const element = BootstrapUtils.getElement(elementOrSelector);
        if (!element) throw new Error(`Failed to find element for popover: ${elementOrSelector}`);

        return new window.bootstrap.Popover(element, options);
    }

    /**
     * Initialize a popover if the element exists, returning null otherwise
     * @param elementOrSelector Element or selector for the popover
     * @param options Popover options
     * @returns Popover instance or null if the element cannot be found
     */
    static initPopoverIfExists(
        elementOrSelector: string | HTMLElement | JQuery,
        options?: any,
    ): PopoverInstance | null {
        const element = BootstrapUtils.getElement(elementOrSelector);
        if (!element) return null;

        return new window.bootstrap.Popover(element, options);
    }

    /**
     * Get an existing popover instance for an element
     * @param elementOrSelector Element or selector for the popover
     * @returns Existing popover instance or null if not found
     */
    static getPopoverInstance(elementOrSelector: string | HTMLElement | JQuery): PopoverInstance | null {
        const element = BootstrapUtils.getElement(elementOrSelector);
        if (!element) return null;

        return window.bootstrap.Popover.getInstance(element);
    }

    /**
     * Initialize a tab
     * @param elementOrSelector Element or selector for the tab
     * @returns Tab instance
     */
    static initTab(elementOrSelector: string | HTMLElement | JQuery): TabInstance {
        const element = BootstrapUtils.getElement(elementOrSelector);
        if (!element) throw new Error(`Failed to find element for tab: ${elementOrSelector}`);

        return new window.bootstrap.Tab(element);
    }

    /**
     * Initialize a tab if the element exists
     * @param elementOrSelector Element or selector for the tab
     * @returns Tab instance or null if element doesn't exist
     */
    static initTabIfExists(elementOrSelector: string | HTMLElement | JQuery): TabInstance | null {
        const element = BootstrapUtils.getElement(elementOrSelector);
        if (!element) return null;

        return new window.bootstrap.Tab(element);
    }

    /**
     * Initialize a collapse
     * @param elementOrSelector Element or selector for the collapse
     * @param options Collapse options
     * @returns Collapse instance
     */
    static initCollapse(elementOrSelector: string | HTMLElement | JQuery, options?: any): CollapseInstance {
        const element = BootstrapUtils.getElement(elementOrSelector);
        if (!element) throw new Error(`Failed to find element for collapse: ${elementOrSelector}`);

        return new window.bootstrap.Collapse(element, options);
    }

    /**
     * Initialize a collapse if the element exists
     * @param elementOrSelector Element or selector for the collapse
     * @param options Collapse options
     * @returns Collapse instance or null if element doesn't exist
     */
    static initCollapseIfExists(
        elementOrSelector: string | HTMLElement | JQuery,
        options?: any,
    ): CollapseInstance | null {
        const element = BootstrapUtils.getElement(elementOrSelector);
        if (!element) return null;

        return new window.bootstrap.Collapse(element, options);
    }

    /**
     * Helper method to get an HTMLElement from various input types
     * @param elementOrSelector Element, jQuery object, or selector
     * @returns HTMLElement or null
     */
    private static getElement(elementOrSelector: string | HTMLElement | JQuery): HTMLElement | null {
        if (!elementOrSelector) return null;

        if (typeof elementOrSelector === 'string') {
            return document.querySelector(elementOrSelector as string);
        }

        if (elementOrSelector instanceof HTMLElement) {
            return elementOrSelector;
        }

        if (elementOrSelector instanceof $) {
            return (elementOrSelector as JQuery)[0] as HTMLElement;
        }

        return null;
    }

    /**
     * Hide an existing popover if it exists
     * @param elementOrSelector Element or selector for the popover
     */
    static hidePopover(elementOrSelector: string | HTMLElement | JQuery): void {
        const popover = BootstrapUtils.getPopoverInstance(elementOrSelector);
        if (popover) popover.hide();
    }

    /**
     * Show an existing popover if it exists
     * @param elementOrSelector Element or selector for the popover
     */
    static showPopover(elementOrSelector: string | HTMLElement | JQuery): void {
        const popover = BootstrapUtils.getPopoverInstance(elementOrSelector);
        if (popover) popover.show();
    }

    /**
     * Show an existing modal if it exists (uses existing instance only)
     * @param elementOrSelector Element or selector for the modal
     */
    static showModalIfExists(elementOrSelector: string | HTMLElement | JQuery): void {
        const modal = BootstrapUtils.getModalInstance(elementOrSelector);
        if (modal) modal.show();
    }
}
