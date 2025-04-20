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

import 'bootstrap';
import {Collapse, Dropdown, Modal, Popover, Tab, Toast, Tooltip} from 'bootstrap';

export class BootstrapUtils {
    /**
     * Initialize a modal
     * @param elementOrSelector Element or selector for the modal
     * @param options Modal options
     * @returns Modal instance
     * @throws Error if the element cannot be found
     */
    static initModal(elementOrSelector: string | HTMLElement | JQuery, options?: Partial<Modal.Options>): Modal {
        const element = BootstrapUtils.getElement(elementOrSelector);
        if (!element) throw new Error(`Failed to find element for modal: ${elementOrSelector}`);

        return new Modal(element, options);
    }

    /**
     * Initialize a modal if the element exists, returning null otherwise
     * @param elementOrSelector Element or selector for the modal
     * @param options Modal options
     * @returns Modal instance or null if the element cannot be found
     */
    static initModalIfExists(
        elementOrSelector: string | HTMLElement | JQuery,
        options?: Partial<Modal.Options>,
    ): Modal | null {
        const element = BootstrapUtils.getElement(elementOrSelector);
        if (!element) return null;

        return new Modal(element, options);
    }

    /**
     * Get an existing modal instance for an element
     * @param elementOrSelector Element or selector for the modal
     * @returns Existing modal instance or null if not found
     */
    static getModalInstance(elementOrSelector: string | HTMLElement | JQuery): Modal | null {
        const element = BootstrapUtils.getElement(elementOrSelector);
        if (!element) return null;

        return Modal.getInstance(element);
    }

    /**
     * Show a modal
     * @param elementOrSelector Element or selector for the modal
     * @param relatedTarget Optional related target element
     */
    static showModal(elementOrSelector: string | HTMLElement | JQuery, relatedTarget?: HTMLElement): void {
        const element = BootstrapUtils.getElement(elementOrSelector);
        if (!element) return;

        const modal = Modal.getInstance(element) || new Modal(element);
        modal.show(relatedTarget);
    }

    /**
     * Hide a modal
     * @param elementOrSelector Element or selector for the modal
     */
    static hideModal(elementOrSelector: string | HTMLElement | JQuery): void {
        const element = BootstrapUtils.getElement(elementOrSelector);
        if (!element) return;

        const modal = Modal.getInstance(element);
        if (modal) modal.hide();
    }

    /**
     * Initialize a toast
     * @param elementOrSelector Element or selector for the toast
     * @param options Toast options
     * @returns Toast instance
     */
    static initToast(elementOrSelector: string | HTMLElement | JQuery, options?: Partial<Toast.Options>): Toast {
        const element = BootstrapUtils.getElement(elementOrSelector);
        if (!element) throw new Error(`Failed to find element for toast: ${elementOrSelector}`);

        return new Toast(element, options);
    }

    /**
     * Initialize a toast if the element exists
     * @param elementOrSelector Element or selector for the toast
     * @param options Toast options
     * @returns Toast instance or null if element doesn't exist
     */
    static initToastIfExists(
        elementOrSelector: string | HTMLElement | JQuery,
        options?: Partial<Toast.Options>,
    ): Toast | null {
        const element = BootstrapUtils.getElement(elementOrSelector);
        if (!element) return null;

        return new Toast(element, options);
    }

    /**
     * Show a toast
     * @param elementOrSelector Element or selector for the toast
     */
    static showToast(elementOrSelector: string | HTMLElement | JQuery): void {
        const element = BootstrapUtils.getElement(elementOrSelector);
        if (!element) return;

        const toast = Toast.getInstance(element) || new Toast(element);
        toast.show();
    }

    /**
     * Hide a toast
     * @param elementOrSelector Element or selector for the toast
     */
    static hideToast(elementOrSelector: string | HTMLElement | JQuery): void {
        const element = BootstrapUtils.getElement(elementOrSelector);
        if (!element) return;

        const toast = Toast.getInstance(element);
        if (toast) toast.hide();
    }

    /**
     * Initialize a dropdown
     * @param elementOrSelector Element or selector for the dropdown
     * @param options Dropdown options
     * @returns Dropdown instance
     * @throws Error if the element cannot be found
     */
    static initDropdown(
        elementOrSelector: string | HTMLElement | JQuery,
        options?: Partial<Dropdown.Options>,
    ): Dropdown {
        const element = BootstrapUtils.getElement(elementOrSelector);
        if (!element) throw new Error(`Failed to find element for dropdown: ${elementOrSelector}`);

        return new Dropdown(element, options);
    }

    /**
     * Initialize a dropdown if the element exists, returning null otherwise
     * @param elementOrSelector Element or selector for the dropdown
     * @param options Dropdown options
     * @returns Dropdown instance or null if the element cannot be found
     */
    static initDropdownIfExists(
        elementOrSelector: string | HTMLElement | JQuery,
        options?: Partial<Dropdown.Options>,
    ): Dropdown | null {
        const element = BootstrapUtils.getElement(elementOrSelector);
        if (!element) return null;

        return new Dropdown(element, options);
    }

    /**
     * Get an existing dropdown instance for an element
     * @param elementOrSelector Element or selector for the dropdown
     * @returns Existing dropdown instance or null if not found
     */
    static getDropdownInstance(elementOrSelector: string | HTMLElement | JQuery): Dropdown | null {
        const element = BootstrapUtils.getElement(elementOrSelector);
        if (!element) return null;

        return Dropdown.getInstance(element);
    }

    /**
     * Show a dropdown
     * @param elementOrSelector Element or selector for the dropdown
     */
    static showDropdown(elementOrSelector: string | HTMLElement | JQuery): void {
        const element = BootstrapUtils.getElement(elementOrSelector);
        if (!element) return;

        const dropdown = Dropdown.getInstance(element) || new Dropdown(element);
        dropdown.show();
    }

    /**
     * Hide a dropdown
     * @param elementOrSelector Element or selector for the dropdown
     */
    static hideDropdown(elementOrSelector: string | HTMLElement | JQuery): void {
        const element = BootstrapUtils.getElement(elementOrSelector);
        if (!element) return;

        const dropdown = Dropdown.getInstance(element);
        if (dropdown) dropdown.hide();
    }

    /**
     * Initialize a tooltip
     * @param elementOrSelector Element or selector for the tooltip
     * @param options Tooltip options
     * @returns Tooltip instance
     */
    static initTooltip(elementOrSelector: string | HTMLElement | JQuery, options?: Partial<Tooltip.Options>): Tooltip {
        const element = BootstrapUtils.getElement(elementOrSelector);
        if (!element) throw new Error(`Failed to find element for tooltip: ${elementOrSelector}`);

        return new Tooltip(element, options);
    }

    /**
     * Initialize a tooltip if the element exists
     * @param elementOrSelector Element or selector for the tooltip
     * @param options Tooltip options
     * @returns Tooltip instance or null if element doesn't exist
     */
    static initTooltipIfExists(
        elementOrSelector: string | HTMLElement | JQuery,
        options?: Partial<Tooltip.Options>,
    ): Tooltip | null {
        const element = BootstrapUtils.getElement(elementOrSelector);
        if (!element) return null;

        return new Tooltip(element, options);
    }

    /**
     * Initialize a popover
     * @param elementOrSelector Element or selector for the popover
     * @param options Popover options
     * @returns Popover instance
     * @throws Error if the element cannot be found
     */
    static initPopover(elementOrSelector: string | HTMLElement | JQuery, options?: Partial<Popover.Options>): Popover {
        const element = BootstrapUtils.getElement(elementOrSelector);
        if (!element) throw new Error(`Failed to find element for popover: ${elementOrSelector}`);

        return new Popover(element, options);
    }

    /**
     * Initialize a popover if the element exists, returning null otherwise
     * @param elementOrSelector Element or selector for the popover
     * @param options Popover options
     * @returns Popover instance or null if the element cannot be found
     */
    static initPopoverIfExists(
        elementOrSelector: string | HTMLElement | JQuery,
        options?: Partial<Popover.Options>,
    ): Popover | null {
        const element = BootstrapUtils.getElement(elementOrSelector);
        if (!element) return null;

        return new Popover(element, options);
    }

    /**
     * Get an existing popover instance for an element
     * @param elementOrSelector Element or selector for the popover
     * @returns Existing popover instance or null if not found
     */
    static getPopoverInstance(elementOrSelector: string | HTMLElement | JQuery): Popover | null {
        const element = BootstrapUtils.getElement(elementOrSelector);
        if (!element) return null;

        return Popover.getInstance(element);
    }

    /**
     * Initialize a tab
     * @param elementOrSelector Element or selector for the tab
     * @returns Tab instance
     */
    static initTab(elementOrSelector: string | HTMLElement | JQuery): Tab {
        const element = BootstrapUtils.getElement(elementOrSelector);
        if (!element) throw new Error(`Failed to find element for tab: ${elementOrSelector}`);

        return new Tab(element);
    }

    /**
     * Initialize a tab if the element exists
     * @param elementOrSelector Element or selector for the tab
     * @returns Tab instance or null if element doesn't exist
     */
    static initTabIfExists(elementOrSelector: string | HTMLElement | JQuery): Tab | null {
        const element = BootstrapUtils.getElement(elementOrSelector);
        if (!element) return null;

        return new Tab(element);
    }

    /**
     * Initialize a collapse
     * @param elementOrSelector Element or selector for the collapse
     * @param options Collapse options
     * @returns Collapse instance
     */
    static initCollapse(
        elementOrSelector: string | HTMLElement | JQuery,
        options?: Partial<Collapse.Options>,
    ): Collapse {
        const element = BootstrapUtils.getElement(elementOrSelector);
        if (!element) throw new Error(`Failed to find element for collapse: ${elementOrSelector}`);

        return new Collapse(element, options);
    }

    /**
     * Initialize a collapse if the element exists
     * @param elementOrSelector Element or selector for the collapse
     * @param options Collapse options
     * @returns Collapse instance or null if element doesn't exist
     */
    static initCollapseIfExists(
        elementOrSelector: string | HTMLElement | JQuery,
        options?: Partial<Collapse.Options>,
    ): Collapse | null {
        const element = BootstrapUtils.getElement(elementOrSelector);
        if (!element) return null;

        return new Collapse(element, options);
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

    // Private static map to track event listeners
    private static eventListenerMap = new WeakMap<HTMLElement, Map<string, EventListener>>();

    /**
     * Private helper that sets an event handler on a single DOM element
     * Handles the removal of existing handlers and tracking of the new one
     *
     * @param domElement The DOM element to attach the event to
     * @param eventName The event name (e.g., 'hidden.bs.modal', 'shown.bs.modal', 'click', etc.)
     * @param handler The event handler function
     */
    private static setDomElementEventHandler(
        domElement: HTMLElement,
        eventName: string,
        handler: (event: Event) => void,
    ): void {
        // Initialize nested map structure if needed
        if (!this.eventListenerMap.has(domElement)) {
            this.eventListenerMap.set(domElement, new Map());
        }

        const elementEvents = this.eventListenerMap.get(domElement)!;

        // Remove existing handler if present
        if (elementEvents.has(eventName)) {
            const oldHandler = elementEvents.get(eventName)!;
            domElement.removeEventListener(eventName, oldHandler);
        }

        // Store and add new handler
        elementEvents.set(eventName, handler);
        domElement.addEventListener(eventName, handler);
    }

    /**
     * Registers an event handler on element(s), removing any previous handler for the same event
     * Similar to jQuery's off().on() pattern
     * Works with both single elements and jQuery collections with multiple elements
     *
     * @param element The element(s) or jQuery object to attach the event to
     * @param eventName The event name (e.g., 'hidden.bs.modal', 'shown.bs.modal', 'click', etc.)
     * @param handler The event handler function
     */
    static setElementEventHandler(
        element: JQuery<HTMLElement> | HTMLElement,
        eventName: string,
        handler: (event: Event) => void,
    ): void {
        // If jQuery object with potentially multiple elements
        if (!(element instanceof HTMLElement)) {
            // Loop through all elements in the jQuery collection
            element.each((_index, domElement) => {
                this.setDomElementEventHandler(domElement, eventName, handler);
            });
            return;
        }

        // Otherwise it's a single DOM element
        this.setDomElementEventHandler(element, eventName, handler);
    }
}
