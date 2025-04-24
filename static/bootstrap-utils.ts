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
 * Bootstrap Utilities
 *
 * This module provides utilities that bridge Bootstrap 5's vanilla JavaScript API
 * with jQuery-based code in the Compiler Explorer codebase. It centralizes Bootstrap
 * API interactions to provide consistent behavior across the application.
 *
 * Key benefits:
 * - Handles conversion between jQuery objects and DOM elements
 * - Provides simplified event handling compatible with jQuery patterns
 * - Maintains consistent API for Bootstrap components
 * - Simplifies Bootstrap component initialization and management
 */

import $ from 'jquery';

import 'bootstrap';
import {Collapse, Dropdown, Modal, Popover, Tab, Toast, Tooltip} from 'bootstrap';

// Private event listener tracking map
const eventListenerMap = new WeakMap<HTMLElement, Map<string, EventListener>>();

/**
 * Helper method to get an HTMLElement from various input types
 *
 * This is a core utility function that bridges jQuery objects with native DOM elements,
 * allowing our codebase to work with both paradigms while the Bootstrap 5 API requires
 * native DOM elements.
 *
 * @param elementOrSelector Element, jQuery object, or selector
 * @returns HTMLElement or null
 */
function getElement(elementOrSelector: string | HTMLElement | JQuery): HTMLElement | null {
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
 * Private helper that sets an event handler on a single DOM element
 * Handles the removal of existing handlers and tracking of the new one
 *
 * @param domElement The DOM element to attach the event to
 * @param eventName The event name (e.g., 'hidden.bs.modal', 'shown.bs.modal', 'click', etc.)
 * @param handler The event handler function
 */
function setDomElementEventHandler(domElement: HTMLElement, eventName: string, handler: (event: Event) => void): void {
    // Initialize nested map structure if needed
    if (!eventListenerMap.has(domElement)) {
        eventListenerMap.set(domElement, new Map());
    }

    const elementEvents = eventListenerMap.get(domElement)!;

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
export function setElementEventHandler(
    element: JQuery<HTMLElement> | HTMLElement,
    eventName: string,
    handler: (event: Event) => void,
): void {
    // If jQuery object with potentially multiple elements
    if (!(element instanceof HTMLElement)) {
        // Loop through all elements in the jQuery collection
        element.each((_index, domElement) => {
            setDomElementEventHandler(domElement, eventName, handler);
        });
        return;
    }

    // Otherwise it's a single DOM element
    setDomElementEventHandler(element, eventName, handler);
}

/**
 * Initialize a modal
 * @param elementOrSelector Element or selector for the modal
 * @param options Modal options
 * @returns Modal instance
 * @throws Error if the element cannot be found
 *
 * Note: When possible, prefer direct Bootstrap 5 API:
 * `new Modal(element, options)`
 */
export function initModal(elementOrSelector: string | HTMLElement | JQuery, options?: Partial<Modal.Options>): Modal {
    const element = getElement(elementOrSelector);
    if (!element) throw new Error(`Failed to find element for modal: ${elementOrSelector}`);

    return new Modal(element, options);
}

/**
 * Initialize a modal if the element exists, returning null otherwise
 * @param elementOrSelector Element or selector for the modal
 * @param options Modal options
 * @returns Modal instance or null if the element cannot be found
 *
 * Note: When possible, prefer direct Bootstrap 5 API:
 * `element ? new Modal(element, options) : null`
 */
export function initModalIfExists(
    elementOrSelector: string | HTMLElement | JQuery,
    options?: Partial<Modal.Options>,
): Modal | null {
    const element = getElement(elementOrSelector);
    if (!element) return null;

    return new Modal(element, options);
}

/**
 * Get an existing modal instance for an element
 * @param elementOrSelector Element or selector for the modal
 * @returns Existing modal instance or null if not found
 */
export function getModalInstance(elementOrSelector: string | HTMLElement | JQuery): Modal | null {
    const element = getElement(elementOrSelector);
    if (!element) return null;

    return Modal.getInstance(element);
}

/**
 * Show a modal
 * @param elementOrSelector Element or selector for the modal
 * @param relatedTarget Optional related target element
 *
 * Note: When possible, prefer direct Bootstrap 5 API:
 * ```
 * const modal = Modal.getInstance(element) || new Modal(element);
 * modal.show(relatedTarget);
 * ```
 */
export function showModal(elementOrSelector: string | HTMLElement | JQuery, relatedTarget?: HTMLElement): void {
    const element = getElement(elementOrSelector);
    if (!element) return;

    const modal = Modal.getInstance(element) || new Modal(element);
    modal.show(relatedTarget);
}

/**
 * Hide a modal
 * @param elementOrSelector Element or selector for the modal
 */
export function hideModal(elementOrSelector: string | HTMLElement | JQuery): void {
    const element = getElement(elementOrSelector);
    if (!element) return;

    const modal = Modal.getInstance(element);
    if (modal) modal.hide();
}

/**
 * Initialize a toast
 * @param elementOrSelector Element or selector for the toast
 * @param options Toast options
 * @returns Toast instance
 *
 * Note: When possible, prefer direct Bootstrap 5 API:
 * `new Toast(element, options)`
 */
export function initToast(elementOrSelector: string | HTMLElement | JQuery, options?: Partial<Toast.Options>): Toast {
    const element = getElement(elementOrSelector);
    if (!element) throw new Error(`Failed to find element for toast: ${elementOrSelector}`);

    return new Toast(element, options);
}

/**
 * Initialize a toast if the element exists
 * @param elementOrSelector Element or selector for the toast
 * @param options Toast options
 * @returns Toast instance or null if element doesn't exist
 *
 * Note: When possible, prefer direct Bootstrap 5 API:
 * `element ? new Toast(element, options) : null`
 */
export function initToastIfExists(
    elementOrSelector: string | HTMLElement | JQuery,
    options?: Partial<Toast.Options>,
): Toast | null {
    const element = getElement(elementOrSelector);
    if (!element) return null;

    return new Toast(element, options);
}

/**
 * Show a toast
 * @param elementOrSelector Element or selector for the toast
 *
 * Note: When possible, prefer direct Bootstrap 5 API:
 * `const toast = Toast.getInstance(element) || new Toast(element); toast.show();`
 */
export function showToast(elementOrSelector: string | HTMLElement | JQuery): void {
    const element = getElement(elementOrSelector);
    if (!element) return;

    const toast = Toast.getInstance(element) || new Toast(element);
    toast.show();
}

/**
 * Hide a toast
 * @param elementOrSelector Element or selector for the toast
 *
 * Note: When possible, prefer direct Bootstrap 5 API:
 * `const toast = Toast.getInstance(element); if (toast) toast.hide();`
 */
export function hideToast(elementOrSelector: string | HTMLElement | JQuery): void {
    const element = getElement(elementOrSelector);
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
export function initDropdown(
    elementOrSelector: string | HTMLElement | JQuery,
    options?: Partial<Dropdown.Options>,
): Dropdown {
    const element = getElement(elementOrSelector);
    if (!element) throw new Error(`Failed to find element for dropdown: ${elementOrSelector}`);

    return new Dropdown(element, options);
}

/**
 * Initialize a dropdown if the element exists, returning null otherwise
 * @param elementOrSelector Element or selector for the dropdown
 * @param options Dropdown options
 * @returns Dropdown instance or null if the element cannot be found
 */
export function initDropdownIfExists(
    elementOrSelector: string | HTMLElement | JQuery,
    options?: Partial<Dropdown.Options>,
): Dropdown | null {
    const element = getElement(elementOrSelector);
    if (!element) return null;

    return new Dropdown(element, options);
}

/**
 * Get an existing dropdown instance for an element
 * @param elementOrSelector Element or selector for the dropdown
 * @returns Existing dropdown instance or null if not found
 */
export function getDropdownInstance(elementOrSelector: string | HTMLElement | JQuery): Dropdown | null {
    const element = getElement(elementOrSelector);
    if (!element) return null;

    return Dropdown.getInstance(element);
}

/**
 * Show a dropdown
 * @param elementOrSelector Element or selector for the dropdown
 */
export function showDropdown(elementOrSelector: string | HTMLElement | JQuery): void {
    const element = getElement(elementOrSelector);
    if (!element) return;

    const dropdown = Dropdown.getInstance(element) || new Dropdown(element);
    dropdown.show();
}

/**
 * Hide a dropdown
 * @param elementOrSelector Element or selector for the dropdown
 *
 * Note: When possible, prefer direct Bootstrap 5 API:
 * `const dropdown = Dropdown.getInstance(element); if (dropdown) dropdown.hide();`
 */
export function hideDropdown(elementOrSelector: string | HTMLElement | JQuery): void {
    const element = getElement(elementOrSelector);
    if (!element) return;

    const dropdown = Dropdown.getInstance(element);
    if (dropdown) dropdown.hide();
}

/**
 * Initialize a tooltip
 * @param elementOrSelector Element or selector for the tooltip
 * @param options Tooltip options
 * @returns Tooltip instance
 *
 * Note: When possible, prefer direct Bootstrap 5 API:
 * `new Tooltip(element, options)`
 */
export function initTooltip(
    elementOrSelector: string | HTMLElement | JQuery,
    options?: Partial<Tooltip.Options>,
): Tooltip {
    const element = getElement(elementOrSelector);
    if (!element) throw new Error(`Failed to find element for tooltip: ${elementOrSelector}`);

    return new Tooltip(element, options);
}

/**
 * Initialize a tooltip if the element exists
 * @param elementOrSelector Element or selector for the tooltip
 * @param options Tooltip options
 * @returns Tooltip instance or null if element doesn't exist
 *
 * Note: When possible, prefer direct Bootstrap 5 API:
 * `element ? new Tooltip(element, options) : null`
 */
export function initTooltipIfExists(
    elementOrSelector: string | HTMLElement | JQuery,
    options?: Partial<Tooltip.Options>,
): Tooltip | null {
    const element = getElement(elementOrSelector);
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
export function initPopover(
    elementOrSelector: string | HTMLElement | JQuery,
    options?: Partial<Popover.Options>,
): Popover {
    const element = getElement(elementOrSelector);
    if (!element) throw new Error(`Failed to find element for popover: ${elementOrSelector}`);

    return new Popover(element, options);
}

/**
 * Initialize a popover if the element exists, returning null otherwise
 * @param elementOrSelector Element or selector for the popover
 * @param options Popover options
 * @returns Popover instance or null if the element cannot be found
 */
export function initPopoverIfExists(
    elementOrSelector: string | HTMLElement | JQuery,
    options?: Partial<Popover.Options>,
): Popover | null {
    const element = getElement(elementOrSelector);
    if (!element) return null;

    return new Popover(element, options);
}

/**
 * Get an existing popover instance for an element
 * @param elementOrSelector Element or selector for the popover
 * @returns Existing popover instance or null if not found
 */
export function getPopoverInstance(elementOrSelector: string | HTMLElement | JQuery): Popover | null {
    const element = getElement(elementOrSelector);
    if (!element) return null;

    return Popover.getInstance(element);
}

/**
 * Initialize a tab
 * @param elementOrSelector Element or selector for the tab
 * @returns Tab instance
 */
export function initTab(elementOrSelector: string | HTMLElement | JQuery): Tab {
    const element = getElement(elementOrSelector);
    if (!element) throw new Error(`Failed to find element for tab: ${elementOrSelector}`);

    return new Tab(element);
}

/**
 * Initialize a tab if the element exists
 * @param elementOrSelector Element or selector for the tab
 * @returns Tab instance or null if element doesn't exist
 */
export function initTabIfExists(elementOrSelector: string | HTMLElement | JQuery): Tab | null {
    const element = getElement(elementOrSelector);
    if (!element) return null;

    return new Tab(element);
}

/**
 * Initialize a collapse
 * @param elementOrSelector Element or selector for the collapse
 * @param options Collapse options
 * @returns Collapse instance
 */
export function initCollapse(
    elementOrSelector: string | HTMLElement | JQuery,
    options?: Partial<Collapse.Options>,
): Collapse {
    const element = getElement(elementOrSelector);
    if (!element) throw new Error(`Failed to find element for collapse: ${elementOrSelector}`);

    return new Collapse(element, options);
}

/**
 * Initialize a collapse if the element exists
 * @param elementOrSelector Element or selector for the collapse
 * @param options Collapse options
 * @returns Collapse instance or null if element doesn't exist
 */
export function initCollapseIfExists(
    elementOrSelector: string | HTMLElement | JQuery,
    options?: Partial<Collapse.Options>,
): Collapse | null {
    const element = getElement(elementOrSelector);
    if (!element) return null;

    return new Collapse(element, options);
}

/**
 * Hide an existing popover if it exists
 * @param elementOrSelector Element or selector for the popover
 */
export function hidePopover(elementOrSelector: string | HTMLElement | JQuery): void {
    const popover = getPopoverInstance(elementOrSelector);
    if (popover) popover.hide();
}

/**
 * Show an existing popover if it exists
 * @param elementOrSelector Element or selector for the popover
 */
export function showPopover(elementOrSelector: string | HTMLElement | JQuery): void {
    const popover = getPopoverInstance(elementOrSelector);
    if (popover) popover.show();
}

/**
 * Show an existing modal if it exists (uses existing instance only)
 * @param elementOrSelector Element or selector for the modal
 */
export function showModalIfExists(elementOrSelector: string | HTMLElement | JQuery): void {
    const modal = getModalInstance(elementOrSelector);
    if (modal) modal.show();
}
