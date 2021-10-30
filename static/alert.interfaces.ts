// Copyright (c) 2021, Compiler Explorer Authors
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

export interface AlertAskOptions {
    /** Function to execute on "yes" button press */
    yes?: () => void;
    /** Function to execute on "no" button press */
    no?: () => void;
    /** HTML markup of "yes" button */
    yesHtml?: string;
    /** Custom HTML class to add to "yes" button */
    yesClass?: string;
    /** HTML markup of "no" button */
    noHtml?: string;
    /** Custom HTML class to add to "no" button */
    noClass?: string;
    /** Function to execute on pane closure */
    onClose?: () => void;
}

export type AlertEnterTextOptions = {
    /** The enter text action returns a value which is captured here */
    yes?: (answer: string) => void;
} & Partial<Pick<AlertAskOptions, 'no' | 'yesHtml' | 'yesClass' | 'noHtml' | 'noClass' | 'onClose'>>;

export interface AlertNotifyOptions {
    /**
     * Which group this notification is from. Sets data-group attribute value
     * Default: ""
     */
    group?: string
    /** If set to true, other notifications within the same group will be removed before sending this one. (Note that
     * this only has any effect if options.group is set).
     * Default: true
     */
    collapseSimilar?: boolean
    /**
     * Space separated list of HTML classes to give to the notification's div element.
     * Default: ""
     */
    alertClass?: string
    /**
     * If set to true, the notification will fade out and be removed automatically.
     * Default: true
     */
    autoDismiss?: boolean
    /**
     * If allow by autoDismiss, controls how long the notification will be visible (in milliseconds) before
     * automatically removed
     * Default: 5000
     */
    dismissTime?: number
}
