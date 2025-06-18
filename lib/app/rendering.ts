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

import express, {Request, Response} from 'express';
import _ from 'underscore';

import {AppArguments} from '../app.interfaces.js';
import {GoldenLayoutRootStruct} from '../clientstate-normalizer.js';
import * as normalizer from '../clientstate-normalizer.js';
import type {ShortLinkMetaData} from '../handlers/handler.interfaces.js';
import * as utils from '../utils.js';
import {
    PugRequireHandler,
    RenderConfig,
    RenderConfigFunction,
    RenderGoldenLayoutHandler,
    ServerDependencies,
    ServerOptions,
} from './server.interfaces.js';
import {getFaviconFilename} from './static-assets.js';
import {isMobileViewer} from './url-handlers.js';

/**
 * Create rendering-related functions
 * @param pugRequireHandler - Handler for Pug requires
 * @param options - Server options
 * @param dependencies - Server dependencies
 * @param appArgs - App arguments
 * @returns Rendering functions
 */
export function createRenderHandlers(
    pugRequireHandler: PugRequireHandler,
    options: ServerOptions,
    dependencies: ServerDependencies,
    appArgs: AppArguments,
): {
    renderConfig: RenderConfigFunction;
    renderGoldenLayout: RenderGoldenLayoutHandler;
    embeddedHandler: express.Handler;
} {
    const {clientOptionsHandler, storageSolution, sponsorConfig} = dependencies;
    const {httpRoot, staticRoot, extraBodyClass} = options;

    /**
     * Renders configuration for templates
     */
    const renderConfig: RenderConfigFunction = (
        extra: Record<string, any>,
        urlOptions?: Record<string, any>,
    ): RenderConfig => {
        const urlOptionsAllowed = ['readOnly', 'hideEditorToolbars', 'language'];
        const filteredUrlOptions = _.mapObject(_.pick(urlOptions || {}, urlOptionsAllowed), val =>
            utils.toProperty(val),
        );
        const allExtraOptions = _.extend({}, filteredUrlOptions, extra);

        if (allExtraOptions.mobileViewer && allExtraOptions.config) {
            const clnormalizer = new normalizer.ClientStateNormalizer();
            clnormalizer.fromGoldenLayout(allExtraOptions.config);
            const clientstate = clnormalizer.normalized;

            const glnormalizer = new normalizer.ClientStateGoldenifier();
            allExtraOptions.slides = glnormalizer.generatePresentationModeMobileViewerSlides(clientstate);
        }

        const options: RenderConfig = _.extend({}, allExtraOptions, clientOptionsHandler.get());
        options.optionsHash = clientOptionsHandler.getHash();
        options.compilerExplorerOptions = JSON.stringify(allExtraOptions);
        options.extraBodyClass = options.embedded ? 'embedded' : extraBodyClass;
        options.httpRoot = httpRoot;
        options.staticRoot = staticRoot;
        options.storageSolution = options.storageSolution || storageSolution;
        options.require = pugRequireHandler;
        options.sponsors = sponsorConfig;
        options.faviconFilename = getFaviconFilename(appArgs.devMode, appArgs.env);
        return options;
    };

    /**
     * Renders GoldenLayout for a given configuration
     */
    const renderGoldenLayout = (
        config: GoldenLayoutRootStruct,
        metadata: ShortLinkMetaData,
        req: Request,
        res: Response,
    ) => {
        const embedded = req.query.embedded === 'true';

        res.render(
            embedded ? 'embed' : 'index',
            renderConfig(
                {
                    embedded: embedded,
                    mobileViewer: isMobileViewer(req),
                    config: config,
                    metadata: metadata,
                    storedStateId: req.params.id || false,
                },
                req.query,
            ),
        );
    };

    /**
     * Handles rendering embedded pages
     */
    const embeddedHandler = (req: Request, res: Response) => {
        res.render(
            'embed',
            renderConfig(
                {
                    embedded: true,
                    mobileViewer: isMobileViewer(req),
                },
                req.query,
            ),
        );
    };

    return {
        renderConfig,
        renderGoldenLayout,
        embeddedHandler,
    };
}
