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

import {Buffer} from 'buffer';
import $ from 'jquery';
import {CompilationResult} from '../types/compilation/compilation.interfaces.js';
import {Artifact, ArtifactType} from '../types/tool.interfaces.js';
import {assert, unwrap} from './assert.js';
import * as BootstrapUtils from './bootstrap-utils.js';
import {Alert} from './widgets/alert.js';

export class ArtifactHandler {
    private alertSystem: Alert;

    constructor(alertSystem: Alert) {
        this.alertSystem = alertSystem;
    }

    handle(result: CompilationResult): void {
        if (result.artifacts) {
            for (const artifact of result.artifacts) {
                this.handleArtifact(artifact);
            }
        } else if (result.execResult) {
            // Handle executor results recursively
            this.handle(result.execResult);
        }
    }

    private handleArtifact(artifact: Artifact): void {
        switch (artifact.type) {
            case ArtifactType.nesrom:
                this.emulateNESROM(artifact);
                break;
            case ArtifactType.bbcdiskimage:
                this.emulateBbcDisk(artifact);
                break;
            case ArtifactType.zxtape:
                this.emulateSpeccyTape(artifact);
                break;
            case ArtifactType.smsrom:
                this.emulateMiracleSMS(artifact);
                break;
            case ArtifactType.timetrace:
                this.offerViewInSpeedscope(artifact);
                break;
            case ArtifactType.c64prg:
                this.emulateC64Prg(artifact);
                break;
            case ArtifactType.heaptracktxt:
                this.offerViewInSpeedscope(artifact);
                break;
            case ArtifactType.gbrom:
                this.emulateGameBoyROM(artifact);
                break;
        }
    }

    private offerViewInSpeedscope(artifact: Artifact): void {
        this.alertSystem.notify(
            'Click ' +
                '<a class="link-primary" target="_blank" id="download_link" style="cursor:pointer;">here</a>' +
                ' to view ' +
                artifact.title +
                ' in Speedscope',
            {
                group: artifact.type,
                collapseSimilar: false,
                dismissTime: 10000,
                onBeforeShow: elem => {
                    elem.find('#download_link').on('click', () => {
                        const tmstr = Date.now();
                        const live_url = 'https://static.ce-cdn.net/speedscope/index.html';
                        const speedscope_url =
                            live_url +
                            '?' +
                            tmstr +
                            '#customFilename=' +
                            encodeURIComponent(artifact.name) +
                            '&b64data=' +
                            encodeURIComponent(artifact.content);
                        window.open(speedscope_url);
                    });
                },
            },
        );
    }

    offerViewInPerfetto(artifact: Artifact): void {
        this.alertSystem.notify(
            'Click ' +
                '<a class="link-primary" target="_blank" id="download_link" style="cursor:pointer;">here</a>' +
                ' to view ' +
                artifact.title +
                ' in Perfetto',
            {
                group: artifact.type,
                collapseSimilar: false,
                dismissTime: 10000,
                onBeforeShow: elem => {
                    elem.find('#download_link').on('click', () => {
                        const perfetto_url = 'https://ui.perfetto.dev';
                        const win = window.open(perfetto_url);
                        if (win) {
                            const timer = setInterval(() => win.postMessage('PING', perfetto_url), 50);

                            const onMessageHandler = (evt: MessageEvent) => {
                                if (evt.data !== 'PONG') return;
                                clearInterval(timer);

                                const data = {
                                    perfetto: {
                                        buffer: Buffer.from(artifact.content, 'base64'),
                                        title: artifact.name,
                                        filename: artifact.name,
                                    },
                                };
                                win.postMessage(data, perfetto_url);
                            };
                            window.addEventListener('message', onMessageHandler);
                        }
                    });
                },
            },
        );
    }

    private emulateMiracleSMS(artifact: Artifact): void {
        const dialog = $('#miracleemu');

        this.alertSystem.notify(
            'Click ' +
                '<a target="_blank" id="miracle_emulink" style="cursor:pointer;" class="link-primary">here</a>' +
                ' to emulate',
            {
                group: 'emulation',
                collapseSimilar: true,
                dismissTime: 10000,
                onBeforeShow: elem => {
                    elem.find('#miracle_emulink').on('click', () => {
                        BootstrapUtils.showModal(dialog);

                        const miracleMenuFrame = dialog.find('#miracleemuframe')[0];
                        assert(miracleMenuFrame instanceof HTMLIFrameElement);
                        if ('contentWindow' in miracleMenuFrame) {
                            const emuwindow = unwrap(miracleMenuFrame.contentWindow);
                            const tmstr = Date.now();
                            emuwindow.location =
                                'https://miracle.xania.org/?' +
                                tmstr +
                                '#b64sms=' +
                                encodeURIComponent(artifact.content);
                        }
                    });
                },
            },
        );
    }

    private emulateSpeccyTape(artifact: Artifact): void {
        const dialog = $('#jsspeccyemu');

        this.alertSystem.notify(
            'Click ' +
                '<a target="_blank" id="jsspeccy_emulink" style="cursor:pointer;" class="link-primary">here</a>' +
                ' to emulate',
            {
                group: 'emulation',
                collapseSimilar: true,
                dismissTime: 10000,
                onBeforeShow: elem => {
                    elem.find('#jsspeccy_emulink').on('click', () => {
                        BootstrapUtils.showModal(dialog);

                        const speccyemuframe = dialog.find('#speccyemuframe')[0];
                        assert(speccyemuframe instanceof HTMLIFrameElement);
                        if ('contentWindow' in speccyemuframe) {
                            const emuwindow = unwrap(speccyemuframe.contentWindow);
                            const tmstr = Date.now();
                            emuwindow.location =
                                'https://static.ce-cdn.net/jsspeccy/index.html?' +
                                tmstr +
                                '#b64tape=' +
                                encodeURIComponent(artifact.content);
                        }
                    });
                },
            },
        );
    }

    private emulateBbcDisk(artifact: Artifact): void {
        const dialog = $('#jsbeebemu');

        this.alertSystem.notify(
            'Click <a target="_blank" id="emulink" style="cursor:pointer;" class="link-primary">here</a> to emulate',
            {
                group: 'emulation',
                collapseSimilar: true,
                dismissTime: 10000,
                onBeforeShow: elem => {
                    elem.find('#emulink').on('click', () => {
                        BootstrapUtils.showModal(dialog);

                        const jsbeebemuframe = dialog.find('#jsbeebemuframe')[0];
                        assert(jsbeebemuframe instanceof HTMLIFrameElement);
                        if ('contentWindow' in jsbeebemuframe) {
                            const emuwindow = unwrap(jsbeebemuframe.contentWindow);
                            const tmstr = Date.now();
                            emuwindow.location =
                                'https://bbc.godbolt.org/?' +
                                tmstr +
                                '#embed&autoboot&disc1=b64data:' +
                                encodeURIComponent(artifact.content);
                        }
                    });
                },
            },
        );
    }

    private emulateNESROM(artifact: Artifact): void {
        const dialog = $('#jsnesemu');

        this.alertSystem.notify(
            'Click <a target="_blank" id="emulink" style="cursor:pointer;" class="link-primary">here</a> to emulate',
            {
                group: 'emulation',
                collapseSimilar: true,
                dismissTime: 10000,
                onBeforeShow: elem => {
                    elem.find('#emulink').on('click', () => {
                        BootstrapUtils.showModal(dialog);

                        const jsnesemuframe = dialog.find('#jsnesemuframe')[0];
                        assert(jsnesemuframe instanceof HTMLIFrameElement);
                        if ('contentWindow' in jsnesemuframe) {
                            const emuwindow = unwrap(jsnesemuframe.contentWindow);
                            const tmstr = Date.now();
                            emuwindow.location =
                                'https://static.ce-cdn.net/jsnes-ceweb/index.html?' +
                                tmstr +
                                '#b64nes=' +
                                encodeURIComponent(artifact.content);
                        }
                    });
                },
            },
        );
    }

    private emulateC64Prg(prg: Artifact): void {
        this.alertSystem.notify(
            'Click <a target="_blank" id="emulink" style="cursor:pointer;" class="link-primary">here</a> to emulate',
            {
                group: 'emulation',
                collapseSimilar: true,
                dismissTime: 10000,
                onBeforeShow: elem => {
                    elem.find('#emulink').on('click', () => {
                        const tmstr = Date.now();
                        const url =
                            'https://static.ce-cdn.net/viciious/viciious.html?' +
                            tmstr +
                            '#filename=' +
                            encodeURIComponent(prg.title) +
                            '&b64c64=' +
                            encodeURIComponent(prg.content);

                        window.open(url, '_blank');
                    });
                },
            },
        );
    }

    private emulateGameBoyROM(prg: Artifact): void {
        const dialog = $('#gbemu');

        this.alertSystem.notify(
            'Click <a target="_blank" id="emulink" style="cursor:pointer;" class="link-primary">here</a> to emulate with a debugger, ' +
                'or <a target="_blank" id="emulink-play" style="cursor:pointer;" class="link-primary">here</a> to emulate just to play.',
            {
                group: 'emulation',
                collapseSimilar: true,
                dismissTime: 10000,
                onBeforeShow: elem => {
                    elem.find('#emulink').on('click', () => {
                        const tmstr = Date.now();
                        const url =
                            'https://static.ce-cdn.net/wasmboy/index.html?' +
                            tmstr +
                            '#rom-name=' +
                            encodeURIComponent(prg.title) +
                            '&rom-data=' +
                            encodeURIComponent(prg.content);
                        window.open(url, '_blank');
                    });

                    elem.find('#emulink-play').on('click', () => {
                        BootstrapUtils.showModal(dialog);

                        const gbemuframe = dialog.find('#gbemuframe')[0];
                        assert(gbemuframe instanceof HTMLIFrameElement);
                        if ('contentWindow' in gbemuframe) {
                            const emuwindow = unwrap(gbemuframe.contentWindow);
                            const tmstr = Date.now();
                            const url =
                                'https://static.ce-cdn.net/wasmboy/iframe/index.html?' +
                                tmstr +
                                '#rom-name=' +
                                encodeURIComponent(prg.title) +
                                '&rom-data=' +
                                encodeURIComponent(prg.content);
                            emuwindow.location = url;
                        }
                    });
                },
            },
        );
    }
}
