// Copyright (c) 2022, Compiler Explorer Authors
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

import {Pane} from './pane';
import * as monaco from 'monaco-editor';
import $ from 'jquery';
import _ from 'underscore';

import {CfgState} from './cfg-view.interfaces';
import {Hub} from '../hub';
import {Container} from 'golden-layout';
import {PaneState} from './pane.interfaces';
import {ga} from '../analytics';
import * as utils from '../utils';

import {
    AnnotatedCfgDescriptor,
    AnnotatedNodeDescriptor,
    CfgDescriptor,
    CFGResult,
} from '../../types/compilation/cfg.interfaces';
import {GraphLayoutCore} from '../graph-layout-core';
import * as MonacoConfig from '../monaco-config';
import TomSelect from 'tom-select';

const ColorTable = {
    red: '#FE5D5D',
    green: '#76E381',
    blue: '#65B7F6',
    grey: '#c5c5c5',
};

type Coordinate = {
    x: number;
    y: number;
};

const DZOOM = 0.1;
const MINZOOM = 0.1;

export class Cfg extends Pane<CfgState> {
    graphDiv: HTMLElement;
    svg: SVGElement;
    blockContainer: HTMLElement;
    graphContainer: HTMLElement;
    graphElement: HTMLElement;
    infoElement: HTMLElement;
    currentPosition: Coordinate = {x: 0, y: 0};
    dragging = false;
    dragStart: Coordinate = {x: 0, y: 0};
    dragStartPosition: Coordinate = {x: 0, y: 0};
    graphDimensions = {width: 0, height: 0};
    functionSelector: TomSelect;
    results: CFGResult;
    state: CfgState & PaneState;
    layout: GraphLayoutCore;
    bbMap: Record<string, HTMLDivElement> = {};

    constructor(hub: Hub, container: Container, state: CfgState & PaneState) {
        if ((state as any).selectedFn) {
            state = {
                id: state.id,
                compilerName: state.compilerName,
                editorid: state.editorid,
                treeid: state.treeid,
                selectedFunction: (state as any).selectedFn,
                zoom: 1,
            };
        }
        super(hub, container, state);
        this.eventHub.emit('cfgViewOpened', this.compilerInfo.compilerId);
        this.eventHub.emit('requestFilters', this.compilerInfo.compilerId);
        this.eventHub.emit('requestCompiler', this.compilerInfo.compilerId);
        const selector = this.domRoot.get()[0].getElementsByClassName('function-selector')[0];
        if (!(selector instanceof HTMLSelectElement)) {
            throw new Error('.function-selector is not an HTMLSelectElement');
        }
        this.functionSelector = new TomSelect(selector, {
            valueField: 'value',
            labelField: 'title',
            searchField: ['title'],
            placeholder: '🔍 Select a function...',
            dropdownParent: 'body',
            plugins: ['dropdown_input'],
            sortField: 'title',
            onChange: e => {
                this.selectFunction(e as any as string);
            },
        });
        this.state = state;
    }

    override getInitialHTML() {
        return $('#cfg').html();
    }

    override getDefaultPaneName() {
        return 'CFG';
    }

    override registerOpeningAnalyticsEvent(): void {
        ga.proxy('send', {
            hitType: 'event',
            eventCategory: 'OpenViewPane',
            eventAction: 'CFGViewPane',
        });
    }

    override registerDynamicElements(state: CfgState) {
        this.graphDiv = this.domRoot.find('.graph')[0];
        this.svg = this.domRoot.find('svg')[0] as SVGElement;
        this.blockContainer = this.domRoot.find('.block-container')[0];
        this.graphContainer = this.domRoot.find('.graph-container')[0];
        this.graphElement = this.domRoot.find('.graph')[0];
        this.infoElement = this.domRoot.find('.cfg-info')[0];
    }

    override registerCallbacks() {
        this.graphContainer.addEventListener('mousedown', e => {
            const div = (e.target as Element).closest('div');
            if (div && (div.classList.contains('block-container') || div.classList.contains('graph-container'))) {
                this.dragging = true;
                this.dragStart = {x: e.clientX, y: e.clientY};
                this.dragStartPosition = {...this.currentPosition};
            } else {
                // pass, let the user select block contents and other text
            }
        });
        this.graphContainer.addEventListener('mouseup', e => {
            this.dragging = false;
        });
        this.graphContainer.addEventListener('mousemove', e => {
            if (this.dragging) {
                this.currentPosition = {
                    x: e.clientX - this.dragStart.x + this.dragStartPosition.x,
                    y: e.clientY - this.dragStart.y + this.dragStartPosition.y,
                };
                this.graphElement.style.left = this.currentPosition.x + 'px';
                this.graphElement.style.top = this.currentPosition.y + 'px';
            }
        });
        this.graphContainer.addEventListener('wheel', e => {
            const delta = DZOOM * -Math.sign(e.deltaY) * Math.max(1, this.state.zoom - 1);
            const prevZoom = this.state.zoom;
            this.state.zoom += delta;
            if (this.state.zoom >= MINZOOM) {
                this.graphElement.style.transform = `scale(${this.state.zoom})`;
                const mouseX = e.clientX - this.graphElement.getBoundingClientRect().x;
                const mouseY = e.clientY - this.graphElement.getBoundingClientRect().y;
                // Amount that the zoom will offset is mouseX / width before zoom * delta * unzoomed width
                // And same for y. The width / height terms cancel.
                this.currentPosition.x -= (mouseX / prevZoom) * delta;
                this.currentPosition.y -= (mouseY / prevZoom) * delta;
                this.graphElement.style.left = this.currentPosition.x + 'px';
                this.graphElement.style.top = this.currentPosition.y + 'px';
            } else {
                this.state.zoom = MINZOOM;
            }
        });
    }

    override onCompiler(compilerId: number, compiler: any, options: unknown, editorId: number, treeId: number): void {
        if (this.compilerInfo.compilerId !== compilerId) return;
        this.compilerInfo.compilerName = compiler ? compiler.name : '';
        this.compilerInfo.editorId = editorId;
        this.compilerInfo.treeId = treeId;
        this.updateTitle();
        if (compiler && !compiler.supportsLLVMOptPipelineView) {
            //this.editor.setValue('<LLVM IR output is not supported for this compiler>');
        }
    }

    override onCompileResult(compilerId: number, compiler: any, result: any) {
        if (this.compilerInfo.compilerId !== compilerId) return;
        this.functionSelector.clear(true);
        this.functionSelector.clearOptions();
        if (result.cfg) {
            const cfg = result.cfg as CFGResult;
            this.results = cfg;
            let selectedFunction: string | null = this.state.selectedFunction;
            const keys = Object.keys(cfg);
            if (keys.length === 0) {
                this.functionSelector.addOption({
                    title: '<No functions available>',
                    value: '<No functions available>',
                });
            }
            for (const fn of keys) {
                this.functionSelector.addOption({
                    title: fn,
                    value: fn,
                });
            }
            if (keys.length > 0) {
                if (selectedFunction === '' || !(selectedFunction !== null && selectedFunction in cfg)) {
                    selectedFunction = keys[0];
                }
                this.functionSelector.setValue(selectedFunction, true);
                this.state.selectedFunction = selectedFunction;
            } else {
                // this.state.selectedFunction won't change, next time the compilation results aren't errors or empty
                // the selected function will still be the same
                selectedFunction = null;
            }
            this.selectFunction(selectedFunction);
        } else {
            // this case can be fallen into with a blank input file
            this.selectFunction(null);
        }
    }

    async createBasicBlocks(fn: CfgDescriptor) {
        for (const node of fn.nodes) {
            const div = document.createElement('div');
            div.classList.add('block');
            div.innerHTML = await monaco.editor.colorize(node.label, 'asm', MonacoConfig.extendConfig({}));
            if (node.id in this.bbMap) {
                throw Error("Duplicate basic block node id's found while drawing cfg");
            }
            this.bbMap[node.id] = div;
            this.blockContainer.appendChild(div);
        }
        for (const node of fn.nodes) {
            const elem = $(this.bbMap[node.id]);
            void this.bbMap[node.id].offsetHeight;
            (node as AnnotatedNodeDescriptor).width = elem.outerWidth() as number;
            (node as AnnotatedNodeDescriptor).height = elem.outerHeight() as number;
        }
    }

    drawEdges() {
        const width = this.layout.getWidth();
        const height = this.layout.getHeight();
        this.svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
        // We want to assembly everything in a document fragment first, then add it to the dom
        // If we add to the dom every iteration the performance is awful, presumably because of layout computation and
        // rendering and whatnot.
        const documentFragment = document.createDocumentFragment();
        for (const block of this.layout.blocks) {
            for (const edge of block.edges) {
                // Sanity check
                if (edge.path.length === 0) {
                    throw Error('Mal-formed edge: Zero segments');
                }
                const points: [number, number][] = [];
                // -1 offset is to create an overlap between the block's bottom border and start of the path, avoid any
                // visual artifacts
                points.push([edge.path[0].start.x, edge.path[0].start.y - 1]);
                for (const segment of edge.path.slice(0, edge.path.length - 1)) {
                    points.push([segment.end.x, segment.end.y]);
                }
                // Edge arrow is going to be a triangle
                const triangleHeight = 7;
                const triangleWidth = 7;
                const endpoint = edge.path[edge.path.length - 1].end;
                // +1 offset to create an overlap with the triangle
                points.push([endpoint.x, endpoint.y - triangleHeight + 1]);
                // Create the poly line
                const line = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
                line.setAttribute('points', points.map(coord => coord.join(',')).join(' '));
                line.setAttribute('fill', 'none');
                line.setAttribute('stroke', ColorTable[edge.color]);
                line.setAttribute('stroke-width', '2');
                documentFragment.appendChild(line);
                // Create teh triangle
                const trianglePoints: [number, number][] = [];
                trianglePoints.push([endpoint.x - triangleWidth / 2, endpoint.y - triangleHeight]);
                trianglePoints.push([endpoint.x + triangleWidth / 2, endpoint.y - triangleHeight]);
                trianglePoints.push([endpoint.x, endpoint.y]);
                trianglePoints.push([endpoint.x - triangleWidth / 2, endpoint.y - triangleHeight]);
                trianglePoints.push([endpoint.x + triangleWidth / 2, endpoint.y - triangleHeight]);
                const triangle = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
                triangle.setAttribute('points', trianglePoints.map(coord => coord.join(',')).join(' '));
                triangle.setAttribute('fill', ColorTable[edge.color]);
                documentFragment.appendChild(triangle);
            }
        }
        this.svg.appendChild(documentFragment);
    }

    applyLayout() {
        const width = this.layout.getWidth();
        const height = this.layout.getHeight();
        this.graphDimensions.width = width;
        this.graphDimensions.height = height;
        this.graphDiv.style.height = height + 'px';
        this.graphDiv.style.width = width + 'px';
        this.svg.style.height = height + 'px';
        this.svg.style.width = width + 'px';
        this.blockContainer.style.height = height + 'px';
        this.blockContainer.style.width = width + 'px';
        for (const block of this.layout.blocks) {
            const elem = this.bbMap[block.data.id];
            elem.style.top = block.coordinates.y + 'px';
            elem.style.left = block.coordinates.x + 'px';
            elem.style.width = block.data.width + 'px';
            elem.style.height = block.data.height + 'px';
        }
    }

    // display the cfg for the specified function if it exists
    // this function does not change or use this.state.selectedFunction
    async selectFunction(name: string | null) {
        this.blockContainer.innerHTML = '';
        this.svg.innerHTML = '';
        if (!name || !(name in this.results)) {
            return;
        }
        const fn = this.results[name];
        this.bbMap = {};
        await this.createBasicBlocks(fn);
        this.layout = new GraphLayoutCore(fn as AnnotatedCfgDescriptor);
        this.applyLayout();
        this.drawEdges();
        this.infoElement.innerHTML = `Layout time: ${Math.round(this.layout.layoutTime)}ms<br/>Basic blocks: ${
            fn.nodes.length
        }`;
    }

    override resize() {
        _.defer(() => {
            const topBarHeight = utils.updateAndCalcTopBarHeight(this.domRoot, this.topBar, this.hideable);
            this.graphContainer.style.width = `${this.domRoot.width() as number}px`;
            this.graphContainer.style.height = `${(this.domRoot.height() as number) - topBarHeight}px`;
        });
    }

    override close(): void {
        this.eventHub.unsubscribe();
        this.eventHub.emit('cfgViewClosed', this.compilerInfo.compilerId);
    }
}
