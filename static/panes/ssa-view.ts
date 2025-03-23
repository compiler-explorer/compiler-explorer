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

import {Pane} from './pane.js';
import * as monaco from 'monaco-editor';
import $ from 'jquery';
import _ from 'underscore';
import * as fileSaver from 'file-saver';

import {CfgState} from './cfg-view.interfaces.js';
import {Hub} from '../hub.js';
import {Container} from 'golden-layout';
import {PaneState} from './pane.interfaces.js';
import * as utils from '../utils.js';

import {
    AnnotatedCfgDescriptor,
    AnnotatedNodeDescriptor,
    CfgDescriptor,
    CFGResult,
} from '../../types/compilation/cfg.interfaces.js';
import {GraphLayoutCore} from '../graph-layout-core.js';
import * as MonacoConfig from '../monaco-config.js';
import TomSelect from 'tom-select';
import {assert, unwrap} from '../assert.js';
import {CompilationResult} from '../compilation/compilation.interfaces.js';
import {CompilerInfo} from '../compiler.interfaces.js';
import {escapeHTML} from '../../shared/common-utils.js';
import { parse } from '../../shared/stacktrace.js';

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

const EST_COMPRESSION_RATIO = 0.022;

function attrs(attributes: Record<string, string | number | null>) {
    return Object.entries(attributes)
        .map(([k, v]) => `${k}="${v}"`)
        .join(' ');
}

function special_round(x: number) {
    assert(x >= 0);
    if (x === 0) {
        return 0;
    }
    const p = Math.pow(10, Math.floor(Math.log10(x)));
    // prettier-ignore
    const candidates = [
        Math.round(x / p) * p - p / 2,
        Math.round(x / p) * p,
        Math.round(x / p) * p + p / 2,
    ];
    return Math.trunc(candidates.sort((a, b) => Math.abs(x - a) - Math.abs(x - b))[0]);
}

function size_to_human(bytes: number) {
    if (bytes < 1000) {
        return special_round(bytes) + ' B';
    } else if (bytes < 1_000_000) {
        return special_round(bytes / 1_000) + ' KB';
    } else if (bytes < 1_000_000_000) {
        return special_round(bytes / 1_000_000) + ' MB';
    } else {
        return special_round(bytes / 1_000_000_000) + ' GB';
    }
}

export class Ssa extends Pane<CfgState> {
    graphDiv: HTMLElement;
    svg: SVGElement;
    blockContainer: HTMLElement;
    graphContainer: HTMLElement;
    graphElement: HTMLElement;
    infoElement: HTMLElement;
    exportPNGButton: JQuery;
    estimatedPNGSize: Element;
    exportSVGButton: JQuery;
    currentPosition: Coordinate = {x: 0, y: 0};
    dragging = false;
    dragStart: Coordinate = {x: 0, y: 0};
    dragStartPosition: Coordinate = {x: 0, y: 0};
    graphDimensions = {width: 0, height: 0};
    functionSelector: TomSelect;
    resetViewButton: JQuery;
    zoomOutButton: JQuery;
    results: CFGResult;
    ssa_result: string;
    state: CfgState & PaneState;
    layout: GraphLayoutCore;
    ssa_layout: GraphLayoutCore;
    bbMap: Record<string, HTMLDivElement> = {};
    tooltipOpen = false;
    readonly extraTransforms: string;
    fictitiousGraphContainer: HTMLDivElement;
    fictitiousBlockContainer: HTMLDivElement;
    zoom = 1;
    // Ugly but I don't see another way
    firstRender = true;
    contentsAreIr = false;
    result: CompilationResult

    constructor(hub: Hub, container: Container, state: CfgState & PaneState) {
        if ((state as any).selectedFn) {
            state = {
                id: state.id,
                compilerName: state.compilerName,
                editorid: state.editorid,
                treeid: state.treeid,
                selectedFunction: (state as any).selectedFn,
            };
        }
        super(hub, container, state);
        this.state = state;
        this.eventHub.emit('cfgViewOpened', this.compilerInfo.compilerId, this.state.isircfg === true);
        this.eventHub.emit('requestCompiler', this.compilerInfo.compilerId);
        this.contentsAreIr = !!this.state.isircfg;
        // This is a workaround for a chrome render bug that's existed since at least 2013
        // https://github.com/compiler-explorer/compiler-explorer/issues/4421
        this.extraTransforms = navigator.userAgent.includes('AppleWebKit') ? ' translateZ(0)' : '';
        this.updateTitle();
    }

    override getInitialHTML() {
        return $('#cfg').html();
    }

    override getDefaultPaneName() {
        // We need to check if this.state exists because this is called in the super constructor before this is actually
        // constructed
        // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
        if (this.state && this.state.isircfg) {
            return `IR CFG`;
        } else {
            return `CFG`;
        }
    }

    override registerButtons() {
        const selector = this.domRoot.get()[0].getElementsByClassName('function-selector')[0];
        assert(selector instanceof HTMLSelectElement, '.function-selector is not an HTMLSelectElement');
        this.functionSelector = new TomSelect(selector, {
            valueField: 'value',
            labelField: 'title',
            searchField: ['title'],
            placeholder: '🔍 Select a function...',
            dropdownParent: 'body',
            plugins: ['dropdown_input'],
            sortField: 'title',
            onChange: (e: string) => this.selectFunction(e),
        });
        this.functionSelector.on('dropdown_close', () => {
            // scroll back to the selection on the next open
            const selection = unwrap(this.functionSelector).getOption(this.state.selectedFunction);
            unwrap(this.functionSelector).setActiveOption(selection);
        });

        this.resetViewButton = this.domRoot.find('.reset-view');
        this.resetViewButton.on('click', () => {
            this.resetView(true);
        });
        this.zoomOutButton = this.domRoot.find('.zoom-out');
        this.zoomOutButton.on('click', () => {
            this.birdsEyeView();
        });
    }

    override registerDynamicElements(state: CfgState) {
        this.graphDiv = this.domRoot.find('.graph')[0];
        this.svg = this.domRoot.find('svg')[0];
        this.blockContainer = this.domRoot.find('.block-container')[0];
        this.graphContainer = this.domRoot.find('.graph-container')[0];
        this.graphElement = this.domRoot.find('.graph')[0];
        this.infoElement = this.domRoot.find('.cfg-info')[0];
        this.exportPNGButton = this.domRoot.find('.export-png').first();
        this.estimatedPNGSize = unwrap(this.exportPNGButton[0].querySelector('.estimated-export-size'));
        this.exportSVGButton = this.domRoot.find('.export-svg').first();
        this.setupFictitiousGraphContainer();
    }

    setupFictitiousGraphContainer() {
        // create a fake .graph-container .graph .block-container where we can compute block dimensions
        // golden layout sets panes to display:none when they aren't the active tab
        // create the .graph-container
        const fictitiousGraphContainer = document.createElement('div');
        fictitiousGraphContainer.setAttribute('class', 'graph-container');
        fictitiousGraphContainer.setAttribute('style', 'position: absolute; bottom: 0; right: 0; width: 0; height: 0;');
        // create the .graph
        const fictitiousGraph = document.createElement('div');
        fictitiousGraph.setAttribute('class', 'graph');
        // create the .block-container
        const fictitousBlockContainer = document.createElement('div');
        fictitousBlockContainer.setAttribute('class', 'block-container');
        // .graph-container -> .graph
        fictitiousGraphContainer.appendChild(fictitiousGraph);
        // .graph -> .block-container
        fictitiousGraph.appendChild(fictitousBlockContainer);
        // finally append to the body
        document.body.appendChild(fictitiousGraphContainer);
        this.fictitiousGraphContainer = fictitiousGraphContainer;
        this.fictitiousBlockContainer = fictitousBlockContainer;
    }

    override registerCallbacks() {
        this.graphContainer.addEventListener('mousedown', e => {
            const div = (unwrap(e.target) as Element).closest('div');
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
                this.setPan({
                    x: e.clientX - this.dragStart.x + this.dragStartPosition.x,
                    y: e.clientY - this.dragStart.y + this.dragStartPosition.y,
                });
            }
        });
        this.graphContainer.addEventListener('wheel', e => {
            const delta = DZOOM * -Math.sign(e.deltaY) * Math.max(1, this.zoom - 1);
            const prevZoom = this.zoom;
            const zoom = this.zoom + delta;
            if (zoom >= MINZOOM) {
                this.setZoom(zoom);
                const mouseX = e.clientX - this.graphElement.getBoundingClientRect().x;
                const mouseY = e.clientY - this.graphElement.getBoundingClientRect().y;
                // Amount that the zoom will offset is mouseX / width before zoom * delta * unzoomed width
                // And same for y. The width / height terms cancel.
                this.setPan({
                    x: this.currentPosition.x - (mouseX / prevZoom) * delta,
                    y: this.currentPosition.y - (mouseY / prevZoom) * delta,
                });
            } else {
                this.setZoom(MINZOOM);
            }
            e.preventDefault();
        });
        this.exportPNGButton.on('click', () => {
            this.exportPNG();
        });
        this.exportSVGButton.on('click', () => {
            this.exportSVG();
        });
        // Dismiss tooltips if you click elsewhere - trigger: focus isn't working for some reason
        $('body').on('click', e => {
            if (this.tooltipOpen) {
                if (!e.target.classList.contains('fold') && $(e.target).parents('.popover.in').length === 0) {
                    this.tooltipOpen = false;
                    $('.fold').popover('hide');
                }
            }
        });
    }

    async exportPNG() {
        fileSaver.saveAs(await this.createPNG(), 'cfg.png');
    }

    exportSVG() {
        fileSaver.saveAs(new Blob([this.createSVG()], {type: 'text/plain;charset=utf-8'}), 'cfg.svg');
    }

    override onCompiler(
        compilerId: number,
        compiler: CompilerInfo | null,
        options: string,
        editorId: number,
        treeId: number,
    ): void {
        if (this.compilerInfo.compilerId !== compilerId) return;
        this.compilerInfo.compilerName = compiler ? compiler.name : '';
        this.compilerInfo.editorId = editorId;
        this.compilerInfo.treeId = treeId;
        this.updateTitle();
        if (compiler && !compiler.optPipeline) {
            //this.editor.setValue('<LLVM IR output is not supported for this compiler>');
        }
    }

    override onCompileResult(compilerId: number, compiler: CompilerInfo, result: CompilationResult) {
        if (this.compilerInfo.compilerId !== compilerId) return;
        this.functionSelector.clear(true);
        this.functionSelector.clearOptions();
        console.log(result)
        const cfg = this.state.isircfg ? result.irOutput?.cfg : result.cfg;

        const gccDump = result.gccDumpOutput;
if (gccDump && gccDump.currentPassOutput) {
    const ssaDump = gccDump.currentPassOutput;
    this.ssa_result = ssaDump
    // console.log(ssaDump); // Now ssaDump contains the SSA (tree) dump text.
} else {
    console.error("SSA dump not found. Check that the dump flags are enabled.");
}

        if (cfg) {
            this.results = cfg;
            this.contentsAreIr = !!this.state.isircfg || !!result.compilationOptions?.includes('-emit-llvm');
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
            const folded_lines: number[] = [];
            const raw_lines = node.label.split('\n');
            const highlighted_asm_untrimmed = await monaco.editor.colorize(
                raw_lines.join('\n'),
                this.contentsAreIr ? 'llvm-ir' : 'asm',
                MonacoConfig.extendConfig({}),
            );
            const highlighted_asm = await monaco.editor.colorize(
                raw_lines
                    .map((line, i) => {
                        if (line.length <= 100) {
                            return line;
                        } else {
                            folded_lines.push(i);
                            return line.slice(0, 100);
                        }
                    })
                    .join('\n'),
                this.contentsAreIr ? 'llvm-ir' : 'asm',
                MonacoConfig.extendConfig({}),
            );
            const untrimmed_lines = highlighted_asm_untrimmed.split('<br/>');
            const lines = highlighted_asm.split('<br/>');
            // highlighted asm has a blank line at the end
            assert(raw_lines.length === untrimmed_lines.length - 1);
            assert(raw_lines.length === lines.length - 1);
            for (const i of folded_lines) {
                lines[i] += `<span class="fold" data-extra="${
                    untrimmed_lines[i]
                        .replace(/"/g, '&quot;') // escape double quotes for the attribute
                        .replace(/\s{2,}/g, '&nbsp;') // clean up occurrences of multiple whitespace
                        .replace(/>(\s|&nbsp;)<\/span>/, '></span>') // Hacky solution to remove whitespace at the start
                }" aria-describedby="wtf">&#8943;</span>`;
            }
            div.innerHTML = lines.join('<br/>');
            for (const fold of div.getElementsByClassName('fold')) {
                $(fold)
                    .popover({
                        content: unwrap(fold.getAttribute('data-extra')),
                        html: true,
                        placement: 'top',
                        template:
                            '<div class="popover cfg-fold-popover" role="tooltip">' +
                            '<div class="arrow"></div>' +
                            '<h3 class="popover-header"></h3>' +
                            '<div class="popover-body"></div>' +
                            '</div>',
                    })
                    .on('show.bs.popover', () => {
                        this.tooltipOpen = true;
                    })
                    .on('hide.bs.popover', () => {
                        this.tooltipOpen = false;
                    });
            }
            // So because this is async there's a race condition here if you rapidly switch functions.
            // This can be triggered by loading an example program. Because the fix going to be tricky I'll defer
            // to another PR. TODO(jeremy-rifkin)
            assert(!(node.id in this.bbMap), "Duplicate basic block node id's found while drawing cfg");
            this.bbMap[node.id] = div;
            this.blockContainer.appendChild(div);
        }
        for (const node of fn.nodes) {
            const fictitiousBlock = this.fictitiousBlockContainer.appendChild(
                this.bbMap[node.id].cloneNode(true),
            ) as HTMLDivElement;
            const elem = $(fictitiousBlock);
            void fictitiousBlock.offsetHeight; // try to trigger a layout recompute
            (node as AnnotatedNodeDescriptor).width = unwrap(elem.outerWidth());
            (node as AnnotatedNodeDescriptor).height = unwrap(elem.outerHeight());
        }
        // remove all children
        this.fictitiousBlockContainer.replaceChildren();
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
                assert(edge.path.length !== 0, 'Mal-formed edge: Zero segments');
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

    // Display the cfg for the specified function if it exists
    // This function sets this.state.selectedFunction if the input is non-null and valid
    // async selectFunction(name: string | null) {
    //     $('.fold').popover('dispose');
    //     this.blockContainer.innerHTML = '';
    //     this.svg.innerHTML = '';
    //     this.estimatedPNGSize.innerHTML = '';
    //     if (!name || !(name in this.results)) {
    //         return;
    //     }
    //     const fn = this.results[name];
    //     this.bbMap = {};
    //     await this.createBasicBlocks(fn);
    //     this.layout = new GraphLayoutCore(fn as AnnotatedCfgDescriptor);
    //     this.applyLayout();
    //     this.drawEdges();
    //     this.infoElement.innerHTML = `Layout time: ${Math.round(this.layout.layoutTime)}ms<br/>Basic blocks: ${
    //         fn.nodes.length
    //     }`;
    //     this.estimatedPNGSize.innerHTML = `(~${size_to_human(
    //         this.layout.getWidth() * this.layout.getHeight() * 4 * EST_COMPRESSION_RATIO,
    //     )})`;
    //     if (this.state.selectedFunction !== name || this.firstRender) {
    //         this.resetView();
    //         this.firstRender = false;
    //     }
    //     this.state.selectedFunction = name;
    //     this.updateState();
    // }

    // parseSSA(ssaCode: string): SSAResult {
    //     const lines = ssaCode.split('\n');
    //     const blocks: SSABlock[] = [];
    //     const edges: SSAEdge[] = [];
    //     let currentBlock: SSABlock | null = null;
        
    //     // Regex to detect a block header. Example: "  <bb 3> :"
    //     const blockHeaderRegex = /^\s*<bb\s+(\d+)>/;
    //     // Regex to detect a goto instruction. Example: "goto <bb 5>;"
    //     const gotoRegex = /goto\s+<bb\s+(\d+)>/;
    //     // Regex to detect a phi function line. Example: "# a_1 = PHI <a_3(2), a_5(4)>"
    //     const phiRegex = /^\s*#\s*(\S+)\s*=\s*PHI\s*<(.+)>/;
        
    //     for (const line of lines) {
    //         // Check if this line marks the beginning of a new basic block.
    //         const blockMatch = line.match(blockHeaderRegex);
    //         if (blockMatch) {
    //             const blockId = blockMatch[1]; // e.g., "3" from "<bb 3>"
    //             currentBlock = { id: blockId, instructions: [] };
    //             blocks.push(currentBlock);
    //             continue;
    //         }
            
    //         // If we have not encountered a block yet, skip processing.
    //         if (!currentBlock) {
    //             continue;
    //         }
            
    //         // Save the line as part of the current block's instructions.
    //         currentBlock.instructions.push(line);
            
    //         // Look for goto instructions to add control edges.
    //         const gotoMatch = line.match(gotoRegex);
    //         if (gotoMatch) {
    //             const targetBlockId = gotoMatch[1];
    //             edges.push({
    //                 source: currentBlock.id,
    //                 target: targetBlockId,
    //                 type: "control"
    //             });
    //         }
            
    //         // Look for phi function lines to add phi edges.
    //         const phiMatch = line.match(phiRegex);
    //         if (phiMatch) {
    //             // phiMatch[1] is the defined variable (e.g., "a_1")
    //             // phiMatch[2] contains the operands, e.g. "a_3(2), a_5(4)"
    //             const phiArgs = phiMatch[2].split(',').map(arg => arg.trim());
    //             for (const arg of phiArgs) {
    //                 // Match the operand block id within parentheses.
    //                 // For example, "a_3(2)" should yield "2".
    //                 const argMatch = arg.match(/\((\d+)\)/);
    //                 if (argMatch) {
    //                     const predBlockId = argMatch[1];
    //                     edges.push({
    //                         source: predBlockId,
    //                         target: currentBlock.id,
    //                         type: "phi",
    //                         info: phiMatch[1] // the phi variable name
    //                     });
    //                 }
    //             }
    //         }
    //     }
        
    //     return { blocks, edges };
    // }
//}

    
parseSSA(ssaCode: string): AnnotatedCfgDescriptor {
    const nodes: AnnotatedNodeDescriptor[] = [];
    const edges: EdgeDescriptor[] = [];

    // Keep a set of "from-to-color" to avoid duplicate edges.
    const edgeSet = new Set<string>();
    function addEdge(from: string, to: string, color: EdgeColor) {
        const key = `${from}-${to}-${color}`;
        if (!edgeSet.has(key)) {
            edges.push({ from, to, arrows: 'to', color });
            edgeSet.add(key);
        }
    }

    const blockMap: Record<string, AnnotatedNodeDescriptor> = {};

    // Track the current block ID and node so we can accumulate code lines.
    let currentBlockId: string | null = null;
    let currentNode: AnnotatedNodeDescriptor | null = null;

    // Track if the current block ended with a goto or return, for fall-through logic.
    let currentBlockEnded = false;

    // Also remember the last block we completed, to add a fall-through edge if needed.
    let lastBlockId: string | null = null;

    // Default node dimensions
    const defaultWidth = 100;
    const defaultHeight = 50;

    // Regexes
    const blockHeaderRegex = /^\s*<bb\s+(\d+)>/;         // E.g. "<bb 2> :"
    const gotoRegex = /goto\s+<bb\s+(\d+)>/;             // E.g. "goto <bb 5>;"
    const returnRegex = /\breturn\b/;                    // Crude detection of a return

    const lines = ssaCode.split('\n');
    for (const line of lines) {
        // Check if line starts a new basic block
        const blockHeaderMatch = line.match(blockHeaderRegex);
        if (blockHeaderMatch) {
            const newBlockId = blockHeaderMatch[1];

            // If we were in a previous block that didn't end with goto/return,
            // add a fall-through edge from it to this new block.
            if (lastBlockId && !currentBlockEnded) {
                addEdge(lastBlockId, newBlockId, 'grey');
            }

            // Mark the new block as current
            currentBlockId = newBlockId;
            currentBlockEnded = false; // reset the "ended" status

            // Create or retrieve the node for this block
            if (!blockMap[currentBlockId]) {
                const node: AnnotatedNodeDescriptor = {
                    id: currentBlockId,
                    label: `Basic Block ${currentBlockId}:\n`,
                    width: defaultWidth,
                    height: defaultHeight
                };
                nodes.push(node);
                blockMap[currentBlockId] = node;
                currentNode = node;
            } else {
                currentNode = blockMap[currentBlockId];
            }

            // Update the lastBlockId to this new one
            lastBlockId = currentBlockId;
            continue;
        }

        // If we're not in a block, skip processing lines
        if (!currentBlockId || !currentNode) continue;

        // Append the current line to the node’s label
        currentNode.label += line + "\n";

        // Check for goto instructions
        const gotoMatch = line.match(gotoRegex);
        if (gotoMatch) {
            currentBlockEnded = true; // block ends with a goto
            const targetBlockId = gotoMatch[1];
            if (!blockMap[targetBlockId]) {
                const newNode: AnnotatedNodeDescriptor = {
                    id: targetBlockId,
                    label: `Basic Block ${targetBlockId}:\n`,
                    width: defaultWidth,
                    height: defaultHeight
                };
                nodes.push(newNode);
                blockMap[targetBlockId] = newNode;
            }
            addEdge(currentBlockId, targetBlockId, 'grey');
        }

        // Check for return statements
        if (returnRegex.test(line)) {
            currentBlockEnded = true; // block ends with a return
        }

    }

    return { nodes, edges };
}

    async selectFunction(name: string | null) {

        console.log(this.ssa_result)

        $('.fold').popover('dispose');
        this.blockContainer.innerHTML = '';
        this.svg.innerHTML = '';
        this.estimatedPNGSize.innerHTML = '';
    
        let ssa_graph_ds: AnnotatedCfgDescriptor = this.parseSSA(this.ssa_result)

        console.log(ssa_graph_ds)

        // Hard-coded demo graph for demonstration purposes
        // const demoGraph: AnnotatedCfgDescriptor = {
        //     nodes: [
        //         { id: 'start', label: 'Start', width: 100, height: 50 },
        //         { id: 'if', label: 'If Condition', width: 120, height: 50 },
        //         { id: 'true', label: 'True Branch', width: 100, height: 50 },
        //         { id: 'false', label: 'False Branch', width: 100, height: 50 },
        //         { id: 'end', label: 'End', width: 100, height: 50 },
        //     ],
        //     edges: [
        //         { from: 'start', to: 'if', color: 'blue', arrows: 'to' },
        //         { from: 'if', to: 'true', color: 'green', arrows: 'to' },
        //         { from: 'if', to: 'false', color: 'red', arrows: 'to' },
        //         { from: 'true', to: 'end', color: 'blue', arrows: 'to' },
        //         { from: 'false', to: 'end', color: 'blue', arrows: 'to' },
        //     ],
        // };
    
        // Instead of using this.results[name], we override with our demoGraph
        const fn = ssa_graph_ds;
    
        // Reset any block mapping from previous layouts
        this.bbMap = {};
        await this.createBasicBlocks(fn);
        this.layout = new GraphLayoutCore(fn as AnnotatedCfgDescriptor);
        this.applyLayout();
        this.drawEdges();

        // if (!name || !(name in this.results)) {
        //     return;
        // }

        // const fn2 = this.results[name];
        // this.bbMap = {};
        // await this.createBasicBlocks(fn2);
        // this.layout = new GraphLayoutCore(fn2 as AnnotatedCfgDescriptor);
        // this.applyLayout();
        // this.drawEdges();
    
        this.infoElement.innerHTML = `Layout time: ${Math.round(this.layout.layoutTime)}ms<br/>Basic blocks: ${fn.nodes.length}`;
        this.estimatedPNGSize.innerHTML = `(~${size_to_human(
            this.layout.getWidth() * this.layout.getHeight() * 4 * EST_COMPRESSION_RATIO
        )})`;
    
        if (this.state.selectedFunction !== name || this.firstRender) {
            this.resetView();
            this.firstRender = false;
        }
        this.state.selectedFunction = name;
        this.updateState();
    }
    

    resetView(resetZoom?: boolean) {
        // If we have selected a new function, or this is the first load, reset zoom and pan to the function entry
        if (this.layout.blocks.length > 0) {
            if (resetZoom) {
                this.setZoom(1);
            }
            const entry_pos = this.layout.blocks[0].coordinates;
            const container_size = this.graphContainer.getBoundingClientRect();
            const entry_size = this.bbMap[this.layout.blocks[0].data.id].getBoundingClientRect();
            this.setPan({
                // entry_size will already have the zoom factored in
                x: -(entry_pos.x * this.zoom) + container_size.width / 2 - entry_size.width / 2,
                y: entry_pos.y * this.zoom,
            });
        }
    }

    birdsEyeView() {
        if (this.layout.blocks.length > 0) {
            const fullW = this.layout.getWidth();
            const fullH = this.layout.getHeight();
            const container_size = this.graphContainer.getBoundingClientRect();
            const zoom = Math.min(container_size.width / fullW, container_size.height / fullH);
            this.setZoom(zoom);
            this.setPan({
                x: container_size.width / 2 - (fullW * zoom) / 2,
                y: container_size.height / 2 - (fullH * zoom) / 2,
            });
        }
    }

    setZoom(zoom: number, superficial?: boolean) {
        this.graphElement.style.transform = `scale(${zoom})${this.extraTransforms}`;
        if (!superficial) {
            this.zoom = zoom;
        }
    }

    setPan(p: Coordinate) {
        this.currentPosition = p;
        this.graphElement.style.left = this.currentPosition.x + 'px';
        this.graphElement.style.top = this.currentPosition.y + 'px';
    }

    createSVG() {
        this.setZoom(1, true);
        let doc = '';
        doc += '<?xml version="1.0"?>';
        doc += '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">';
        doc += `<svg ${attrs({
            xmlns: 'http://www.w3.org/2000/svg',
            version: '1.1',
            width: this.svg.style.width,
            height: this.svg.style.height,
            viewBox: this.svg.getAttribute('viewBox'),
        })}>`;
        doc += '<style>.code{font: 16px Consolas;}</style>';
        // insert the background
        doc += `<rect ${attrs({
            x: '0',
            y: '0',
            width: this.svg.style.width,
            height: this.svg.style.height,
            fill: window.getComputedStyle(this.graphContainer).backgroundColor,
        })} />`;
        // just grab the edges/arrows directly
        doc += this.svg.innerHTML;
        // the blocks we'll have to copy over
        for (const block of this.layout.blocks) {
            const block_elem = this.bbMap[block.data.id];
            const block_style = window.getComputedStyle(block_elem);
            const block_bounding_box = block_elem.getBoundingClientRect();
            doc += `<rect ${attrs({
                x: block.coordinates.x,
                y: block.coordinates.y,
                width: block.data.width,
                height: block.data.height,
                fill: block_style.background,
                stroke: block_style.borderColor,
                'stroke-width': block_style.borderWidth,
            })} />`;
            for (const [_, span] of block_elem.querySelectorAll('span[class]').entries()) {
                const text = new DOMParser().parseFromString(span.innerHTML, 'text/html').documentElement.textContent;
                if (!text || text.trim() === '') {
                    continue;
                }
                const span_style = window.getComputedStyle(span);
                const span_box = span.getBoundingClientRect();
                const top = span_box.top - block_bounding_box.top;
                const left = span_box.left - block_bounding_box.left;
                doc += `<text ${attrs({
                    x: block.coordinates.x + left,
                    y: block.coordinates.y + top + span_box.height / 2 + parseInt(block_style.paddingTop),
                    class: 'code',
                    fill: span_style.color,
                })}>${escapeHTML(text)}</text>`;
            }
        }
        doc += '</svg>';
        this.setZoom(this.zoom, true);
        return doc;
    }

    async createPNG() {
        const svg_blob = new Blob([this.createSVG()], {type: 'image/svg+xml;charset=utf-8'});
        const svg_url = URL.createObjectURL(svg_blob);
        const image = new Image();
        const width = this.layout.getWidth();
        const height = this.layout.getHeight();
        image.width = width;
        image.height = height;
        const canvas = await new Promise<HTMLCanvasElement>((resolve, reject) => {
            image.onerror = reject;
            image.onload = () => {
                const canvas = document.createElement('canvas');
                canvas.width = width;
                canvas.height = height;
                const ctx = canvas.getContext('2d');
                if (!ctx) {
                    throw new Error('Null ctx');
                }
                ctx.drawImage(image, 0, 0, width, height);
                resolve(canvas);
            };
            image.src = svg_url;
        });
        return await new Promise<Blob>((resolve, reject) => {
            canvas.toBlob(blob => {
                if (blob) {
                    resolve(blob);
                } else {
                    reject(blob);
                }
            }, 'image/png');
        });
    }

    override resize() {
        _.defer(() => {
            const topBarHeight = utils.updateAndCalcTopBarHeight(this.domRoot, this.topBar, this.hideable);
            this.graphContainer.style.width = `${unwrap(this.domRoot.width())}px`;
            this.graphContainer.style.height = `${unwrap(this.domRoot.height()) - topBarHeight}px`;
            $('.fold').popover('hide');
        });
    }

    override getCurrentState(): CfgState & PaneState {
        const state = {
            id: this.compilerInfo.compilerId,
            compilerName: this.compilerInfo.compilerName,
            editorid: this.compilerInfo.editorId,
            treeid: this.compilerInfo.treeId,
            selectedFunction: this.state.selectedFunction,
            isircfg: this.state.isircfg,
        };
        this.paneRenaming.addState(state);
        return state;
    }

    override close(): void {
        this.eventHub.unsubscribe();
        this.eventHub.emit('cfgViewClosed', this.compilerInfo.compilerId, this.state.isircfg === true);
        this.fictitiousGraphContainer.remove();
    }
}

// interface SSABlock {
//     id: string; // The block id (e.g. "2" for <bb 2>)
//     instructions: string[];
// }

// interface SSAEdge {
//     source: string; // Block id where the edge originates
//     target: string; // Block id where the edge goes
//     type: "control" | "phi";
//     info?: string; // Additional info (for phi edges, this might be the phi variable)
// }

// interface SSAResult {
//     blocks: SSABlock[];
//     edges: SSAEdge[];
// }

// function parseSSA(ssaCode: string): SSAResult {
//     const lines = ssaCode.split('\n');
//     const blocks: SSABlock[] = [];
//     const edges: SSAEdge[] = [];
//     let currentBlock: SSABlock | null = null;
    
//     // Regex to detect a block header. Example: "  <bb 3> :"
//     const blockHeaderRegex = /^\s*<bb\s+(\d+)>/;
//     // Regex to detect a goto instruction. Example: "goto <bb 5>;"
//     const gotoRegex = /goto\s+<bb\s+(\d+)>/;
//     // Regex to detect a phi function line. Example: "# a_1 = PHI <a_3(2), a_5(4)>"
//     const phiRegex = /^\s*#\s*(\S+)\s*=\s*PHI\s*<(.+)>/;
    
//     for (const line of lines) {
//         // Check if this line marks the beginning of a new basic block.
//         const blockMatch = line.match(blockHeaderRegex);
//         if (blockMatch) {
//             const blockId = blockMatch[1]; // e.g., "3" from "<bb 3>"
//             currentBlock = { id: blockId, instructions: [] };
//             blocks.push(currentBlock);
//             continue;
//         }
        
//         // If we have not encountered a block yet, skip processing.
//         if (!currentBlock) {
//             continue;
//         }
        
//         // Save the line as part of the current block's instructions.
//         currentBlock.instructions.push(line);
        
//         // Look for goto instructions to add control edges.
//         const gotoMatch = line.match(gotoRegex);
//         if (gotoMatch) {
//             const targetBlockId = gotoMatch[1];
//             edges.push({
//                 source: currentBlock.id,
//                 target: targetBlockId,
//                 type: "control"
//             });
//         }
        
//         // Look for phi function lines to add phi edges.
//         const phiMatch = line.match(phiRegex);
//         if (phiMatch) {
//             // phiMatch[1] is the defined variable (e.g., "a_1")
//             // phiMatch[2] contains the operands, e.g. "a_3(2), a_5(4)"
//             const phiArgs = phiMatch[2].split(',').map(arg => arg.trim());
//             for (const arg of phiArgs) {
//                 // Match the operand block id within parentheses.
//                 // For example, "a_3(2)" should yield "2".
//                 const argMatch = arg.match(/\((\d+)\)/);
//                 if (argMatch) {
//                     const predBlockId = argMatch[1];
//                     edges.push({
//                         source: predBlockId,
//                         target: currentBlock.id,
//                         type: "phi",
//                         info: phiMatch[1] // the phi variable name
//                     });
//                 }
//             }
//         }
//     }
    
//     return { blocks, edges };
// }

// interface SSABlock {
//     id: string; // The block id (e.g. "2" for <bb 2>)
//     instructions: string[];
// }

// interface SSAEdge {
//     source: string; // Block id where the edge originates
//     target: string; // Block id where the edge goes
//     type: "control" | "phi";
//     info?: string; // Additional info (for phi edges, this might be the phi variable)
// }

// interface SSAResult {
//     blocks: SSABlock[];
//     edges: SSAEdge[];
// }

export type EdgeColor = 'red' | 'green' | 'blue' | 'grey';

export type EdgeDescriptor = {
    from: string;
    to: string;
    arrows: string; // <- useless
    color: EdgeColor;
};

export type NodeDescriptor = {
    id: string; // typically label for the bb
    label: string; // really the source
};

export type AnnotatedNodeDescriptor = NodeDescriptor & {
    width: number; // in pixels
    height: number; // in pixels
};

type CfgDescriptor_<ND> = {
    edges: EdgeDescriptor[];
    nodes: ND[];
};

export type CfgDescriptor = CfgDescriptor_<NodeDescriptor>;
export type AnnotatedCfgDescriptor = CfgDescriptor_<AnnotatedNodeDescriptor>;