// Copyright (c) 2017, Najjar Chedy
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

import $ from 'jquery';
import * as vis from 'vis-network';
import _ from 'underscore';
import {ga} from '../analytics';
import {Toggles} from '../widgets/toggles';
import TomSelect from 'tom-select';
import {Container} from 'golden-layout';
import {CfgOptions, CfgState} from './cfg-view.interfaces';
import {Hub} from '../hub';
import {Pane} from './pane';

interface NodeInfo {
    edges: string[];
    dagEdges: string[];
    index: string;
    id: number;
    level: number;
    state: number;
    inCount: number;
}

export class Cfg extends Pane<CfgState> {
    defaultCfgOutput: object;
    llvmCfgPlaceholder: object;
    binaryModeSupport: object;
    savedPos: any;
    savedScale: any;
    needsMove: boolean;
    currentFunc: string;
    functions: Record<string, vis.Data>;
    networkOpts: vis.Options;
    cfgVisualiser: vis.Network;
    _binaryFilter: boolean;
    functionPicker: TomSelect;
    toggles: Toggles;
    toggleNavigationButton: JQuery;
    toggleNavigationTitle: string;
    togglePhysicsButton: JQuery;
    togglePhysicsTitle: string;
    options: Required<CfgOptions>;

    constructor(hub: Hub, container: Container, state: CfgState) {
        super(hub, container, state);

        this.llvmCfgPlaceholder = {
            nodes: [
                {
                    id: 0,
                    shape: 'box',
                    label: '-emit-llvm currently not supported',
                },
            ],
            edges: [],
        };
        this.binaryModeSupport = {
            nodes: [
                {
                    id: 0,
                    shape: 'box',
                    label: 'Cfg mode cannot be used when the binary filter is set',
                },
            ],
            edges: [],
        };

        this.savedPos = state.pos;
        this.savedScale = state.scale;
        this.needsMove = this.savedPos && this.savedScale;

        this.currentFunc = state.selectedFn || '';
        this.functions = {};

        this._binaryFilter = false;

        const pickerEl = this.domRoot.find('.function-picker')[0] as HTMLInputElement;
        this.functionPicker = new TomSelect(pickerEl, {
            sortField: 'name',
            valueField: 'name',
            labelField: 'name',
            searchField: ['name'],
            dropdownParent: 'body',
            plugins: ['input_autogrow'],
            onChange: (e: any) => {
                // TomSelect says it's an Event, but we receive strings
                const val = e as string;
                if (val in this.functions) {
                    const selectedFn = this.functions[val];
                    this.currentFunc = val;
                    this.showCfgResults({
                        nodes: selectedFn.nodes,
                        edges: selectedFn.edges,
                    });
                    if (selectedFn.nodes && selectedFn.nodes.length > 0) {
                        this.cfgVisualiser.selectNodes([selectedFn.nodes[0].id]);
                    }
                    this.resize();
                    this.updateState();
                }
            },
        });

        this.updateButtons();
    }

    override getInitialHTML(): string {
        return $('#cfg').html();
    }

    override registerOpeningAnalyticsEvent(): void {
        ga.proxy('send', {
            hitType: 'event',
            eventCategory: 'OpenViewPane',
            eventAction: 'Cfg',
        });
    }

    override onCompileResult(compilerId: number, compiler: any, result: any) {
        if (this.compilerInfo.compilerId === compilerId) {
            let functionNames: string[] = [];
            if (compiler.supportsCfg && !$.isEmptyObject(result.cfg)) {
                this.functions = result.cfg;
                functionNames = Object.keys(this.functions);
                if (functionNames.indexOf(this.currentFunc) === -1) {
                    this.currentFunc = functionNames[0];
                }
                const selectedFn = this.functions[this.currentFunc];
                this.showCfgResults({
                    nodes: selectedFn.nodes,
                    edges: selectedFn.edges,
                });
                if (selectedFn.nodes && selectedFn.nodes.length > 0) {
                    this.cfgVisualiser.selectNodes([selectedFn.nodes[0].id]);
                }
            } else {
                // We don't reset the current function here as we would lose the saved one if this happened at the beginning
                // (Hint: It *does* happen)
                if (!result.compilationOptions?.includes('-emit-llvm')) {
                    this.showCfgResults(this._binaryFilter ? this.binaryModeSupport : this.defaultCfgOutput);
                } else {
                    this.showCfgResults(this._binaryFilter ? this.binaryModeSupport : this.llvmCfgPlaceholder);
                }
            }

            this.functionPicker.clearOptions();
            this.functionPicker.addOption(
                functionNames.length
                    ? this.adaptStructure(functionNames)
                    : {name: 'The input does not contain functions'}
            );
            this.functionPicker.refreshOptions(false);

            this.functionPicker.clear();
            this.functionPicker.addItem(
                functionNames.length ? this.currentFunc : 'The input does not contain any function',
                true
            );
            this.updateState();
        }
    }

    override registerDynamicElements(state: CfgState) {
        this.defaultCfgOutput = {nodes: [{id: 0, shape: 'box', label: 'No Output'}], edges: []};
        // Note that this might be outdated if no functions were present when creating the link, but that's handled
        // by selectize
        this.options = {
            navigation: state.options?.navigation ?? false,
            physics: state.options?.physics ?? false,
        };

        this.networkOpts = {
            autoResize: true,
            locale: 'en',
            edges: {
                arrows: {to: {enabled: true}},
                smooth: {
                    enabled: true,
                    type: 'dynamic',
                    roundness: 1,
                },
                physics: true,
            },
            nodes: {
                font: {face: 'Consolas, "Liberation Mono", Courier, monospace', align: 'left'},
            },
            layout: {
                hierarchical: {
                    enabled: true,
                    direction: 'UD',
                    nodeSpacing: 100,
                    levelSeparation: 150,
                },
            },
            physics: {
                enabled: this.options.physics,
                hierarchicalRepulsion: {
                    nodeDistance: 160,
                },
            },
            interaction: {
                navigationButtons: this.options.navigation,
                keyboard: {
                    enabled: true,
                    speed: {x: 10, y: 10, zoom: 0.03},
                    bindToWindow: false,
                },
            },
        };

        this.cfgVisualiser = new vis.Network(
            this.domRoot.find('.graph-placeholder')[0],
            this.defaultCfgOutput,
            this.networkOpts
        );
    }

    override onCompiler(compilerId: number, compiler: any) {
        if (compilerId === this.compilerInfo.compilerId) {
            this.compilerInfo.compilerName = compiler ? compiler.name : '';
            this.updateTitle();
        }
    }

    onFiltersChange(compilerId: number, filters: any) {
        if (this.compilerInfo.compilerId === compilerId) {
            this._binaryFilter = filters.binary;
        }
    }

    override registerButtons(state: CfgState) {
        this.toggles = new Toggles(this.domRoot.find('.options'), this.options);

        this.toggleNavigationButton = this.domRoot.find('.toggle-navigation');
        this.toggleNavigationTitle = this.toggleNavigationButton.prop('title') as string;

        this.togglePhysicsButton = this.domRoot.find('.toggle-physics');
        this.togglePhysicsTitle = this.togglePhysicsButton.prop('title');

        this.topBar = this.domRoot.find('.top-bar');
    }

    override registerCallbacks() {
        this.cfgVisualiser.on('dragEnd', this.updateState.bind(this));
        this.cfgVisualiser.on('zoom', this.updateState.bind(this));

        this.eventHub.on('filtersChange', this.onFiltersChange, this);

        this.eventHub.emit('cfgViewOpened', this.compilerInfo.compilerId);
        this.eventHub.emit('requestFilters', this.compilerInfo.compilerId);
        this.eventHub.emit('requestCompiler', this.compilerInfo.compilerId);

        this.togglePhysicsButton.on('click', () => {
            this.networkOpts.physics.enabled = this.togglePhysicsButton.hasClass('active');
            // change only physics.enabled option to preserve current node locations
            this.cfgVisualiser.setOptions({
                physics: {enabled: this.networkOpts.physics.enabled},
            });
        });

        this.toggleNavigationButton.on('click', () => {
            this.networkOpts.interaction.navigationButtons = this.toggleNavigationButton.hasClass('active');
            this.cfgVisualiser.setOptions({
                interaction: {
                    navigationButtons: this.networkOpts.interaction.navigationButtons,
                },
            });
        });
        this.toggles.on('change', () => {
            this.updateButtons();
            this.updateState();
        });
    }

    updateButtons() {
        const formatButtonTitle = (button: JQuery, title: string) => {
            button.prop('title', '[' + (button.hasClass('active') ? 'ON' : 'OFF') + '] ' + title);
        };
        formatButtonTitle(this.togglePhysicsButton, this.togglePhysicsTitle);
        formatButtonTitle(this.toggleNavigationButton, this.toggleNavigationTitle);
    }

    override resize() {
        const height = (this.domRoot.height() as number) - (this.topBar.outerHeight(true) ?? 0);
        if ((this.cfgVisualiser as any).canvas !== undefined) {
            this.cfgVisualiser.setSize('100%', height.toString());
            this.cfgVisualiser.redraw();
        }
    }

    override getDefaultPaneName() {
        return 'Graph Viewer';
    }

    assignLevels(data: vis.Data) {
        const nodes: NodeInfo[] = [];
        const idToIdx: string[] = [];
        for (const i in data.nodes) {
            const node = data.nodes[i];
            idToIdx[node.id] = i;
            nodes.push({
                edges: [],
                dagEdges: [],
                index: i,
                id: node.id,
                level: 0,
                state: 0,
                inCount: 0,
            });
        }
        const isEdgeValid = (edge: vis.Edge) => edge.from && edge.to && edge.from in idToIdx && edge.to in idToIdx;
        for (const edge of data.edges as vis.Edge[]) {
            if (edge.from && edge.to && isEdgeValid(edge)) {
                nodes[idToIdx[edge.from]].edges.push(idToIdx[edge.to]);
            }
        }

        const dfs = (node: NodeInfo) => {
            // choose which edges will be back-edges
            node.state = 1;

            node.edges.forEach(targetIndex => {
                const target = nodes[targetIndex];
                if (target.state !== 1) {
                    if (target.state === 0) {
                        dfs(target);
                    }
                    node.dagEdges.push(targetIndex);
                    target.inCount += 1;
                }
            });
            node.state = 2;
        };
        const markLevels = (node: NodeInfo) => {
            node.dagEdges.forEach(targetIndex => {
                const target = nodes[targetIndex];
                target.level = Math.max(target.level, node.level + 1);
                if (--target.inCount === 0) {
                    markLevels(target);
                }
            });
        };
        nodes.forEach(node => {
            if (node.state === 0) {
                dfs(node);
                node.level = 1;
                markLevels(node);
            }
        });
        if (data.nodes) {
            for (const node of nodes) {
                data.nodes[node.index]['level'] = node.level;
            }
        }

        for (const edge of data.edges as vis.Edge[]) {
            if (edge.from && edge.to && isEdgeValid(edge)) {
                const nodeA = nodes[idToIdx[edge.from]];
                const nodeB = nodes[idToIdx[edge.to]];
                if (nodeA.level >= nodeB.level) {
                    edge.physics = false;
                } else {
                    edge.physics = true;
                    const diff = nodeB.level - nodeA.level;
                    edge.length = diff * (200 - 5 * Math.min(5, diff));
                }
            } else {
                edge.physics = false;
            }
        }
    }

    showCfgResults(data: vis.Data) {
        this.assignLevels(data);
        this.cfgVisualiser.setData(data);
        /* FIXME: This does not work.
         * It's here because I suspected that not having content in the constructor was
         * breaking the move, but it does not seem like it
         */
        if (this.needsMove) {
            this.cfgVisualiser.moveTo({
                position: this.savedPos,
                animation: false,
                scale: this.savedScale,
            });
            this.needsMove = false;
        }
    }

    override onCompilerClose(compilerId: number) {
        if (this.compilerInfo.compilerId === compilerId) {
            // We can't immediately close as an outer loop somewhere in GoldenLayout is iterating over
            // the hierarchy. We can't modify while it's being iterated over.
            this.close();
            _.defer(() => {
                this.container.close();
            }, this);
        }
    }

    override close() {
        this.eventHub.unsubscribe();
        this.eventHub.emit('cfgViewClosed', this.compilerInfo.compilerId);
        this.cfgVisualiser.destroy();
    }

    getEffectiveOptions() {
        return this.toggles.get();
    }

    override getCurrentState(): CfgState {
        return {
            ...super.getCurrentState(),
            selectedFn: this.currentFunc,
            pos: this.cfgVisualiser.getViewPosition(),
            scale: this.cfgVisualiser.getScale(),
            options: this.getEffectiveOptions(),
        };
    }

    adaptStructure(names: string[]) {
        return names.map(name => {
            return {name};
        });
    }
}
