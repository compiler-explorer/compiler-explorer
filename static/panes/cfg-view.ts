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

import * as vis from 'vis-network';
import _ from 'underscore';
import { Toggles } from '../toggles';
import { ga } from '../analytics';
import TomSelect from 'tom-select';
import { Container } from 'golden-layout';
import { CfgState } from './cfg-view.interfaces';
import { PaneRenaming } from '../pane-renaming';

interface NodeInfo {
    edges: string[],
    dagEdges: number[],
    index: string,
    id: number,
    level: number,
    state: number,
    inCount: number,
}

export class Cfg {
    container: Container;
    eventHub: any;
    domRoot: JQuery;
    defaultCfgOutput: object;
    binaryModeSupport: object;
    savedPos: any;
    savedScale: any;
    needsMove: boolean;
    currentFunc: string;
    functions: { [key: string]: any };
    networkOpts: any;
    cfgVisualiser: any;
    compilerId: number;
    _compilerName = '';
    _editorid: number;
    _binaryFilter: boolean;
    functionPicker: TomSelect;
    supportsCfg = false;
    toggles: Toggles;
    toggleNavigationButton: JQuery;
    toggleNavigationTitle: string;
    togglePhysicsButton: JQuery;
    togglePhysicsTitle: string;
    topBar: JQuery;
    paneName: string;

    constructor(hub: any, container: Container, state: CfgState) {
        this.container = container;
        this.eventHub = hub.createEventHub();
        this.domRoot = container.getElement();
        this.domRoot.html($('#cfg').html());
        this.defaultCfgOutput = { nodes: [{ id: 0, shape: 'box', label: 'No Output' }], edges: [] };
        this.binaryModeSupport = {
            nodes: [{
                id: 0,
                shape: 'box',
                label: 'Cfg mode cannot be used when the binary filter is set',
            }], edges: [],
        };
        // Note that this might be outdated if no functions were present when creating the link, but that's handled
        // by selectize
        state.options = state.options || {};
        this.savedPos = state.pos;
        this.savedScale = state.scale;
        this.needsMove = this.savedPos && this.savedScale;

        this.currentFunc = state.selectedFn || '';
        this.functions = {};

        this.networkOpts = {
            autoResize: true,
            locale: 'en',
            edges: {
                arrows: { to: { enabled: true } },
                smooth: {
                    enabled: true,
                    type: 'dynamic',
                    roundness: 1,
                },
                physics: true,
            },
            nodes: {
                font: { face: 'Consolas, "Liberation Mono", Courier, monospace', align: 'left' },
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
                enabled: !!state.options.physics,
                hierarchicalRepulsion: {
                    nodeDistance: 160,
                },
            },
            interaction: {
                navigationButtons: !!state.options.navigation,
                keyboard: {
                    enabled: true,
                    speed: { x: 10, y: 10, zoom: 0.03 },
                    bindToWindow: false,
                },
            },
        };

        this.cfgVisualiser = new vis.Network(this.domRoot.find('.graph-placeholder')[0],
            this.defaultCfgOutput, this.networkOpts);


        this.initButtons(state);

        this.compilerId = state.id;
        this._editorid = state.editorid;
        this._binaryFilter = false;

        const pickerEl = this.domRoot.find('.function-picker')[0] as HTMLInputElement;
        this.functionPicker = new TomSelect(pickerEl, {
            sortField: 'name',
            valueField: 'name',
            labelField: 'name',
            searchField: ['name'],
            dropdownParent: 'body',
            plugins: ['input_autogrow'],
            onChange: (val: string) => {
                const selectedFn = this.functions[val];
                if (selectedFn) {
                    this.currentFunc = val;
                    this.showCfgResults({
                        nodes: selectedFn.nodes,
                        edges: selectedFn.edges,
                    });
                    this.cfgVisualiser.selectNodes([selectedFn.nodes[0].id]);
                    this.resize();
                    this.saveState();
                }
            },
        } as any
            // The current version of tom-select has a very restrictive type definition
            // that forces to pass the whole options object. This is a workaround to make it type check
        );

        this.initCallbacks();
        this.updateButtons();
        this.updateTitle();
        ga.proxy('send', {
            hitType: 'event',
            eventCategory: 'OpenViewPane',
            eventAction: 'Cfg',
        });
    }

    onCompileResult(compilerId: number, compiler: any, result: any) {
        if (this.compilerId === compilerId) {
            let functionNames: string[] = [];
            if (this.supportsCfg && !$.isEmptyObject(result.cfg)) {
                this.functions = result.cfg;
                functionNames = Object.keys(this.functions);
                if (functionNames.indexOf(this.currentFunc) === -1) {
                    this.currentFunc = functionNames[0];
                }
                this.showCfgResults({
                    nodes: this.functions[this.currentFunc].nodes,
                    edges: this.functions[this.currentFunc].edges,
                });
                this.cfgVisualiser.selectNodes([this.functions[this.currentFunc].nodes[0].id]);
            } else {
                // We don't reset the current function here as we would lose the saved one if this happened at the beginning
                // (Hint: It *does* happen)
                this.showCfgResults(this._binaryFilter ? this.binaryModeSupport : this.defaultCfgOutput);
            }

            this.functionPicker.clearOptions();
            this.functionPicker.addOption(functionNames.length ?
                this.adaptStructure(functionNames) : { name: 'The input does not contain functions' });
            this.functionPicker.refreshOptions(false);

            this.functionPicker.clear();
            this.functionPicker.addItem(functionNames.length ?
                this.currentFunc : 'The input does not contain any function', true);
            this.saveState();
        }
    }
    onCompiler(compilerId: number, compiler: any) {
        if (compilerId === this.compilerId) {
            this._compilerName = compiler ? compiler.name : '';
            this.supportsCfg = compiler.supportsCfg;
            this.updateTitle();
        }
    }

    onFiltersChange(compilerId: number, filters: any) {
        if (this.compilerId === compilerId) {
            this._binaryFilter = filters.binary;
        }
    }

    initButtons(state: any) {
        this.toggles = new Toggles(this.domRoot.find('.options'), state.options);

        this.toggleNavigationButton = this.domRoot.find('.toggle-navigation');
        this.toggleNavigationTitle = this.toggleNavigationButton.prop('title') as string;

        this.togglePhysicsButton = this.domRoot.find('.toggle-physics');
        this.togglePhysicsTitle = this.togglePhysicsButton.prop('title');

        this.topBar = this.domRoot.find('.top-bar');
    }

    initCallbacks() {
        this.cfgVisualiser.on('dragEnd', this.saveState.bind(this));
        this.cfgVisualiser.on('zoom', this.saveState.bind(this));

        this.eventHub.on('compilerClose', this.onCompilerClose, this);
        this.eventHub.on('compileResult', this.onCompileResult, this);
        this.eventHub.on('compiler', this.onCompiler, this);
        this.eventHub.on('filtersChange', this.onFiltersChange, this);

        this.container.on('destroy', this.close, this);
        this.container.on('resize', this.resize, this);
        this.container.on('shown', this.resize, this);
        PaneRenaming.registerCallback(this);
        this.eventHub.emit('cfgViewOpened', this.compilerId);
        this.eventHub.emit('requestFilters', this.compilerId);
        this.eventHub.emit('requestCompiler', this.compilerId);

        this.togglePhysicsButton.on('click', () => {
            this.networkOpts.physics.enabled = this.togglePhysicsButton.hasClass('active');
            // change only physics.enabled option to preserve current node locations
            this.cfgVisualiser.setOptions({
                physics: { enabled: this.networkOpts.physics.enabled },
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
            this.saveState();
        });
    }

    updateButtons() {
        const formatButtonTitle = (button: JQuery, title: string) => {
            button.prop('title', '[' + (button.hasClass('active') ? 'ON' : 'OFF') + '] ' + title);
        };
        formatButtonTitle(this.togglePhysicsButton, this.togglePhysicsTitle);
        formatButtonTitle(this.toggleNavigationButton, this.toggleNavigationTitle);
    }

    resize() {
        if (this.cfgVisualiser.canvas) {
            const height = this.domRoot.height() as number - (this.topBar.outerHeight(true) ?? 0);
            this.cfgVisualiser.setSize('100%', height.toString());
            this.cfgVisualiser.redraw();
        }
    }

    getPaneName() {
        return `Graph Viewer ${this._compilerName}` +
            `(Editor #${this._editorid}, ` +
            `Compiler #${this.compilerId})`;
    }

    updateTitle() {
        const name = this.paneName ? this.paneName : this.getPaneName();
        this.container.setTitle(_.escape(name));
    }

    assignLevels(data: any) {
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
        const isEdgeValid = (edge: any) => edge.from in idToIdx && edge.to in idToIdx;
        data.edges.forEach((edge: any) => {
            if (isEdgeValid(edge)) {
                nodes[idToIdx[edge.from]].edges.push(idToIdx[edge.to]);
            }
        });

        const dfs = (node: any) => { // choose which edges will be back-edges
            node.state = 1;
            node.edges.forEach((targetIndex: number) => {
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
        const markLevels = (node: any) => {
            node.dagEdges.forEach((targetIndex: number) => {
                const target = nodes[targetIndex];
                target.level = Math.max(target.level, node.level + 1);
                if (--target.inCount === 0) {
                    markLevels(target);
                }
            });
        };
        nodes.forEach((node: any) => {
            if (node.state === 0) {
                dfs(node);
                node.level = 1;
                markLevels(node);
            }
        });
        nodes.forEach((node: any) => {
            data.nodes[node.index]['level'] = node.level;
        });
        data.edges.forEach((edge: any) => {
            if (isEdgeValid(edge)) {
                const nodeA = nodes[idToIdx[edge.from]];
                const nodeB = nodes[idToIdx[edge.to]];
                if (nodeA.level >= nodeB.level) {
                    edge.physics = false;
                } else {
                    edge.physics = true;
                    const diff = (nodeB.level - nodeA.level);
                    edge.length = diff * (200 - 5 * (Math.min(5, diff)));
                }
            } else {
                edge.physics = false;
            }
        });
    }

    showCfgResults(data: any) {
        this.assignLevels(data);
        this.cfgVisualiser.setData(data);
        /* FIXME: This does not work. It's here because I suspected that not having content in the constructor was
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

    onCompilerClose(compilerId: number) {
        if (this.compilerId === compilerId) {
            // We can't immediately close as an outer loop somewhere in GoldenLayout is iterating over
            // the hierarchy. We can't modify while it's being iterated over.
            this.close();
            _.defer((self: Cfg) => {
                self.container.close();
            }, this);
        }
    }

    close() {
        this.eventHub.unsubscribe();
        this.eventHub.emit('cfgViewClosed', this.compilerId);
        this.cfgVisualiser.destroy();
    }

    saveState() {
        this.container.setState(this.currentState());
    }

    getEffectiveOptions() {
        return this.toggles.get();
    }

    currentState(): CfgState {
        return {
            id: this.compilerId,
            editorid: this._editorid,
            selectedFn: this.currentFunc,
            pos: this.cfgVisualiser.getViewPosition(),
            scale: this.cfgVisualiser.getScale(),
            options: this.getEffectiveOptions(),
        };
    }

    adaptStructure(names: string[]) {
        return _.map(names, name => {
            return { name };
        });
    }
}

